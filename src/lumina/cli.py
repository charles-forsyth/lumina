import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import google.auth
import google.auth.exceptions
from rich.console import Console
from rich.logging import RichHandler

from .config import settings
from .core import ImageGenerator

console = Console()
logger = logging.getLogger("lumina")


def get_project_id(project_id_arg: Optional[str]) -> Optional[str]:
    if project_id_arg:
        return project_id_arg
    if settings.project_id:
        return settings.project_id

    try:
        _, project = google.auth.default()
        if project:
            return project
    except google.auth.exceptions.DefaultCredentialsError:
        pass
    except Exception as e:
        logger.debug(f"Unexpected error during auth discovery: {e}")
        pass
    return None


def run_init():
    """
    Initialize the application configuration.
    Creates a secure .env file in ~/.config/lumina/
    """
    config_dir = Path.home() / ".config" / "lumina"
    env_file = config_dir / ".env"

    if env_file.exists():
        console.print(f"[yellow]Configuration already exists at {env_file}[/yellow]")
        return

    try:
        config_dir.mkdir(parents=True, exist_ok=True)

        default_config = (
            "# Secure Configuration for Lumina\n"
            "# Permissions set to 600 (User Read/Write Only)\n\n"
            "# AUTHENTICATION (Choose One)\n"
            "API_KEY=\n"
            "PROJECT_ID=\n\n"
            "LOCATION=us-central1\n"
            "MODEL_NAME=gemini-3-pro-image-preview\n"
            f"OUTPUT_DIR={Path.home() / 'Pictures' / 'Lumina_Generated'}\n"
            "ASPECT_RATIO=1:1\n"
            "IMAGE_SIZE=1K\n"
            "SAFETY_FILTER_LEVEL=BLOCK_ONLY_HIGH\n"
            "PERSON_GENERATION=allow_all\n"
            "ADD_WATERMARK=true\n"
        )

        env_file.write_text(default_config)
        env_file.chmod(0o600)

        console.print(f"[green]Initialized configuration at {env_file}[/green]")
        console.print("Please edit this file to add your API_KEY or PROJECT_ID.")

    except Exception as e:
        console.print(f"[red]Failed to initialize configuration: {e}[/red]")
        sys.exit(1)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lumina",
        description="Lumina: Modernized Gemini Image Generation CLI",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
EXAMPLES:

1. Basic Generation:
   lumina -p "A majestic lion on Mars"

2. Styles & Variations:
   lumina -p "A city street" \
       --style "Cyberpunk" --style "Neon" \
       --variation "Rainy" --variation "Cinematic Lighting"

3. Image Editing (Inpainting/Modification):
   lumina -p "Add a red hat to the cat" -i cat.png

4. High Quality (4K, 16:9):
   lumina -p "Space battle fleet" \
       --aspect-ratio "16:9" --image-size "4K"

5. Piping from Stdin:
   echo "A cyberpunk street food vendor" | lumina

6. Strict Count (Nano Banana):
   lumina -p "A robot" --count 4
""",
    )

    # Global/Common Args
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging."
    )

    # Subparsers
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Init Command
    subparsers.add_parser("init", help="Initialize configuration.")

    # Generation Arguments (Top Level)
    parser.add_argument(
        "--prompt",
        "-p",
        help="The text prompt to generate an image from. Required unless piping stdin.",
    )
    parser.add_argument(
        "--image",
        "-i",
        action="append",
        type=Path,
        help=(
            "Reference image(s) for editing/composition. "
            "Can be specified multiple times."
        ),
    )
    parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=1,
        help="Number of images to generate (Nano Banana strict). Default: 1",
    )
    parser.add_argument(
        "--style",
        action="append",
        help="Artistic styles to apply (e.g., 'Cyberpunk', 'Watercolor').",
    )
    parser.add_argument(
        "--variation",
        action="append",
        help="Visual variations to apply (e.g., 'Cinematic Lighting').",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        help="Directory to save output. Defaults to ~/Pictures/Lumina_Generated.",
    )
    parser.add_argument(
        "--filename",
        "-f",
        help="Specific filename for output image (e.g., 'result.png').",
    )
    parser.add_argument("--api-key", help="Google AI Studio API Key (overrides env).")
    parser.add_argument("--project-id", help="GCP Project ID (overrides env).")
    parser.add_argument("--location", help="GCP Location (default: us-central1).")
    parser.add_argument(
        "--model-name",
        help="Vertex AI Model (default: gemini-3-pro-image-preview).",
    )
    parser.add_argument(
        "--aspect-ratio",
        help="Aspect ratio. Options: '1:1', '16:9', '9:16', '4:3', '3:4'.",
    )
    parser.add_argument(
        "--image-size", help="Image resolution. Options: '1K', '2K', '4K'."
    )
    parser.add_argument(
        "--negative-prompt",
        help="Items to exclude from the image (e.g., 'blur, distortion').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible results.",
    )

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        force=True,
    )

    # Suppress noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("google.genai").setLevel(logging.WARNING)
    logging.getLogger("google.auth").setLevel(logging.WARNING)
    logging.getLogger("google.api_core").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    if args.command == "init":
        run_init()
        return

    # Handle Stdin for Prompt
    prompt = args.prompt
    if not prompt:
        # Check if there is data in stdin (and it's not a TTY)
        if not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
            if prompt:
                logger.debug("Reading prompt from stdin...")

    # If still no prompt, show error
    if not prompt:
        console.print("[yellow]No command or prompt provided.[/yellow]")
        console.print(
            "Use [bold]--prompt[/bold] or pipe text via stdin.\n"
            "Run [bold]lumina --help[/bold] for usage."
        )
        sys.exit(0)

    # --- Generation Logic ---

    # Resolve Configuration
    resolved_api_key = args.api_key or settings.api_key
    resolved_project_id = get_project_id(args.project_id)
    resolved_location = args.location or settings.location
    resolved_model_name = args.model_name or settings.model_name
    resolved_output_dir = args.output_dir or settings.output_dir
    resolved_aspect_ratio = args.aspect_ratio or settings.aspect_ratio
    resolved_image_size = args.image_size or settings.image_size

    # Validate Auth
    if not resolved_api_key and not resolved_project_id:
        console.print(
            "[bold red]Authentication missing.[/bold red] Provide either "
            "--api-key (or API_KEY in env)"
            "\nOR --project-id (or PROJECT_ID/ADC)."
        )
        sys.exit(1)

    # "Nano Banana" Prompt Augmentation
    full_prompt = prompt
    if args.style:
        style_text = ", ".join(args.style)
        full_prompt += f", in the style of {style_text}"
    if args.variation:
        var_text = ", ".join(args.variation)
        full_prompt += f", with variations in {var_text}"

    if resolved_api_key:
        logger.debug("Using Authentication: API Key")
    else:
        logger.debug(
            f"Using Authentication: Vertex AI (Project: {resolved_project_id})"
        )

    logger.debug(f"Model: {resolved_model_name}")
    logger.debug(f"Full Prompt: {full_prompt}")

    generator = ImageGenerator(
        model_name=resolved_model_name,
        api_key=resolved_api_key,
        project_id=resolved_project_id,
        location=resolved_location,
    )

    try:
        files = generator.generate(
            prompt=full_prompt,
            reference_images=args.image,
            count=args.count,
            aspect_ratio=resolved_aspect_ratio,
            image_size=resolved_image_size,
            negative_prompt=args.negative_prompt,
            person_generation=settings.person_generation,
            safety_filter_level=settings.safety_filter_level,
            add_watermark=settings.add_watermark,
            seed=args.seed,
            output_dir=resolved_output_dir,
            filename=args.filename,
        )
        console.print(
            f"[bold green]Successfully generated {len(files)} images.[/bold green]"
        )
        for f in files:
            console.print(f"  - {f}")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
