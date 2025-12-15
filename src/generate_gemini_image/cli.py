import logging
import sys
from pathlib import Path
from typing import List, Optional

import google.auth
import typer
from rich.console import Console
from rich.logging import RichHandler

from .config import settings
from .core import ImageGenerator

app = typer.Typer(
    help="Modernized Gemini Image Generation CLI",
    context_settings={"help_option_names": ["-h", "--help"]},
)
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("generate-gemini-image")


def get_project_id(project_id_arg: Optional[str]) -> Optional[str]:
    if project_id_arg:
        return project_id_arg
    if settings.project_id:
        return settings.project_id

    try:
        _, project = google.auth.default()
        if project:
            return project
    except Exception:
        pass
    return None


@app.command()
def init():
    """
    Initialize the application configuration.
    Creates a secure .env file in ~/.config/generate-gemini-image/
    """
    config_dir = Path.home() / ".config" / "generate-gemini-image"
    env_file = config_dir / ".env"

    if env_file.exists():
        console.print(f"[yellow]Configuration already exists at {env_file}[/yellow]")
        return

    try:
        config_dir.mkdir(parents=True, exist_ok=True)
        
        default_config = (
            "# Secure Configuration for Generate Gemini Image\n"
            "# Permissions set to 600 (User Read/Write Only)\n\n"
            "# AUTHENTICATION (Choose One)\n"
            "API_KEY=\n"
            "PROJECT_ID=\n\n"
            "LOCATION=us-central1\n"
            "MODEL_NAME=gemini-3-pro-image-preview\n"
            f"OUTPUT_DIR={Path.home() / 'Pictures' / 'Gemini_Generated'}\n"
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
        raise typer.Exit(code=1) from e


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    prompt: Optional[str] = typer.Option(
        None,
        "--prompt",
        "-p",
        help="The text prompt to generate an image from. Required unless piping stdin.",
    ),
    image: Optional[List[Path]] = typer.Option(
        None,
        "--image",
        "-i",
        help=(
            "Reference image(s) for editing/composition. "
            "Can be specified multiple times."
        ),
    ),
    count: int = typer.Option(
        1,
        "--count",
        "-n",
        help="Number of images to generate (Nano Banana strict).",
    ),
    styles: Optional[List[str]] = typer.Option(
        None,
        "--style",
        help=(
            "Artistic styles to apply. Examples: 'Cyberpunk', 'Watercolor', "
            "'Oil Painting', 'Photorealistic', 'Anime', 'Sketch', 'Vintage'."
        ),
    ),
    variations: Optional[List[str]] = typer.Option(
        None,
        "--variation",
        help=(
            "Visual variations to apply. Examples: 'Cinematic Lighting', 'Moody', "
            "'Golden Hour', 'Minimalist', 'High Contrast', 'Pastel Colors'."
        ),
    ),
    output_dir: Path = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory to save output. Defaults to ~/Pictures/Gemini_Generated.",
    ),
    filename: Optional[str] = typer.Option(
        None,
        "--filename",
        "-f",
        help="Specific filename for output image (e.g., 'result.png').",
    ),
    api_key: str = typer.Option(
        None, "--api-key", help="Google AI Studio API Key (overrides env)."
    ),
    project_id: str = typer.Option(
        None, "--project-id", help="GCP Project ID (overrides env)."
    ),
    location: str = typer.Option(
        None, "--location", help="GCP Location (default: us-central1)."
    ),
    model_name: str = typer.Option(
        None,
        "--model-name",
        help="Vertex AI Model (default: gemini-3-pro-image-preview).",
    ),
    aspect_ratio: str = typer.Option(
        None,
        help="Aspect ratio. Options: '1:1', '16:9', '9:16', '4:3', '3:4'.",
    ),
    image_size: str = typer.Option(
        None,
        help="Image resolution. Options: '1K', '2K', '4K'.",
    ),
    negative_prompt: str = typer.Option(
        None,
        help="Items to exclude from the image (e.g., 'blur, distortion, people').",
    ),
    seed: int = typer.Option(
        None,
        help="Random seed for reproducible results (if supported by model).",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging."
    ),
):
    """
    Generate images using Gemini 3 Pro (Nano Banana Pro).
    
    This tool allows you to create high-quality images from text prompts,
    edit existing images, and combine multiple images using Google's generative AI.

    EXAMPLES:
    
    1. Basic Generation:
       $ generate-gemini-image -p "A majestic lion on Mars"

    2. Styles & Variations:
       $ generate-gemini-image -p "A city street" \
           --style "Cyberpunk" --style "Neon" \
           --variation "Rainy" --variation "Cinematic Lighting"

    3. Image Editing (Inpainting/Modification):
       $ generate-gemini-image -p "Add a red hat to the cat" -i cat.png

    4. High Quality (4K, 16:9):
       $ generate-gemini-image -p "Space battle fleet" \
           --aspect-ratio "16:9" --image-size "4K"

    5. Piping from Stdin:
       $ echo "A cyberpunk street food vendor" | generate-gemini-image
    """
    # If a subcommand (like 'init') is invoked, just return and let it run.
    if ctx.invoked_subcommand is not None:
        return

    # Handle Stdin for Prompt
    if not prompt:
        # Check if there is data in stdin (and it's not a TTY)
        if not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
            if prompt:
                logger.info("Reading prompt from stdin...")

    # If still no prompt, show error
    if not prompt:
        console.print("[yellow]No command or prompt provided.[/yellow]")
        console.print(
            "Use [bold]--prompt[/bold] or pipe text via stdin.\n"
            "Run [bold]generate-gemini-image --help[/bold] for usage."
        )
        raise typer.Exit(code=0)

    # --- Generation Logic ---
    if verbose:
        logger.setLevel(logging.DEBUG)

    # Resolve Configuration
    resolved_api_key = api_key or settings.api_key
    resolved_project_id = get_project_id(project_id)
    resolved_location = location or settings.location
    resolved_model_name = model_name or settings.model_name
    resolved_output_dir = output_dir or settings.output_dir
    resolved_aspect_ratio = aspect_ratio or settings.aspect_ratio
    resolved_image_size = image_size or settings.image_size
    
    # Validate Auth
    if not resolved_api_key and not resolved_project_id:
         console.print(
            "[bold red]Authentication missing.[/bold red] Provide either "
            "--api-key (or API_KEY in env)\n"
            "OR --project-id (or PROJECT_ID/ADC)."
        )
         raise typer.Exit(code=1)

    # "Nano Banana" Prompt Augmentation
    full_prompt = prompt
    if styles:
        style_text = ", ".join(styles)
        full_prompt += f", in the style of {style_text}"
    if variations:
        var_text = ", ".join(variations)
        full_prompt += f", with variations in {var_text}"

    if resolved_api_key:
        logger.info("Using Authentication: API Key")
    else:
        logger.info(f"Using Authentication: Vertex AI (Project: {resolved_project_id})")
        
    logger.info(f"Model: {resolved_model_name}")
    logger.info(f"Full Prompt: {full_prompt}")

    generator = ImageGenerator(
        model_name=resolved_model_name,
        api_key=resolved_api_key,
        project_id=resolved_project_id,
        location=resolved_location,
    )

    try:
        files = generator.generate(
            prompt=full_prompt,
            reference_images=image,
            count=count,
            aspect_ratio=resolved_aspect_ratio,
            image_size=resolved_image_size,
            negative_prompt=negative_prompt,
            person_generation=settings.person_generation,
            safety_filter_level=settings.safety_filter_level,
            add_watermark=settings.add_watermark,
            seed=seed,
            output_dir=resolved_output_dir,
            filename=filename,
        )
        console.print(
            f"[bold green]Successfully generated {len(files)} images.[/bold green]"
        )
        for f in files:
            console.print(f"  - {f}")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
