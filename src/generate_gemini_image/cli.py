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

app = typer.Typer(help="Modernized Gemini Image Generation CLI")
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