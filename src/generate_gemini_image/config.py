from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    project_id: Optional[str] = Field(None, description="Google Cloud Project ID")
    location: str = Field("us-central1", description="Google Cloud Location")
    model_name: str = Field(
        "gemini-2.5-flash-image", description="Vertex AI Model Name"
    )
    output_dir: Path = Field(Path("."), description="Output directory for images")
    
    # Image Generation Defaults
    aspect_ratio: str = Field("1:1", description="Default aspect ratio")
    safety_filter_level: str = Field("block_some", description="Safety filter level")
    person_generation: str = Field("allow_all", description="Person generation setting")
    add_watermark: bool = Field(True, description="Add invisible watermark")

    model_config = SettingsConfigDict(
        env_file=[
            str(Path.home() / ".config" / "generate-gemini-image" / ".env"),
            ".env"
        ],
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()