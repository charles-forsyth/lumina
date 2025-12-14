import logging
import os
from pathlib import Path
from typing import List, Optional

from google import genai
from google.genai import types

from .config import settings
from .utils import ensure_directory, sanitize_filename

logger = logging.getLogger(__name__)


class ImageGenerator:
    def __init__(self, project_id: str, location: str, model_name: str):
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self._client = None
        
        # Ensure environment variables are set for Vertex AI if needed
        if project_id:
            os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
        if location:
            os.environ["GOOGLE_CLOUD_LOCATION"] = location
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

    @property
    def client(self):
        if not self._client:
            try:
                self._client = genai.Client(
                    vertexai=True, 
                    project=self.project_id, 
                    location=self.location
                )
            except Exception as e:
                logger.error(f"Failed to initialize GenAI Client: {e}")
                raise
        return self._client

    def generate(
        self,
        prompt: str,
        count: int = 1,
        aspect_ratio: str = "1:1",
        image_size: str = "1K",
        negative_prompt: Optional[str] = None,
        person_generation: str = "allow_all",
        safety_filter_level: str = "block_some",
        add_watermark: bool = True,
        seed: Optional[int] = None,
        output_dir: Path = Path("."),
    ) -> List[Path]:

        logger.info(f"Generating image with prompt: '{prompt}'")
        logger.info(f"Model: {self.model_name} | Size: {image_size} | Ratio: {aspect_ratio}")

        # Ensure output directory exists
        ensure_directory(output_dir)
        
        saved_files = []

        # Gemini 3 Pro generates one image per request typically, or we loop for count
        # The prompt loop handles the count requirement strictly
        for i in range(count):
            try:
                # Configure the generation request
                # Note: negative_prompt, person_generation, etc. might need different handling 
                # or might not be fully supported in the preview config yet same as legacy.
                # We focus on the core image config.
                
                config = types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"], # Request both for "Thinking"
                    image_config=types.ImageConfig(
                        aspect_ratio=aspect_ratio,
                        image_size=image_size
                    ),
                    safety_settings=[
                        types.SafetySetting(
                            category="HARM_CATEGORY_HARASSMENT",
                            threshold=safety_filter_level.upper()
                        ),
                        # Add other categories as needed, mapping the simple string to full enum
                    ] if safety_filter_level else None
                )

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[prompt],
                    config=config,
                )

                # Process Response
                for part in response.parts:
                    if part.text:
                        logger.debug(f"Thinking/Text: {part.text[:200]}...")
                    
                    if part.inline_data or (hasattr(part, 'as_image') and part.as_image()):
                        img = part.as_image()
                        
                        filename = sanitize_filename(prompt)
                        # Uniqueify if multiple counts or parts
                        if count > 1 or len(response.parts) > 1:
                            name_stem = Path(filename).stem
                            suffix = Path(filename).suffix
                            filename = f"{name_stem}_{i}_{sanitize_filename('img')[-6:]}{suffix}"
                        
                        output_path = output_dir / filename
                        img.save(output_path)
                        saved_files.append(output_path)
                        logger.info(f"Saved: {output_path}")

            except Exception as e:
                logger.error(f"Image generation failed for iteration {i+1}: {e}")
                # We continue to try other iterations if one fails, or could raise
                if i == count - 1 and not saved_files:
                     raise

        return saved_files