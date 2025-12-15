import logging
import os
from pathlib import Path
from typing import List, Optional, Union

from google import genai
from PIL import Image

from .utils import ensure_directory, sanitize_filename

logger = logging.getLogger(__name__)


class ImageGenerator:
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        location: str = "us-central1"
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.project_id = project_id
        self.location = location
        self._client = None

        # Setup Environment for Vertex AI if strictly needed (legacy compat)
        if not self.api_key and self.project_id:
            os.environ["GOOGLE_CLOUD_PROJECT"] = self.project_id
            os.environ["GOOGLE_CLOUD_LOCATION"] = self.location
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

    @property
    def client(self):
        if not self._client:
            try:
                if self.api_key:
                    logger.info("Initializing Client with API Key (Google AI Studio)")
                    self._client = genai.Client(api_key=self.api_key, vertexai=False)
                else:
                    logger.info("Initializing Client with Vertex AI (GCP)")
                    if not self.project_id:
                        raise ValueError("Project ID is required for Vertex AI.")
                    self._client = genai.Client(
                        vertexai=True,
                        project=self.project_id,
                        location=self.location,
                    )
            except Exception as e:
                logger.error(f"Failed to initialize GenAI Client: {e}")
                raise
        return self._client

    def _resolve_safety_threshold(self, level: str) -> str:
        """
        Maps user-friendly or legacy safety levels to valid Vertex AI/Gemini enums.
        """
        level = level.upper()
        mapping = {
            "BLOCK_SOME": "BLOCK_ONLY_HIGH",  # Legacy/Default fix
            "BLOCK_MOST": "BLOCK_LOW_AND_ABOVE",
            "BLOCK_FEW": "BLOCK_ONLY_HIGH",
            "BLOCK_NONE": "BLOCK_NONE",
        }
        return mapping.get(level, level)

    def generate(
        self,
        prompt: str,
        reference_images: Optional[List[Path]] = None,
        count: int = 1,
        aspect_ratio: str = "1:1",
        image_size: str = "1K",
        negative_prompt: Optional[str] = None,
        person_generation: str = "allow_all",
        safety_filter_level: str = "BLOCK_ONLY_HIGH",
        add_watermark: bool = True,
        seed: Optional[int] = None,
        output_dir: Path = Path("."),
    ) -> List[Path]:

        # 1. Handle Negative Prompt (Append logic)
        final_prompt = prompt
        if negative_prompt:
            final_prompt += f" \n(Exclude: {negative_prompt})"
            logger.info(f"Appended negative prompt: {negative_prompt}")

        # 2. Handle Person Generation (Prompt guidance)
        if person_generation == "dont_allow":
            final_prompt += " \n(Do not include people in this image.)"
        elif person_generation == "allow_adult":
            final_prompt += " \n(If people are included, they must be adults.)"

        logger.info(f"Generating image with prompt: '{final_prompt}'")
        logger.info(
            f"Model: {self.model_name} | Size: {image_size} | Ratio: {aspect_ratio}"
        )

        # Prepare Content (Text + Images)
        contents: List[Union[str, Image.Image]] = [final_prompt]
        
        if reference_images:
            logger.info(f"Using {len(reference_images)} reference image(s).")
            for img_path in reference_images:
                if not img_path.exists():
                    logger.warning(f"Reference image not found: {img_path}")
                    continue
                try:
                    img = Image.open(img_path)
                    contents.append(img)
                except Exception as e:
                    logger.error(f"Failed to load image {img_path}: {e}")

        # Ensure output directory exists
        ensure_directory(output_dir)

        saved_files = []
        
        valid_threshold = self._resolve_safety_threshold(safety_filter_level)
        logger.debug(
            f"Resolved Safety Threshold: {safety_filter_level} -> {valid_threshold}"
        )

        # Gemini 3 Pro generates one image per request typically
        for i in range(count):
            try:
                # Use dict for config to avoid import issues with specific types
                config = {
                    "response_modalities": ["TEXT", "IMAGE"],
                    "image_config": {
                        "aspect_ratio": aspect_ratio,
                        "image_size": image_size,
                    },
                    "safety_settings": [
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": valid_threshold,
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "threshold": valid_threshold,
                        },
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "threshold": valid_threshold,
                        },
                         {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "threshold": valid_threshold,
                        }
                    ],
                }
                
                # Attempt to inject seed if provided (Note: might not be honored by all models)
                if seed is not None:
                    # 'seed' usually goes into generation_config, but image models structure varies.
                    # We'll try adding it to the top level config dict if the SDK supports flat mapping,
                    # or creating a 'generation_config' key.
                    # Safest for now is to try top-level or omit if unsure, but we'll try:
                    # config['seed'] = seed  <-- SDK might complain if invalid field.
                    pass 

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )

                # Process Response
                for part in response.parts:
                    if part.text:
                        logger.debug(f"Thinking/Text: {part.text[:200]}...")

                    if part.inline_data or (
                        hasattr(part, "as_image") and part.as_image()
                    ):
                        img = part.as_image()

                        filename = sanitize_filename(prompt) # Use original prompt for filename
                        # Uniqueify if multiple counts or parts
                        if count > 1 or len(response.parts) > 1:
                            name_stem = Path(filename).stem
                            suffix = Path(filename).suffix
                            s_img = sanitize_filename("img")[-6:]
                            filename = f"{name_stem}_{i}_{s_img}{suffix}"

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