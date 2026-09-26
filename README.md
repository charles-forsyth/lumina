# Lumina: Modernized Gemini Image Generation CLI

**Lumina** is a robust, secure, and distribution-ready CLI tool for generating and editing images using Google's **Gemini 3 Pro** (aka Nano Banana Pro) models.

## Features

*   **Nano Banana Pro**: Uses the GA `gemini-3-pro-image` model via the `google-genai` SDK (Nano Banana 2, `gemini-3.1-flash-image`, is a cheaper option via `--model-name`).
*   **Image Editing**: Modify existing images or create composites using reference images (`--image`).
*   **Dual Auth**: Support for both **API Key** (Google AI Studio) and **Vertex AI** (GCP).
*   **Flexible CLI**: Support for named arguments (`--prompt`), piping from stdin (`|`), and rich output.
*   **Nano Banana Enhancements**: Strict count adherence, style/variation prompt augmentation.
*   **Secure Configuration**: Uses `pydantic-settings` to load credentials from environment variables (`.env` support).
*   **Reproducible**: Managed with `uv` and `pyproject.toml`.

## Installation

### Using `uv` (Recommended)

You can install this tool directly from the repository:

```bash
uv tool install git+https://github.com/charles-forsyth/lumina.git
```

To update later:

```bash
uv tool update lumina
```

### Initial Setup

After installation, run the initialization command to create your secure configuration file:

```bash
lumina init
```

This will create `~/.config/lumina/.env`. **Edit this file to set your authentication:**

**Option A: API Key (Simpler)**
Get a key from [Google AI Studio](https://aistudio.google.com/).
```env
API_KEY=your_api_key_here
```

**Option B: Vertex AI (Enterprise)**
Use your Google Cloud Project.
```env
PROJECT_ID=your_gcp_project_id
```

## Usage Examples

### 1. Basic Generation
Generate a single image with default settings.
```bash
lumina -p "A futuristic city on Mars"
```

### 2. Styles and Variations
Apply artistic styles and variations to your prompt.
```bash
lumina -p "A portrait of a wizard" \
    --style "Oil Painting" --style "Classical" \
    --variation "Dramatic Lighting" --variation "Moody"
```
*   **Styles**: Cyberpunk, Watercolor, Sketch, Anime, 3D Render, Vintage, Minimalist...
*   **Variations**: Cinematic Lighting, Golden Hour, High Contrast, Pastel Colors, Dark Fantasy...

### 3. Image Editing (Inpainting)
Modify an existing image by providing it as context. The model will interpret your prompt as an edit instruction.
```bash
lumina -p "Add sunglasses to the cat" -i cat.png
```

### 4. Multi-Image Composition
Combine elements from multiple images.
```bash
lumina -p "Combine the style of image 1 with the subject of image 2" \
    -i style_ref.png -i subject_ref.png
```

### 5. High Quality (4K)
Generate a high-resolution, wide-format image.
```bash
lumina -p "Space battle fleet" \
    --aspect-ratio "16:9" \
    --image-size "4K"
```

### 6. Strict Count (Nano Banana)
Generate exactly `N` images (by running the request multiple times if needed).
```bash
lumina -p "A cute robot" --count 4
```

### 7. Piping from Stdin
Great for chaining commands or reading from files.
```bash
echo "A cyberpunk street food vendor" | lumina
```
```bash
cat prompt.txt | lumina
```

### 8. Specific Filename
Save the output to a specific file instead of auto-generating a name.
```bash
lumina -p "Logo" --filename "company_logo.png"
```

## Configuration Reference (`.env`)

| Setting | Description | Default |
| :--- | :--- | :--- |
| `API_KEY` | Google AI Studio Key | None |
| `PROJECT_ID` | GCP Project ID | None |
| `MODEL_NAME` | Model ID | `gemini-3-pro-image` |
| `OUTPUT_DIR` | Output folder | `~/Pictures/Lumina_Generated` |
| `ASPECT_RATIO` | Default shape | `1:1` |
| `IMAGE_SIZE` | Resolution (`1K`, `2K`, `4K`) | `1K` |
| `SAFETY_FILTER_LEVEL` | Content filtering (`BLOCK_NONE`, `BLOCK_ONLY_HIGH`) | `BLOCK_ONLY_HIGH` |

## License

Private / Internal Use.