# Modernized Gemini Image Generation CLI

A robust, secure, and distribution-ready CLI tool for generating images using Google's **Gemini 3 Pro** (aka Nano Banana Pro) models.

## Features

*   **Gemini 3 Pro**: Uses the latest `gemini-3-pro-image-preview` model via `google-genai` SDK.
*   **Dual Auth**: Support for both **API Key** (Google AI Studio) and **Vertex AI** (GCP).
*   **Flexible CLI**: Support for named arguments (`--prompt`), piping from stdin (`|`), and rich output.
*   **Nano Banana Enhancements**: Strict count adherence, style/variation prompt augmentation.
*   **Secure Configuration**: Uses `pydantic-settings` to load credentials from environment variables (`.env` support).
*   **Reproducible**: Managed with `uv` and `pyproject.toml`.

## Installation

### Using `uv` (Recommended)

You can install this tool directly from the repository:

```bash
uv tool install git+https://github.com/charles-forsyth/generate-gemini-image.git
```

To update later:

```bash
uv tool update generate-gemini-image
```

### Initial Setup

After installation, run the initialization command to create your secure configuration file:

```bash
generate-gemini-image init
```

This will create `~/.config/generate-gemini-image/.env`. **Edit this file to set your authentication: **

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
generate-gemini-image --prompt "A futuristic city on Mars"
# Short flag
generate-gemini-image -p "A futuristic city on Mars"
```

### 2. Piping from Stdin
Great for chaining commands or reading from files.
```bash
echo "A cyberpunk street food vendor" | generate-gemini-image
```
```bash
cat prompt.txt | generate-gemini-image
```

### 3. Multiple Images & Strict Count
Generate exactly 4 images using the Nano Banana strict count feature.
```bash
generate-gemini-image -p "A cute robot" --count 4
```

### 4. Styles and Variations
Apply artistic styles and variations to your prompt.
```bash
generate-gemini-image -p "A portrait of a wizard" \
    --style "oil painting" --style "classical" \
    --variation "dramatic lighting" --variation "moody"
```

### 5. High Resolution & Aspect Ratio
Generate a 4K, 16:9 cinematic image.
```bash
generate-gemini-image -p "Space battle fleet" \
    --aspect-ratio "16:9" \
    --image-size "4K"
```

### 6. Override Configuration
Override default settings for a single run.
```bash
generate-gemini-image -p "Test" \
    --model-name "gemini-2.5-flash-image" \
    --output-dir "./my-images"
```

## Configuration Reference (`.env`)

| Setting | Description | Default |
| :--- | :--- | :--- |
| `API_KEY` | Google AI Studio Key | None |
| `PROJECT_ID` | GCP Project ID | None |
| `MODEL_NAME` | Model ID | `gemini-3-pro-image-preview` |
| `OUTPUT_DIR` | Output folder | `~/Pictures/Gemini_Generated` |
| `ASPECT_RATIO` | Default shape | `1:1` |
| `IMAGE_SIZE` | Resolution (`1K`, `2K`, `4K`) | `1K` |
| `SAFETY_FILTER_LEVEL` | Content filtering (`BLOCK_NONE`, `BLOCK_ONLY_HIGH`) | `BLOCK_ONLY_HIGH` |

## License

Private / Internal Use.
