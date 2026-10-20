# Lumina: Modernized Gemini Image Generation CLI

**Lumina** (`v0.4.1`) is a CLI tool for generating and editing images using Google's **Gemini 3 Pro Image** (`gemini-3-pro-image`, Nano Banana Pro) and **Gemini 3.1 Flash Image** (`gemini-3.1-flash-image`, Nano Banana 2) models.

See [SPEC.md](SPEC.md) for the full system specification, architecture, configuration schema, and requirement traceability.

## Features

* **Nano Banana Pro & Flash:** Defaults to GA `gemini-3-pro-image` via the `google-genai` SDK; pass `--model-name gemini-3.1-flash-image` for faster, lower-cost generation.
* **Image Editing & Multi-Image Composition:** Modify existing images or combine multiple reference images (`-i ref1.png -i ref2.png`).
* **Prompt Optimization (`--optimize`):** Rewrite terse prompts or raw data into detailed visual descriptions using `gemini-3.8-flash`.
* **Dual Authentication:** Supports a dedicated Gemini API key (`API_KEY` in `~/.config/lumina/.env`, restricted to `generativelanguage.googleapis.com`) or Vertex AI (`PROJECT_ID` + ADC) without probing GCE metadata when `API_KEY` is configured.
* **Strict Count Adherence (`--count N`):** Loops automatically until all `N` requested images are generated.

## Installation

### Using `uv` (Recommended)

```bash
uv tool install --python 3.12 git+https://github.com/charles-forsyth/lumina.git
```

To update later:

```bash
uv tool upgrade lumina
```

### Initial Setup

Create your secure configuration file (`0600` permissions):

```bash
lumina init
```

Edit `~/.config/lumina/.env` to configure authentication:

**Option A: Gemini API Key (Recommended)**
Create a key restricted to `generativelanguage.googleapis.com`:
```env
API_KEY=your_api_key_here
```

**Option B: Vertex AI (GCP ADC)**
```env
PROJECT_ID=your_gcp_project_id
LOCATION=us-central1
```

## Usage Examples

### 1. Basic Generation
```bash
lumina -p "A futuristic observatory on a mountain ridge at dusk"
```

### 2. Styles, Variations, and Prompt Optimization
```bash
lumina -p "A portrait of an owl" \
    --style "Oil Painting" --style "Classical" \
    --variation "Dramatic Lighting" \
    --optimize
```

### 3. Image Editing (Inpainting / Modification)
```bash
lumina -p "Add brass goggles to the owl" -i owl.png
```

### 4. Multi-Image Composition
```bash
lumina -p "Combine the lighting of image 1 with the subject of image 2" \
    -i style_ref.png -i subject_ref.png
```

### 5. High Resolution & Widescreen (2K / 4K)
```bash
lumina -p "Deep space relay station" --aspect-ratio "16:9" --image-size "4K"
```

### 6. Fast / Lower-Cost Tier (Nano Banana 2)
```bash
lumina -p "Storyboard sketch of a server room" --model-name gemini-3.1-flash-image
```

### 7. Piping from Stdin & Custom Output Path
```bash
echo "Minimalist geometric brass compass" | lumina -o ./out -f compass.png
```

## Configuration Reference (`~/.config/lumina/.env`)

| Setting | Description | Default |
| :--- | :--- | :--- |
| `API_KEY` | Gemini API Key (`generativelanguage.googleapis.com`) | `None` |
| `PROJECT_ID` | GCP Project ID (Vertex AI fallback) | `None` |
| `LOCATION` | GCP Region (Vertex AI fallback) | `us-central1` |
| `MODEL_NAME` | Image model ID | `gemini-3-pro-image` |
| `OUTPUT_DIR` | Default output folder | `~/Pictures/Lumina_Generated` |
| `ASPECT_RATIO` | Default aspect ratio (`1:1`, `16:9`, `9:16`, `4:3`, `3:4`) | `1:1` |
| `IMAGE_SIZE` | Resolution (`1K`, `2K`, `4K`) | `1K` |
| `SAFETY_FILTER_LEVEL` | Content filtering (`BLOCK_NONE`, `BLOCK_ONLY_HIGH`) | `BLOCK_ONLY_HIGH` |
| `PERSON_GENERATION` | Person policy (`allow_all`, `allow_adult`, `dont_allow`) | `allow_all` |
| `ADD_WATERMARK` | Invisible SynthID watermark (Vertex AI) | `true` |

## Development

```bash
uv sync --python 3.12 --extra dev
uv run --python 3.12 pytest
uv run --python 3.12 ruff check .
```

## License

MIT License
