# lumina specification

Version 0.4.1 (2026-10-20; code 0.4.1). Owner: Chuck Forsyth.
Repository: `charles-forsyth/lumina` (local checkout `~/Projects/Gen-Image-IVXX`). License: MIT.

## 1. Purpose and scope

`lumina` is a Python CLI for generating and editing images with Google's Gemini image models:
- **Nano Banana Pro (`gemini-3-pro-image`)** — default high-fidelity image generation and multi-image editing/composition model.
- **Nano Banana 2 (`gemini-3.1-flash-image`)** — faster, lower-cost image generation tier (used by Mirage for scene stills via `--model-name gemini-3.1-flash-image`).
- **Gemini 3.8 Flash (`gemini-3.8-flash`)** — text model used when `--optimize` (`--opt`) is passed to rewrite raw prompts or tabular data into detailed visual prompts before image synthesis.

`lumina` supports both standalone CLI usage and non-interactive orchestration (Mirage shells out to `lumina` for character portraits and scene stills).

## 2. Design decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | Dedicated per-tool config `~/.config/lumina/.env` (`0600`) | Loaded via `pydantic-settings`; `lumina init` scaffolds the file with `0600` permissions. |
| D2 | API-key-first authentication without GCE metadata probing | When `API_KEY` (`--api-key` or `settings.api_key`) is present, `ImageGenerator` initializes `genai.Client(api_key=..., vertexai=False)` and `get_project_id(..., has_api_key=True)` skips `google.auth.default()`. This avoids 3 GCE metadata server retry timeouts (~6 s delay) on non-GCP workstations. |
| D3 | Strict `--count` loop (`generate`) | Gemini image models return 1 image per `generate_content` call. To guarantee exact count adherence (`--count N`), `ImageGenerator.generate()` loops until `N` images are produced (up to `count * 2` attempts). |
| D4 | Reference image composition via PIL (`--image` repeatable) | Passing `-i img1.png -i img2.png` loads each image with Pillow and prepends them to the `contents` list alongside the text prompt, enabling inpainting, style transfer, and multi-image composition through the same `generate_content` endpoint. |
| D5 | Suppress `AFC is enabled` SDK log noise | `AFCFilter` strips noisy automatic-function-calling log lines emitted by `google-genai`. |

## 3. Architecture

```text
lumina CLI (src/lumina/cli.py)
  |-- Settings (src/lumina/config.py)
  |     `-- ~/.config/lumina/.env (or ./.env)
  |-- Command & input routing (cli.py)
  |     |-- `lumina init` -> run_init() creates ~/.config/lumina/.env (0600)
  |     |-- Prompt resolution: --prompt / -p or piped stdin
  |     `-- Auth resolution: API_KEY (Google AI Studio) or PROJECT_ID / ADC (Vertex AI)
  `-- ImageGenerator (src/lumina/core.py)
        |-- Lazy genai.Client property (vertexai=False with api_key, else vertexai=True)
        |-- Optional optimize_prompt() via gemini-3.8-flash (--optimize / --opt)
        |-- Style & variation prompt augmentation (--style, --variation, --negative-prompt)
        `-- generate() loop (up to count * 2 attempts)
              |-- Loads reference_images via PIL.Image.open()
              |-- types.GenerateContentConfig(response_modalities=["IMAGE"], image_config=ImageConfig(aspect_ratio, image_size))
              `-- part.as_image().save(output_path)
```

### Module map

| File | Responsibility |
|---|---|
| `src/lumina/cli.py` | Argument parser (`get_parser`), `AFCFilter`, `get_project_id()`, `run_init()`, `main()`. |
| `src/lumina/config.py` | `Settings` (`pydantic-settings` model) defining defaults for auth, model, output directory, aspect ratio, image size, safety filter, and watermark. |
| `src/lumina/core.py` | `OPTIMIZATION_MODEL = "gemini-3.8-flash"`, `ImageGenerator` (`client`, `optimize_prompt`, `generate`). |
| `src/lumina/utils.py` | `ensure_directory()`, `sanitize_filename()` (slugifies prompt + timestamp when `--filename` is omitted). |

## 4. Configuration (`~/.config/lumina/.env`)

| Variable | Field | Default | Description |
|---|---|---|---|
| `API_KEY` | `settings.api_key` | `None` | Gemini API key restricted to `generativelanguage.googleapis.com`. Preferred auth method. |
| `PROJECT_ID` | `settings.project_id` | `None` | GCP project ID for Vertex AI fallback when `API_KEY` is unset. |
| `LOCATION` | `settings.location` | `"us-central1"` | GCP region for Vertex AI fallback. |
| `MODEL_NAME` | `settings.model_name` | `"gemini-3-pro-image"` | Image generation model (`gemini-3-pro-image` or `gemini-3.1-flash-image`). |
| `OUTPUT_DIR` | `settings.output_dir` | `~/Pictures/Lumina_Generated` | Directory where generated images are written. |
| `ASPECT_RATIO` | `settings.aspect_ratio` | `"1:1"` | Aspect ratio (`1:1`, `16:9`, `9:16`, `4:3`, `3:4`). |
| `IMAGE_SIZE` | `settings.image_size` | `"1K"` | Output resolution tier (`1K`, `2K`, `4K`). |
| `SAFETY_FILTER_LEVEL` | `settings.safety_filter_level` | `"BLOCK_ONLY_HIGH"` | Safety threshold (`BLOCK_NONE`, `BLOCK_ONLY_HIGH`, etc.). |
| `PERSON_GENERATION` | `settings.person_generation` | `"allow_all"` | Person generation policy. |
| `ADD_WATERMARK` | `settings.add_watermark` | `True` | SynthID watermark flag (note: only passed in Vertex AI mode; Gemini Developer API rejects `person_generation` / ` safety_filter_level` in `ImageConfig`). |

## 5. CLI interface

```text
lumina [init]
       [--verbose | -v]
       [--prompt PROMPT | -p PROMPT]
       [--image PATH | -i PATH ...]
       [--count N | -n N]
       [--style STYLE ...]
       [--variation VARIATION ...]
       [--output-dir DIR | -o DIR]
       [--filename NAME | -f NAME]
       [--api-key KEY]
       [--project-id ID]
       [--location REGION]
       [--model-name MODEL]
       [--aspect-ratio {1:1,16:9,9:16,4:3,3:4}]
       [--image-size {1K,2K,4K}]
       [--negative-prompt TEXT]
       [--optimize | --opt]
       [--seed INT]
```

Filename behavior when `--count > 1`:
- If `--filename logo.png` is passed with `--count 3`, files are saved as `logo_1.png`, `logo_2.png`, `logo_3.png`.
- If `--filename` is omitted, `sanitize_filename(prompt)` produces `<slug>_<timestamp>.png` (or `<slug>_<timestamp>_<idx>.png`).

## 6. Cost model

| Operation | Model | List price (USD) |
|---|---|---|
| Standard / high-res still | `gemini-3-pro-image` (Nano Banana Pro) | ~$0.134 / image |
| Fast scene still | `gemini-3.1-flash-image` (Nano Banana 2) | ~$0.067 / image |
| Prompt optimization (`--optimize`) | `gemini-3.8-flash` | <$0.001 / call |

## 7. Requirements traceability

| ID | Requirement | Verified by |
|---|---|---|
| REQ-AUTH-1 | Vertex AI client initializes with `vertexai=True, project=..., location=...` when `api_key` is omitted. | `tests/test_core.py::test_init` |
| REQ-AUTH-2 | Gemini API client initializes with `api_key=..., vertexai=False` when `api_key` is provided. | `tests/test_core.py::test_init_api_key` |
| REQ-AUTH-3 | `get_project_id()` does not call `google.auth.default()` when `has_api_key=True`, avoiding GCE metadata server timeouts. | `tests/test_core.py::test_get_project_id_skips_gce_auth_when_api_key_set` |
| REQ-GEN-1 | Single image generation creates the output directory and writes the image file. | `tests/test_core.py::test_generate_images` |
| REQ-GEN-2 | Multi-image generation (`count=2`) loops until the requested number of images is saved. | `tests/test_core.py::test_generate_multiple_images_loop` |

## 8. Known gaps and backlog

- **K1 (Confirmed):** `Settings.model_config` reads both `~/.config/lumina/.env` and `./.env`. Because `API_KEY` is a generic variable name, running `lumina` inside another project directory that has its own `.env` containing an `API_KEY` variable could override `~/.config/lumina/.env`. Restricting `env_file` to `~/.config/lumina/.env` (or prefixing `LUMINA_API_KEY`) would harden isolation.
- **K2 (Confirmed):** The Gemini API returns JPEG bytes inside `part.as_image()`, which `part.as_image().save(output_path)` writes directly even when `--filename` ends with `.png` (resulting in a JPEG file with a `.png` extension). Converting via ` PIL.Image.open(io.BytesIO(part.inline_data.data)).save(output_path)` would match the file container to the requested extension.
