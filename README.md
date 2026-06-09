# Screengrab Failure Classifier Demo

This repo classifies why a browser/screen-emulation run failed while navigating
a CRM. The input is an ordered screenshot sequence, usually around 40 frames.
The output is a structured failure classification.

Categories:

- `2FA`: MFA, 2FA, authenticator, one-time code, or verification challenge.
- `expired_credentials`: expired password, reset/change password, invalid
  credentials, or expired session.
- `ui_change`: unexpected page, changed layout, missing element, permission
  page, moved page, or not-found page.
- `other`: anything that does not fit the categories above.

There is one workflow:

1. Load screenshots.
2. Sort them.
3. Normalize image size.
4. Compute visual deltas.
5. Select initial keyframes.
6. Classify the selected frames.
7. Validate the classifier JSON.
8. If the result is uncertain, run one fallback pass with more frames.
9. Return the final classification.

The Streamlit app chooses its backend from `screengrab_demo/inference_config.py`.
It is currently set to `api`, which uses Together-hosted Qwen. Set
`INFERENCE_BACKEND = "local"` in that file to use local vLLM instead. The vLLM
and Together CLIs still run their own explicit backends. `MockClassifier`
remains in the repo for tests and pipeline development.

## Fixed Workflow

The pipeline does not expose runtime tuning for keyframe count, image size,
confidence threshold, OCR, or fallback behavior.

The fixed values are:

- Image max edge: `1280` pixels.
- Initial keyframe limit: `12` frames.
- Initial even-spacing count: `6` frames.
- Visual-delta threshold: `0.12`.
- Fallback confidence threshold: `0.60`.
- Fallback keyframe limit: `40` frames.
- OCR: not used.

The first pass intentionally does not send all 40 screenshots to the VLM. CRM
screenshots are usually repetitive, and the useful evidence is normally in a
small number of frames. The fallback pass still expands coverage when the first
result is uncertain.

## Step By Step

1. `frame_io` loads image files, zip uploads, or Streamlit uploads.
2. Filenames are sorted with natural ordering, so `frame2.png` comes before
   `frame10.png`.
3. Each image is converted to RGB and resized only if its longest edge exceeds
   `1280` pixels.
4. `vision` computes `delta_to_previous` for each frame by comparing each frame
   to the previous frame.
5. `keyframes.select_initial_keyframes` selects:
   - first frame
   - last frame
   - frames above the fixed visual-delta threshold
   - evenly spaced frames for sequence coverage
   - no more than 12 total frames
6. The selected frames are passed to a classifier.
7. The classifier must return `ClassificationResult`.
8. `pipeline.fallback_reason_for` triggers fallback when:
   - confidence is below `0.60`
   - category is `other`
   - no evidence frames are returned
9. `keyframes.select_fallback_keyframes` selects up to 40 frames.
10. The classifier runs once more on the fallback frames.
11. The pipeline returns the final `PipelineResult`.

## Classifier Contract

All classifiers implement the same interface:

```python
class Classifier(Protocol):
    def classify(
        self,
        frames: Sequence[SelectedFrame],
        *,
        context: Mapping[str, str] | None = None,
    ) -> ClassificationResult:
        ...
```

The current implementations are:

- `MockClassifier` in `screengrab_demo/classifier.py`
- `VLLMClassifier` in `inference/vllm_classifier.py`

The result shape is:

```json
{
  "category": "2FA",
  "confidence": 0.92,
  "evidence_frames": [12, 13],
  "visible_text": ["Enter verification code", "Authenticator app"],
  "reason": "The selected frames show a verification-code challenge after login."
}
```

Rules:

- `category` must be `2FA`, `expired_credentials`, `ui_change`, or `other`.
- `confidence` must be between `0.0` and `1.0`.
- `evidence_frames` must use original frame numbers, not selected-frame
  positions.

## Repository Layout

```text
.
|-- app.py                         # Streamlit UI using configured backend
|-- api_infernce/
|   |-- README.md                  # Together API run instructions
|   |-- run_together.py            # CLI for Together-hosted Qwen
|   `-- together_classifier.py     # Together API classifier adapter
|-- inference/
|   |-- README.md                  # vLLM run instructions
|   |-- helpers.py                 # compatibility re-exports
|   |-- run_vllm.py                # CLI for real vLLM inference
|   `-- vllm_classifier.py         # OpenAI-compatible vLLM classifier adapter
|-- screengrab_demo/
|   |-- classifier.py              # classifier protocol and MockClassifier
|   |-- frame_io.py                # frame loading, sorting, normalization
|   |-- inference_config.py        # app backend selection: api or local
|   |-- keyframes.py               # fixed keyframe selection
|   |-- model_io.py                # prompt, data URL, JSON parsing helpers
|   |-- models.py                  # dataclasses and validation
|   |-- pipeline.py                # orchestration and fallback
|   `-- vision.py                  # visual-delta helper
|-- tests/                         # unit tests
|-- pyproject.toml                 # uv project dependencies
`-- uv.lock                        # resolved dependency lockfile
```

## Install

```bash
uv sync
```

Dependencies:

- `streamlit`
- `pillow`
- `openai`

`vllm` is not installed by this project. It should be installed separately on
the GPU host that serves Qwen3-VL.

## Run Tests

```bash
uv run python -m unittest discover -s tests -v
```

The tests cover sorting, keyframe selection, fallback triggering, mock
classification, multimodal request helper construction, response parsing, and
the Together adapter's fixed model behavior.

## Streamlit Backend Selection

The app backend is selected in `screengrab_demo/inference_config.py`:

```python
INFERENCE_BACKEND = "api"
```

Allowed values:

- `api`: Together-hosted Qwen through `api_infernce`.
- `local`: local vLLM through `inference`.

The current value is `api`, so set your Together API key before running
Streamlit:

```bash
export TOGETHER_API_KEY=...
```

If you switch the value to `local`, start vLLM first:

```bash
vllm serve Qwen/Qwen3-VL-32B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --limit-mm-per-prompt image=40
```

## Run the Streamlit App

Run Streamlit:

```bash
uv run streamlit run app.py
```

Then:

1. Upload screenshots.
2. Inspect the selected screenshot gallery.
3. Wait for classification through the configured backend.
4. Inspect the category, confidence, evidence frames, visible text, and fallback
   status.

## Run Real vLLM Inference

Start vLLM on a GPU host:

```bash
vllm serve Qwen/Qwen3-VL-32B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --limit-mm-per-prompt image=40
```

Run the CLI from this repo:

```bash
uv run python -m inference.run_vllm example/credentials_expired \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen3-VL-32B-Instruct \
  --context crm=ExampleCRM \
  --context last_action="submitted login form"
```

What this does:

1. Loads images from `example/credentials_expired`.
2. Runs the fixed shared pipeline.
3. Sends selected frames as labeled `image_url` data URLs.
4. Instructs Qwen3-VL to return only classification JSON.
5. Parses and validates the response locally.
6. Runs fallback if required.
7. Prints the final `PipelineResult` as JSON.

The vLLM-specific commands are also documented in `inference/README.md`.

## Run Together API Inference

Set your Together API key:

```bash
export TOGETHER_API_KEY=...
```

Run the alternate API inference CLI:

```bash
uv run python -m api_infernce.run_together example_screenshots/credentials_expired \
  --context crm=ExampleCRM \
  --context last_action="submitted login form"
```

This uses the same fixed pipeline, but calls Together's OpenAI-compatible API
with:

- base URL: `https://api.together.ai/v1`
- model: `Qwen/Qwen3.5-397B-A17B`

The Together-specific commands are documented in `api_infernce/README.md`.

## Current Status

Implemented:

- local Streamlit UI
- frame loading from uploads, zips, and paths
- natural sorting
- image normalization
- visual-delta calculation
- fixed keyframe selection
- mock classifier
- Streamlit app using configured backend
- vLLM classifier adapter
- vLLM inference CLI
- Together Qwen classifier adapter
- Together API inference CLI
- response parsing and validation
- fallback pass
- unit tests

Not implemented yet:

- labeled evaluation-set runner
- accuracy and latency reporting over fixture datasets

## References

- vLLM multimodal input docs:
  https://docs.vllm.ai/en/stable/features/multimodal_inputs/
- Qwen3-VL repo:
  https://github.com/QwenLM/Qwen3-VL
