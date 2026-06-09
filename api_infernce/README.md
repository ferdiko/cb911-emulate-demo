# Together API Inference

This folder contains an alternate hosted-API inference path. It runs the same
fixed screengrab pipeline as the vLLM path, but calls Qwen through Together's
OpenAI-compatible API.

Folder name note: this directory is named `api_infernce` to match the requested
name.

## Fixed API Target

- Base URL: `https://api.together.ai/v1`
- Model: `Qwen/Qwen3.5-397B-A17B`
- API key environment variable: `TOGETHER_API_KEY`

Together's docs describe their OpenAI compatibility layer as using the OpenAI
client with `base_url="https://api.together.ai/v1"`. Their current vision docs
show image input through `chat.completions.create` with mixed `text` and
`image_url` content blocks, and their serverless catalog currently recommends
`Qwen/Qwen3.5-397B-A17B` for vision.

## Run

From the repo root:

```bash
export TOGETHER_API_KEY=...

uv run python -m api_infernce.run_together example_screenshots/credentials_expired \
  --context crm=ExampleCRM \
  --context last_action="submitted login form"
```

The CLI only accepts the screenshot path and optional automation context. The
pipeline behavior is fixed:

1. Load images.
2. Sort naturally.
3. Normalize to max edge `1280`.
4. Compute visual deltas.
5. Select up to `12` initial keyframes.
6. Call Qwen through Together.
7. Parse and validate JSON.
8. If uncertain, select up to `40` fallback frames and call Together again.
9. Print the final `PipelineResult` as JSON.

## Files

- `together_classifier.py`: `TogetherQwenClassifier`, implementing the shared
  classifier interface.
- `run_together.py`: CLI entry point for the fixed pipeline.
