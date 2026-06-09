# vLLM Inference

This folder contains the real classifier adapter for a vLLM
OpenAI-compatible server. The Streamlit app and this CLI both use the same
fixed workflow and call Qwen3-VL through vLLM.

## Files

- `helpers.py`: builds multimodal chat content, encodes frames as data URLs, and
  parses the model's JSON response.
- `vllm_classifier.py`: `VLLMClassifier`, which implements the shared
  classifier interface.
- `run_vllm.py`: CLI for loading screenshots, running the fixed pipeline,
  calling vLLM, and printing the structured result.

## Install Client Dependencies

From the repo root:

```bash
uv sync
```

This installs the client-side dependencies, including the `openai` Python
client used to call vLLM's OpenAI-compatible API.

## Start vLLM

Run this on the GPU host:

```bash
vllm serve Qwen/Qwen3-VL-32B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --limit-mm-per-prompt image=40
```

Install vLLM separately on the GPU host. It is intentionally not a project
dependency because it is a heavy server runtime.

## Run Inference

From the repo root:

```bash
uv run python -m inference.run_vllm example/credentials_expired \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen3-VL-32B-Instruct \
  --context crm=ExampleCRM \
  --context last_action="submitted login form"
```

The CLI only accepts values needed to reach the vLLM server and provide
automation context:

- screenshot path
- model name
- base URL
- API key
- `KEY=VALUE` context entries

The pipeline behavior itself is fixed:

1. Load images.
2. Sort naturally.
3. Normalize to max edge `1280`.
4. Compute visual deltas.
5. Select up to `12` initial keyframes.
6. Call Qwen3-VL through vLLM.
7. Parse and validate JSON.
8. If uncertain, select up to `40` fallback frames and call Qwen3-VL again.
9. Print the final `PipelineResult` as JSON.
