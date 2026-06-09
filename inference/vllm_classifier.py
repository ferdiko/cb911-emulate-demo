"""Classifier implementation backed by a vLLM OpenAI-compatible server."""

from __future__ import annotations

from typing import Mapping, Sequence

from screengrab_demo.model_io import JSON_SYSTEM_PROMPT, build_multimodal_content, parse_classification_response
from screengrab_demo.models import ClassificationResult, SelectedFrame


DEFAULT_MODEL = "Qwen/Qwen3-VL-32B-Instruct"
DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_API_KEY = "EMPTY"
REQUEST_TIMEOUT_SECONDS = 120.0
MAX_RESPONSE_TOKENS = 512
TEMPERATURE = 0.0



class VLLMClassifier:
    """Adapter that satisfies `screengrab_demo.classifier.Classifier`."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = DEFAULT_API_KEY,
    ) -> None:
        self.model = model
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError("Install the `openai` package to use VLLMClassifier.") from exc

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

    def classify(
        self,
        frames: Sequence[SelectedFrame],
        *,
        context: Mapping[str, str] | None = None,
    ) -> ClassificationResult:
        content = build_multimodal_content(frames, context=context)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": JSON_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_RESPONSE_TOKENS,
        )
        text = response.choices[0].message.content
        if not text:
            raise RuntimeError("vLLM returned an empty response")
        return parse_classification_response(text)
