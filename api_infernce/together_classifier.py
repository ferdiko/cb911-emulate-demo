"""Classifier implementation backed by Together's OpenAI-compatible API."""

from __future__ import annotations

import os
from typing import Mapping, Sequence

from screengrab_demo.model_io import JSON_SYSTEM_PROMPT, build_multimodal_content, parse_classification_response
from screengrab_demo.models import ClassificationResult, SelectedFrame


TOGETHER_MODEL = "Qwen/Qwen3.5-397B-A17B"
TOGETHER_BASE_URL = "https://api.together.ai/v1"
TOGETHER_API_KEY_ENV = "TOGETHER_API_KEY"
REQUEST_TIMEOUT_SECONDS = 120.0
MAX_RESPONSE_TOKENS = 512
TEMPERATURE = 0.0


def _load_streamlit_secret(name: str) -> str | None:
    try:
        import streamlit as st
    except ModuleNotFoundError:
        return None

    try:
        value = st.secrets.get(name)
    except Exception:
        return None
    if not value:
        return None
    return str(value)


def load_together_api_key() -> str | None:
    return os.environ.get(TOGETHER_API_KEY_ENV) or _load_streamlit_secret(TOGETHER_API_KEY_ENV)


class TogetherQwenClassifier:
    """Adapter that satisfies `screengrab_demo.classifier.Classifier`."""

    def __init__(self, *, client: object | None = None) -> None:
        if client is not None:
            self.client = client
            return

        api_key = load_together_api_key()
        if not api_key:
            raise RuntimeError(
                f"Set {TOGETHER_API_KEY_ENV} as an environment variable or Streamlit secret "
                "to use TogetherQwenClassifier."
            )

        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError("Install the `openai` package to use TogetherQwenClassifier.") from exc

        self.client = OpenAI(
            api_key=api_key,
            base_url=TOGETHER_BASE_URL,
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
            model=TOGETHER_MODEL,
            messages=[
                {"role": "system", "content": JSON_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_RESPONSE_TOKENS,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content
        if not text:
            raise RuntimeError("Together returned an empty response")
        return parse_classification_response(text)
