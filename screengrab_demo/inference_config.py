"""Code-level inference backend selection."""

from __future__ import annotations

from screengrab_demo.classifier import Classifier


INFERENCE_BACKEND = "api"


def build_classifier() -> Classifier:
    if INFERENCE_BACKEND == "api":
        from api_infernce.together_classifier import TogetherQwenClassifier

        return TogetherQwenClassifier()
    if INFERENCE_BACKEND == "local":
        from inference.vllm_classifier import VLLMClassifier

        return VLLMClassifier()
    raise ValueError(f"Unknown inference backend: {INFERENCE_BACKEND}")


def backend_label() -> str:
    if INFERENCE_BACKEND == "api":
        return "Together API"
    if INFERENCE_BACKEND == "local":
        return "Local vLLM"
    return INFERENCE_BACKEND


def backend_requirement() -> str:
    if INFERENCE_BACKEND == "api":
        return "Set TOGETHER_API_KEY before running inference."
    if INFERENCE_BACKEND == "local":
        return "Make sure vLLM is running at http://localhost:8000/v1."
    return "Set INFERENCE_BACKEND to 'api' or 'local'."
