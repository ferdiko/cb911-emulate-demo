"""Pipeline orchestration for frame selection and classification."""

from __future__ import annotations

from typing import Mapping, Sequence

from .classifier import Classifier
from .keyframes import select_fallback_keyframes, select_initial_keyframes
from .models import ClassificationResult, Frame, PipelineResult


FALLBACK_CONFIDENCE_THRESHOLD = 0.60


def run_pipeline(
    frames: Sequence[Frame],
    classifier: Classifier,
    *,
    context: Mapping[str, str] | None = None,
) -> PipelineResult:
    if not frames:
        raise ValueError("at least one frame is required")

    selected = select_initial_keyframes(frames)
    classification = classifier.classify(selected, context=context)

    fallback_reason = fallback_reason_for(classification)
    if fallback_reason:
        fallback_selected = select_fallback_keyframes(frames)
        if len(fallback_selected) <= len(selected):
            return PipelineResult(
                classification=classification,
                selected_frames=selected,
                pass_name="initial",
                used_fallback=False,
                fallback_reason=None,
            )
        fallback_classification = classifier.classify(fallback_selected, context=context)
        return PipelineResult(
            classification=fallback_classification,
            selected_frames=fallback_selected,
            pass_name="fallback",
            used_fallback=True,
            fallback_reason=fallback_reason,
        )

    return PipelineResult(
        classification=classification,
        selected_frames=selected,
        pass_name="initial",
        used_fallback=False,
        fallback_reason=None,
    )


def fallback_reason_for(result: ClassificationResult) -> str | None:
    if result.confidence < FALLBACK_CONFIDENCE_THRESHOLD:
        return (
            f"confidence {result.confidence:.2f} below threshold "
            f"{FALLBACK_CONFIDENCE_THRESHOLD:.2f}"
        )
    if result.category == "other":
        return "category is other"
    if not result.evidence_frames:
        return "no evidence frames returned"
    return None
