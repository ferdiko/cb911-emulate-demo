"""Shared data models for the classification pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


ALLOWED_CATEGORIES = ("2FA", "expired_credentials", "ui_change", "other")


@dataclass(frozen=True)
class Frame:
    """A single screenshot and its derived metadata.

    The `image` field intentionally uses `Any` so the core pipeline does not
    depend on Pillow. Image loading code can store a PIL Image there.
    """

    number: int
    name: str
    image: Any | None = None
    data: bytes | None = None
    width: int | None = None
    height: int | None = None
    content_type: str | None = None
    delta_to_previous: float = 0.0


@dataclass(frozen=True)
class SelectedFrame:
    frame: Frame
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ClassificationResult:
    category: str
    confidence: float
    evidence_frames: tuple[int, ...] = ()
    visible_text: tuple[str, ...] = ()
    reason: str = ""
    raw_response: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.category not in ALLOWED_CATEGORIES:
            allowed = ", ".join(ALLOWED_CATEGORIES)
            raise ValueError(f"category must be one of: {allowed}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")


@dataclass(frozen=True)
class PipelineResult:
    classification: ClassificationResult
    selected_frames: tuple[SelectedFrame, ...]
    pass_name: str
    used_fallback: bool
    fallback_reason: str | None = None
