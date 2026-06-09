"""Classifier boundary.

The production classifier will call Qwen3-VL through vLLM. For now, the mock
classifier gives deterministic behavior for pipeline development and tests.
"""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence

from .models import ClassificationResult, SelectedFrame


class Classifier(Protocol):
    def classify(
        self,
        frames: Sequence[SelectedFrame],
        *,
        context: Mapping[str, str] | None = None,
    ) -> ClassificationResult:
        ...


class MockClassifier:
    """A deterministic placeholder for the future vLLM classifier."""

    keyword_categories = (
        (
            "2FA",
            (
                "2fa",
                "mfa",
                "verification code",
                "verify your identity",
                "authenticator",
                "one-time",
                "otp",
                "security code",
            ),
        ),
        (
            "expired_credentials",
            (
                "expired password",
                "password expired",
                "credentials expired",
                "reset password",
                "change your password",
                "incorrect password",
                "session expired",
            ),
        ),
        (
            "ui_change",
            (
                "not found",
                "page moved",
                "new layout",
                "unexpected page",
                "element not found",
                "permission denied",
            ),
        ),
    )

    def __init__(self, *, forced_result: ClassificationResult | None = None) -> None:
        self.forced_result = forced_result

    def classify(
        self,
        frames: Sequence[SelectedFrame],
        *,
        context: Mapping[str, str] | None = None,
    ) -> ClassificationResult:
        if self.forced_result is not None:
            return self.forced_result

        evidence_text = _collect_evidence_text(frames, context)
        lowered = evidence_text.casefold()

        for category, keywords in self.keyword_categories:
            hits = [keyword for keyword in keywords if keyword in lowered]
            if hits:
                evidence_frames = tuple(
                    selected.frame.number
                    for selected in frames
                    if any(keyword in selected.frame.name.casefold() for keyword in hits)
                )
                if not evidence_frames:
                    evidence_frames = tuple(selected.frame.number for selected in frames[:1])
                return ClassificationResult(
                    category=category,
                    confidence=0.72,
                    evidence_frames=evidence_frames,
                    reason=f"Mock classifier matched keyword(s): {', '.join(hits[:3])}.",
                )

        return ClassificationResult(
            category="other",
            confidence=0.25,
            evidence_frames=(),
            reason="Mock classifier found no category-specific keywords.",
        )


def _collect_evidence_text(
    frames: Sequence[SelectedFrame],
    context: Mapping[str, str] | None,
) -> str:
    chunks: list[str] = []
    if context:
        chunks.extend(str(value) for value in context.values())
    for selected in frames:
        chunks.append(selected.frame.name)
    return "\n".join(chunks)
