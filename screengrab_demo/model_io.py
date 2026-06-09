"""Shared multimodal request and response helpers."""

from __future__ import annotations

import base64
import json
import mimetypes
from dataclasses import asdict
from typing import Any, Mapping, Sequence

from .frame_io import image_to_png_bytes
from .models import ALLOWED_CATEGORIES, ClassificationResult, Frame, PipelineResult, SelectedFrame


JSON_SYSTEM_PROMPT = (
    "You are a strict JSON API. Respond with exactly one JSON object and no "
    "markdown, no prose, no code fences, and no surrounding text."
)


class InvalidClassificationResponse(ValueError):
    """Raised when the model response cannot be parsed into the schema."""

    def __init__(self, message: str, *, raw_text: str | None = None) -> None:
        self.raw_text = raw_text
        if raw_text:
            message = f"{message}. Raw response preview: {_response_preview(raw_text)}"
        super().__init__(message)


def build_multimodal_content(
    frames: Sequence[SelectedFrame],
    *,
    context: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Build OpenAI-compatible chat content for multimodal serving."""

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": build_classification_prompt(frames, context=context),
        }
    ]

    for selected in frames:
        label = (
            f"Frame {selected.frame.number}: {selected.frame.name}\n"
            f"Selection reasons: {', '.join(selected.reasons) or 'none'}"
        )
        content.append({"type": "text", "text": label})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": frame_to_data_url(selected.frame)},
            }
        )

    return content


def build_classification_prompt(
    frames: Sequence[SelectedFrame],
    *,
    context: Mapping[str, str] | None = None,
) -> str:
    """Return the text instruction paired with the image list."""

    context_lines = []
    if context:
        context_lines = [f"- {key}: {value}" for key, value in context.items() if value]
    context_block = "\n".join(context_lines) if context_lines else "- none provided"
    frame_numbers = ", ".join(str(selected.frame.number) for selected in frames)

    return f"""Classify why a browser automation run failed while navigating a CRM.

Allowed categories:
- 2FA: the screen asks for MFA, 2FA, an authenticator code, one-time passcode, or identity verification.
- expired_credentials: the screen shows expired credentials, password reset/change requirements, incorrect password, or expired session.
- ui_change: the automation likely reached an unexpected page, missing element, changed layout, permission page, or moved/not-found page.
- other: the evidence does not fit the categories above.

Automation context:
{context_block}

You will receive selected frames from the failed run. They retain their original frame numbers: {frame_numbers}.

Return only valid JSON with this exact shape:
{{
  "category": "2FA | expired_credentials | ui_change | other",
  "confidence": 0.0,
  "evidence_frames": [1],
  "visible_text": ["short text visible in evidence frames"],
  "reason": "brief explanation grounded in the frames"
}}

Rules:
- `category` must be one of: {", ".join(ALLOWED_CATEGORIES)}.
- `confidence` must be a number from 0.0 to 1.0.
- `evidence_frames` must use original frame numbers, not selected-frame positions.
- Prefer `other` when the evidence is ambiguous.
- Do not include markdown fences or any text outside the JSON object.
"""


def frame_to_data_url(frame: Frame) -> str:
    """Encode a frame as a data URL suitable for OpenAI-compatible vision input."""

    if frame.image is not None:
        payload = image_to_png_bytes(frame.image)
        content_type = "image/png"
    elif frame.data is not None:
        payload = frame.data
        content_type = frame.content_type or mimetypes.guess_type(frame.name)[0] or "image/png"
    else:
        raise ValueError(f"frame {frame.number} has neither image nor raw data")

    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def parse_classification_response(text: str) -> ClassificationResult:
    """Parse model text into a validated ClassificationResult."""

    data = _load_json_object(text)
    category = _normalize_category(_required(data, "category"))
    confidence = _parse_confidence(_required(data, "confidence"))
    evidence_frames = _parse_int_tuple(data.get("evidence_frames", ()))
    visible_text = _parse_str_tuple(data.get("visible_text", ()))
    reason = str(data.get("reason", "")).strip()

    return ClassificationResult(
        category=category,
        confidence=confidence,
        evidence_frames=evidence_frames,
        visible_text=visible_text,
        reason=reason,
        raw_response=data,
    )


def pipeline_result_to_dict(result: PipelineResult) -> dict[str, Any]:
    """Convert a pipeline result to a CLI-friendly JSON object."""

    return {
        "classification": asdict(result.classification),
        "pass_name": result.pass_name,
        "used_fallback": result.used_fallback,
        "fallback_reason": result.fallback_reason,
        "selected_frames": [
            {
                "number": selected.frame.number,
                "name": selected.frame.name,
                "reasons": list(selected.reasons),
                "delta_to_previous": selected.frame.delta_to_previous,
            }
            for selected in result.selected_frames
        ],
    }


def _load_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise InvalidClassificationResponse(
            "response did not contain a JSON object",
            raw_text=text,
        )

    try:
        parsed = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        raise InvalidClassificationResponse(
            f"response JSON was invalid: {exc}",
            raw_text=text,
        ) from exc

    if not isinstance(parsed, dict):
        raise InvalidClassificationResponse("response JSON must be an object")
    return parsed


def _required(data: Mapping[str, Any], key: str) -> Any:
    if key not in data:
        raise InvalidClassificationResponse(f"missing required key: {key}")
    return data[key]


def _normalize_category(value: Any) -> str:
    normalized = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "2fa": "2FA",
        "mfa": "2FA",
        "expired_credentials": "expired_credentials",
        "credential_expired": "expired_credentials",
        "credentials_expired": "expired_credentials",
        "expired": "expired_credentials",
        "ui_change": "ui_change",
        "ui_changes": "ui_change",
        "changed_ui": "ui_change",
        "other": "other",
    }
    if normalized in aliases:
        return aliases[normalized]
    allowed = ", ".join(ALLOWED_CATEGORIES)
    raise InvalidClassificationResponse(f"unknown category {value!r}; expected one of: {allowed}")


def _parse_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidClassificationResponse("confidence must be numeric") from exc
    if confidence > 1.0 and confidence <= 100.0:
        confidence = confidence / 100.0
    if not 0.0 <= confidence <= 1.0:
        raise InvalidClassificationResponse("confidence must be between 0.0 and 1.0")
    return confidence


def _parse_int_tuple(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        raise InvalidClassificationResponse("evidence_frames must be a list")
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise InvalidClassificationResponse("evidence_frames must contain integers") from exc


def _parse_str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        raise InvalidClassificationResponse("visible_text must be a list")
    return tuple(str(item) for item in value)


def _response_preview(text: str, *, max_length: int = 500) -> str:
    preview = " ".join(text.split())
    if len(preview) > max_length:
        return preview[:max_length] + "..."
    return preview
