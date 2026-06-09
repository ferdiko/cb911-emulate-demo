"""Deterministic keyframe selection."""

from __future__ import annotations

from typing import Sequence

from .models import Frame, SelectedFrame


INITIAL_KEYFRAME_LIMIT = 12
FALLBACK_KEYFRAME_LIMIT = 40
INITIAL_EVENLY_SPACED_COUNT = 6
VISUAL_DELTA_THRESHOLD = 0.12


def select_keyframes(frames: Sequence[Frame]) -> tuple[SelectedFrame, ...]:
    """Select the fixed first-pass keyframe set."""

    return select_initial_keyframes(frames)


def select_initial_keyframes(frames: Sequence[Frame]) -> tuple[SelectedFrame, ...]:
    """Select the fixed first-pass keyframe set."""

    return _select_keyframes(
        frames,
        max_keyframes=INITIAL_KEYFRAME_LIMIT,
        evenly_spaced_count=INITIAL_EVENLY_SPACED_COUNT,
        delta_threshold=VISUAL_DELTA_THRESHOLD,
    )


def select_fallback_keyframes(frames: Sequence[Frame]) -> tuple[SelectedFrame, ...]:
    """Select the fixed fallback keyframe set."""

    fallback_limit = min(FALLBACK_KEYFRAME_LIMIT, len(frames))
    return _select_keyframes(
        frames,
        max_keyframes=fallback_limit,
        evenly_spaced_count=fallback_limit,
        delta_threshold=0.0,
    )


def _select_keyframes(
    frames: Sequence[Frame],
    *,
    max_keyframes: int,
    evenly_spaced_count: int,
    delta_threshold: float,
) -> tuple[SelectedFrame, ...]:
    """Select a bounded, ordered set of frames for the classifier."""

    if not frames:
        return ()

    if max_keyframes <= 0:
        raise ValueError("max_keyframes must be positive")
    if len(frames) > 1 and max_keyframes < 2:
        raise ValueError("max_keyframes must be at least 2 for multi-frame input")

    reasons_by_index: dict[int, list[str]] = {}

    def mark(index: int, reason: str) -> None:
        reasons_by_index.setdefault(index, [])
        if reason not in reasons_by_index[index]:
            reasons_by_index[index].append(reason)

    mark(0, "first_frame")
    mark(len(frames) - 1, "last_frame")

    for index, frame in sorted(
        enumerate(frames),
        key=lambda item: item[1].delta_to_previous,
        reverse=True,
    ):
        if frame.delta_to_previous >= delta_threshold:
            mark(index, f"visual_delta:{frame.delta_to_previous:.3f}")

    for index in evenly_spaced_indices(len(frames), evenly_spaced_count):
        mark(index, "even_spacing")

    selected_indices = _cap_indices(reasons_by_index, frames, max_keyframes)
    return tuple(
        SelectedFrame(frame=frames[index], reasons=tuple(reasons_by_index[index]))
        for index in selected_indices
    )


def evenly_spaced_indices(total: int, count: int) -> tuple[int, ...]:
    if total <= 0 or count <= 0:
        return ()
    if count >= total:
        return tuple(range(total))
    if count == 1:
        return (0,)

    last = total - 1
    return tuple(round(i * last / (count - 1)) for i in range(count))


def _cap_indices(
    reasons_by_index: dict[int, list[str]],
    frames: Sequence[Frame],
    max_keyframes: int,
) -> tuple[int, ...]:
    if len(reasons_by_index) <= max_keyframes:
        return tuple(sorted(reasons_by_index))

    forced = {0, len(frames) - 1}
    scored: list[tuple[float, int]] = []
    for index, reasons in reasons_by_index.items():
        if index in forced:
            continue
        score = _score_index(frames[index], reasons)
        scored.append((score, index))

    remaining_slots = max(0, max_keyframes - len(forced))
    picked = forced | {index for _, index in sorted(scored, reverse=True)[:remaining_slots]}
    return tuple(sorted(picked))


def _score_index(frame: Frame, reasons: Sequence[str]) -> float:
    score = frame.delta_to_previous
    if any(reason.startswith("visual_delta:") for reason in reasons):
        score += 5.0
    if "even_spacing" in reasons:
        score += 1.0
    return score
