"""Visual-delta helpers for screenshots."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Sequence

from .models import Frame


def annotate_visual_deltas(frames: Sequence[Frame]) -> list[Frame]:
    """Return frames with `delta_to_previous` filled in."""

    if not frames:
        return []

    annotated = [replace(frames[0], delta_to_previous=0.0)]
    for previous, current in zip(frames, frames[1:]):
        delta = visual_delta(previous.image, current.image) if previous.image and current.image else 0.0
        annotated.append(replace(current, delta_to_previous=delta))
    return annotated


def visual_delta(previous: object, current: object, *, sample_size: tuple[int, int] = (96, 96)) -> float:
    """Return a 0-1 approximate visual difference between two PIL images."""

    ImageChops, ImageStat = _load_pillow_delta_modules()
    left = previous.convert("RGB").resize(sample_size)  # type: ignore[attr-defined]
    right = current.convert("RGB").resize(sample_size)  # type: ignore[attr-defined]
    diff = ImageChops.difference(left, right)
    stat = ImageStat.Stat(diff)
    rms = math.sqrt(sum(channel * channel for channel in stat.rms) / len(stat.rms))
    return max(0.0, min(1.0, rms / 255.0))


def _load_pillow_delta_modules() -> tuple[object, object]:
    try:
        from PIL import ImageChops, ImageStat
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required for visual-delta calculation.") from exc
    return ImageChops, ImageStat
