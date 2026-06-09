"""Frame loading, sorting, and image normalization helpers."""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

from .models import Frame
from .vision import annotate_visual_deltas


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
IMAGE_MAX_EDGE = 1280


class MissingImageDependencyError(RuntimeError):
    """Raised when image loading is requested without Pillow installed."""


def natural_sort_key(value: str) -> tuple[object, ...]:
    """Return a key that sorts numeric filename fragments numerically."""

    parts = re.split(r"(\d+)", value.casefold())
    return tuple(int(part) if part.isdigit() else part for part in parts)


def is_image_name(name: str) -> bool:
    return Path(name).suffix.casefold() in IMAGE_SUFFIXES


def load_frames_from_paths(
    paths: Iterable[str | Path],
) -> list[Frame]:
    """Load image files from disk in natural filename order."""

    sorted_paths = sorted((Path(path) for path in paths), key=lambda p: natural_sort_key(p.name))
    loaded: list[tuple[str, bytes, str | None]] = []
    for path in sorted_paths:
        if not is_image_name(path.name):
            continue
        loaded.append((path.name, path.read_bytes(), _content_type_for_name(path.name)))
    return frames_from_named_bytes(loaded)


def load_frames_from_uploads(
    uploads: Sequence[object],
) -> list[Frame]:
    """Load frames from Streamlit-style uploaded files.

    Any uploaded zip files are expanded; image uploads are loaded directly.
    """

    named_bytes: list[tuple[str, bytes, str | None]] = []
    for upload in uploads:
        name = getattr(upload, "name", "uploaded")
        data = upload.getvalue() if hasattr(upload, "getvalue") else upload.read()
        if Path(name).suffix.casefold() == ".zip":
            named_bytes.extend(_frames_from_zip_bytes(data))
        elif is_image_name(name):
            named_bytes.append((name, data, _content_type_for_name(name)))

    return frames_from_named_bytes(named_bytes)


def frames_from_named_bytes(
    items: Iterable[tuple[str, bytes, str | None]],
) -> list[Frame]:
    """Create frames from `(name, bytes, content_type)` tuples."""

    Image = _load_pillow_image_class()
    sorted_items = sorted(items, key=lambda item: natural_sort_key(item[0]))
    frames: list[Frame] = []

    for number, (name, data, content_type) in enumerate(sorted_items, start=1):
        image = Image.open(io.BytesIO(data))
        image.load()
        image = normalize_image(image)
        frames.append(
            Frame(
                number=number,
                name=name,
                image=image,
                data=data,
                width=image.width,
                height=image.height,
                content_type=content_type,
            )
        )

    return annotate_visual_deltas(frames)


def normalize_image(image: object) -> object:
    """Convert to RGB and shrink large images while preserving aspect ratio."""

    normalized = image.convert("RGB")  # type: ignore[attr-defined]
    longest_edge = max(normalized.size)  # type: ignore[arg-type]
    if longest_edge <= IMAGE_MAX_EDGE:
        return normalized

    resized = normalized.copy()
    resized.thumbnail((IMAGE_MAX_EDGE, IMAGE_MAX_EDGE))
    return resized


def image_to_png_bytes(image: object) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")  # type: ignore[attr-defined]
    return buffer.getvalue()


def _frames_from_zip_bytes(data: bytes) -> list[tuple[str, bytes, str | None]]:
    items: list[tuple[str, bytes, str | None]] = []
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for name in archive.namelist():
            if name.endswith("/") or not is_image_name(name):
                continue
            items.append((Path(name).name, archive.read(name), _content_type_for_name(name)))
    return items


def _content_type_for_name(name: str) -> str | None:
    suffix = Path(name).suffix.casefold()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".bmp":
        return "image/bmp"
    return None


def _load_pillow_image_class() -> object:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise MissingImageDependencyError(
            "Pillow is required for image loading. Install dependencies from requirements.txt."
        ) from exc
    return Image
