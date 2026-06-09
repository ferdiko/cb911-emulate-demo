"""CLI for running the screengrab pipeline against Together-hosted Qwen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from screengrab_demo.frame_io import IMAGE_SUFFIXES, load_frames_from_paths
from screengrab_demo.model_io import pipeline_result_to_dict
from screengrab_demo.pipeline import run_pipeline

from .together_classifier import TogetherQwenClassifier


def main() -> None:
    args = parse_args()
    paths = collect_image_paths(args.frames)
    if not paths:
        raise SystemExit(f"No image files found in {args.frames}")

    frames = load_frames_from_paths(paths)
    result = run_pipeline(
        frames,
        TogetherQwenClassifier(),
        context=parse_context(args.context),
    )

    print(json.dumps(pipeline_result_to_dict(result), indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", type=Path, help="Directory containing screenshots, or a single screenshot file.")
    parser.add_argument(
        "--context",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Automation context to include in the prompt. May be provided multiple times.",
    )
    return parser.parse_args()


def collect_image_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.casefold() in IMAGE_SUFFIXES else []
    if not path.is_dir():
        return []
    return [
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.casefold() in IMAGE_SUFFIXES
    ]


def parse_context(items: list[str]) -> dict[str, str]:
    context: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Invalid --context value {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"Invalid --context value {item!r}; key cannot be empty")
        context[key] = value.strip()
    return context


if __name__ == "__main__":
    main()
