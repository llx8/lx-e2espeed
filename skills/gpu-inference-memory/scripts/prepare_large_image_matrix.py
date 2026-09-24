#!/usr/bin/env python3
"""Resize real images to an exact maximum long edge while preserving aspect ratio."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def target_size(width: int, height: int, long_edge: int) -> tuple[int, int]:
    if width <= 0 or height <= 0 or long_edge <= 0:
        raise ValueError("Image dimensions and long edge must be positive")
    if width == height:
        raise ValueError("Use non-square images so the short edge remains below the long edge")
    scale = long_edge / max(width, height)
    resized = (max(1, round(width * scale)), max(1, round(height * scale)))
    if max(resized) != long_edge or min(resized) >= long_edge:
        raise RuntimeError(f"Invalid target size: {resized}")
    return resized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--long-edge", type=int, default=13000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source in args.images:
        if not source.is_file():
            raise FileNotFoundError(source)
        with Image.open(source) as image:
            size = target_size(image.width, image.height, args.long_edge)
            output = args.output_dir / f"{source.stem}_long{args.long_edge}_{size[0]}x{size[1]}.png"
            image.resize(size, Image.Resampling.LANCZOS).save(output)
            print(f"{source} -> {output} ({size[0]}x{size[1]})")


if __name__ == "__main__":
    main()
