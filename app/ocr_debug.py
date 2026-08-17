from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from app.plate_recognition import OcrSegment


def save_debug_images(
    output_dir: Path,
    source: np.ndarray,
    crop: np.ndarray,
    variants: Sequence[np.ndarray],
    segments: Sequence[OcrSegment],
    variant_names: Sequence[str] | None = None,
) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    items = [("01-source.jpg", source), ("02-roi.jpg", crop)]
    names = list(variant_names or ())
    items.extend(
        (
            f"{index + 3:02d}-variant-{_safe_name(names[index] if index < len(names) else str(index))}.jpg",
            image,
        )
        for index, image in enumerate(variants)
    )
    for filename, image in items:
        path = output_dir / filename
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Debug görseli yazılamadı: {path}")
        paths.append(path)

    for variant_index, variant in enumerate(variants):
        annotated = variant.copy()
        for segment in segments:
            if segment.variant_index != variant_index:
                continue
            x1, y1, x2, y2 = (round(value) for value in segment.box)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"{segment.text} {segment.confidence:.2f}",
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        label = names[variant_index] if variant_index < len(names) else str(variant_index)
        path = output_dir / (
            f"{len(items) + variant_index + 1:02d}-result-{_safe_name(label)}.jpg"
        )
        if not cv2.imwrite(str(path), annotated):
            raise OSError(f"Debug görseli yazılamadı: {path}")
        paths.append(path)
    return tuple(paths)


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isascii() and character.isalnum() else "-"
        for character in value
    ).strip("-") or "unnamed"
