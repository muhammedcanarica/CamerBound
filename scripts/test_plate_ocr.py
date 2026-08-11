from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.camera import Direction
from app.config import load_config
from app.plate_recognition import (
    PaddleOcrProvider,
    OcrModelNotFound,
    TurkishPlateValidator,
    correct_plate_candidate,
    crop_roi,
    normalize_plate_text,
    preprocess_variants,
    select_best_candidate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Tek görselde lokal plaka OCR testi.")
    parser.add_argument("image")
    parser.add_argument("--direction", choices=("ENTRY", "EXIT"), default="ENTRY")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        parser.error(f"Görsel okunamadı: {args.image}")

    config = load_config().plate_recognition
    direction = Direction(args.direction)
    crop = crop_roi(image, config.roi_for(direction))
    if crop is None:
        parser.error("ROI görselden çıkarılamadı.")

    try:
        provider = PaddleOcrProvider(config.model_root)
    except OcrModelNotFound as exc:
        print(str(exc))
        return 2
    segments = provider.recognize(preprocess_variants(crop))
    if not segments:
        print("OCR sonucu bulunamadı.")
        return 1

    for segment in segments:
        normalized = normalize_plate_text(segment.text)
        corrected = correct_plate_candidate(segment.text)
        print(
            f"raw={segment.text!r} normalized={normalized!r} "
            f"corrected={corrected!r} valid={bool(corrected and TurkishPlateValidator.is_valid(corrected))} "
            f"confidence={segment.confidence:.3f}"
        )

    best = select_best_candidate(segments, camera_id=0)
    print(f"best={best}")
    return 0 if best is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
