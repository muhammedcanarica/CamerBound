from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.camera import Direction
from app.config import load_config
from app.plate_recognition import (
    LOW_LIGHT_THRESHOLD,
    OCR_VARIANT_NAMES,
    PaddleOcrProvider,
    OcrModelNotFound,
    TurkishPlateValidator,
    correct_plate_candidate,
    crop_roi,
    normalize_plate_text,
    preprocess_variants,
    roi_mean_brightness,
    select_best_candidate,
)
from app.ocr_debug import save_debug_images
from app.plate_detector import (
    OpenVinoPlateDetector,
    crop_padded_plate,
    select_plate_detections,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Tek görselde lokal plaka OCR testi.")
    parser.add_argument("image")
    parser.add_argument("--direction", choices=("ENTRY", "EXIT"), default="ENTRY")
    parser.add_argument(
        "--detector-mode",
        choices=("config", "on", "off"),
        default="config",
        help="Detector kullanımını config değerinden alır veya saha karşılaştırması için zorlar.",
    )
    parser.add_argument(
        "--save-debug",
        type=Path,
        metavar="DIR",
        help="ROI, ön işleme varyantları ve OCR kutularını bu klasöre kaydeder.",
    )
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
        provider = PaddleOcrProvider(
            config.model_root,
            backend=config.ocr_backend,
        )
    except OcrModelNotFound as exc:
        print(str(exc))
        return 2
    detector_config = config.plate_detector
    detector_enabled = (
        detector_config.enabled
        if args.detector_mode == "config"
        else args.detector_mode == "on"
    )
    detector_started_at = time.perf_counter()
    detections = []
    ocr_crops = [crop]
    fallback = not detector_enabled
    if detector_enabled:
        try:
            detector = OpenVinoPlateDetector(detector_config)
            detections = detector.detect(crop)
            selected = select_plate_detections(
                detections,
                detector_config.max_plate_candidates_per_frame,
            )
            ocr_crops = [
                plate_crop
                for detection in selected
                if (
                    plate_crop := crop_padded_plate(
                        crop,
                        detection,
                        detector_config.crop_padding_ratio,
                    )
                )
                is not None
            ]
            fallback = False
        except Exception as exc:
            if not detector_config.fallback_to_roi_ocr:
                print(f"Plate detector kullanılamıyor: {exc}")
                return 2
            print(
                "Plate detector kullanılamıyor; ROI OCR fallback aktif: "
                f"{type(exc).__name__}"
            )
            ocr_crops = [crop]
            fallback = True
    detector_ms = (time.perf_counter() - detector_started_at) * 1000
    if detector_enabled and not fallback and not ocr_crops:
        print(
            f"direction={direction.value} detector_ms={detector_ms:.1f} "
            f"plates={len(detections)} OCR'a uygun plate crop bulunamadı."
        )
        return 1

    processing_started_at = time.perf_counter()
    brightness = roi_mean_brightness(crop)
    variants = []
    for ocr_crop in ocr_crops:
        crop_brightness = roi_mean_brightness(ocr_crop)
        variants.extend(
            preprocess_variants(ocr_crop, brightness=crop_brightness)
        )
    preprocess_ms = (time.perf_counter() - processing_started_at) * 1000
    inference_started_at = time.perf_counter()
    segments = provider.recognize(variants)
    inference_ms = (time.perf_counter() - inference_started_at) * 1000
    total_ocr_ms = (time.perf_counter() - processing_started_at) * 1000
    print(
        f"image={Path(args.image).resolve()} direction={direction.value} "
        f"source={image.shape[1]}x{image.shape[0]} roi={crop.shape[1]}x{crop.shape[0]} "
        f"brightness={brightness:.1f} "
        f"low_light={'yes' if brightness < LOW_LIGHT_THRESHOLD else 'no'} "
        f"detector={'on' if detector_enabled else 'off'} "
        f"detector_ms={detector_ms:.1f} plates={len(detections)} "
        f"plate_crops={','.join(f'{item.shape[1]}x{item.shape[0]}' for item in ocr_crops)} "
        f"fallback={'roi' if fallback else 'no'} "
        f"variants={len(variants)} preprocess_ms={preprocess_ms:.1f} "
        f"inference_ms={inference_ms:.1f} total_ocr_ms={total_ocr_ms:.1f}"
    )
    print("ocr_variants:")
    for index, variant in enumerate(variants):
        name = (
            OCR_VARIANT_NAMES[index]
            if index < len(OCR_VARIANT_NAMES)
            else f"variant-{index}"
        )
        print(f"  index={index} name={name} size={variant.shape[1]}x{variant.shape[0]}")
    if args.save_debug:
        paths = save_debug_images(
            args.save_debug.resolve(),
            image,
            ocr_crops[0],
            variants,
            segments,
        )
        print("debug_files:")
        for path in paths:
            print(f"  {path}")
    if not segments:
        print("OCR sonucu bulunamadı.")
        return 1

    for segment in segments:
        normalized = normalize_plate_text(segment.text)
        corrected = correct_plate_candidate(segment.text)
        print(
            f"raw={segment.text!r} normalized={normalized!r} "
            f"corrected={corrected!r} valid={bool(corrected and TurkishPlateValidator.is_valid(corrected))} "
            f"confidence={segment.confidence:.3f} variant={segment.variant_index} box={segment.box}"
        )

    best = select_best_candidate(segments, camera_id=0)
    if best is None:
        print("best=NONE (Türk plaka formatına uyan aday yok)")
    else:
        print(
            f"best={best.plate} confidence={best.confidence:.3f} "
            f"raw={best.raw_text!r}"
        )
    return 0 if best is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
