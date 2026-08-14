from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.camera import Direction
from app.config import load_config
from app.ocr_debug import save_debug_images
from app.plate_detector import (
    OpenVinoPlateDetector,
    PlateDetection,
    crop_padded_plate,
    detector_recovery_tiles,
    select_plate_detections,
)
from app.plate_recognition import (
    LOW_LIGHT_THRESHOLD,
    OcrModelNotFound,
    OcrSegment,
    PaddleOcrProvider,
    PlateCandidate,
    TurkishPlateValidator,
    correct_plate_candidate,
    crop_roi,
    normalize_plate_text,
    preprocess_roi_fallback_variants,
    preprocess_variants,
    roi_mean_brightness,
    select_best_candidate,
)


@dataclass(slots=True)
class OcrRun:
    label: str
    crops: list[np.ndarray]
    variants: list[np.ndarray]
    segments: list[OcrSegment]
    candidate: PlateCandidate | None
    preprocess_ms: float
    inference_ms: float


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gerçek OpenVINO + PaddleOCR saha tanılama aracı."
    )
    parser.add_argument("image")
    parser.add_argument("--direction", choices=("ENTRY", "EXIT"), default="ENTRY")
    parser.add_argument(
        "--mode",
        choices=("production", "detector-only", "detector-ocr", "roi-ocr", "compare"),
        help="Çalıştırılacak açık pipeline aşaması (varsayılan: production).",
    )
    parser.add_argument(
        "--detector-mode",
        choices=("config", "on", "off"),
        help="Geriye dönük uyumluluk: on=detector-ocr, off=roi-ocr.",
    )
    parser.add_argument(
        "--save-debug",
        type=Path,
        metavar="DIR",
        help="Detector girişleri, ROI/crop'lar, varyantlar ve OCR kutularını kaydeder.",
    )
    return parser


def _resolved_mode(args: argparse.Namespace) -> str:
    if args.mode:
        return args.mode
    return {"on": "detector-ocr", "off": "roi-ocr"}.get(
        args.detector_mode,
        "production",
    )


def _print_detections(
    detector: OpenVinoPlateDetector,
    detections: Sequence[PlateDetection],
    roi: np.ndarray,
    elapsed_ms: float,
) -> None:
    diagnostics = detector.last_diagnostics
    if diagnostics is not None:
        highest = (
            "none"
            if diagnostics.highest_plate_confidence is None
            else f"{diagnostics.highest_plate_confidence:.3f}"
        )
        print(
            f"detector_variant={diagnostics.detector_variant} "
            f"detections={len(detections)} raw_candidates={diagnostics.raw_candidate_count} "
            f"plate_class_candidates={diagnostics.plate_class_candidate_count} "
            f"highest_plate_confidence={highest} "
            f"confidence_rejected={diagnostics.confidence_rejected_count} "
            f"bbox_rejected={diagnostics.bbox_rejected_count} "
            f"detector_input={diagnostics.input_width}x{diagnostics.input_height} "
            f"input_layout={diagnostics.input_layout} input_dtype={diagnostics.input_dtype} "
            f"roi_size={roi.shape[1]}x{roi.shape[0]} "
            f"resize_scale_x={diagnostics.resize_scale_x:.4f} "
            f"resize_scale_y={diagnostics.resize_scale_y:.4f} "
            f"aspect_distortion_ratio={diagnostics.aspect_distortion_ratio:.2f} "
            f"tiled_recovery={'yes' if diagnostics.tiled_recovery_pass else 'no'} "
            f"recovery_tiles={diagnostics.recovery_tile_count} detector_ms={elapsed_ms:.1f}"
        )
    for index, detection in enumerate(detections):
        print(
            f"detector_bbox[{index}] confidence={detection.confidence:.3f} "
            f"x={detection.x} y={detection.y} "
            f"width={detection.width} height={detection.height}"
        )


def _raw_detector_compare(
    detector: OpenVinoPlateDetector,
    roi: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, np.ndarray]]]:
    height, width = roi.shape[:2]
    direct, _ = detector._infer_detections(  # development diagnostic only
        roi,
        coordinate_width=width,
        coordinate_height=height,
    )
    side = max(width, height)
    square = np.full((side, side, 3), 114, dtype=np.uint8)
    y_offset = (side - height) // 2
    square[y_offset : y_offset + height, :width] = roi
    letterbox, _ = detector._infer_detections(
        square,
        coordinate_width=side,
        coordinate_height=side,
    )
    tiles = list(detector_recovery_tiles(roi))
    tile_detections: list[PlateDetection] = []
    for x_offset, tile in tiles:
        detected, _ = detector._infer_detections(
            tile,
            coordinate_width=tile.shape[1],
            coordinate_height=tile.shape[0],
        )
        tile_detections.extend(
            replace(item, x=item.x + x_offset) for item in detected
        )
    print(
        "detector_compare "
        f"direct={len(direct)} direct_best={_best_confidence(direct)} "
        f"letterbox={len(letterbox)} letterbox_best={_best_confidence(letterbox)} "
        f"tiles={len(tiles)} tiled={len(tile_detections)} "
        f"tiled_best={_best_confidence(tile_detections)} "
        f"direct_aspect_distortion={max(width / height, height / width):.2f}"
    )
    return square, tiles


def _best_confidence(detections: Sequence[PlateDetection]) -> str:
    return "none" if not detections else f"{max(item.confidence for item in detections):.3f}"


def _run_ocr(
    provider: PaddleOcrProvider,
    crops: Sequence[np.ndarray],
    *,
    label: str,
    fallback: bool,
    min_confidence: float,
) -> OcrRun:
    started = time.perf_counter()
    variants: list[np.ndarray] = []
    for crop in crops:
        brightness = roi_mean_brightness(crop)
        variants.extend(
            preprocess_roi_fallback_variants(crop, brightness=brightness)
            if fallback
            else preprocess_variants(crop, brightness=brightness)
        )
    preprocess_ms = (time.perf_counter() - started) * 1000.0
    inference_started = time.perf_counter()
    segments: list[OcrSegment] = []
    attempted: list[np.ndarray] = []
    candidate = None
    if fallback:
        for variant_index, variant in enumerate(variants):
            attempted.append(variant)
            batch = provider.recognize([variant])
            segments.extend(replace(item, variant_index=variant_index) for item in batch)
            candidate = select_best_candidate(segments, camera_id=0)
            if candidate is not None and candidate.confidence >= min_confidence:
                break
    else:
        attempted = variants
        segments = provider.recognize(variants)
        candidate = select_best_candidate(segments, camera_id=0)
    inference_ms = (time.perf_counter() - inference_started) * 1000.0
    return OcrRun(
        label,
        list(crops),
        attempted,
        segments,
        candidate,
        preprocess_ms,
        inference_ms,
    )


def _print_ocr(run: OcrRun) -> None:
    print(
        f"ocr_stage={run.label} crops={','.join(f'{c.shape[1]}x{c.shape[0]}' for c in run.crops)} "
        f"variants={len(run.variants)} preprocess_ms={run.preprocess_ms:.1f} "
        f"inference_ms={run.inference_ms:.1f}"
    )
    for segment in run.segments:
        normalized = normalize_plate_text(segment.text)
        corrected = correct_plate_candidate(segment.text)
        valid = bool(corrected and TurkishPlateValidator.is_valid(corrected))
        print(
            f"raw_stage={run.label} raw={segment.text!r} normalized={normalized!r} "
            f"corrected={corrected!r} valid={valid} confidence={segment.confidence:.3f} "
            f"variant={segment.variant_index} box={segment.box}"
        )
    if run.candidate is None:
        print(f"best_stage={run.label} best=NONE rejection=no-valid-turkish-plate")
    else:
        print(
            f"best_stage={run.label} best={run.candidate.plate} "
            f"confidence={run.candidate.confidence:.3f} raw={run.candidate.raw_text!r}"
        )


def _save_detector_debug(
    output_dir: Path,
    source: np.ndarray,
    roi: np.ndarray,
    detector: OpenVinoPlateDetector,
    detections: Sequence[PlateDetection],
    letterbox: np.ndarray | None,
    tiles: Sequence[tuple[int, np.ndarray]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay = roi.copy()
    for item in detections:
        cv2.rectangle(
            overlay,
            (item.x, item.y),
            (item.x + item.width, item.y + item.height),
            (0, 255, 0),
            2,
        )
    inputs = {
        "01-source.jpg": source,
        "02-roi.jpg": roi,
        "03-detector-overlay.jpg": overlay,
        "04-detector-input-direct.jpg": cv2.resize(
            roi,
            (detector._input_width, detector._input_height),
        ),
    }
    if letterbox is not None:
        inputs["05-detector-input-letterbox.jpg"] = cv2.resize(
            letterbox,
            (detector._input_width, detector._input_height),
        )
    for index, (_offset, tile) in enumerate(tiles):
        inputs[f"{6 + index:02d}-detector-input-tile-{index}.jpg"] = cv2.resize(
            tile,
            (detector._input_width, detector._input_height),
        )
    for filename, image in inputs.items():
        path = output_dir / filename
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Debug görseli yazılamadı: {path}")


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    mode = _resolved_mode(args)
    image = cv2.imread(args.image)
    if image is None:
        parser.error(f"Görsel okunamadı: {args.image}")
    config = load_config().plate_recognition
    direction = Direction(args.direction)
    roi = crop_roi(image, config.roi_for(direction))
    if roi is None:
        parser.error("ROI görselden çıkarılamadı.")
    print(
        f"image={Path(args.image).resolve()} mode={mode} direction={direction.value} "
        f"source={image.shape[1]}x{image.shape[0]} roi={roi.shape[1]}x{roi.shape[0]} "
        f"brightness={roi_mean_brightness(roi):.1f} "
        f"low_light={'yes' if roi_mean_brightness(roi) < LOW_LIGHT_THRESHOLD else 'no'}"
    )

    detector = OpenVinoPlateDetector(config.plate_detector)
    detector_started = time.perf_counter()
    detections = detector.detect(roi)
    detector_ms = (time.perf_counter() - detector_started) * 1000.0
    _print_detections(detector, detections, roi, detector_ms)
    letterbox = None
    tiles: list[tuple[int, np.ndarray]] = []
    if mode == "compare":
        letterbox, tiles = _raw_detector_compare(detector, roi)
    if args.save_debug:
        _save_detector_debug(
            args.save_debug.resolve(),
            image,
            roi,
            detector,
            detections,
            letterbox,
            tiles,
        )
    if mode == "detector-only":
        return 0 if detections else 1

    try:
        provider = PaddleOcrProvider(
            config.model_root,
            backend=config.ocr_backend,
            cpu_threads=config.ocr_cpu_threads,
        )
    except OcrModelNotFound as exc:
        print(str(exc))
        return 2

    selected = select_plate_detections(
        detections,
        config.plate_detector.max_plate_candidates_per_frame,
    )
    detector_crops = [
        crop
        for item in selected
        if (
            crop := crop_padded_plate(
                roi,
                item,
                (
                    config.plate_detector.tiled_recovery_crop_padding_ratio
                    if detector.last_diagnostics is not None
                    and detector.last_diagnostics.detector_variant == "tiled"
                    else config.plate_detector.crop_padding_ratio
                ),
            )
        )
        is not None
    ]
    runs: list[OcrRun] = []
    if mode in {"production", "detector-ocr", "compare"} and detector_crops:
        runs.append(
            _run_ocr(
                provider,
                detector_crops,
                label="detector-ocr",
                fallback=False,
                min_confidence=config.min_confidence,
            )
        )
    elif mode in {"production", "detector-ocr"}:
        print("ocr_stage=detector-ocr skipped=no-usable-detector-crop")
    if mode in {"roi-ocr", "compare"}:
        runs.append(
            _run_ocr(
                provider,
                [roi],
                label="roi-ocr",
                fallback=True,
                min_confidence=config.min_confidence,
            )
        )
    for run in runs:
        _print_ocr(run)
        if args.save_debug:
            save_debug_images(
                args.save_debug.resolve() / run.label,
                image,
                run.crops[0],
                run.variants,
                run.segments,
            )
    return 0 if any(run.candidate is not None for run in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
