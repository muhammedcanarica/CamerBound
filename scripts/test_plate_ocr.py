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
    OCR_VARIANT_NAMES,
    SHADOW_COMPARISON_VARIANT_NAMES,
    SHADOW_OCR_VARIANT_NAMES,
    CropQualityMetrics,
    OcrModelNotFound,
    OcrImageProfile,
    OcrSegment,
    PaddleOcrProvider,
    PlateCandidate,
    TurkishPlateValidator,
    correct_plate_candidate_with_cost,
    classify_crop_quality,
    crop_roi,
    normalize_plate_text,
    measure_crop_quality,
    preprocess_roi_fallback_variants,
    preprocess_shadow_comparison_variants,
    preprocess_shadow_variants,
    preprocess_variants,
    recognize_detector_crops,
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
    variant_names: list[str]
    metrics: list[CropQualityMetrics]
    profiles: list[OcrImageProfile]
    current_variant_count: int
    shadow_variant_count: int
    inference_calls: int
    text_detection_box_counts: list[int | None]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gerçek OpenVINO + PaddleOCR saha tanılama aracı."
    )
    parser.add_argument("image")
    parser.add_argument(
        "--compare-image",
        action="append",
        default=[],
        metavar="PATH",
        help="Aynı kamera/açıdan ek shadow veya sun frame'i karşılaştırır.",
    )
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
    strategy: str,
    min_confidence: float,
    crop_detections: Sequence[PlateDetection | None] = (),
) -> OcrRun:
    metrics = [measure_crop_quality(crop) for crop in crops]
    profiles = [classify_crop_quality(item) for item in metrics]
    if strategy == "adaptive":
        result = recognize_detector_crops(
            provider,
            crops,
            camera_id=0,
            min_confidence=min_confidence,
            crop_detections=crop_detections,
        )
        return OcrRun(
            label=label,
            crops=list(crops),
            variants=list(result.variants),
            segments=list(result.segments),
            candidate=result.candidate,
            preprocess_ms=(
                result.current_preprocess_ms + result.shadow_preprocess_ms
            ),
            inference_ms=(
                result.current_inference_ms + result.shadow_inference_ms
            ),
            variant_names=list(result.variant_names),
            metrics=list(result.quality_metrics),
            profiles=list(result.profiles),
            current_variant_count=result.current_variant_count,
            shadow_variant_count=result.shadow_variant_count,
            inference_calls=result.inference_calls,
            text_detection_box_counts=list(result.text_detection_box_counts),
        )

    started = time.perf_counter()
    variants: list[np.ndarray] = []
    variant_names: list[str] = []
    for crop_index, crop in enumerate(crops):
        brightness = roi_mean_brightness(crop)
        if strategy == "roi":
            crop_variants = preprocess_roi_fallback_variants(
                crop,
                brightness=brightness,
            )
            names = [f"roi-fallback-{index}" for index in range(len(crop_variants))]
        elif strategy == "shadow-color":
            crop_variants = [preprocess_shadow_comparison_variants(crop)[0]]
            names = [SHADOW_COMPARISON_VARIANT_NAMES[0]]
        elif strategy == "shadow":
            crop_variants = preprocess_shadow_variants(
                crop,
                profile=profiles[crop_index],
            )
            names = list(SHADOW_OCR_VARIANT_NAMES[: len(crop_variants)])
        else:
            crop_variants = preprocess_variants(crop, brightness=brightness)
            names = list(OCR_VARIANT_NAMES[: len(crop_variants)])
        for name, variant in zip(names, crop_variants):
            variants.append(variant)
            variant_names.append(
                name if len(crops) == 1 else f"crop-{crop_index}-{name}"
            )
    preprocess_ms = (time.perf_counter() - started) * 1000.0
    inference_started = time.perf_counter()
    segments: list[OcrSegment] = []
    attempted: list[np.ndarray] = []
    attempted_names: list[str] = []
    text_detection_box_counts: list[int | None] = []
    inference_calls = 0
    candidate = None
    if strategy == "roi":
        for variant_index, variant in enumerate(variants):
            attempted.append(variant)
            attempted_names.append(variant_names[variant_index])
            batch = provider.recognize([variant])
            inference_calls += 1
            text_detection_box_counts.extend(
                _reported_box_counts(provider, 1, batch)
            )
            segments.extend(replace(item, variant_index=variant_index) for item in batch)
            candidate = select_best_candidate(segments, camera_id=0)
            if candidate is not None and candidate.confidence >= min_confidence:
                break
    else:
        attempted = variants
        attempted_names = variant_names
        segments = provider.recognize(variants)
        inference_calls = 1 if variants else 0
        text_detection_box_counts = list(
            _reported_box_counts(provider, len(variants), segments)
        )
        candidate = select_best_candidate(segments, camera_id=0)
    inference_ms = (time.perf_counter() - inference_started) * 1000.0
    return OcrRun(
        label=label,
        crops=list(crops),
        variants=attempted,
        segments=segments,
        candidate=candidate,
        preprocess_ms=preprocess_ms,
        inference_ms=inference_ms,
        variant_names=attempted_names,
        metrics=metrics,
        profiles=profiles,
        current_variant_count=(len(attempted) if strategy == "current" else 0),
        shadow_variant_count=(
            len(attempted)
            if strategy in {"shadow", "shadow-color"}
            else 0
        ),
        inference_calls=inference_calls,
        text_detection_box_counts=text_detection_box_counts,
    )


def _reported_box_counts(
    provider: PaddleOcrProvider,
    variant_count: int,
    segments: Sequence[OcrSegment],
) -> tuple[int | None, ...]:
    reported = getattr(provider, "last_detection_box_counts", ())
    if isinstance(reported, tuple) and len(reported) == variant_count:
        return reported
    counts: list[int | None] = [None] * variant_count
    for segment in segments:
        if 0 <= segment.variant_index < variant_count:
            counts[segment.variant_index] = (counts[segment.variant_index] or 0) + 1
    return tuple(counts)


def _print_ocr(run: OcrRun) -> None:
    for index, (metrics, profile) in enumerate(zip(run.metrics, run.profiles)):
        print(
            f"crop_metrics[{index}] plate_crop={metrics.width}x{metrics.height} "
            f"profile={profile.value} mean={metrics.luma_mean:.1f} "
            f"median={metrics.luma_median:.1f} p10={metrics.luma_p10:.1f} "
            f"p90={metrics.luma_p90:.1f} dynamic_range={metrics.dynamic_range:.1f} "
            f"stddev={metrics.grayscale_stddev:.1f} "
            f"local_contrast={metrics.local_contrast:.1f} "
            f"sharpness={metrics.laplacian_sharpness:.1f} "
            f"black_ratio={metrics.saturated_black_ratio:.3f} "
            f"white_ratio={metrics.saturated_white_ratio:.3f}"
        )
    print(
        f"ocr_stage={run.label} crops={','.join(f'{c.shape[1]}x{c.shape[0]}' for c in run.crops)} "
        f"variants={len(run.variants)} preprocess_ms={run.preprocess_ms:.1f} "
        f"inference_ms={run.inference_ms:.1f} inference_calls={run.inference_calls} "
        f"current_variants={run.current_variant_count} "
        f"shadow_variants={run.shadow_variant_count}"
    )
    segments_by_variant: dict[int, list[OcrSegment]] = {}
    for segment in run.segments:
        segments_by_variant.setdefault(segment.variant_index, []).append(segment)
    for variant_index, variant_name in enumerate(run.variant_names):
        box_count = (
            run.text_detection_box_counts[variant_index]
            if variant_index < len(run.text_detection_box_counts)
            else None
        )
        if variant_index not in segments_by_variant:
            reason = "no-text-detected" if box_count == 0 else "no-recognized-text"
            print(
                f"raw_stage={run.label} variant={variant_name} "
                f"text_detection_boxes={'unknown' if box_count is None else box_count} "
                f"raw=NONE rejection_reason={reason}"
            )
    for segment in run.segments:
        normalized = normalize_plate_text(segment.text)
        corrected_with_cost = correct_plate_candidate_with_cost(segment.text)
        corrected = corrected_with_cost[0] if corrected_with_cost else None
        correction_cost = corrected_with_cost[1] if corrected_with_cost else None
        valid = bool(corrected and TurkishPlateValidator.is_valid(corrected))
        rejection_reason = "none"
        if not valid:
            rejection_reason = "invalid-turkish-plate"
        elif segment.confidence < 0.65:
            rejection_reason = "below-min-confidence"
        variant_name = (
            run.variant_names[segment.variant_index]
            if segment.variant_index < len(run.variant_names)
            else f"variant-{segment.variant_index}"
        )
        box_count = (
            run.text_detection_box_counts[segment.variant_index]
            if segment.variant_index < len(run.text_detection_box_counts)
            else None
        )
        print(
            f"raw_stage={run.label} raw={segment.text!r} normalized={normalized!r} "
            f"corrected={corrected!r} correction_cost={correction_cost!r} "
            f"valid={valid} confidence={segment.confidence:.3f} "
            f"variant={variant_name} variant_index={segment.variant_index} "
            f"text_detection_boxes={'unknown' if box_count is None else box_count} "
            f"box={segment.box} rejection_reason={rejection_reason}"
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
    detector_crops: Sequence[np.ndarray],
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
    inputs: dict[str, np.ndarray] = {
        "01-source.jpg": source,
        "02-roi.jpg": roi,
        "03-detector-overlay.jpg": overlay,
    }
    next_index = 4
    for crop_index, crop in enumerate(detector_crops):
        inputs[f"{next_index:02d}-detector-crop-original-{crop_index}.jpg"] = crop
        next_index += 1
    inputs[f"{next_index:02d}-detector-input-direct.jpg"] = cv2.resize(
            roi,
            (detector._input_width, detector._input_height),
        )
    next_index += 1
    if letterbox is not None:
        inputs[f"{next_index:02d}-detector-input-letterbox.jpg"] = cv2.resize(
            letterbox,
            (detector._input_width, detector._input_height),
        )
        next_index += 1
    for index, (_offset, tile) in enumerate(tiles):
        inputs[f"{next_index + index:02d}-detector-input-tile-{index}.jpg"] = cv2.resize(
            tile,
            (detector._input_width, detector._input_height),
        )
    for filename, image in inputs.items():
        path = output_dir / filename
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Debug görseli yazılamadı: {path}")


def _process_image(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    image_path: str,
    debug_output: Path | None,
) -> int:
    mode = _resolved_mode(args)
    image = cv2.imread(image_path)
    if image is None:
        parser.error(f"Görsel okunamadı: {image_path}")
    config = load_config().plate_recognition
    direction = Direction(args.direction)
    roi = crop_roi(image, config.roi_for(direction))
    if roi is None:
        parser.error("ROI görselden çıkarılamadı.")
    print(
        f"image={Path(image_path).resolve()} mode={mode} direction={direction.value} "
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
    selected = select_plate_detections(
        detections,
        config.plate_detector.max_plate_candidates_per_frame,
    )
    detector_crops: list[np.ndarray] = []
    crop_detections: list[PlateDetection] = []
    for item in selected:
        crop = crop_padded_plate(
            roi,
            item,
            (
                config.plate_detector.tiled_recovery_crop_padding_ratio
                if detector.last_diagnostics is not None
                and detector.last_diagnostics.detector_variant == "tiled"
                else config.plate_detector.crop_padding_ratio
            ),
        )
        if crop is not None:
            detector_crops.append(crop)
            crop_detections.append(item)
    if debug_output is not None:
        _save_detector_debug(
            debug_output,
            image,
            roi,
            detector,
            detections,
            letterbox,
            tiles,
            detector_crops,
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

    runs: list[OcrRun] = []
    if mode in {"production", "detector-ocr"} and detector_crops:
        runs.append(
            _run_ocr(
                provider,
                detector_crops,
                label="detector-ocr",
                strategy="adaptive",
                min_confidence=config.min_confidence,
                crop_detections=crop_detections,
            )
        )
    elif mode in {"production", "detector-ocr"}:
        print("ocr_stage=detector-ocr skipped=no-usable-detector-crop")
    if mode == "compare" and detector_crops:
        runs.extend(
            (
                _run_ocr(
                    provider,
                    detector_crops,
                    label="current-ocr",
                    strategy="current",
                    min_confidence=config.min_confidence,
                    crop_detections=crop_detections,
                ),
                _run_ocr(
                    provider,
                    detector_crops,
                    label="shadow-color-ocr",
                    strategy="shadow-color",
                    min_confidence=config.min_confidence,
                    crop_detections=crop_detections,
                ),
                _run_ocr(
                    provider,
                    detector_crops,
                    label="shadow-ocr",
                    strategy="shadow",
                    min_confidence=config.min_confidence,
                    crop_detections=crop_detections,
                ),
            )
        )
    elif mode == "compare":
        print("ocr_stage=current-ocr skipped=no-usable-detector-crop")
        print("ocr_stage=shadow-color-ocr skipped=no-usable-detector-crop")
        print("ocr_stage=shadow-ocr skipped=no-usable-detector-crop")
    if mode in {"roi-ocr", "compare"}:
        runs.append(
            _run_ocr(
                provider,
                [roi],
                label="roi-ocr",
                strategy="roi",
                min_confidence=config.min_confidence,
            )
        )
    for run in runs:
        _print_ocr(run)
        if debug_output is not None:
            save_debug_images(
                debug_output / run.label,
                image,
                run.crops[0],
                run.variants,
                run.segments,
                run.variant_names,
            )
    candidates = [run.candidate for run in runs if run.candidate is not None]
    if candidates:
        best = max(
            candidates,
            key=lambda item: (
                item.variant_support,
                -item.correction_cost,
                item.confidence,
                item.plate,
            ),
        )
        print(
            f"BEST CANDIDATE plate={best.plate} confidence={best.confidence:.3f} "
            f"raw={best.raw_text!r} correction_cost={best.correction_cost}"
        )
    else:
        print("BEST CANDIDATE none")
    return 0 if any(run.candidate is not None for run in runs) else 1


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    image_paths = [args.image, *args.compare_image]
    statuses: list[int] = []
    for index, image_path in enumerate(image_paths):
        debug_output = args.save_debug.resolve() if args.save_debug else None
        if debug_output is not None and len(image_paths) > 1:
            debug_output = debug_output / f"{index + 1:02d}-{Path(image_path).stem}"
        print(f"=== IMAGE {index + 1}/{len(image_paths)} ===")
        statuses.append(
            _process_image(
                args,
                parser,
                image_path,
                debug_output,
            )
        )
    return 0 if any(status == 0 for status in statuses) else max(statuses)


if __name__ == "__main__":
    raise SystemExit(main())
