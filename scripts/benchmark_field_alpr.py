from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.camera import Direction
from app.config import load_config
from app.database import Database
from app.ocr_models import read_model_name, select_ocr_backend
from app.plate_capture import PlateCaptureService
from app.plate_detector import (
    OpenVinoPlateDetector,
    PlateDetection,
    crop_padded_plate,
    plate_detection_geometry_quality,
    plate_detection_ranking_score,
    select_plate_detections,
)
from app.plate_recognition import (
    OcrImageProfile,
    OcrSegment,
    PaddleOcrProvider,
    PlateDetectionProcessor,
    PlateCandidate,
    PlateRecognitionProcessor,
    TurkishPlateValidator,
    correct_plate_candidate_with_cost,
    crop_roi,
    normalize_plate_text,
    preprocess_roi_fallback_variants,
    recognize_detector_crops,
    recognize_ocr_search_tiles,
    roi_mean_brightness,
    select_best_candidate,
)
from app.plate_service import PlateService


@dataclass(frozen=True, slots=True)
class FieldSample:
    image: str
    direction: str
    expected_plate: str | None
    condition: str
    plate_present: bool = True


@dataclass(frozen=True, slots=True)
class SampleResult:
    image: str
    direction: str
    condition: str
    expected_plate: str | None
    candidate: str | None
    confidence: float | None
    detector_hit: bool
    detector_variant: str
    detector_detections: int
    detector_ms: float
    ocr_ms: float
    end_to_end_ms: float
    used_roi_fallback: bool
    raw_segment_count: int
    valid_candidate: bool
    low_confidence: bool
    exact: bool
    character_accuracy: float
    crop_profiles: tuple[str, ...]
    inference_calls: int
    rescue_tiles: int = 0
    ocr_provider_mode: str = "text-detection-and-recognition"
    crop_resolutions: tuple[str, ...] = ()
    detector_bboxes: tuple[dict[str, object], ...] = ()
    preprocessing_variants: tuple[str, ...] = ()
    variant_resolutions: tuple[str, ...] = ()
    text_detection_box_counts: tuple[int | None, ...] = ()
    raw_ocr_segments: tuple[dict[str, object], ...] = ()
    rejection_reason: str = "none"
    detector_crop_rescue_attempted: bool = False


class RecognitionOnlyProvider:
    """Development-only adapter for detector-crop recognition A/B tests."""

    def __init__(self, model_root: Path, backend: str, cpu_threads: int) -> None:
        from paddleocr import TextRecognition

        selection = select_ocr_backend(model_root, backend, load_onnx=True)
        recognition_dir = selection.model_root / "recognition"
        options: dict[str, object] = {
            "model_name": read_model_name(recognition_dir),
            "model_dir": str(recognition_dir),
            "device": "cpu",
            "enable_hpi": False,
            "cpu_threads": max(1, int(cpu_threads)),
        }
        if selection.backend.value == "onnx":
            options["engine"] = "onnxruntime"
        else:
            options["enable_mkldnn"] = False
        self._predictor = TextRecognition(**options)
        self.last_detection_box_counts: tuple[int, ...] = ()

    def recognize(self, images: Sequence[np.ndarray]) -> list[OcrSegment]:
        if not images:
            return []
        segments: list[OcrSegment] = []
        results = self._predictor.predict(input=list(images))
        for index, result in enumerate(results):
            payload = getattr(result, "json", {})
            data = payload.get("res", payload) if isinstance(payload, dict) else {}
            text = data.get("rec_text") if isinstance(data, dict) else None
            score = data.get("rec_score", 0.0) if isinstance(data, dict) else 0.0
            if isinstance(text, str) and text.strip():
                image = images[index]
                segments.append(
                    OcrSegment(
                        text=text,
                        confidence=float(score),
                        box=(0.0, 0.0, float(image.shape[1]), float(image.shape[0])),
                        variant_index=index,
                    )
                )
        self.last_detection_box_counts = tuple(1 for _ in images)
        return segments


def image_resolution(image: np.ndarray) -> str:
    return f"{image.shape[1]}x{image.shape[0]}"


def raw_ocr_trace(
    segments: Sequence[OcrSegment],
    variant_names: Sequence[str],
    detection_box_counts: Sequence[int | None],
    min_confidence: float,
) -> tuple[dict[str, object], ...]:
    trace: list[dict[str, object]] = []
    segments_by_variant: dict[int, list[OcrSegment]] = {}
    for segment in segments:
        segments_by_variant.setdefault(segment.variant_index, []).append(segment)
    for variant_index, variant_name in enumerate(variant_names):
        box_count = (
            detection_box_counts[variant_index]
            if variant_index < len(detection_box_counts)
            else None
        )
        variant_segments = segments_by_variant.get(variant_index, [])
        if not variant_segments:
            trace.append(
                {
                    "variant": variant_name,
                    "variant_index": variant_index,
                    "text_detection_box_count": box_count,
                    "raw_text": None,
                    "confidence": None,
                    "normalized": None,
                    "corrected": None,
                    "correction_cost": None,
                    "valid": False,
                    "rejection_reason": (
                        "no-text-detected"
                        if box_count == 0
                        else "no-recognized-text"
                    ),
                }
            )
            continue
        for segment in variant_segments:
            corrected_with_cost = correct_plate_candidate_with_cost(segment.text)
            corrected = corrected_with_cost[0] if corrected_with_cost else None
            correction_cost = (
                corrected_with_cost[1] if corrected_with_cost else None
            )
            valid = bool(corrected and TurkishPlateValidator.is_valid(corrected))
            rejection_reason = "none"
            if not valid:
                rejection_reason = "invalid-turkish-plate"
            elif segment.confidence < min_confidence:
                rejection_reason = "below-min-confidence"
            trace.append(
                {
                    "variant": variant_name,
                    "variant_index": variant_index,
                    "text_detection_box_count": box_count,
                    "raw_text": segment.text,
                    "confidence": segment.confidence,
                    "normalized": normalize_plate_text(segment.text),
                    "corrected": corrected,
                    "correction_cost": correction_cost,
                    "valid": valid,
                    "rejection_reason": rejection_reason,
                    "box": segment.box,
                }
            )
    return tuple(trace)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local, gitignored real-field OpenVINO + PaddleOCR benchmark."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--label", default="field")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Optional earlier benchmark JSON used to emit metric deltas.",
    )
    parser.add_argument(
        "--recognition-only-ab",
        action="store_true",
        help="Also benchmark direct recognition on the exact detector crops.",
    )
    parser.add_argument(
        "--decision-trace",
        action="store_true",
        help=(
            "Replay real OCR outputs from distinct samples through production "
            "confirmation, stabilization and an isolated PlateService database."
        ),
    )
    return parser.parse_args()


def load_manifest(path: Path) -> tuple[FieldSample, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Field manifest must contain a non-empty JSON array.")
    samples: list[FieldSample] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest entry {index} must be an object.")
        sample = FieldSample(
            image=str(item["image"]),
            direction=str(item.get("direction", "ENTRY")).upper(),
            expected_plate=(
                str(item["expected_plate"]).upper()
                if item.get("expected_plate") is not None
                else None
            ),
            condition=str(item.get("condition", "unspecified")),
            plate_present=bool(item.get("plate_present", True)),
        )
        Direction(sample.direction)
        if sample.expected_plate is not None and not TurkishPlateValidator.is_valid(
            sample.expected_plate
        ):
            raise ValueError(
                f"Manifest entry {index} has invalid expected_plate: "
                f"{sample.expected_plate}"
            )
        samples.append(sample)
    return tuple(samples)


def resolve_image(path: str, manifest_path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    project_candidate = (PROJECT_ROOT / candidate).resolve()
    if project_candidate.is_file():
        return project_candidate
    return (manifest_path.parent / candidate).resolve()


def detector_crops(
    image: np.ndarray,
    direction: Direction,
    detector: OpenVinoPlateDetector,
    config,
) -> tuple[np.ndarray, list[PlateDetection], list[np.ndarray], float]:
    roi = crop_roi(image, config.roi_for(direction))
    if roi is None:
        raise ValueError("Configured ROI is empty for field image.")
    started = time.perf_counter()
    detections = detector.detect(roi)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    selected = select_plate_detections(
        detections,
        config.plate_detector.max_plate_candidates_per_frame,
        roi_width=roi.shape[1],
        roi_height=roi.shape[0],
    )
    padding = (
        config.plate_detector.tiled_recovery_crop_padding_ratio
        if detector.last_diagnostics is not None
        and detector.last_diagnostics.detector_variant == "tiled"
        else config.plate_detector.crop_padding_ratio
    )
    crops: list[np.ndarray] = []
    crop_detections: list[PlateDetection] = []
    for detection in selected:
        crop = crop_padded_plate(roi, detection, padding)
        if crop is not None:
            crops.append(crop)
            crop_detections.append(detection)
    return roi, crop_detections, crops, elapsed_ms


def recognize_roi_fallback(provider, roi: np.ndarray, min_confidence: float):
    search = recognize_ocr_search_tiles(provider, roi, 0, min_confidence)
    segments: list[OcrSegment] = list(search.segments)
    candidate = search.candidate
    calls = search.inference_calls
    started = time.perf_counter()
    if candidate is not None and candidate.confidence >= min_confidence:
        return candidate, segments, calls, search.inference_ms, calls
    variants = preprocess_roi_fallback_variants(
        roi, brightness=roi_mean_brightness(roi)
    )
    for fallback_index, variant in enumerate(variants):
        calls += 1
        batch = provider.recognize([variant])
        variant_index = len(search.tiles) + fallback_index
        segments.extend(
            OcrSegment(
                item.text,
                item.confidence,
                item.box,
                variant_index,
            )
            for item in batch
        )
        candidate = select_best_candidate(segments, camera_id=0)
        if candidate is not None and candidate.confidence >= min_confidence:
            break
    return (
        candidate,
        segments,
        calls,
        search.inference_ms + (time.perf_counter() - started) * 1000.0,
        search.inference_calls,
    )


def run_sample(
    sample,
    manifest_path,
    detector,
    provider,
    config,
    *,
    fallback_provider=None,
) -> SampleResult:
    path = resolve_image(sample.image, manifest_path)
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Field image could not be read: {path}")
    direction = Direction(sample.direction)
    total_started = time.perf_counter()
    roi, crop_detections, crops, detector_ms = detector_crops(
        image, direction, detector, config
    )
    diagnostics = detector.last_diagnostics
    used_roi_fallback = not crops
    detector_crop_rescue_attempted = False
    if crops:
        ocr = recognize_detector_crops(
            provider,
            crops,
            camera_id=0,
            min_confidence=config.min_confidence,
            crop_detections=crop_detections,
        )
        candidate = ocr.candidate
        segments = list(ocr.segments)
        inference_calls = ocr.inference_calls
        ocr_ms = ocr.current_inference_ms + ocr.shadow_inference_ms
        profiles = tuple(item.value for item in ocr.profiles)
        variant_names = list(ocr.variant_names)
        variants = list(ocr.variants)
        detection_box_counts = list(
            tuple(None for _ in variants)
            if isinstance(provider, RecognitionOnlyProvider)
            else ocr.text_detection_box_counts
        )
        rescue_tiles = 0
        if candidate is None or candidate.confidence < config.min_confidence:
            detector_crop_rescue_attempted = True
            search = recognize_ocr_search_tiles(
                fallback_provider or provider,
                roi,
                0,
                config.min_confidence,
            )
            rescue_offset = len(variants)
            attempted_tiles = search.tiles[: search.inference_calls]
            variants.extend(tile.image for tile in attempted_tiles)
            variant_names.extend(
                "detector-crop-rescue-tile-"
                f"{index}[x={tile.roi_box[0]},w={tile.roi_box[2]}]"
                for index, tile in enumerate(attempted_tiles)
            )
            rescue_segments = [
                replace(
                    segment,
                    variant_index=segment.variant_index + rescue_offset,
                )
                for segment in search.segments
            ]
            segments.extend(rescue_segments)
            detection_box_counts.extend(search.text_detection_box_counts)
            rescue_candidate = select_best_candidate(rescue_segments, camera_id=0)
            if (
                rescue_candidate is not None
                and rescue_candidate.confidence >= config.min_confidence
            ):
                candidate = rescue_candidate
                used_roi_fallback = True
            elif candidate is None and rescue_candidate is not None:
                candidate = rescue_candidate
                used_roi_fallback = True
            rescue_tiles = search.inference_calls
            inference_calls += search.inference_calls
            ocr_ms += search.inference_ms
    else:
        (
            candidate,
            segments,
            inference_calls,
            ocr_ms,
            rescue_tiles,
        ) = recognize_roi_fallback(
            fallback_provider or provider,
            roi,
            config.min_confidence,
        )
        profiles = (OcrImageProfile.NORMAL.value,)
        variant_names = ()
        variants = ()
        detection_box_counts = ()
    expected = sample.expected_plate
    actual = candidate.plate if candidate is not None else None
    return SampleResult(
        image=sample.image,
        direction=sample.direction,
        condition=sample.condition,
        expected_plate=expected,
        candidate=actual,
        confidence=candidate.confidence if candidate is not None else None,
        detector_hit=bool(crops),
        detector_variant=(
            diagnostics.detector_variant if diagnostics is not None else "unknown"
        ),
        detector_detections=len(crop_detections),
        detector_ms=detector_ms,
        ocr_ms=ocr_ms,
        end_to_end_ms=(time.perf_counter() - total_started) * 1000.0,
        used_roi_fallback=used_roi_fallback,
        raw_segment_count=len(segments),
        valid_candidate=candidate is not None,
        low_confidence=(
            candidate is not None and candidate.confidence < config.min_confidence
        ),
        exact=expected is not None and actual == expected,
        character_accuracy=character_accuracy(expected, actual),
        crop_profiles=profiles,
        inference_calls=inference_calls,
        rescue_tiles=rescue_tiles,
        ocr_provider_mode=(
            "recognition-only"
            if isinstance(provider, RecognitionOnlyProvider) and crops
            else "text-detection-and-recognition"
        ),
        crop_resolutions=tuple(image_resolution(crop) for crop in crops),
        detector_bboxes=tuple(
            {
                "x": detection.x,
                "y": detection.y,
                "width": detection.width,
                "height": detection.height,
                "confidence": detection.confidence,
                "aspect_ratio": detection.width / max(1, detection.height),
                "area": detection.area,
                "roi_area_ratio": detection.area / max(1, roi.shape[1] * roi.shape[0]),
                "relative_x": detection.x / max(1, roi.shape[1]),
                "relative_y": detection.y / max(1, roi.shape[0]),
                "geometry_quality": plate_detection_geometry_quality(
                    detection,
                    roi_width=roi.shape[1],
                    roi_height=roi.shape[0],
                ),
                "ranking_score": plate_detection_ranking_score(
                    detection,
                    roi_width=roi.shape[1],
                    roi_height=roi.shape[0],
                ),
            }
            for detection in crop_detections
        ),
        preprocessing_variants=tuple(variant_names),
        variant_resolutions=tuple(image_resolution(variant) for variant in variants),
        text_detection_box_counts=tuple(detection_box_counts),
        raw_ocr_segments=raw_ocr_trace(
            segments,
            variant_names,
            detection_box_counts,
            config.min_confidence,
        ),
        rejection_reason=(
            "no-ocr-text"
            if not segments
            else "no-valid-plate"
            if candidate is None
            else "below-min-confidence"
            if candidate.confidence < config.min_confidence
            else "none"
        ),
        detector_crop_rescue_attempted=detector_crop_rescue_attempted,
    )


class RecordedOcrSequenceProvider:
    """Replay real model segments once per distinct frame for downstream tracing."""

    def __init__(self, results: Sequence[SampleResult]) -> None:
        self._batches = [
            [
                OcrSegment(
                    text=str(item["raw_text"]),
                    confidence=float(item["confidence"]),
                    box=tuple(item.get("box", (0.0, 0.0, 1.0, 1.0))),
                    variant_index=int(item["variant_index"]),
                )
                for item in result.raw_ocr_segments
                if item.get("raw_text") is not None
            ]
            for result in results
        ]
        self._box_counts = [result.text_detection_box_counts for result in results]
        self.last_detection_box_counts: tuple[int, ...] = ()

    def recognize(self, images: Sequence[np.ndarray]) -> list[OcrSegment]:
        if not self._batches:
            return []
        batch = self._batches.pop(0)
        counts = self._box_counts.pop(0)
        if len(counts) == len(images) and all(value is not None for value in counts):
            self.last_detection_box_counts = tuple(int(value) for value in counts)
        else:
            self.last_detection_box_counts = ()
        return batch


def production_decision_trace(
    samples: Sequence[FieldSample],
    results: Sequence[SampleResult],
    manifest_path: Path,
    detector: OpenVinoPlateDetector,
    config,
) -> dict[str, object]:
    expected = {sample.expected_plate for sample in samples}
    if len(samples) < 2 or None in expected or len(expected) != 1:
        raise ValueError(
            "--decision-trace requires at least two distinct samples with one "
            "shared expected_plate."
        )
    if any(result.candidate is None for result in results):
        raise ValueError("--decision-trace requires a candidate for every sample.")

    debug_root = PROJECT_ROOT / ".test-tmp"
    debug_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="field-decision-trace-", dir=debug_root
    ) as temporary:
        root = Path(temporary)
        database = Database(root / "trace.db")
        database.initialize()
        direction = Direction(samples[0].direction)
        with database.connection() as connection:
            connection.execute(
                """
                UPDATE cameras
                SET name = 'Field trace', direction = ?, enabled = 0
                WHERE id = 1
                """,
                (direction.value,),
            )
        capture_service = PlateCaptureService(root / "captures", root)
        plate_service = PlateService(
            database,
            duplicate_cooldown_seconds=120,
            capture_service=capture_service,
        )
        processor = PlateRecognitionProcessor(
            RecordedOcrSequenceProvider(results),
            plate_service,
            config,
        )
        detection_processor = PlateDetectionProcessor(config, detector)
        base_observed_at = time.monotonic()
        base_captured_at = datetime.now(timezone.utc)
        frame_trace: list[dict[str, object]] = []
        for index, (sample, result) in enumerate(zip(samples, results), 1):
            image_path = resolve_image(sample.image, manifest_path)
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise FileNotFoundError(f"Field image could not be read: {image_path}")
            observed_at = base_observed_at + index * 0.25
            captured_at = base_captured_at + timedelta(milliseconds=index * 250)
            detection = detection_processor.prepare_job(
                1,
                direction,
                frame,
                captured_at=captured_at,
                observed_at=observed_at,
                received_at=observed_at,
                frame_id=index,
                detector_source="field-decision-trace",
                allow_zero_detection_fallback=False,
            )
            if detection.job is None:
                raise RuntimeError(
                    f"Distinct trace frame {index} did not produce an OCR job."
                )
            outcome = processor.process_ocr_job(detection.job, queue_depth=0)
            frame_trace.append(
                {
                    "frame_id": index,
                    "image": sample.image,
                    "detector_hit": bool(detection.detections),
                    "crop_resolutions": list(result.crop_resolutions),
                    "ocr_provider": result.ocr_provider_mode,
                    "raw_ocr_segments": list(result.raw_ocr_segments),
                    "candidate": (
                        outcome.candidate.plate
                        if outcome.candidate is not None
                        else None
                    ),
                    "confirmation": (
                        f"{outcome.confirmation_count}/"
                        f"{outcome.confirmation_required}"
                    ),
                    "state": outcome.state.value,
                    "rejection_reason": outcome.suppression_reason or "none",
                }
            )
        deadline = processor.next_pending_deadline()
        finalized = processor.finalize_due(
            None if deadline is None else deadline + 0.01
        )
        final_outcome = finalized[0][1] if finalized else None
        with database.connection() as connection:
            persisted = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT plate, direction, camera_id, confidence, timestamp, image_path
                    FROM plate_records ORDER BY id
                    """
                ).fetchall()
            ]
        return {
            "isolation": (
                "Real detector jobs and recorded real Paddle outputs, replayed once "
                "per distinct frame through production confirmation, stabilization "
                "and an isolated real PlateService database."
            ),
            "frames": frame_trace,
            "finalization": {
                "state": final_outcome.state.value if final_outcome else "none",
                "candidate": (
                    final_outcome.candidate.plate
                    if final_outcome is not None
                    and final_outcome.candidate is not None
                    else None
                ),
                "confirmation": (
                    f"{final_outcome.confirmation_count}/"
                    f"{final_outcome.confirmation_required}"
                    if final_outcome is not None
                    else "0/0"
                ),
                "rejection_reason": (
                    final_outcome.suppression_reason or "none"
                    if final_outcome is not None
                    else "no-due-decision"
                ),
            },
            "persisted_records": persisted,
        }


def character_accuracy(expected: str | None, actual: str | None) -> float:
    if expected is None:
        return 1.0 if actual is None else 0.0
    distance = edit_distance(expected, actual or "")
    return max(0.0, 1.0 - distance / max(1, len(expected)))


def edit_distance(first: str, second: str) -> int:
    previous = list(range(len(second) + 1))
    for first_index, first_char in enumerate(first, 1):
        current = [first_index]
        for second_index, second_char in enumerate(second, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[second_index] + 1,
                    previous[second_index - 1] + (first_char != second_char),
                )
            )
        previous = current
    return previous[-1]


def confusion_pairs(expected: str, actual: str) -> Counter[str]:
    if len(expected) != len(actual):
        return Counter()
    return Counter(
        f"{wanted}/{seen}"
        for wanted, seen in zip(expected, actual)
        if wanted != seen
    )


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def summarize(label: str, samples, results) -> dict[str, object]:
    positives = [sample for sample in samples if sample.plate_present]
    negatives = [sample for sample in samples if not sample.plate_present]
    positive_results = [
        result for sample, result in zip(samples, results) if sample.plate_present
    ]
    detector_hits = sum(result.detector_hit for result in positive_results)
    conditional = [result for result in positive_results if result.detector_hit]
    exact = sum(result.exact for result in positive_results)
    false_reads = sum(
        result.candidate is not None and not result.exact for result in positive_results
    )
    no_reads = sum(result.candidate is None for result in positive_results)
    false_detections = sum(
        result.detector_hit
        for sample, result in zip(samples, results)
        if not sample.plate_present
    )
    confusions: Counter[str] = Counter()
    for result in positive_results:
        if result.expected_plate and result.candidate:
            confusions.update(confusion_pairs(result.expected_plate, result.candidate))
    detector_times = [result.detector_ms for result in results]
    ocr_times = [result.ocr_ms for result in results]
    total_times = [result.end_to_end_ms for result in results]
    rescue_results = [result for result in results if result.rescue_tiles > 0]
    detector_crop_rescues = [
        result for result in results if result.detector_crop_rescue_attempted
    ]

    def ratio(value: int, denominator: int) -> float:
        return value / denominator if denominator else 0.0

    return {
        "label": label,
        "sample_count": len(samples),
        "positive_count": len(positives),
        "detector": {
            "recall": ratio(detector_hits, len(positives)),
            "primary_recall": ratio(
                sum(
                    result.detector_hit and result.detector_variant == "raw"
                    for result in positive_results
                ),
                len(positives),
            ),
            "tiled_recovery_gain": ratio(
                sum(
                    result.detector_hit and result.detector_variant == "tiled"
                    for result in positive_results
                ),
                len(positives),
            ),
            "negative_count": len(negatives),
            "false_detections": false_detections,
            "false_detection_rate": (
                ratio(false_detections, len(negatives)) if negatives else None
            ),
        },
        "ocr": {
            "conditional_exact_accuracy": ratio(
                sum(result.exact for result in conditional), len(conditional)
            ),
            "no_text_rate": ratio(
                sum(result.raw_segment_count == 0 for result in positive_results),
                len(positives),
            ),
            "invalid_candidate_rate": ratio(
                sum(
                    result.raw_segment_count > 0 and not result.valid_candidate
                    for result in positive_results
                ),
                len(positives),
            ),
            "low_confidence_rate": ratio(
                sum(result.low_confidence for result in positive_results),
                len(positives),
            ),
            "character_accuracy": statistics.fmean(
                result.character_accuracy for result in positive_results
            ),
            "confusions": dict(sorted(confusions.items())),
        },
        "ocr_rescue": {
            "detector_miss_samples": sum(
                not result.detector_hit for result in positive_results
            ),
            "attempted_samples": len(rescue_results),
            "successful_exact_reads": sum(result.exact for result in rescue_results),
            "attempted_tiles": sum(result.rescue_tiles for result in rescue_results),
            "detector_crop_invalid_samples": len(detector_crop_rescues),
            "detector_crop_recovered_exact": sum(
                result.exact for result in detector_crop_rescues
            ),
            "false_detection_suppressed_recovery_count": len(
                detector_crop_rescues
            ),
            "ocr_mean_ms": (
                statistics.fmean(result.ocr_ms for result in rescue_results)
                if rescue_results
                else None
            ),
        },
        "end_to_end": {
            "correct_read_rate": ratio(exact, len(positives)),
            "no_read_rate": ratio(no_reads, len(positives)),
            "false_read_rate": ratio(false_reads, len(positives)),
            "exact_candidate_precision": ratio(exact, exact + false_reads),
            "exact_candidate_recall": ratio(exact, len(positives)),
            "ambiguous_discard_rate": None,
            "note": "Still-image benchmark measures candidates, not multi-frame saves.",
        },
        "performance_ms": {
            "detector_mean": statistics.fmean(detector_times),
            "detector_p50": statistics.median(detector_times),
            "detector_p95": percentile(detector_times, 0.95),
            "ocr_mean": statistics.fmean(ocr_times),
            "ocr_p50": statistics.median(ocr_times),
            "ocr_p95": percentile(ocr_times, 0.95),
            "end_to_end_mean": statistics.fmean(total_times),
            "end_to_end_p50": statistics.median(total_times),
            "end_to_end_p95": percentile(total_times, 0.95),
            "cpu_inference_calls": sum(result.inference_calls for result in results),
        },
    }


def comparison(baseline: dict[str, object], current: dict[str, object]) -> dict[str, float]:
    paths = {
        "detector_recall": ("detector", "recall"),
        "ocr_conditional_exact_accuracy": ("ocr", "conditional_exact_accuracy"),
        "character_accuracy": ("ocr", "character_accuracy"),
        "correct_read_rate": ("end_to_end", "correct_read_rate"),
        "no_read_rate": ("end_to_end", "no_read_rate"),
        "false_read_rate": ("end_to_end", "false_read_rate"),
        "detector_mean_ms": ("performance_ms", "detector_mean"),
        "ocr_mean_ms": ("performance_ms", "ocr_mean"),
        "end_to_end_mean_ms": ("performance_ms", "end_to_end_mean"),
    }
    deltas: dict[str, float] = {}
    for name, path in paths.items():
        before = baseline[path[0]][path[1]]
        after = current[path[0]][path[1]]
        deltas[f"{name}_before"] = float(before)
        deltas[f"{name}_after"] = float(after)
        deltas[f"{name}_delta"] = float(after) - float(before)
    return deltas


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    samples = load_manifest(manifest_path)
    config = load_config().plate_recognition
    detector = OpenVinoPlateDetector(config.plate_detector)
    provider = PaddleOcrProvider(
        config.model_root,
        backend=config.ocr_backend,
        cpu_threads=config.ocr_cpu_threads,
    )
    results = [
        run_sample(sample, manifest_path, detector, provider, config)
        for sample in samples
    ]
    report: dict[str, object] = {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "pipeline": summarize(args.label, samples, results),
        "samples": [asdict(result) for result in results],
    }
    if args.decision_trace:
        report["production_decision_trace"] = production_decision_trace(
            samples,
            results,
            manifest_path,
            detector,
            config,
        )
    if args.baseline is not None:
        baseline_payload = json.loads(args.baseline.resolve().read_text(encoding="utf-8"))
        report["comparison"] = comparison(
            baseline_payload["pipeline"],
            report["pipeline"],
        )
    if args.recognition_only_ab:
        recognition_provider = RecognitionOnlyProvider(
            config.model_root, config.ocr_backend, config.ocr_cpu_threads
        )
        recognition_results = [
            run_sample(
                sample,
                manifest_path,
                detector,
                recognition_provider,
                config,
                fallback_provider=provider,
            )
            for sample in samples
        ]
        report["recognition_only"] = summarize(
            f"{args.label}-recognition-only", samples, recognition_results
        )
        report["recognition_only_samples"] = [
            asdict(result) for result in recognition_results
        ]
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
