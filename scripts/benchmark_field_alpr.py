from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.camera import Direction
from app.config import load_config
from app.ocr_models import read_model_name, select_ocr_backend
from app.plate_detector import (
    OpenVinoPlateDetector,
    PlateDetection,
    crop_padded_plate,
    select_plate_detections,
)
from app.plate_recognition import (
    OcrImageProfile,
    OcrSegment,
    PaddleOcrProvider,
    PlateCandidate,
    TurkishPlateValidator,
    crop_roi,
    preprocess_roi_fallback_variants,
    recognize_detector_crops,
    roi_mean_brightness,
    select_best_candidate,
)


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
        detections, config.plate_detector.max_plate_candidates_per_frame
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
    segments: list[OcrSegment] = []
    candidate = None
    calls = 0
    started = time.perf_counter()
    variants = preprocess_roi_fallback_variants(
        roi, brightness=roi_mean_brightness(roi)
    )
    for variant_index, variant in enumerate(variants):
        calls += 1
        batch = provider.recognize([variant])
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
    return candidate, segments, calls, (time.perf_counter() - started) * 1000.0


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
    else:
        candidate, segments, inference_calls, ocr_ms = recognize_roi_fallback(
            fallback_provider or provider, roi, config.min_confidence
        )
        profiles = (OcrImageProfile.NORMAL.value,)
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
    )


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
