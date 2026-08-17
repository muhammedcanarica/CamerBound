from __future__ import annotations

import re
import os
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Callable, Protocol, Sequence

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot

from app.camera import CameraService, Direction
from app.config import NormalizedRoi, PlateRecognitionConfig, application_root
from app.ocr_models import (
    OcrBackend,
    OcrModelError,
    OcrModelNotFound,
    read_model_name,
    select_ocr_backend,
)
from app.plate_detector import (
    DetectorDiagnostics,
    OpenVinoPlateDetector,
    PlateDetection,
    PlateDetector,
    PlateDetectorError,
    crop_padded_plate,
    select_plate_detections,
)
from app.plate_service import (
    DuplicatePlateDetection,
    PlateRecord,
    PlateService,
)
from app.time_utils import as_utc, utc_now


class RecognitionStatus(StrEnum):
    STOPPED = "STOPPED"
    INITIALIZING = "INITIALIZING"
    ACTIVE = "ACTIVE"
    UNAVAILABLE = "UNAVAILABLE"
    ERROR = "ERROR"


class OcrJobType(StrEnum):
    """Explicit OCR work class; crop dimensions never determine priority."""

    DETECTOR_CROP = "DETECTOR_CROP"
    DETECTOR_ERROR_FALLBACK = "DETECTOR_ERROR_FALLBACK"
    ZERO_DETECTION_FALLBACK = "ZERO_DETECTION_FALLBACK"
    STATIC_ZERO_DETECTION_RESCUE = "STATIC_ZERO_DETECTION_RESCUE"

    @property
    def priority(self) -> int:
        return {
            OcrJobType.DETECTOR_CROP: 3,
            OcrJobType.DETECTOR_ERROR_FALLBACK: 2,
            OcrJobType.ZERO_DETECTION_FALLBACK: 1,
            OcrJobType.STATIC_ZERO_DETECTION_RESCUE: 0,
        }[self]


class OcrImageProfile(StrEnum):
    NORMAL = "NORMAL"
    LOW_LIGHT = "LOW_LIGHT"
    SHADOW_LOW_CONTRAST = "SHADOW_LOW_CONTRAST"
    OVEREXPOSED = "OVEREXPOSED"


LOGGER = logging.getLogger(__name__)
PLATE_PRESENCE_RELEASE_SECONDS = 15
OCR_DIAGNOSTIC_LOG_INTERVAL_SECONDS = 5.0
MOTION_ANALYSIS_WIDTH = 160
MOTION_PIXEL_DIFFERENCE_THRESHOLD = 20
MOTION_CONTINUE_THRESHOLD_RATIO = 0.60
RECENT_PROCESSED_FRAME_ID_LIMIT = 256
LIVE_FRAMES_PER_REPLAY_FRAME = 2
LOW_LIGHT_THRESHOLD = 85.0
LOW_LIGHT_GAMMA = 0.72
SHADOW_DYNAMIC_RANGE_THRESHOLD = 80.0
SHADOW_GRAYSCALE_STDDEV_THRESHOLD = 30.0
SHADOW_LOCAL_CONTRAST_THRESHOLD = 12.0
SHADOW_MAX_MEAN_LUMINANCE = 200.0
SHADOW_MIN_RECOVERABLE_DYNAMIC_RANGE = 12.0
SHADOW_MIN_RECOVERABLE_LOCAL_CONTRAST = 2.0
OVEREXPOSED_P10_THRESHOLD = 220.0
OVEREXPOSED_WHITE_RATIO_THRESHOLD = 0.60
SATURATED_BLACK_MAX_VALUE = 5
SATURATED_WHITE_MIN_VALUE = 250
OCR_VARIANT_NAMES = (
    "adaptive-color",
    "adaptive-clahe",
    "upscaled-2x-clahe-sharpened",
    "low-light-gamma-clahe-sharpened",
)
SHADOW_OCR_VARIANT_NAMES = (
    "shadow-gamma-gray",
)
SHADOW_COMPARISON_VARIANT_NAMES = (
    "shadow-gamma-color",
    "shadow-gamma-gray",
    "shadow-gamma-clahe-gray",
)
ROI_FALLBACK_MAX_WIDTH = 960
RISKY_CONFIRMATIONS_REQUIRED = 3
POST_SAVE_CONFIRMATIONS_REQUIRED = 4
POST_SAVE_NEAR_DUPLICATE_WINDOW_SECONDS = 45 * 60
RECENT_SAVED_EVIDENCE_LIMIT = 32
CONFIRMATION_OBSERVATION_LIMIT_PER_CAMERA = 128
CONFIRMED_PLATE_LIMIT = 256
PENDING_DECISION_CAMERA_LIMIT = 16
PENDING_DECISION_OBSERVATION_LIMIT = 32
PENDING_DECISION_PLATE_LIMIT = 8
ZERO_DETECTION_FALLBACK_MAX_ATTEMPTS = 2
ZERO_DETECTION_LIVE_FALLBACK_MAX_ATTEMPTS = 1
ZERO_DETECTION_FALLBACK_MIN_SPACING_MS = 1_000
ZERO_DETECTION_EVENT_STATE_LIMIT = 32
COMPLETED_MOTION_EVENT_LIMIT = 32
RAW_RECOGNITION_CAPTURE_ENV = "CAMERBOUND_CAPTURE_NEXT_RECOGNITION_FRAME"


@dataclass(frozen=True, slots=True)
class CropQualityMetrics:
    width: int
    height: int
    luma_mean: float
    luma_median: float
    luma_p10: float
    luma_p90: float
    dynamic_range: float
    grayscale_stddev: float
    local_contrast: float
    laplacian_sharpness: float
    saturated_black_ratio: float
    saturated_white_ratio: float


def _log_plate_detector_diagnostics(
    camera_id: int,
    source: str,
    detector: PlateDetector,
    roi_crop: np.ndarray,
    last_logged_at: dict[tuple[int, str], float],
) -> None:
    if not LOGGER.isEnabledFor(logging.DEBUG):
        return
    diagnostics = getattr(detector, "last_diagnostics", None)
    if not isinstance(diagnostics, DetectorDiagnostics):
        return
    now = time.monotonic()
    key = (camera_id, source)
    previous = last_logged_at.get(key)
    if previous is not None and now - previous < OCR_DIAGNOSTIC_LOG_INTERVAL_SECONDS:
        return
    last_logged_at[key] = now
    brightness = (
        "not-evaluated"
        if diagnostics.raw_brightness is None
        else f"{diagnostics.raw_brightness:.1f}"
    )
    shadow_metric = (
        "not-evaluated"
        if diagnostics.shadow_metric is None
        else f"{diagnostics.shadow_metric:.1f}"
    )
    LOGGER.debug(
        "Plate detector diagnostics camera_id=%s source=%s detector_variant=%s "
        "brightness=%s raw_brightness=%s shadow_metric=%s enhanced_pass=%s "
        "detections=%s raw_candidates=%s plate_class_candidates=%s "
        "highest_plate_confidence=%s confidence_rejected=%s bbox_rejected=%s "
        "detector_input=%sx%s input_layout=%s input_dtype=%s roi_size=%sx%s "
        "resize_scale_x=%.4f resize_scale_y=%.4f aspect_distortion_ratio=%.2f "
        "tiled_recovery=%s recovery_tiles=%s raw_detector_ms=%.1f "
        "enhanced_detector_ms=%.1f tiled_detector_ms=%.1f detector_latency_ms=%.1f",
        camera_id,
        source,
        diagnostics.detector_variant,
        brightness,
        brightness,
        shadow_metric,
        "yes" if diagnostics.enhanced_pass else "no",
        diagnostics.detections,
        diagnostics.raw_candidate_count,
        diagnostics.plate_class_candidate_count,
        (
            "none"
            if diagnostics.highest_plate_confidence is None
            else f"{diagnostics.highest_plate_confidence:.3f}"
        ),
        diagnostics.confidence_rejected_count,
        diagnostics.bbox_rejected_count,
        diagnostics.input_width,
        diagnostics.input_height,
        diagnostics.input_layout,
        diagnostics.input_dtype,
        roi_crop.shape[1],
        roi_crop.shape[0],
        diagnostics.resize_scale_x,
        diagnostics.resize_scale_y,
        diagnostics.aspect_distortion_ratio,
        "yes" if diagnostics.tiled_recovery_pass else "no",
        diagnostics.recovery_tile_count,
        diagnostics.raw_detector_ms,
        diagnostics.enhanced_detector_ms,
        diagnostics.tiled_detector_ms,
        (
            diagnostics.raw_detector_ms
            + diagnostics.enhanced_detector_ms
            + diagnostics.tiled_detector_ms
        ),
    )


@dataclass(frozen=True, slots=True)
class OcrSegment:
    text: str
    confidence: float
    box: tuple[float, float, float, float]
    variant_index: int = 0


@dataclass(frozen=True, slots=True)
class PlateCandidate:
    plate: str
    confidence: float
    raw_text: str
    camera_id: int
    normalized_raw_text: str = ""
    correction_cost: int = 0
    variant_support: int = 1
    supporting_variants: tuple[int, ...] = ()
    frame_id: int | None = None
    observed_at: float | None = None
    job_type: OcrJobType | None = None
    detector_bbox: tuple[float, float, float, float] | None = None
    detector_crop_evidence: bool = False
    spatial_alias_evidence: bool = False


@dataclass(frozen=True, slots=True)
class DetectorCropOcrResult:
    variants: tuple[np.ndarray, ...]
    variant_names: tuple[str, ...]
    segments: tuple[OcrSegment, ...]
    candidate: PlateCandidate | None
    quality_metrics: tuple[CropQualityMetrics, ...]
    profiles: tuple[OcrImageProfile, ...]
    current_variant_count: int
    shadow_variant_count: int
    inference_calls: int
    text_detection_box_counts: tuple[int | None, ...]
    current_preprocess_ms: float
    current_inference_ms: float
    shadow_preprocess_ms: float
    shadow_inference_ms: float


class OcrProvider(Protocol):
    def recognize(self, images: Sequence[np.ndarray]) -> list[OcrSegment]: ...


class PaddleOcrProvider:
    """Offline PaddleOCR adapter selecting ONNX Runtime or native Paddle."""

    def __init__(
        self,
        model_root: Path,
        backend: str | OcrBackend = OcrBackend.AUTO,
        *,
        pipeline_factory=None,
        cpu_threads: int = 4,
    ) -> None:
        selection = select_ocr_backend(model_root, backend, load_onnx=True)
        self.backend = selection.backend
        self.backend_label = selection.label
        detection_dir = selection.model_root / "detection"
        recognition_dir = selection.model_root / "recognition"

        os.environ.setdefault(
            "PADDLE_PDX_CACHE_HOME",
            str(application_root() / "data" / "paddlex-cache"),
        )
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        if pipeline_factory is None:
            try:
                if self.backend is OcrBackend.PADDLE:
                    import paddle  # noqa: F401 - verifies the native runtime
                from paddleocr import PaddleOCR
            except (ImportError, OSError) as exc:
                dependency = (
                    "PaddlePaddle/PaddleOCR"
                    if self.backend is OcrBackend.PADDLE
                    else "PaddleOCR/ONNX Runtime"
                )
                raise RuntimeError(f"{dependency} yüklenemedi.") from exc
            pipeline_factory = PaddleOCR

        options = {
            "text_detection_model_name": read_model_name(detection_dir),
            "text_detection_model_dir": str(detection_dir),
            "text_recognition_model_name": read_model_name(recognition_dir),
            "text_recognition_model_dir": str(recognition_dir),
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "device": "cpu",
            "enable_hpi": False,
            "cpu_threads": max(1, int(cpu_threads)),
        }
        if self.backend is OcrBackend.ONNX:
            options["engine"] = "onnxruntime"
        else:
            # The current Windows Paddle dev build fails in oneDNN/PIR for these
            # official PP-OCRv5 models; the regular CPU executor is compatible.
            options["enable_mkldnn"] = False
        self._pipeline = pipeline_factory(**options)
        self.last_detection_box_counts: tuple[int, ...] = ()

    def recognize(self, images: Sequence[np.ndarray]) -> list[OcrSegment]:
        segments: list[OcrSegment] = []
        detection_box_counts = [0] * len(images)
        if not images:
            self.last_detection_box_counts = ()
            return segments
        results = self._pipeline.predict(input=list(images))
        for result_index, result in enumerate(results):
            payload = getattr(result, "json", {})
            if not isinstance(payload, dict):
                continue
            data = payload.get("res", payload)
            if not isinstance(data, dict):
                continue
            texts = data.get("rec_texts", [])
            scores = data.get("rec_scores", [])
            boxes = data.get("rec_boxes", [])
            if result_index < len(detection_box_counts):
                try:
                    detection_box_counts[result_index] = len(boxes)
                except TypeError:
                    detection_box_counts[result_index] = 0
            for index, text in enumerate(texts):
                if not isinstance(text, str) or not text.strip():
                    continue
                score = _safe_score(scores[index] if index < len(scores) else 0.0)
                box = _safe_box(boxes[index] if index < len(boxes) else None, index)
                segments.append(
                    OcrSegment(
                        text=text,
                        confidence=score,
                        box=box,
                        variant_index=result_index,
                    )
                )
        self.last_detection_box_counts = tuple(detection_box_counts)
        return segments


class TurkishPlateValidator:
    PLATE_PATTERN = re.compile(r"^(?P<province>\d{2})(?P<letters>[A-Z]{1,3})(?P<number>\d{2,4})$")

    @classmethod
    def is_valid(cls, plate: str) -> bool:
        match = cls.PLATE_PATTERN.fullmatch(plate)
        if match is None:
            return False
        province = int(match.group("province"))
        return 1 <= province <= 81


PROVINCE_DIGIT_CORRECTIONS = {"O": "0", "I": "1", "L": "1", "S": "5", "B": "8"}
LETTER_CORRECTIONS = {"0": "O", "1": "I"}
SUFFIX_DIGIT_CORRECTIONS = {
    "O": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
}


def normalize_plate_text(raw_text: str) -> str:
    return "".join(character for character in raw_text.upper() if character.isascii() and character.isalnum())


def correct_plate_candidate(raw_text: str) -> str | None:
    corrected = correct_plate_candidate_with_cost(raw_text)
    return corrected[0] if corrected is not None else None


def correct_plate_candidate_with_cost(raw_text: str) -> tuple[str, int] | None:
    normalized = normalize_plate_text(raw_text)
    if len(normalized) < 5 or len(normalized) > 9:
        return None

    province, province_cost = _translate(normalized[:2], PROVINCE_DIGIT_CORRECTIONS, str.isdigit)
    if province is None:
        return None

    corrected: list[tuple[int, str]] = []
    remainder = normalized[2:]
    for letter_count in range(1, 4):
        if len(remainder) - letter_count not in range(2, 5):
            continue
        letters, letter_cost = _translate(
            remainder[:letter_count], LETTER_CORRECTIONS, _is_ascii_letter
        )
        suffix, suffix_cost = _translate(
            remainder[letter_count:], SUFFIX_DIGIT_CORRECTIONS, str.isdigit
        )
        if letters is None or suffix is None:
            continue
        plate = province + letters + suffix
        if TurkishPlateValidator.is_valid(plate):
            corrected.append((province_cost + letter_cost + suffix_cost, plate))
    if not corrected:
        return None
    corrected.sort(key=lambda item: (item[0], item[1]))
    cost, plate = corrected[0]
    return plate, cost


class ConfirmationTracker:
    def __init__(self, required: int, window_seconds: float) -> None:
        self.required = max(1, required)
        self.window_seconds = max(0.1, window_seconds)
        self._observations: dict[int, deque[_ConfirmationObservation]] = defaultdict(deque)
        self._confirmed_until: dict[tuple[int, str], float] = {}
        self._anonymous_frame_id = 0

    def observe(self, candidate: PlateCandidate, observed_at: float) -> PlateCandidate | None:
        return self.observe_progress(candidate, observed_at).candidate

    def observe_progress(
        self,
        candidate: PlateCandidate,
        observed_at: float,
        *,
        frame_id: int | None = None,
        post_save_near_plate: str | None = None,
        spatial_alias: bool = False,
    ) -> ConfirmationProgress:
        key = (candidate.camera_id, candidate.plate)
        confirmed_until = self._confirmed_until.get(key, 0.0)
        if confirmed_until >= observed_at:
            return ConfirmationProgress(
                None,
                0,
                self.required,
                suppression_reason="active-presence",
            )

        if frame_id is None:
            self._anonymous_frame_id -= 1
            frame_id = self._anonymous_frame_id
        observations = self._observations[candidate.camera_id]
        if not any(item.frame_id == frame_id for item in observations):
            observations.append(
                _ConfirmationObservation(observed_at, frame_id, candidate)
            )
        ordered = sorted(observations, key=lambda item: (item.observed_at, item.frame_id))
        newest_at = ordered[-1].observed_at
        cutoff = newest_at - self.window_seconds
        observations.clear()
        observations.extend(item for item in ordered if item.observed_at >= cutoff)
        while len(observations) > CONFIRMATION_OBSERVATION_LIMIT_PER_CAMERA:
            observations.popleft()

        exact = [item for item in observations if item.candidate.plate == candidate.plate]
        near_groups: dict[str, list[_ConfirmationObservation]] = defaultdict(list)
        for item in observations:
            if item.candidate.plate != candidate.plate and plates_are_near_conflicts(
                candidate.plate, item.candidate.plate
            ):
                near_groups[item.candidate.plate].append(item)
        runner_up_votes = max((len(items) for items in near_groups.values()), default=0)
        correction_risk = any(item.candidate.correction_cost > 0 for item in exact)
        effective_required = self.required
        if correction_risk or near_groups:
            effective_required = max(effective_required, RISKY_CONFIRMATIONS_REQUIRED)
        if post_save_near_plate is not None:
            effective_required = max(effective_required, POST_SAVE_CONFIRMATIONS_REQUIRED)

        candidate_votes = len(exact)
        ambiguity_margin_ok = not near_groups or candidate_votes - runner_up_votes >= 2
        detector_votes = sum(
            1 for item in exact if item.candidate.detector_crop_evidence
        )
        detector_evidence_ok = (
            post_save_near_plate is None
            or detector_votes >= POST_SAVE_CONFIRMATIONS_REQUIRED
        )
        ready = (
            candidate_votes >= effective_required
            and ambiguity_margin_ok
            and detector_evidence_ok
        )
        near_conflicts = tuple(sorted(near_groups))
        if not ready:
            return ConfirmationProgress(
                None,
                candidate_votes,
                effective_required,
                near_conflicts=near_conflicts,
                runner_up_votes=runner_up_votes,
                suppression_reason=(
                    "near-duplicate-ambiguity"
                    if near_groups or post_save_near_plate is not None
                    else None
                ),
            )

        spatial_alias_votes = sum(
            1 for item in exact if item.candidate.spatial_alias_evidence
        )
        if spatial_alias and spatial_alias_votes >= RISKY_CONFIRMATIONS_REQUIRED:
            return ConfirmationProgress(
                None,
                candidate_votes,
                effective_required,
                near_conflicts=(post_save_near_plate,) if post_save_near_plate else (),
                runner_up_votes=runner_up_votes,
                suppression_reason="near-duplicate-ambiguity",
                suppress=True,
            )

        confidence = sum(item.candidate.confidence for item in exact) / len(exact)
        retained = [
            item
            for item in observations
            if not (
                item.candidate.plate == candidate.plate
                or plates_are_near_conflicts(candidate.plate, item.candidate.plate)
            )
        ]
        observations.clear()
        observations.extend(retained)
        self._confirmed_until[key] = newest_at + self.window_seconds
        while len(self._confirmed_until) > CONFIRMED_PLATE_LIMIT:
            self._confirmed_until.pop(next(iter(self._confirmed_until)))
        return ConfirmationProgress(
            replace(candidate, confidence=confidence),
            candidate_votes,
            effective_required,
            near_conflicts=near_conflicts,
            runner_up_votes=runner_up_votes,
        )


@dataclass(frozen=True, slots=True)
class _ConfirmationObservation:
    observed_at: float
    frame_id: int
    candidate: PlateCandidate


@dataclass(frozen=True, slots=True)
class ConfirmationProgress:
    candidate: PlateCandidate | None
    observed_count: int
    required_count: int
    near_conflicts: tuple[str, ...] = ()
    runner_up_votes: int = 0
    suppression_reason: str | None = None
    suppress: bool = False


@dataclass(slots=True)
class _PlatePresence:
    last_seen: float
    record_claimed: bool = False


class PlatePresenceTracker:
    """Track active plate appearances independently for each camera."""

    def __init__(self, release_seconds: float = PLATE_PRESENCE_RELEASE_SECONDS) -> None:
        self.release_seconds = max(0.1, release_seconds)
        self._presences: dict[tuple[int, str], _PlatePresence] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(candidate: PlateCandidate) -> tuple[int, str]:
        return candidate.camera_id, normalize_plate_text(candidate.plate)

    def observe(self, candidate: PlateCandidate, observed_at: float) -> None:
        """Refresh last_seen and start a new event after the release interval."""
        key = self._key(candidate)
        with self._lock:
            presence = self._presences.get(key)
            if (
                presence is None
                or observed_at - presence.last_seen > self.release_seconds
            ):
                self._presences[key] = _PlatePresence(last_seen=observed_at)
                return
            presence.last_seen = max(presence.last_seen, observed_at)

    def claim_record(self, candidate: PlateCandidate) -> bool:
        """Atomically reserve the single record allowed for the active event."""
        key = self._key(candidate)
        with self._lock:
            presence = self._presences.get(key)
            if presence is None:
                return False
            if presence.record_claimed:
                return False
            presence.record_claimed = True
            return True

    def release_record_claim(self, candidate: PlateCandidate) -> None:
        """Allow a retry when persistence failed before producing a record."""
        key = self._key(candidate)
        with self._lock:
            presence = self._presences.get(key)
            if presence is not None:
                presence.record_claimed = False


@dataclass(frozen=True, slots=True)
class _SavedPlateEvidence:
    plate: str
    detected_at: datetime
    detector_bbox: tuple[float, float, float, float] | None


@dataclass(frozen=True, slots=True)
class _PlateEvidenceObservation:
    candidate: PlateCandidate
    frame_key: int
    observed_at: float


@dataclass(frozen=True, slots=True)
class _RepresentativeFrame:
    candidate: PlateCandidate
    captured_at: datetime
    full_frame: object
    quality_score: float
    detections: tuple[PlateDetection, ...]
    used_roi_fallback: bool


@dataclass(slots=True)
class PendingPlateDecision:
    camera_id: int
    direction: Direction
    last_updated_monotonic: float
    cleanup_deadline_monotonic: float
    observations: deque[_PlateEvidenceObservation] = field(default_factory=deque)
    frame_keys: set[int] = field(default_factory=set)
    representatives: dict[str, _RepresentativeFrame] = field(default_factory=dict)
    provisional_plate: str | None = None
    deadline_monotonic: float | None = None
    min_hold_until_monotonic: float | None = None


@dataclass(frozen=True, slots=True)
class _PlateDecisionEvaluation:
    winner: PlateCandidate | None
    winner_votes: int
    runner_up_votes: int
    required_votes: int
    near_conflicts: tuple[str, ...]
    reliable: bool
    early_finalizable: bool
    suppression_reason: str | None = None


class RecognitionState(StrEnum):
    NO_OCR_TEXT = "NO_OCR_TEXT"
    NO_VALID_PLATE = "NO_VALID_PLATE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
    STABILIZING = "STABILIZING"
    AMBIGUOUS_DISCARDED = "AMBIGUOUS_DISCARDED"
    SAVED = "SAVED"
    DUPLICATE_SUPPRESSED = "DUPLICATE_SUPPRESSED"


@dataclass(frozen=True, slots=True)
class RecognitionOutcome:
    candidate: PlateCandidate | None
    record: PlateRecord | None
    state: RecognitionState
    confirmation_count: int = 0
    confirmation_required: int = 0
    duplicate: bool = False
    detections: tuple[PlateDetection, ...] = ()
    used_roi_fallback: bool = False
    suppression_reason: str | None = None


@dataclass(frozen=True, slots=True)
class FrameSnapshot:
    frame_id: int
    camera_id: int
    direction: Direction
    captured_at: datetime
    observed_at: float
    received_at: float
    full_frame: np.ndarray
    motion_score: float = 0.0


@dataclass(frozen=True, slots=True)
class MotionEvent:
    event_id: int
    camera_id: int
    direction: Direction
    started_at: float
    ended_at: float
    enqueued_at: float
    frames: tuple[FrameSnapshot, ...]


@dataclass(slots=True)
class _ActiveMotionEvent:
    event_id: int
    camera_id: int
    direction: Direction
    started_at: float
    last_motion_at: float
    frames: deque[FrameSnapshot]


@dataclass(slots=True)
class _ZeroDetectionEventState:
    attempt_count: int = 0
    last_attempt_at: float | None = None
    attempted_frame_ids: set[int] = field(default_factory=set)
    detector_crop_seen: bool = False


class PreDetectionFrameBuffer:
    """Thread-safe per-camera RAM ring and lightweight motion-event builder."""

    def __init__(self, config: PlateRecognitionConfig) -> None:
        self.config = config
        self._rings: dict[int, deque[FrameSnapshot]] = defaultdict(deque)
        self._previous_motion_frames: dict[int, np.ndarray] = {}
        self._events: dict[int, _ActiveMotionEvent] = {}
        self._next_event_id = 1
        self._lock = threading.Lock()

    def ingest(
        self,
        snapshot: FrameSnapshot,
        *,
        motion_score: float | None = None,
    ) -> tuple[MotionEvent, ...]:
        completed: list[MotionEvent] = []
        with self._lock:
            ring = self._rings[snapshot.camera_id]
            if motion_score is None:
                motion_score = self._motion_score(snapshot)
            snapshot = FrameSnapshot(
                frame_id=snapshot.frame_id,
                camera_id=snapshot.camera_id,
                direction=snapshot.direction,
                captured_at=snapshot.captured_at,
                observed_at=snapshot.observed_at,
                received_at=snapshot.received_at,
                full_frame=snapshot.full_frame,
                motion_score=motion_score,
            )
            ring.append(snapshot)
            self._trim_ring(ring, snapshot.observed_at)

            active = self._events.get(snapshot.camera_id)
            threshold = self.config.motion_changed_pixel_ratio
            motion_active = motion_score >= (
                threshold
                if active is None
                else threshold * MOTION_CONTINUE_THRESHOLD_RATIO
            )
            if active is None and motion_active:
                pre_roll_cutoff = (
                    snapshot.observed_at - self.config.motion_pre_roll_ms / 1000.0
                )
                pinned = deque(
                    item for item in ring if item.observed_at >= pre_roll_cutoff
                )
                active = _ActiveMotionEvent(
                    event_id=self._next_event_id,
                    camera_id=snapshot.camera_id,
                    direction=snapshot.direction,
                    started_at=snapshot.observed_at,
                    last_motion_at=snapshot.observed_at,
                    frames=pinned,
                )
                self._next_event_id += 1
                self._events[snapshot.camera_id] = active
                LOGGER.debug(
                    "Motion started camera_id=%s event_id=%s pre_roll_frames=%s",
                    snapshot.camera_id,
                    active.event_id,
                    len(pinned),
                )
            elif active is not None:
                if not active.frames or active.frames[-1].frame_id != snapshot.frame_id:
                    active.frames.append(snapshot)
                if motion_active:
                    active.last_motion_at = snapshot.observed_at

            if active is not None:
                maximum_event_frames = (
                    self.config.pre_detection_buffer_max_frames_per_camera
                )
                while len(active.frames) > maximum_event_frames:
                    active.frames.popleft()
                quiet_seconds = max(
                    self.config.motion_quiet_ms,
                    self.config.motion_post_roll_ms,
                ) / 1000.0
                maximum_seconds = self.config.motion_event_max_duration_ms / 1000.0
                quiet_complete = (
                    not motion_active
                    and snapshot.observed_at - active.last_motion_at >= quiet_seconds
                )
                duration_complete = (
                    snapshot.observed_at - active.started_at >= maximum_seconds
                )
                if quiet_complete or duration_complete:
                    completed.append(self._finish_event(active, snapshot.observed_at))
                    self._events.pop(snapshot.camera_id, None)
        return tuple(completed)

    def snapshots(self, camera_id: int) -> tuple[FrameSnapshot, ...]:
        with self._lock:
            return tuple(self._rings.get(camera_id, ()))

    def active_event_id(self, camera_id: int) -> int | None:
        with self._lock:
            event = self._events.get(camera_id)
            return event.event_id if event is not None else None

    def active_event_frame_count(self, camera_id: int) -> int:
        with self._lock:
            event = self._events.get(camera_id)
            return len(event.frames) if event is not None else 0

    def ring_depth(self, camera_id: int) -> int:
        with self._lock:
            return len(self._rings.get(camera_id, ()))

    def clear(self) -> None:
        with self._lock:
            self._rings.clear()
            self._previous_motion_frames.clear()
            self._events.clear()

    def _motion_score(self, snapshot: FrameSnapshot) -> float:
        roi_crop = crop_roi(snapshot.full_frame, self.config.roi_for(snapshot.direction))
        if roi_crop is None:
            return 0.0
        gray = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2GRAY)
        scale = min(1.0, MOTION_ANALYSIS_WIDTH / max(1, gray.shape[1]))
        if scale < 1.0:
            gray = cv2.resize(
                gray,
                (MOTION_ANALYSIS_WIDTH, max(1, round(gray.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        previous = self._previous_motion_frames.get(snapshot.camera_id)
        self._previous_motion_frames[snapshot.camera_id] = gray
        if previous is None or previous.shape != gray.shape:
            return 0.0
        changed = cv2.absdiff(previous, gray) >= MOTION_PIXEL_DIFFERENCE_THRESHOLD
        return float(np.count_nonzero(changed) / changed.size)

    def _trim_ring(self, ring: deque[FrameSnapshot], now: float) -> None:
        cutoff = now - self.config.pre_detection_buffer_duration_ms / 1000.0
        while ring and ring[0].observed_at < cutoff:
            ring.popleft()
        while len(ring) > self.config.pre_detection_buffer_max_frames_per_camera:
            ring.popleft()

    @staticmethod
    def _finish_event(active: _ActiveMotionEvent, ended_at: float) -> MotionEvent:
        event = MotionEvent(
            event_id=active.event_id,
            camera_id=active.camera_id,
            direction=active.direction,
            started_at=active.started_at,
            ended_at=ended_at,
            enqueued_at=time.monotonic(),
            frames=tuple(active.frames),
        )
        LOGGER.debug(
            "Motion ended camera_id=%s event_id=%s duration_ms=%.1f frames=%s",
            event.camera_id,
            event.event_id,
            (event.ended_at - event.started_at) * 1000.0,
            len(event.frames),
        )
        return event


def select_replay_frames(
    frames: Sequence[FrameSnapshot],
    maximum: int,
    roi_for: Callable[[Direction], NormalizedRoi],
) -> tuple[FrameSnapshot, ...]:
    """Select a sharp frame from each temporal bin for broad event coverage."""
    if maximum <= 0 or not frames:
        return ()
    ordered = sorted(frames, key=lambda item: (item.observed_at, item.frame_id))
    if len(ordered) <= maximum:
        return tuple(ordered)
    selected: list[FrameSnapshot] = []
    for bin_index in range(maximum):
        start = bin_index * len(ordered) // maximum
        end = (bin_index + 1) * len(ordered) // maximum
        bucket = ordered[start:max(start + 1, end)]
        selected.append(
            max(
                bucket,
                key=lambda item: _snapshot_roi_sharpness(item, roi_for),
            )
        )
    return tuple(sorted(selected, key=lambda item: item.observed_at))


def _snapshot_roi_sharpness(
    snapshot: FrameSnapshot,
    roi_for: Callable[[Direction], NormalizedRoi],
) -> float:
    crop = crop_roi(snapshot.full_frame, roi_for(snapshot.direction))
    if crop is None:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


@dataclass(slots=True)
class _ReplayEventWork:
    event: MotionEvent
    remaining: deque[FrameSnapshot]


@dataclass(frozen=True, slots=True)
class ReplayFrameItem:
    event_id: int
    snapshot: FrameSnapshot
    event_frames: int


class ReplayEventBuffer:
    """Camera-fair bounded queue of temporally selected historical snapshots."""

    def __init__(self, config: PlateRecognitionConfig) -> None:
        self.config = config
        self._events: dict[int, deque[_ReplayEventWork]] = defaultdict(deque)
        self._camera_order: deque[int] = deque()
        self._known_camera_ids: set[int] = set()
        self._lock = threading.Lock()
        self.dropped_count = 0
        self.stale_count = 0

    def add(self, event: MotionEvent) -> bool:
        selected = select_replay_frames(
            event.frames,
            self.config.max_replay_frames_per_event,
            self.config.roi_for,
        )
        if not selected:
            return False
        queued_event = MotionEvent(
            event_id=event.event_id,
            camera_id=event.camera_id,
            direction=event.direction,
            started_at=event.started_at,
            ended_at=event.ended_at,
            enqueued_at=event.enqueued_at,
            frames=selected,
        )
        with self._lock:
            if event.camera_id not in self._known_camera_ids:
                self._known_camera_ids.add(event.camera_id)
                self._camera_order.append(event.camera_id)
            camera_events = self._events[event.camera_id]
            if len(camera_events) >= self.config.max_pending_replay_events_per_camera:
                camera_events.popleft()
                self.dropped_count += 1
            camera_events.append(_ReplayEventWork(queued_event, deque(selected)))
            depth = len(camera_events)
        LOGGER.debug(
            "Replay event queued camera_id=%s event_id=%s event_frames=%s "
            "replay_selected_frame_count=%s queue_depth=%s",
            event.camera_id,
            event.event_id,
            len(event.frames),
            len(selected),
            depth,
        )
        return True

    def take(self, now: float | None = None) -> FrameSnapshot | None:
        item = self.take_with_event(now)
        return item.snapshot if item is not None else None

    def take_with_event(self, now: float | None = None) -> ReplayFrameItem | None:
        now = time.monotonic() if now is None else now
        maximum_age = self.config.replay_event_max_age_ms / 1000.0
        with self._lock:
            for _ in range(len(self._camera_order)):
                camera_id = self._camera_order.popleft()
                self._camera_order.append(camera_id)
                camera_events = self._events[camera_id]
                while camera_events and now - camera_events[0].event.enqueued_at > maximum_age:
                    stale = camera_events.popleft()
                    self.stale_count += 1
                    LOGGER.debug(
                        "Replay event dropped camera_id=%s event_id=%s reason=stale",
                        camera_id,
                        stale.event.event_id,
                    )
                if not camera_events:
                    continue
                work = camera_events[0]
                snapshot = work.remaining.popleft()
                if not work.remaining:
                    camera_events.popleft()
                return ReplayFrameItem(
                    work.event.event_id,
                    snapshot,
                    len(work.event.frames),
                )
        return None

    def pending_event_count(self, camera_id: int | None = None) -> int:
        with self._lock:
            if camera_id is not None:
                return len(self._events.get(camera_id, ()))
            return sum(len(items) for items in self._events.values())

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self._camera_order.clear()
            self._known_camera_ids.clear()


@dataclass(frozen=True, slots=True)
class OcrJob:
    """A bounded, in-memory OCR observation tied to its original camera frame."""

    camera_id: int
    direction: Direction
    captured_at: datetime
    observed_at: float
    received_at: float
    queued_at: float
    full_frame: np.ndarray
    roi_crop: np.ndarray
    ocr_crops: tuple[np.ndarray, ...]
    detections: tuple[PlateDetection, ...]
    used_roi_fallback: bool
    fallback_reason: str | None
    detector_ms: float
    quality_score: float
    job_type: OcrJobType = OcrJobType.DETECTOR_CROP
    frame_id: int | None = None
    detector_source: str = "live"
    ocr_crop_detections: tuple[PlateDetection | None, ...] = ()


@dataclass(frozen=True, slots=True)
class DetectionJobResult:
    job: OcrJob | None
    detections: tuple[PlateDetection, ...]
    used_roi_fallback: bool
    fallback_reason: str | None
    detector_ms: float
    roi_brightness: float
    fallback_skipped_reason: str | None = None
    motion_event_id: int | None = None
    fallback_attempt: int = 0
    time_since_previous_attempt_ms: float | None = None
    motion_score: float = 0.0
    event_frames: int = 0
    ring_depth: int = 0


@dataclass(frozen=True, slots=True)
class OcrBufferAddResult:
    accepted: bool
    replaced: int
    dropped: int
    camera_depth: int
    total_depth: int
    replaced_job_type: OcrJobType | None = None
    drop_reason: str | None = None
    coalesced: bool = False
    stale_discarded: int = 0


class OcrJobBuffer:
    """Priority-aware, camera-fair bounded RAM buffer for OCR work."""

    def __init__(
        self,
        max_per_camera: int,
        max_age_ms: int,
        detector_crop_max_age_ms: int | None = None,
    ) -> None:
        self.max_per_camera = max(1, max_per_camera)
        self.max_age_seconds = max(0.1, max_age_ms / 1000.0)
        self.detector_crop_max_age_seconds = max(
            self.max_age_seconds,
            (
                max_age_ms
                if detector_crop_max_age_ms is None
                else detector_crop_max_age_ms
            )
            / 1000.0,
        )
        self._jobs: dict[int, deque[OcrJob]] = defaultdict(deque)
        self._camera_order: deque[int] = deque()
        self._known_camera_ids: set[int] = set()
        self._condition = threading.Condition()
        self.dropped_count = 0
        self.replaced_count = 0
        self.stale_count = 0

    def add(self, job: OcrJob) -> OcrBufferAddResult:
        replaced = 0
        dropped = 0
        accepted = True
        replaced_job_type: OcrJobType | None = None
        drop_reason: str | None = None
        coalesced = False
        with self._condition:
            if job.camera_id not in self._known_camera_ids:
                self._known_camera_ids.add(job.camera_id)
                self._camera_order.append(job.camera_id)
            camera_jobs = self._jobs[job.camera_id]
            stale_discarded = self._discard_stale_locked(
                camera_jobs, time.monotonic()
            )
            same_type_pending = sum(
                item.job_type is job.job_type for item in camera_jobs
            )
            pending_limit = (
                ZERO_DETECTION_FALLBACK_MAX_ATTEMPTS
                if job.job_type is OcrJobType.ZERO_DETECTION_FALLBACK
                else 1
            )
            if (
                job.job_type is not OcrJobType.DETECTOR_CROP
                and same_type_pending >= pending_limit
            ):
                accepted = False
                dropped = 1
                coalesced = True
                drop_reason = "fallback-pending"
                self.dropped_count += 1
            elif len(camera_jobs) >= self.max_per_camera:
                weakest_index = min(
                    range(len(camera_jobs)),
                    key=lambda index: (
                        camera_jobs[index].job_type.priority,
                        camera_jobs[index].quality_score,
                        index,
                    ),
                )
                weakest = camera_jobs[weakest_index]
                higher_priority = job.job_type.priority > weakest.job_type.priority
                better_same_priority = (
                    job.job_type.priority == weakest.job_type.priority
                    and job.quality_score > weakest.quality_score
                )
                if higher_priority or better_same_priority:
                    replaced_job_type = weakest.job_type
                    del camera_jobs[weakest_index]
                    camera_jobs.append(job)
                    replaced = 1
                    self.replaced_count += 1
                else:
                    accepted = False
                    dropped = 1
                    drop_reason = (
                        "lower-priority"
                        if job.job_type.priority < weakest.job_type.priority
                        else "quality-not-better"
                    )
                    self.dropped_count += 1
            else:
                camera_jobs.append(job)
            camera_depth = len(camera_jobs)
            total_depth = sum(len(items) for items in self._jobs.values())
            if accepted:
                self._condition.notify()
        result = OcrBufferAddResult(
            accepted=accepted,
            replaced=replaced,
            dropped=dropped,
            camera_depth=camera_depth,
            total_depth=total_depth,
            replaced_job_type=replaced_job_type,
            drop_reason=drop_reason,
            coalesced=coalesced,
            stale_discarded=stale_discarded,
        )

        if LOGGER.isEnabledFor(logging.DEBUG) and (not accepted or replaced):
            LOGGER.debug(
                "OCR buffer decision camera_id=%s job_type=%s source=%s frame_id=%s "
                "priority=%s accepted=%s queue_depth=%s camera_depth=%s "
                "replaced_job_type=%s drop_reason=%s coalesced=%s stale_count=%s",
                job.camera_id,
                job.job_type.value,
                job.detector_source,
                job.frame_id,
                job.job_type.priority,
                "yes" if accepted else "no",
                total_depth,
                camera_depth,
                replaced_job_type.value if replaced_job_type is not None else "none",
                drop_reason or "none",
                "yes" if coalesced else "no",
                self.stale_count,
            )
        return result

    def take(
        self,
        stop_event: threading.Event | None = None,
        *,
        wait: bool = False,
        wake_at: float | None = None,
    ) -> OcrJob | None:
        with self._condition:
            while True:
                if stop_event is not None and stop_event.is_set():
                    return None
                now = time.monotonic()
                if wake_at is not None and now >= wake_at:
                    return None
                for camera_id in tuple(self._camera_order):
                    self._discard_stale_locked(self._jobs[camera_id], now)
                for job_type in (
                    OcrJobType.DETECTOR_CROP,
                    OcrJobType.DETECTOR_ERROR_FALLBACK,
                    OcrJobType.ZERO_DETECTION_FALLBACK,
                    OcrJobType.STATIC_ZERO_DETECTION_RESCUE,
                ):
                    for _ in range(len(self._camera_order)):
                        camera_id = self._camera_order.popleft()
                        self._camera_order.append(camera_id)
                        camera_jobs = self._jobs[camera_id]
                        job_index = next(
                            (
                                index
                                for index, pending in enumerate(camera_jobs)
                                if pending.job_type is job_type
                            ),
                            None,
                        )
                        if job_index is not None:
                            job = camera_jobs[job_index]
                            del camera_jobs[job_index]
                            return job
                if not wait:
                    return None
                wait_seconds = 0.05
                if wake_at is not None:
                    wait_seconds = min(wait_seconds, max(0.0, wake_at - now))
                self._condition.wait(wait_seconds)

    def _discard_stale_locked(
        self,
        camera_jobs: deque[OcrJob],
        now: float,
    ) -> int:
        kept = [
            job
            for job in camera_jobs
            if now - job.queued_at <= self._max_age_seconds(job)
        ]
        discarded = len(camera_jobs) - len(kept)
        if not discarded:
            return 0
        camera_jobs.clear()
        camera_jobs.extend(kept)
        self.stale_count += discarded
        return discarded

    def _max_age_seconds(self, job: OcrJob) -> float:
        if job.job_type is OcrJobType.DETECTOR_CROP:
            return self.detector_crop_max_age_seconds
        return self.max_age_seconds

    def pending_count(self, camera_id: int | None = None) -> int:
        with self._condition:
            if camera_id is not None:
                return len(self._jobs.get(camera_id, ()))
            return sum(len(items) for items in self._jobs.values())

    def clear(self) -> None:
        with self._condition:
            self._jobs.clear()
            self._camera_order.clear()
            self._known_camera_ids.clear()
            self._condition.notify_all()

    def wake_all(self) -> None:
        with self._condition:
            self._condition.notify_all()


class PlateDetectionProcessor:
    """Detector-only stage. It never initializes or invokes PaddleOCR."""

    def __init__(
        self,
        config: PlateRecognitionConfig,
        detector: PlateDetector | None,
    ) -> None:
        self.config = config
        self.detector = detector
        self._last_zero_detection_fallback_at: dict[int, float] = {}
        self._zero_detection_events: dict[
            tuple[int, int], _ZeroDetectionEventState
        ] = {}
        self._detector_crop_frame_ids: dict[int, deque[int]] = defaultdict(deque)
        self._detector_crop_frame_id_sets: dict[int, set[int]] = defaultdict(set)
        self._last_detector_error_logged_at: dict[int, float] = {}
        self._last_detector_diagnostic_at: dict[tuple[int, str], float] = {}
        self._first_static_miss_at: dict[int, float] = {}
        self._last_static_rescue_at: dict[int, float] = {}
        self._last_valid_candidate_at: dict[int, float] = {}
        self._rescue_state_lock = threading.Lock()
        self._raw_capture_pending = (
            os.getenv(RAW_RECOGNITION_CAPTURE_ENV, "").strip() == "1"
        )

    def prepare_job(
        self,
        camera_id: int,
        direction: Direction,
        frame: object,
        *,
        captured_at: datetime,
        observed_at: float,
        received_at: float,
        frame_id: int | None = None,
        detector_source: str = "live",
        allow_zero_detection_fallback: bool = True,
        zero_detection_fallback_event_id: int | None = None,
        motion_score: float | None = None,
        event_frames: int = 0,
        ring_depth: int = 0,
    ) -> DetectionJobResult:
        roi_crop = crop_roi(frame, self.config.roi_for(direction))
        if roi_crop is None:
            return DetectionJobResult(None, (), False, None, 0.0, 0.0)
        self._capture_raw_frame_once(camera_id, direction, frame)

        detector_started_at = time.perf_counter()
        detector_config = self.config.plate_detector
        detections: list[PlateDetection] = []
        selected: list[PlateDetection] = []
        ocr_crop_detections: list[PlateDetection | None] = []
        used_roi_fallback = not detector_config.enabled
        fallback_reason = "detector-disabled" if used_roi_fallback else None
        ocr_crops: list[np.ndarray] = []
        fallback_skipped_reason: str | None = None
        tiled_recovery = False
        fallback_attempt = 0
        time_since_previous_attempt_ms: float | None = None

        if detector_config.enabled and self.detector is not None:
            try:
                detections = self.detector.detect(roi_crop)
                tiled_recovery = (
                    getattr(self.detector, "last_diagnostics", None) is not None
                    and getattr(
                        self.detector.last_diagnostics,
                        "detector_variant",
                        None,
                    )
                    == "tiled"
                )
                _log_plate_detector_diagnostics(
                    camera_id,
                    detector_source,
                    self.detector,
                    roi_crop,
                    self._last_detector_diagnostic_at,
                )
            except Exception as exc:
                self._log_detector_error(
                    camera_id,
                    exc,
                    detector_started_at,
                    detector_config.fallback_to_roi_ocr,
                )
                if not detector_config.fallback_to_roi_ocr:
                    raise
                used_roi_fallback = True
                fallback_reason = "detector-error"
                ocr_crops = [roi_crop]
            if not used_roi_fallback:
                selected = select_plate_detections(
                    detections,
                    detector_config.max_plate_candidates_per_frame,
                )
                for detection in selected:
                    plate_crop = crop_padded_plate(
                        roi_crop,
                        detection,
                        (
                            detector_config.tiled_recovery_crop_padding_ratio
                            if tiled_recovery
                            else detector_config.crop_padding_ratio
                        ),
                    )
                    if plate_crop is not None:
                        ocr_crops.append(plate_crop)
                        ocr_crop_detections.append(detection)
                if ocr_crops and zero_detection_fallback_event_id is not None:
                    self._event_state(
                        camera_id,
                        zero_detection_fallback_event_id,
                    ).detector_crop_seen = True
                if ocr_crops and frame_id is not None:
                    self._remember_detector_crop_frame(camera_id, frame_id)
                if ocr_crops:
                    self._first_static_miss_at.pop(camera_id, None)
                if not ocr_crops:
                    (
                        should_fallback,
                        fallback_skipped_reason,
                        fallback_attempt,
                        time_since_previous_attempt_ms,
                    ) = (
                        self._should_run_zero_detection_fallback(
                            camera_id,
                            observed_at,
                            detector_config.zero_detection_roi_fallback_enabled,
                            detector_config.zero_detection_roi_fallback_interval_ms,
                            allow_zero_detection_fallback,
                            zero_detection_fallback_event_id,
                            frame_id,
                            motion_score,
                            event_frames,
                            ring_depth,
                        )
                    )
                    if should_fallback:
                        used_roi_fallback = True
                        fallback_reason = "zero-detection"
                        ocr_crops = [roi_crop]
                    elif zero_detection_fallback_event_id is None:
                        static_rescue, static_skip_reason = (
                            self._should_run_static_zero_detection_rescue(
                                camera_id,
                                observed_at,
                                detector_config.static_zero_detection_rescue_enabled,
                                detector_config.static_zero_detection_rescue_interval_ms,
                                allow_zero_detection_fallback,
                            )
                        )
                        if static_rescue:
                            used_roi_fallback = True
                            fallback_reason = "static-zero-detection-rescue"
                            fallback_skipped_reason = None
                            ocr_crops = [roi_crop]
                        else:
                            fallback_skipped_reason = static_skip_reason
        elif detector_config.enabled:
            if not detector_config.fallback_to_roi_ocr:
                raise PlateDetectorError(
                    "Plate detector kullanılamıyor ve ROI OCR fallback kapalı."
                )
            used_roi_fallback = True
            fallback_reason = "detector-error"
            ocr_crops = [roi_crop]
        else:
            ocr_crops = [roi_crop]

        detector_ms = (time.perf_counter() - detector_started_at) * 1000.0
        detection_tuple = tuple(detections)
        brightness = roi_mean_brightness(roi_crop)
        if not ocr_crops:
            return DetectionJobResult(
                None,
                detection_tuple,
                used_roi_fallback,
                fallback_reason,
                detector_ms,
                brightness,
                fallback_skipped_reason,
                zero_detection_fallback_event_id,
                fallback_attempt,
                time_since_previous_attempt_ms,
                motion_score or 0.0,
                event_frames,
                ring_depth,
            )

        full_frame = frame
        if isinstance(frame, np.ndarray) and frame.flags.writeable:
            full_frame = frame.copy()
            full_frame.setflags(write=False)
        if not isinstance(full_frame, np.ndarray):
            return DetectionJobResult(
                None,
                detection_tuple,
                used_roi_fallback,
                fallback_reason,
                detector_ms,
                brightness,
            )
        job_type = OcrJobType.DETECTOR_CROP
        if fallback_reason == "zero-detection":
            job_type = OcrJobType.ZERO_DETECTION_FALLBACK
        elif fallback_reason == "static-zero-detection-rescue":
            job_type = OcrJobType.STATIC_ZERO_DETECTION_RESCUE
        elif used_roi_fallback:
            # Detector-disabled and detector-unavailable paths share the protected
            # detector-error fallback priority tier; fallback_reason stays explicit.
            job_type = OcrJobType.DETECTOR_ERROR_FALLBACK
        job = OcrJob(
            camera_id=camera_id,
            direction=direction,
            captured_at=captured_at,
            observed_at=observed_at,
            received_at=received_at,
            queued_at=time.monotonic(),
            full_frame=full_frame,
            roi_crop=roi_crop,
            ocr_crops=tuple(ocr_crops),
            detections=detection_tuple,
            used_roi_fallback=used_roi_fallback,
            fallback_reason=fallback_reason,
            detector_ms=detector_ms,
            quality_score=ocr_job_quality_score(ocr_crops, selected),
            job_type=job_type,
            frame_id=frame_id,
            detector_source=detector_source,
            ocr_crop_detections=tuple(ocr_crop_detections),
        )
        return DetectionJobResult(
            job,
            detection_tuple,
            used_roi_fallback,
            fallback_reason,
            detector_ms,
            brightness,
            fallback_skipped_reason,
            zero_detection_fallback_event_id,
            fallback_attempt,
            time_since_previous_attempt_ms,
            motion_score or 0.0,
            event_frames,
            ring_depth,
        )

    def _should_run_static_zero_detection_rescue(
        self,
        camera_id: int,
        observed_at: float,
        enabled: bool,
        interval_ms: int,
        allowed_for_source: bool,
    ) -> tuple[bool, str]:
        if not enabled:
            return False, "static-rescue-disabled"
        if not allowed_for_source:
            return False, "replay-static-rescue"
        interval_seconds = max(
            interval_ms,
            self.config.recognition_interval_ms,
        ) / 1000.0
        with self._rescue_state_lock:
            first_miss_at = self._first_static_miss_at.setdefault(
                camera_id,
                observed_at,
            )
            if observed_at - first_miss_at < interval_seconds:
                return False, "static-rescue-warmup"
            last_candidate_at = self._last_valid_candidate_at.get(camera_id)
            if (
                last_candidate_at is not None
                and observed_at - last_candidate_at < PLATE_PRESENCE_RELEASE_SECONDS
            ):
                return False, "static-rescue-recent-recognition"
            last_rescue_at = self._last_static_rescue_at.get(camera_id)
            if (
                last_rescue_at is not None
                and observed_at - last_rescue_at < interval_seconds
            ):
                return False, "static-rescue-cooldown"
            self._last_static_rescue_at[camera_id] = observed_at
        LOGGER.debug(
            "Static zero-detection rescue camera_id=%s "
            "fallback_reason=static-zero-detection-rescue "
            "cooldown_ms=%s priority=%s",
            camera_id,
            int(interval_seconds * 1000.0),
            OcrJobType.STATIC_ZERO_DETECTION_RESCUE.priority,
        )
        return True, "none"

    def _capture_raw_frame_once(
        self,
        camera_id: int,
        direction: Direction,
        frame: object,
    ) -> None:
        if not self._raw_capture_pending or not isinstance(frame, np.ndarray):
            return
        self._raw_capture_pending = False
        try:
            output_dir = application_root() / "debug" / "recognition-frames"
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            output_path = output_dir / (
                f"camera-{camera_id}-{direction.value.lower()}-{timestamp}.jpg"
            )
            if cv2.imwrite(str(output_path), frame):
                LOGGER.info("One-shot recognition frame captured path=%s", output_path)
            else:
                LOGGER.warning(
                    "One-shot recognition frame capture failed path=%s",
                    output_path,
                )
        except OSError as exc:
            LOGGER.warning(
                "One-shot recognition frame capture failed error_type=%s",
                type(exc).__name__,
            )

    def note_valid_candidate(self, camera_id: int, observed_at: float) -> None:
        with self._rescue_state_lock:
            self._last_valid_candidate_at[camera_id] = observed_at

    def _should_run_zero_detection_fallback(
        self,
        camera_id: int,
        observed_at: float,
        enabled: bool,
        interval_ms: int,
        allowed_for_source: bool,
        motion_event_id: int | None,
        frame_id: int | None,
        motion_score: float | None,
        event_frames: int,
        ring_depth: int,
    ) -> tuple[bool, str | None, int, float | None]:
        if not enabled:
            return False, "disabled", 0, None
        if not allowed_for_source:
            return False, "replay-zero-detection", 0, None
        if motion_event_id is None:
            return False, "no-meaningful-motion", 0, None
        state = self._event_state(camera_id, motion_event_id)
        if frame_id is not None and frame_id in state.attempted_frame_ids:
            return False, "same-frame", state.attempt_count, None
        if state.attempt_count >= ZERO_DETECTION_LIVE_FALLBACK_MAX_ATTEMPTS:
            return False, "event-end-attempt-reserved", state.attempt_count, None
        if (
            motion_score is not None
            and motion_score
            < self.config.motion_changed_pixel_ratio
            * MOTION_CONTINUE_THRESHOLD_RATIO
        ):
            return False, "no-meaningful-motion", state.attempt_count, None
        last_fallback_at = self._last_zero_detection_fallback_at.get(camera_id)
        time_since_previous_attempt_ms = (
            None
            if last_fallback_at is None
            else max(0.0, (observed_at - last_fallback_at) * 1000.0)
        )
        spacing_ms = max(interval_ms, ZERO_DETECTION_FALLBACK_MIN_SPACING_MS)
        if (
            last_fallback_at is not None
            and time_since_previous_attempt_ms < spacing_ms
        ):
            return (
                False,
                "fallback-spacing",
                state.attempt_count,
                time_since_previous_attempt_ms,
            )
        self._last_zero_detection_fallback_at[camera_id] = observed_at
        state.attempt_count += 1
        state.last_attempt_at = observed_at
        if frame_id is not None:
            state.attempted_frame_ids.add(frame_id)
        LOGGER.debug(
            "Zero-detection fallback camera_id=%s event_id=%s "
            "fallback_attempt=%s/%s frame_id=%s motion_score=%.4f "
            "fallback_reason=zero-detection fallback_skipped_reason=none "
            "time_since_previous_attempt_ms=%s event_frames=%s ring_depth=%s",
            camera_id,
            motion_event_id,
            state.attempt_count,
            ZERO_DETECTION_FALLBACK_MAX_ATTEMPTS,
            frame_id,
            motion_score or 0.0,
            (
                "none"
                if time_since_previous_attempt_ms is None
                else f"{time_since_previous_attempt_ms:.1f}"
            ),
            event_frames,
            ring_depth,
        )
        return True, None, state.attempt_count, time_since_previous_attempt_ms

    def _event_state(
        self,
        camera_id: int,
        event_id: int,
    ) -> _ZeroDetectionEventState:
        key = (camera_id, event_id)
        state = self._zero_detection_events.get(key)
        if state is not None:
            return state
        if len(self._zero_detection_events) >= ZERO_DETECTION_EVENT_STATE_LIMIT:
            oldest_key = next(iter(self._zero_detection_events))
            self._zero_detection_events.pop(oldest_key, None)
        state = _ZeroDetectionEventState()
        self._zero_detection_events[key] = state
        return state

    @property
    def zero_detection_event_state_count(self) -> int:
        return len(self._zero_detection_events)

    def prepare_event_end_fallback(
        self,
        event: MotionEvent,
        *,
        ring_depth: int,
    ) -> DetectionJobResult:
        key = (event.camera_id, event.event_id)
        state = self._event_state(event.camera_id, event.event_id)
        detector_config = self.config.plate_detector
        base_result = {
            "detections": (),
            "used_roi_fallback": False,
            "fallback_reason": None,
            "detector_ms": 0.0,
            "roi_brightness": 0.0,
            "motion_event_id": event.event_id,
            "fallback_attempt": state.attempt_count,
            "motion_score": 0.0,
            "event_frames": len(event.frames),
            "ring_depth": ring_depth,
        }
        try:
            if not detector_config.zero_detection_roi_fallback_enabled:
                return DetectionJobResult(
                    job=None,
                    fallback_skipped_reason="disabled",
                    **base_result,
                )
            detector_crop_frames = self._detector_crop_frame_id_sets.get(
                event.camera_id,
                set(),
            )
            if state.detector_crop_seen or any(
                snapshot.frame_id in detector_crop_frames
                for snapshot in event.frames
            ):
                return DetectionJobResult(
                    job=None,
                    fallback_skipped_reason="detector-success",
                    **base_result,
                )
            if state.attempt_count >= ZERO_DETECTION_FALLBACK_MAX_ATTEMPTS:
                return DetectionJobResult(
                    job=None,
                    fallback_skipped_reason="fallback-limit",
                    **base_result,
                )

            spacing_ms = max(
                detector_config.zero_detection_roi_fallback_interval_ms,
                ZERO_DETECTION_FALLBACK_MIN_SPACING_MS,
            )
            previous_attempt_at = self._last_zero_detection_fallback_at.get(
                event.camera_id
            )
            candidates = [
                snapshot
                for snapshot in event.frames
                if snapshot.frame_id not in state.attempted_frame_ids
                and snapshot.motion_score
                >= self.config.motion_changed_pixel_ratio
                * MOTION_CONTINUE_THRESHOLD_RATIO
                and (
                    previous_attempt_at is None
                    or (snapshot.observed_at - previous_attempt_at) * 1000.0
                    >= spacing_ms
                )
                and crop_roi(
                    snapshot.full_frame,
                    self.config.roi_for(snapshot.direction),
                )
                is not None
            ]
            if not candidates:
                has_meaningful_unused = any(
                    snapshot.frame_id not in state.attempted_frame_ids
                    and snapshot.motion_score
                    >= self.config.motion_changed_pixel_ratio
                    * MOTION_CONTINUE_THRESHOLD_RATIO
                    for snapshot in event.frames
                )
                return DetectionJobResult(
                    job=None,
                    fallback_skipped_reason=(
                        "fallback-spacing"
                        if has_meaningful_unused and previous_attempt_at is not None
                        else "no-meaningful-motion"
                    ),
                    **base_result,
                )

            selected = max(
                candidates,
                key=lambda snapshot: (
                    _snapshot_roi_sharpness(snapshot, self.config.roi_for),
                    snapshot.observed_at,
                    snapshot.motion_score,
                    snapshot.frame_id,
                ),
            )
            time_since_previous_attempt_ms = (
                None
                if previous_attempt_at is None
                else max(
                    0.0,
                    (selected.observed_at - previous_attempt_at) * 1000.0,
                )
            )
            state.attempt_count += 1
            state.last_attempt_at = selected.observed_at
            state.attempted_frame_ids.add(selected.frame_id)
            self._last_zero_detection_fallback_at[event.camera_id] = (
                selected.observed_at
            )
            roi_crop = crop_roi(
                selected.full_frame,
                self.config.roi_for(selected.direction),
            )
            if roi_crop is None:
                return DetectionJobResult(
                    job=None,
                    fallback_skipped_reason="no-meaningful-motion",
                    **base_result,
                )
            full_frame = selected.full_frame
            if full_frame.flags.writeable:
                full_frame = full_frame.copy()
                full_frame.setflags(write=False)
            job = OcrJob(
                camera_id=event.camera_id,
                direction=event.direction,
                captured_at=selected.captured_at,
                observed_at=selected.observed_at,
                received_at=selected.received_at,
                queued_at=time.monotonic(),
                full_frame=full_frame,
                roi_crop=roi_crop,
                ocr_crops=(roi_crop,),
                detections=(),
                used_roi_fallback=True,
                fallback_reason="zero-detection-event-end",
                detector_ms=0.0,
                quality_score=ocr_job_quality_score((roi_crop,), ()),
                job_type=OcrJobType.ZERO_DETECTION_FALLBACK,
                frame_id=selected.frame_id,
                detector_source="event-end",
                ocr_crop_detections=(),
            )
            LOGGER.debug(
                "Zero-detection fallback camera_id=%s event_id=%s "
                "fallback_attempt=%s/%s frame_id=%s motion_score=%.4f "
                "fallback_reason=event-end fallback_skipped_reason=none "
                "time_since_previous_attempt_ms=%s event_frames=%s ring_depth=%s",
                event.camera_id,
                event.event_id,
                state.attempt_count,
                ZERO_DETECTION_FALLBACK_MAX_ATTEMPTS,
                selected.frame_id,
                selected.motion_score,
                (
                    "none"
                    if time_since_previous_attempt_ms is None
                    else f"{time_since_previous_attempt_ms:.1f}"
                ),
                len(event.frames),
                ring_depth,
            )
            return DetectionJobResult(
                job=job,
                detections=(),
                used_roi_fallback=True,
                fallback_reason="zero-detection-event-end",
                detector_ms=0.0,
                roi_brightness=roi_mean_brightness(roi_crop),
                motion_event_id=event.event_id,
                fallback_attempt=state.attempt_count,
                time_since_previous_attempt_ms=time_since_previous_attempt_ms,
                motion_score=selected.motion_score,
                event_frames=len(event.frames),
                ring_depth=ring_depth,
            )
        finally:
            self._zero_detection_events.pop(key, None)

    def _remember_detector_crop_frame(self, camera_id: int, frame_id: int) -> None:
        recent_set = self._detector_crop_frame_id_sets[camera_id]
        if frame_id in recent_set:
            return
        recent = self._detector_crop_frame_ids[camera_id]
        recent.append(frame_id)
        recent_set.add(frame_id)
        while len(recent) > RECENT_PROCESSED_FRAME_ID_LIMIT:
            recent_set.discard(recent.popleft())

    def _log_detector_error(
        self,
        camera_id: int,
        exc: Exception,
        now: float,
        fallback_enabled: bool,
    ) -> None:
        last_logged_at = self._last_detector_error_logged_at.get(camera_id)
        if (
            last_logged_at is not None
            and now - last_logged_at < OCR_DIAGNOSTIC_LOG_INTERVAL_SECONDS
        ):
            return
        self._last_detector_error_logged_at[camera_id] = now
        LOGGER.warning(
            "Plate detector inference failed camera_id=%s fallback=%s error_type=%s",
            camera_id,
            "roi" if fallback_enabled else "disabled",
            type(exc).__name__,
        )


class PlateRecognitionProcessor:
    def __init__(
        self,
        provider: OcrProvider,
        plate_service: PlateService,
        config: PlateRecognitionConfig,
        detector: PlateDetector | None = None,
    ) -> None:
        self.provider = provider
        self.plate_service = plate_service
        self.config = config
        self.detector = detector
        self.confirmations = ConfirmationTracker(
            config.confirmations_required,
            config.confirmation_window_seconds,
        )
        self.presences = PlatePresenceTracker()
        self._last_ocr_started_at: dict[int, float] = {}
        self._last_diagnostic_logged_at: dict[int, float] = {}
        self._last_pipeline_state: dict[int, tuple[RecognitionState, str | None]] = {}
        self._last_detector_error_logged_at: dict[int, float] = {}
        self._last_detector_diagnostic_at: dict[tuple[int, str], float] = {}
        self._last_zero_detection_fallback_at: dict[int, float] = {}
        self._observation_frame_ids: dict[int, deque[int]] = defaultdict(deque)
        self._observation_frame_id_sets: dict[int, set[int]] = defaultdict(set)
        self._recent_saved_evidence: dict[int, deque[_SavedPlateEvidence]] = defaultdict(
            lambda: deque(maxlen=RECENT_SAVED_EVIDENCE_LIMIT)
        )
        self._pending_decisions: dict[int, PendingPlateDecision] = {}
        self._evidence_sequence = 0

    def process(
        self,
        camera_id: int,
        direction: Direction,
        frame: object,
        detected_at: datetime | None = None,
        monotonic_at: float | None = None,
        frame_received_at: float | None = None,
    ) -> RecognitionOutcome:
        roi_crop = crop_roi(frame, self.config.roi_for(direction))
        if roi_crop is None:
            return self._outcome(camera_id, RecognitionState.NO_OCR_TEXT)

        recognition_started_at = time.perf_counter()
        previous_ocr_started_at = self._last_ocr_started_at.get(camera_id)
        self._last_ocr_started_at[camera_id] = recognition_started_at
        roi_brightness = roi_mean_brightness(roi_crop)
        detector_config = self.config.plate_detector
        detections: list[PlateDetection] = []
        detector_ms = 0.0
        used_roi_fallback = not detector_config.enabled
        fallback_reason = "detector-disabled" if used_roi_fallback else None
        ocr_crops: list[np.ndarray] = []
        ocr_crop_detections: list[PlateDetection | None] = []
        tiled_recovery = False

        if detector_config.enabled and self.detector is not None:
            try:
                detections = self.detector.detect(roi_crop)
                tiled_recovery = (
                    getattr(self.detector, "last_diagnostics", None) is not None
                    and getattr(
                        self.detector.last_diagnostics,
                        "detector_variant",
                        None,
                    )
                    == "tiled"
                )
                _log_plate_detector_diagnostics(
                    camera_id,
                    "live",
                    self.detector,
                    roi_crop,
                    self._last_detector_diagnostic_at,
                )
            except Exception as exc:
                self._log_detector_error(
                    camera_id,
                    exc,
                    recognition_started_at,
                    detector_config.fallback_to_roi_ocr,
                )
                if not detector_config.fallback_to_roi_ocr:
                    raise
                used_roi_fallback = True
                fallback_reason = "detector-error"
                ocr_crops = [roi_crop]
            detector_finished_at = time.perf_counter()
            detector_ms = (
                detector_finished_at - recognition_started_at
            ) * 1000.0
            if not used_roi_fallback:
                selected = select_plate_detections(
                    detections,
                    detector_config.max_plate_candidates_per_frame,
                )
                for detection in selected:
                    plate_crop = crop_padded_plate(
                        roi_crop,
                        detection,
                        (
                            detector_config.tiled_recovery_crop_padding_ratio
                            if tiled_recovery
                            else detector_config.crop_padding_ratio
                        ),
                    )
                    if plate_crop is not None:
                        ocr_crops.append(plate_crop)
                        ocr_crop_detections.append(detection)
                if not ocr_crops and self._should_run_zero_detection_fallback(
                    camera_id,
                    monotonic_at,
                    detector_config.zero_detection_roi_fallback_enabled,
                    detector_config.zero_detection_roi_fallback_interval_ms,
                ):
                    used_roi_fallback = True
                    fallback_reason = "zero-detection"
                    ocr_crops = [roi_crop]
            ocr_started_at = detector_finished_at
        elif detector_config.enabled:
            if not detector_config.fallback_to_roi_ocr:
                raise PlateDetectorError(
                    "Plate detector kullanılamıyor ve ROI OCR fallback kapalı."
                )
            used_roi_fallback = True
            fallback_reason = "detector-error"
            ocr_crops = [roi_crop]
            ocr_started_at = recognition_started_at
        else:
            ocr_crops = [roi_crop]
            ocr_started_at = recognition_started_at

        if not ocr_crops:
            self._log_ocr_diagnostics(
                camera_id=camera_id,
                direction=direction,
                frame=frame,
                crop=roi_crop,
                variants=(),
                brightness=roi_brightness,
                ocr_started_at=recognition_started_at,
                previous_ocr_started_at=previous_ocr_started_at,
                processing_duration_ms=0.0,
                inference_duration_ms=0.0,
                detector_ms=detector_ms,
                detections=detections,
                plate_crops=(),
                total_recognition_ms=detector_ms,
                frame_received_at=frame_received_at,
                candidate_found=False,
                used_roi_fallback=used_roi_fallback,
                fallback_reason=fallback_reason,
            )
            return self._outcome(
                camera_id,
                RecognitionState.NO_OCR_TEXT,
                detections=tuple(detections),
                used_roi_fallback=used_roi_fallback,
            )

        variants: list[np.ndarray] = []
        variant_detections: dict[int, PlateDetection] = {}
        for crop_index, ocr_crop in enumerate(ocr_crops):
            brightness = roi_mean_brightness(ocr_crop)
            crop_variants = preprocess_variants(ocr_crop, brightness=brightness)
            detection = (
                ocr_crop_detections[crop_index]
                if crop_index < len(ocr_crop_detections)
                else None
            )
            for variant in crop_variants:
                if detection is not None:
                    variant_detections[len(variants)] = detection
                variants.append(variant)
        inference_started_at = time.perf_counter()
        segments = self.provider.recognize(variants)
        ocr_finished_at = time.perf_counter()
        processing_duration_ms = (ocr_finished_at - ocr_started_at) * 1000.0
        inference_duration_ms = (ocr_finished_at - inference_started_at) * 1000.0
        candidate = select_best_candidate(
            segments,
            camera_id,
            variant_detections=variant_detections,
        )
        self._log_ocr_diagnostics(
            camera_id=camera_id,
            direction=direction,
            frame=frame,
            crop=roi_crop,
            variants=variants,
            brightness=roi_brightness,
            ocr_started_at=recognition_started_at,
            previous_ocr_started_at=previous_ocr_started_at,
            processing_duration_ms=processing_duration_ms,
            inference_duration_ms=inference_duration_ms,
            detector_ms=detector_ms,
            detections=detections,
            plate_crops=ocr_crops,
            total_recognition_ms=(ocr_finished_at - recognition_started_at) * 1000.0,
            frame_received_at=frame_received_at,
            candidate_found=candidate is not None,
            used_roi_fallback=used_roi_fallback,
            fallback_reason=fallback_reason,
        )
        detection_context = {
            "detections": tuple(detections),
            "used_roi_fallback": used_roi_fallback,
        }
        return self._complete_observation(
            camera_id=camera_id,
            direction=direction,
            segments=segments,
            candidate=candidate,
            observed_at=(
                monotonic_at if monotonic_at is not None else time.monotonic()
            ),
            detected_at=detected_at,
            full_frame=frame,
            frame_id=None,
            job_type=(
                OcrJobType.DETECTOR_CROP
                if not used_roi_fallback
                else OcrJobType.DETECTOR_ERROR_FALLBACK
            ),
            quality_score=0.0,
            **detection_context,
        )

    def process_ocr_job(
        self,
        job: OcrJob,
        *,
        queue_depth: int,
    ) -> RecognitionOutcome:
        """Run preprocessing, OCR and persistence for an already detected frame."""
        queue_wait_ms = max(
            0.0,
            (time.monotonic() - job.queued_at) * 1000.0,
        )
        ocr_started_at = time.perf_counter()
        segments: list[OcrSegment] = []
        candidate: PlateCandidate | None = None
        attempted_variants: list[np.ndarray] = []
        variant_names: list[str] = []
        crop_metrics: list[CropQualityMetrics] = []
        profiles: list[OcrImageProfile] = []
        text_detection_box_counts: list[int | None] = []
        current_variant_count = 0
        shadow_variant_count = 0
        inference_calls = 0
        shadow_preprocess_ms = 0.0
        shadow_inference_ms = 0.0
        if job.job_type is OcrJobType.DETECTOR_CROP:
            detector_ocr = recognize_detector_crops(
                self.provider,
                job.ocr_crops,
                job.camera_id,
                self.config.min_confidence,
                crop_detections=job.ocr_crop_detections,
            )
            attempted_variants = list(detector_ocr.variants)
            variant_names = list(detector_ocr.variant_names)
            segments = list(detector_ocr.segments)
            candidate = detector_ocr.candidate
            crop_metrics = list(detector_ocr.quality_metrics)
            profiles = list(detector_ocr.profiles)
            text_detection_box_counts = list(
                detector_ocr.text_detection_box_counts
            )
            current_variant_count = detector_ocr.current_variant_count
            shadow_variant_count = detector_ocr.shadow_variant_count
            inference_calls = detector_ocr.inference_calls
            preprocess_ms = (
                detector_ocr.current_preprocess_ms
                + detector_ocr.shadow_preprocess_ms
            )
            inference_ms = (
                detector_ocr.current_inference_ms
                + detector_ocr.shadow_inference_ms
            )
            shadow_preprocess_ms = detector_ocr.shadow_preprocess_ms
            shadow_inference_ms = detector_ocr.shadow_inference_ms
        else:
            # Full ROI detection is the expensive safety path. Try the compact
            # color image first and only pay for enhanced contrast when needed.
            crop_metrics = [measure_crop_quality(crop) for crop in job.ocr_crops]
            profiles = [classify_crop_quality(item) for item in crop_metrics]
            preprocess_started_at = time.perf_counter()
            variants: list[np.ndarray] = []
            for crop in job.ocr_crops:
                variants.extend(
                    preprocess_roi_fallback_variants(
                        crop,
                        brightness=roi_mean_brightness(crop),
                    )
                )
            preprocess_ms = (
                time.perf_counter() - preprocess_started_at
            ) * 1000.0
            inference_started_at = time.perf_counter()
            for variant_index, variant in enumerate(variants):
                attempted_variants.append(variant)
                variant_names.append(f"roi-fallback-{variant_index}")
                batch = self.provider.recognize([variant])
                inference_calls += 1
                text_detection_box_counts.extend(
                    _provider_detection_box_counts(self.provider, 1, batch)
                )
                segments.extend(
                    OcrSegment(
                        text=segment.text,
                        confidence=segment.confidence,
                        box=segment.box,
                        variant_index=variant_index,
                    )
                    for segment in batch
                )
                candidate = select_best_candidate(segments, job.camera_id)
                if (
                    candidate is not None
                    and candidate.confidence >= self.config.min_confidence
                ):
                    break
            inference_ms = (
                time.perf_counter() - inference_started_at
            ) * 1000.0
            current_variant_count = len(attempted_variants)
        ocr_finished_at = time.perf_counter()
        if candidate is None:
            candidate = select_best_candidate(
                segments,
                job.camera_id,
            )
        outcome = self._complete_observation(
            camera_id=job.camera_id,
            direction=job.direction,
            segments=segments,
            candidate=candidate,
            observed_at=job.observed_at,
            detected_at=job.captured_at,
            full_frame=job.full_frame,
            detections=job.detections,
            used_roi_fallback=job.used_roi_fallback,
            frame_id=job.frame_id,
            job_type=job.job_type,
            quality_score=job.quality_score,
        )
        self._log_queued_ocr_diagnostics(
            job=job,
            variants=attempted_variants,
            segments=segments,
            queue_depth=queue_depth,
            ocr_started_at=ocr_started_at,
            preprocess_ms=preprocess_ms,
            inference_ms=inference_ms,
            queue_wait_ms=queue_wait_ms,
            candidate=candidate,
            outcome=outcome,
            variant_names=variant_names,
            crop_metrics=crop_metrics,
            profiles=profiles,
            text_detection_box_counts=text_detection_box_counts,
            current_variant_count=current_variant_count,
            shadow_variant_count=shadow_variant_count,
            inference_calls=inference_calls,
            shadow_preprocess_ms=shadow_preprocess_ms,
            shadow_inference_ms=shadow_inference_ms,
        )
        return outcome

    def _complete_observation(
        self,
        *,
        camera_id: int,
        direction: Direction,
        segments: Sequence[OcrSegment],
        candidate: PlateCandidate | None,
        observed_at: float,
        detected_at: datetime | None,
        full_frame: object,
        detections: tuple[PlateDetection, ...],
        used_roi_fallback: bool,
        frame_id: int | None,
        job_type: OcrJobType,
        quality_score: float,
    ) -> RecognitionOutcome:
        detection_context = {
            "detections": detections,
            "used_roi_fallback": used_roi_fallback,
        }
        if not segments:
            return self._outcome(
                camera_id,
                RecognitionState.NO_OCR_TEXT,
                **detection_context,
            )
        if candidate is None:
            return self._outcome(
                camera_id,
                RecognitionState.NO_VALID_PLATE,
                **detection_context,
            )
        if candidate.confidence < self.config.min_confidence:
            return self._outcome(
                camera_id,
                RecognitionState.LOW_CONFIDENCE,
                candidate=candidate,
                **detection_context,
            )

        candidate = replace(
            candidate,
            frame_id=frame_id,
            observed_at=observed_at,
            job_type=job_type,
            detector_crop_evidence=job_type is OcrJobType.DETECTOR_CROP,
        )

        if frame_id is not None and not self._claim_observation_frame(
            camera_id, frame_id
        ):
            LOGGER.debug(
                "Duplicate OCR observation skipped camera_id=%s frame_id=%s",
                camera_id,
                frame_id,
            )
            return self._outcome(
                camera_id,
                RecognitionState.DUPLICATE_SUPPRESSED,
                candidate=candidate,
                duplicate=True,
                suppression_reason="active-presence",
                **detection_context,
            )

        detection_time = as_utc(detected_at) if detected_at is not None else utc_now()
        post_save_near_plate = self._post_save_near_plate(
            candidate,
            detection_time,
        )
        spatial_alias = self._has_spatial_alias(
            candidate,
            detection_time,
            post_save_near_plate,
        )
        candidate = replace(candidate, spatial_alias_evidence=spatial_alias)
        self.presences.observe(candidate, observed_at)
        processing_now = time.monotonic()
        progress = self.confirmations.observe_progress(
            candidate,
            observed_at,
            frame_id=frame_id,
            post_save_near_plate=post_save_near_plate,
            spatial_alias=spatial_alias,
        )
        self._log_confirmation_decision(
            candidate,
            progress,
            post_save_near_plate,
            spatial_alias,
        )
        pending_decision: PendingPlateDecision | None = None
        if self.config.plate_stabilization_window_ms > 0:
            pending_decision = self._add_stabilization_evidence(
                candidate=candidate,
                camera_id=camera_id,
                direction=direction,
                observed_at=observed_at,
                captured_at=detection_time,
                full_frame=full_frame,
                quality_score=quality_score,
                detections=detections,
                used_roi_fallback=used_roi_fallback,
                processing_now=processing_now,
                base_confirmed_plate=(
                    progress.candidate.plate
                    if progress.candidate is not None
                    else None
                ),
            )
        if pending_decision is not None:
            evaluation = self._evaluate_pending_decision(pending_decision)
            if (
                evaluation.early_finalizable
                and pending_decision.min_hold_until_monotonic is not None
                and processing_now >= pending_decision.min_hold_until_monotonic
            ):
                return self._finalize_pending_decision(camera_id)
            leader = evaluation.winner or candidate
            state = (
                RecognitionState.STABILIZING
                if pending_decision.provisional_plate is not None
                else RecognitionState.AWAITING_CONFIRMATION
            )
            return self._outcome(
                camera_id,
                state,
                candidate=leader,
                confirmation_count=evaluation.winner_votes,
                confirmation_required=evaluation.required_votes,
                suppression_reason=evaluation.suppression_reason,
                **detection_context,
            )
        confirmed = progress.candidate
        if confirmed is None:
            state = (
                RecognitionState.DUPLICATE_SUPPRESSED
                if progress.suppress
                else RecognitionState.AWAITING_CONFIRMATION
            )
            return self._outcome(
                camera_id,
                state,
                candidate=candidate,
                confirmation_count=progress.observed_count,
                confirmation_required=progress.required_count,
                duplicate=progress.suppress,
                suppression_reason=progress.suppression_reason,
                **detection_context,
            )
        if not self.presences.claim_record(confirmed):
            return self._outcome(
                camera_id,
                RecognitionState.DUPLICATE_SUPPRESSED,
                candidate=confirmed,
                confirmation_count=progress.observed_count,
                confirmation_required=progress.required_count,
                duplicate=True,
                suppression_reason="active-presence",
                **detection_context,
            )
        try:
            record = self.plate_service.save_plate_detection(
                confirmed.plate,
                camera_id,
                confirmed.confidence,
                detection_time,
                full_frame,
            )
        except DuplicatePlateDetection as exc:
            LOGGER.debug(
                "Duplicate plate detection suppressed camera_id=%s "
                "suppression_reason=%s latest_movement_direction=%s",
                camera_id,
                exc.reason.value,
                (
                    exc.latest_direction.value
                    if exc.latest_direction is not None
                    else "none"
                ),
            )
            return self._outcome(
                camera_id,
                RecognitionState.DUPLICATE_SUPPRESSED,
                candidate=confirmed,
                confirmation_count=progress.observed_count,
                confirmation_required=progress.required_count,
                duplicate=True,
                suppression_reason=exc.reason.value,
                **detection_context,
            )
        except Exception:
            self.presences.release_record_claim(confirmed)
            raise
        self._remember_saved_evidence(confirmed, detection_time)
        LOGGER.info("Plate detection saved for camera_id=%s", camera_id)
        return self._outcome(
            camera_id,
            RecognitionState.SAVED,
            candidate=confirmed,
            record=record,
            confirmation_count=progress.observed_count,
            confirmation_required=progress.required_count,
            **detection_context,
        )

    @property
    def pending_decision_count(self) -> int:
        return len(self._pending_decisions)

    def next_pending_deadline(self) -> float | None:
        deadlines = [
            (
                decision.deadline_monotonic
                if decision.provisional_plate is not None
                else decision.cleanup_deadline_monotonic
            )
            for decision in self._pending_decisions.values()
        ]
        return min((value for value in deadlines if value is not None), default=None)

    def finalize_due(self, now: float | None = None) -> list[tuple[int, RecognitionOutcome]]:
        current = time.monotonic() if now is None else now
        outcomes: list[tuple[int, RecognitionOutcome]] = []
        for camera_id, decision in tuple(self._pending_decisions.items()):
            if decision.provisional_plate is None:
                if current >= decision.cleanup_deadline_monotonic:
                    self._pending_decisions.pop(camera_id, None)
                continue
            if (
                decision.deadline_monotonic is not None
                and current >= decision.deadline_monotonic
            ):
                outcomes.append(
                    (
                        camera_id,
                        self._finalize_pending_decision(camera_id),
                    )
                )
        return outcomes

    def clear_pending_decisions(self) -> None:
        self._pending_decisions.clear()

    def _add_stabilization_evidence(
        self,
        *,
        candidate: PlateCandidate,
        camera_id: int,
        direction: Direction,
        observed_at: float,
        captured_at: datetime,
        full_frame: object,
        quality_score: float,
        detections: tuple[PlateDetection, ...],
        used_roi_fallback: bool,
        processing_now: float,
        base_confirmed_plate: str | None,
    ) -> PendingPlateDecision:
        decision = self._pending_decisions.get(camera_id)
        if decision is not None and decision.direction is not direction:
            LOGGER.debug(
                "Pending plate decision discarded camera_id=%s reason=direction-changed",
                camera_id,
            )
            self._pending_decisions.pop(camera_id, None)
            decision = None
        if decision is None:
            if len(self._pending_decisions) >= PENDING_DECISION_CAMERA_LIMIT:
                oldest_camera = min(
                    self._pending_decisions,
                    key=lambda key: self._pending_decisions[key].last_updated_monotonic,
                )
                self._pending_decisions.pop(oldest_camera, None)
                LOGGER.debug(
                    "Pending plate decision evicted camera_id=%s reason=camera-limit",
                    oldest_camera,
                )
            decision = PendingPlateDecision(
                camera_id=camera_id,
                direction=direction,
                last_updated_monotonic=processing_now,
                cleanup_deadline_monotonic=(
                    processing_now + self.config.confirmation_window_seconds
                ),
            )
            self._pending_decisions[camera_id] = decision

        frame_key = candidate.frame_id
        if frame_key is None:
            self._evidence_sequence -= 1
            frame_key = self._evidence_sequence
        if frame_key in decision.frame_keys:
            return decision
        if len(decision.observations) >= PENDING_DECISION_OBSERVATION_LIMIT:
            LOGGER.debug(
                "Pending plate observation ignored camera_id=%s reason=evidence-limit",
                camera_id,
            )
            return decision

        observation = _PlateEvidenceObservation(
            candidate=candidate,
            frame_key=frame_key,
            observed_at=observed_at,
        )
        decision.observations.append(observation)
        decision.frame_keys.add(frame_key)
        decision.last_updated_monotonic = processing_now
        if decision.provisional_plate is None:
            decision.cleanup_deadline_monotonic = (
                processing_now + self.config.confirmation_window_seconds
            )

        representative = _RepresentativeFrame(
            candidate=candidate,
            captured_at=captured_at,
            full_frame=full_frame,
            quality_score=quality_score,
            detections=detections,
            used_roi_fallback=used_roi_fallback,
        )
        previous = decision.representatives.get(candidate.plate)
        if previous is not None:
            if self._representative_score(representative) > self._representative_score(
                previous
            ):
                decision.representatives[candidate.plate] = representative
        elif len(decision.representatives) < PENDING_DECISION_PLATE_LIMIT:
            decision.representatives[candidate.plate] = representative
        else:
            weakest_plate = min(
                decision.representatives,
                key=lambda plate: self._representative_score(
                    decision.representatives[plate]
                ),
            )
            if self._representative_score(representative) > self._representative_score(
                decision.representatives[weakest_plate]
            ):
                decision.representatives.pop(weakest_plate, None)
                decision.representatives[candidate.plate] = representative

        counts: dict[str, int] = defaultdict(int)
        for item in decision.observations:
            counts[item.candidate.plate] += 1
        if (
            decision.provisional_plate is None
            and base_confirmed_plate is not None
            and counts[base_confirmed_plate] > 0
        ):
            provisional = base_confirmed_plate
            window_seconds = self.config.plate_stabilization_window_ms / 1000.0
            min_hold_seconds = self.config.plate_stabilization_min_hold_ms / 1000.0
            decision.provisional_plate = provisional
            decision.deadline_monotonic = processing_now + window_seconds
            decision.min_hold_until_monotonic = processing_now + min_hold_seconds
            LOGGER.debug(
                "Plate provisionally confirmed camera_id=%s candidate=%s "
                "deadline_ms=%s min_hold_ms=%s observations=%s",
                camera_id,
                provisional,
                self.config.plate_stabilization_window_ms,
                self.config.plate_stabilization_min_hold_ms,
                len(decision.observations),
            )
        return decision

    def _evaluate_pending_decision(
        self,
        decision: PendingPlateDecision,
    ) -> _PlateDecisionEvaluation:
        if not decision.observations:
            return _PlateDecisionEvaluation(None, 0, 0, 1, (), False, False)
        provisional = decision.provisional_plate
        counts: dict[str, int] = defaultdict(int)
        groups: dict[str, list[_PlateEvidenceObservation]] = defaultdict(list)
        for item in decision.observations:
            counts[item.candidate.plate] += 1
            groups[item.candidate.plate].append(item)
        anchor = provisional or max(counts, key=lambda plate: (counts[plate], plate))
        cluster = {
            plate: items
            for plate, items in groups.items()
            if plate == anchor or plates_are_near_conflicts(anchor, plate)
        }
        winner_plate = max(
            cluster,
            key=lambda plate: (
                len(cluster[plate]),
                sum(item.candidate.correction_cost == 0 for item in cluster[plate]),
                sum(item.candidate.variant_support for item in cluster[plate]),
                sum(item.candidate.confidence for item in cluster[plate])
                / len(cluster[plate]),
                plate,
            ),
        )
        winner_items = cluster[winner_plate]
        winner_votes = len(winner_items)
        competing = {
            plate: items for plate, items in cluster.items() if plate != winner_plate
        }
        runner_up_votes = max((len(items) for items in competing.values()), default=0)
        correction_risk = any(item.candidate.correction_cost > 0 for item in winner_items)
        required_votes = self.config.confirmations_required
        if correction_risk or competing:
            required_votes = max(required_votes, RISKY_CONFIRMATIONS_REQUIRED)

        representative = decision.representatives.get(winner_plate)
        winner = representative.candidate if representative is not None else winner_items[-1].candidate
        post_save_near_plate = None
        if representative is not None:
            post_save_near_plate = self._post_save_near_plate(
                winner,
                representative.captured_at,
            )
        detector_votes = sum(
            item.candidate.detector_crop_evidence for item in winner_items
        )
        spatial_alias_votes = sum(
            item.candidate.spatial_alias_evidence for item in winner_items
        )
        post_save_safe = True
        suppression_reason: str | None = None
        if post_save_near_plate is not None:
            required_votes = max(required_votes, POST_SAVE_CONFIRMATIONS_REQUIRED)
            post_save_safe = detector_votes >= POST_SAVE_CONFIRMATIONS_REQUIRED
            if spatial_alias_votes >= RISKY_CONFIRMATIONS_REQUIRED:
                post_save_safe = False
                suppression_reason = "near-duplicate-ambiguity"

        margin_ok = not competing or winner_votes - runner_up_votes >= 2
        trailing_reversal = False
        if provisional is not None and winner_plate != provisional:
            relevant_order = [
                item.candidate.plate
                for item in sorted(
                    (item for items in cluster.values() for item in items),
                    key=lambda item: (item.observed_at, item.frame_key),
                )
            ]
            trailing_reversal = (
                len(relevant_order) >= RISKY_CONFIRMATIONS_REQUIRED
                and all(
                    plate == winner_plate
                    for plate in relevant_order[-RISKY_CONFIRMATIONS_REQUIRED:]
                )
            )
        reliable = (
            winner_votes >= required_votes
            and post_save_safe
            and (margin_ok or trailing_reversal)
        )
        early_finalizable = (
            reliable
            and not competing
            and not correction_risk
            and post_save_near_plate is None
            and detector_votes >= RISKY_CONFIRMATIONS_REQUIRED
            and winner_votes >= RISKY_CONFIRMATIONS_REQUIRED
        )
        near_conflicts = tuple(sorted(competing))
        if not reliable and suppression_reason is None and competing:
            suppression_reason = "near-duplicate-ambiguity"
        confidence = sum(item.candidate.confidence for item in winner_items) / winner_votes
        winner = replace(winner, confidence=confidence)
        return _PlateDecisionEvaluation(
            winner=winner,
            winner_votes=winner_votes,
            runner_up_votes=runner_up_votes,
            required_votes=required_votes,
            near_conflicts=near_conflicts,
            reliable=reliable,
            early_finalizable=early_finalizable,
            suppression_reason=suppression_reason,
        )

    def _finalize_pending_decision(
        self,
        camera_id: int,
    ) -> RecognitionOutcome:
        decision = self._pending_decisions.get(camera_id)
        if decision is None:
            return self._outcome(camera_id, RecognitionState.AMBIGUOUS_DISCARDED)
        evaluation = self._evaluate_pending_decision(decision)
        self._pending_decisions.pop(camera_id, None)
        winner = evaluation.winner
        if winner is None or not evaluation.reliable:
            LOGGER.debug(
                "Pending plate decision discarded camera_id=%s winner=%s votes=%s/%s "
                "runner_up_votes=%s near_conflicts=%s suppression_reason=%s",
                camera_id,
                winner.plate if winner is not None else "none",
                evaluation.winner_votes,
                evaluation.required_votes,
                evaluation.runner_up_votes,
                ",".join(evaluation.near_conflicts) or "none",
                evaluation.suppression_reason or "insufficient-evidence",
            )
            return self._outcome(
                camera_id,
                RecognitionState.AMBIGUOUS_DISCARDED,
                candidate=winner,
                confirmation_count=evaluation.winner_votes,
                confirmation_required=evaluation.required_votes,
                suppression_reason=(
                    evaluation.suppression_reason or "insufficient-evidence"
                ),
            )

        representative = decision.representatives.get(winner.plate)
        if representative is None:
            LOGGER.warning(
                "Pending plate decision discarded camera_id=%s winner=%s "
                "reason=representative-frame-evicted",
                camera_id,
                winner.plate,
            )
            return self._outcome(
                camera_id,
                RecognitionState.AMBIGUOUS_DISCARDED,
                candidate=winner,
                confirmation_count=evaluation.winner_votes,
                confirmation_required=evaluation.required_votes,
                suppression_reason="representative-frame-missing",
            )
        if not self.presences.claim_record(winner):
            return self._outcome(
                camera_id,
                RecognitionState.DUPLICATE_SUPPRESSED,
                candidate=winner,
                confirmation_count=evaluation.winner_votes,
                confirmation_required=evaluation.required_votes,
                duplicate=True,
                suppression_reason="active-presence",
                detections=representative.detections,
                used_roi_fallback=representative.used_roi_fallback,
            )
        try:
            record = self.plate_service.save_plate_detection(
                winner.plate,
                camera_id,
                winner.confidence,
                representative.captured_at,
                representative.full_frame,
            )
        except DuplicatePlateDetection as exc:
            return self._outcome(
                camera_id,
                RecognitionState.DUPLICATE_SUPPRESSED,
                candidate=winner,
                confirmation_count=evaluation.winner_votes,
                confirmation_required=evaluation.required_votes,
                duplicate=True,
                suppression_reason=exc.reason.value,
                detections=representative.detections,
                used_roi_fallback=representative.used_roi_fallback,
            )
        except Exception:
            self.presences.release_record_claim(winner)
            raise
        self._remember_saved_evidence(winner, representative.captured_at)
        LOGGER.info("Stabilized plate decision saved for camera_id=%s", camera_id)
        return self._outcome(
            camera_id,
            RecognitionState.SAVED,
            candidate=winner,
            record=record,
            confirmation_count=evaluation.winner_votes,
            confirmation_required=evaluation.required_votes,
            detections=representative.detections,
            used_roi_fallback=representative.used_roi_fallback,
        )

    @staticmethod
    def _representative_score(
        representative: _RepresentativeFrame,
    ) -> tuple[int, float, int, int, float]:
        candidate = representative.candidate
        return (
            int(candidate.detector_crop_evidence),
            representative.quality_score,
            candidate.variant_support,
            -candidate.correction_cost,
            candidate.confidence,
        )

    def _post_save_near_plate(
        self,
        candidate: PlateCandidate,
        detection_time: datetime,
    ) -> str | None:
        for plate, _ in self.plate_service.get_recent_camera_plates(
            candidate.camera_id,
            detection_time,
            POST_SAVE_NEAR_DUPLICATE_WINDOW_SECONDS,
        ):
            if plates_are_near_conflicts(candidate.plate, plate):
                return plate
        return None

    def _has_spatial_alias(
        self,
        candidate: PlateCandidate,
        detection_time: datetime,
        near_plate: str | None,
    ) -> bool:
        if near_plate is None or candidate.detector_bbox is None:
            return False
        cutoff = detection_time.timestamp() - POST_SAVE_NEAR_DUPLICATE_WINDOW_SECONDS
        for evidence in self._recent_saved_evidence[candidate.camera_id]:
            if evidence.detected_at.timestamp() < cutoff:
                continue
            if evidence.plate != near_plate or evidence.detector_bbox is None:
                continue
            if plate_boxes_match(candidate.detector_bbox, evidence.detector_bbox):
                return True
        return False

    def _remember_saved_evidence(
        self,
        candidate: PlateCandidate,
        detection_time: datetime,
    ) -> None:
        self._recent_saved_evidence[candidate.camera_id].append(
            _SavedPlateEvidence(
                candidate.plate,
                detection_time,
                candidate.detector_bbox,
            )
        )

    def _log_confirmation_decision(
        self,
        candidate: PlateCandidate,
        progress: ConfirmationProgress,
        near_duplicate_match: str | None,
        spatial_match: bool,
    ) -> None:
        LOGGER.debug(
            "OCR confirmation decision camera_id=%s frame_id=%s candidate=%s "
            "raw_text=%s correction_cost=%s variant_support=%s base_required=%s "
            "effective_required=%s near_conflicts=%s candidate_votes=%s "
            "runner_up_votes=%s decision=%s suppression_reason=%s "
            "near_duplicate_match=%s spatial_match=%s job_type=%s",
            candidate.camera_id,
            candidate.frame_id,
            candidate.plate,
            candidate.raw_text,
            candidate.correction_cost,
            candidate.variant_support,
            self.confirmations.required,
            progress.required_count,
            ",".join(progress.near_conflicts) or "none",
            progress.observed_count,
            progress.runner_up_votes,
            "suppress" if progress.suppress else "confirm" if progress.candidate else "wait",
            progress.suppression_reason or "none",
            near_duplicate_match or "none",
            "yes" if spatial_match else "no",
            candidate.job_type.value if candidate.job_type is not None else "unknown",
        )

    def _claim_observation_frame(self, camera_id: int, frame_id: int) -> bool:
        recent_set = self._observation_frame_id_sets[camera_id]
        if frame_id in recent_set:
            return False
        recent = self._observation_frame_ids[camera_id]
        recent.append(frame_id)
        recent_set.add(frame_id)
        while len(recent) > RECENT_PROCESSED_FRAME_ID_LIMIT:
            recent_set.discard(recent.popleft())
        return True

    def _outcome(
        self,
        camera_id: int,
        state: RecognitionState,
        *,
        candidate: PlateCandidate | None = None,
        record: PlateRecord | None = None,
        confirmation_count: int = 0,
        confirmation_required: int = 0,
        duplicate: bool = False,
        detections: tuple[PlateDetection, ...] = (),
        used_roi_fallback: bool = False,
        suppression_reason: str | None = None,
    ) -> RecognitionOutcome:
        signature = (state, candidate.plate if candidate is not None else None)
        if (
            LOGGER.isEnabledFor(logging.DEBUG)
            and self._last_pipeline_state.get(camera_id) != signature
        ):
            LOGGER.debug(
                "OCR pipeline state camera_id=%s state=%s confirmation=%s/%s",
                camera_id,
                state.value,
                confirmation_count,
                confirmation_required,
            )
        self._last_pipeline_state[camera_id] = signature
        return RecognitionOutcome(
            candidate=candidate,
            record=record,
            state=state,
            confirmation_count=confirmation_count,
            confirmation_required=confirmation_required,
            duplicate=duplicate,
            detections=detections,
            used_roi_fallback=used_roi_fallback,
            suppression_reason=suppression_reason,
        )

    def _log_ocr_diagnostics(
        self,
        *,
        camera_id: int,
        direction: Direction,
        frame: object,
        crop: np.ndarray,
        variants: Sequence[np.ndarray],
        brightness: float,
        ocr_started_at: float,
        previous_ocr_started_at: float | None,
        processing_duration_ms: float,
        inference_duration_ms: float,
        detector_ms: float,
        detections: Sequence[PlateDetection],
        plate_crops: Sequence[np.ndarray],
        total_recognition_ms: float,
        frame_received_at: float | None,
        candidate_found: bool,
        used_roi_fallback: bool,
        fallback_reason: str | None,
    ) -> None:
        if not LOGGER.isEnabledFor(logging.DEBUG):
            return
        last_logged_at = self._last_diagnostic_logged_at.get(camera_id)
        if (
            last_logged_at is not None
            and ocr_started_at - last_logged_at
            < OCR_DIAGNOSTIC_LOG_INTERVAL_SECONDS
        ):
            return
        self._last_diagnostic_logged_at[camera_id] = ocr_started_at
        actual_interval_ms = (
            None
            if previous_ocr_started_at is None
            else (ocr_started_at - previous_ocr_started_at) * 1000.0
        )
        frame_wait_ms = (
            None
            if frame_received_at is None
            else max(0.0, (ocr_started_at - frame_received_at) * 1000.0)
        )
        LOGGER.debug(
            "OCR diagnostics camera_id=%s direction=%s frame=%s roi=%s "
            "brightness=%.1f low_light=%s variants=%s variant_sizes=%s "
            "detector_ms=%.1f plates=%s det_conf=%s plate_crops=%s "
            "ocr_ms=%.1f processing_ms=%.1f inference_ms=%.1f "
            "total_recognition_ms=%.1f fallback=%s fallback_reason=%s "
            "actual_interval_ms=%s frame_wait_ms=%s candidate=%s",
            camera_id,
            direction.value,
            _image_resolution(frame),
            _image_resolution(crop),
            brightness,
            "yes" if brightness < LOW_LIGHT_THRESHOLD else "no",
            len(variants),
            ",".join(_image_resolution(variant) for variant in variants),
            detector_ms,
            len(detections),
            (
                "none"
                if not detections
                else f"{max(item.confidence for item in detections):.3f}"
            ),
            (
                "none"
                if not plate_crops
                else ",".join(_image_resolution(item) for item in plate_crops)
            ),
            processing_duration_ms,
            processing_duration_ms,
            inference_duration_ms,
            total_recognition_ms,
            "roi" if used_roi_fallback else "no",
            fallback_reason or "none",
            (
                "first"
                if actual_interval_ms is None
                else f"{actual_interval_ms:.1f}"
            ),
            "unknown" if frame_wait_ms is None else f"{frame_wait_ms:.1f}",
            "yes" if candidate_found else "no",
        )

    def _should_run_zero_detection_fallback(
        self,
        camera_id: int,
        observed_at: float | None,
        enabled: bool,
        interval_ms: int,
    ) -> bool:
        if not enabled:
            return False
        now = observed_at if observed_at is not None else time.monotonic()
        last_fallback_at = self._last_zero_detection_fallback_at.get(camera_id)
        if (
            last_fallback_at is not None
            and (now - last_fallback_at) * 1000.0 < interval_ms
        ):
            return False
        self._last_zero_detection_fallback_at[camera_id] = now
        return True

    def _log_queued_ocr_diagnostics(
        self,
        *,
        job: OcrJob,
        variants: Sequence[np.ndarray],
        segments: Sequence[OcrSegment],
        queue_depth: int,
        ocr_started_at: float,
        preprocess_ms: float,
        inference_ms: float,
        queue_wait_ms: float,
        candidate: PlateCandidate | None,
        outcome: RecognitionOutcome,
        variant_names: Sequence[str],
        crop_metrics: Sequence[CropQualityMetrics],
        profiles: Sequence[OcrImageProfile],
        text_detection_box_counts: Sequence[int | None],
        current_variant_count: int,
        shadow_variant_count: int,
        inference_calls: int,
        shadow_preprocess_ms: float,
        shadow_inference_ms: float,
    ) -> None:
        if not LOGGER.isEnabledFor(logging.DEBUG):
            return
        last_logged_at = self._last_diagnostic_logged_at.get(job.camera_id)
        if (
            last_logged_at is not None
            and ocr_started_at - last_logged_at
            < OCR_DIAGNOSTIC_LOG_INTERVAL_SECONDS
        ):
            return
        self._last_diagnostic_logged_at[job.camera_id] = ocr_started_at
        now = time.monotonic()
        end_to_end_ms = max(0.0, (now - job.observed_at) * 1000.0)
        brightness = roi_mean_brightness(job.roi_crop)
        candidate_rejection_reason = "none"
        if outcome.state in {
            RecognitionState.NO_OCR_TEXT,
            RecognitionState.NO_VALID_PLATE,
            RecognitionState.LOW_CONFIDENCE,
            RecognitionState.AMBIGUOUS_DISCARDED,
            RecognitionState.DUPLICATE_SUPPRESSED,
        }:
            candidate_rejection_reason = (
                outcome.suppression_reason or outcome.state.value.lower()
            )
        LOGGER.debug(
            "OCR worker diagnostics camera_id=%s direction=%s brightness=%.1f "
            "low_light=%s profiles=%s crop_quality=%s variants=%s "
            "current_variants=%s shadow_variants=%s inference_calls=%s "
            "queue_depth=%s queue_wait_ms=%.1f "
            "preprocess_ms=%.1f inference_ms=%.1f end_to_end_ms=%.1f "
            "shadow_preprocess_ms=%.1f shadow_inference_ms=%.1f "
            "job_type=%s priority=%s source=%s frame_id=%s fallback_reason=%s "
            "candidate=%s recognition_state=%s candidate_rejection_reason=%s "
            "confirmation=%s/%s detector_bboxes=%s text_detection_boxes=%s "
            "raw_ocr_segments=%s variant_trace=%s",
            job.camera_id,
            job.direction.value,
            brightness,
            (
                "yes"
                if OcrImageProfile.LOW_LIGHT in profiles
                else "no"
            ),
            ",".join(profile.value for profile in profiles) or "none",
            _format_crop_quality_metrics(crop_metrics),
            len(variants),
            current_variant_count,
            shadow_variant_count,
            inference_calls,
            queue_depth,
            queue_wait_ms,
            preprocess_ms,
            inference_ms,
            end_to_end_ms,
            shadow_preprocess_ms,
            shadow_inference_ms,
            job.job_type.value,
            job.job_type.priority,
            job.detector_source,
            job.frame_id,
            job.fallback_reason or "none",
            candidate.plate if candidate is not None else "none",
            outcome.state.value,
            candidate_rejection_reason,
            outcome.confirmation_count,
            outcome.confirmation_required,
            _format_detector_bboxes(job.ocr_crop_detections),
            _format_detection_box_counts(
                variant_names,
                text_detection_box_counts,
            ),
            _format_raw_ocr_segments(segments),
            _format_variant_ocr_trace(
                variant_names,
                segments,
                self.config.min_confidence,
                text_detection_box_counts,
            ),
        )

    def _log_detector_error(
        self,
        camera_id: int,
        exc: Exception,
        now: float,
        fallback_enabled: bool,
    ) -> None:
        last_logged_at = self._last_detector_error_logged_at.get(camera_id)
        if (
            last_logged_at is not None
            and now - last_logged_at < OCR_DIAGNOSTIC_LOG_INTERVAL_SECONDS
        ):
            return
        self._last_detector_error_logged_at[camera_id] = now
        LOGGER.warning(
            "Plate detector inference failed camera_id=%s fallback=%s error_type=%s",
            camera_id,
            "roi" if fallback_enabled else "disabled",
            type(exc).__name__,
        )


def ocr_job_quality_score(
    crops: Sequence[np.ndarray],
    detections: Sequence[PlateDetection] = (),
) -> float:
    """Cheap score favoring confident, larger and sharper plate observations."""
    confidence = max((item.confidence for item in detections), default=0.0)
    best_visual_score = 0.0
    for crop in crops:
        if not isinstance(crop, np.ndarray) or crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        area = float(crop.shape[0] * crop.shape[1])
        visual_score = float(np.log1p(area)) + min(sharpness, 2_000.0) / 100.0
        best_visual_score = max(best_visual_score, visual_score)
    return confidence * 100.0 + best_visual_score


@dataclass(slots=True)
class _PendingFrame:
    direction: Direction
    frame: object
    received_at: float
    observed_at: float
    captured_at: datetime
    frame_id: int = 0
    motion_event_id: int | None = None
    motion_score: float = 0.0
    event_frames: int = 0
    ring_depth: int = 0


class PlateOcrWorker:
    """Single-consumer OCR stage; the provider is never used outside this thread."""

    def __init__(
        self,
        plate_service: PlateService,
        config: PlateRecognitionConfig,
        provider_factory: Callable[[], OcrProvider],
        job_buffer: OcrJobBuffer,
        stop_event: threading.Event,
        on_outcome: Callable[[int, RecognitionOutcome], None],
        on_status: Callable[[RecognitionStatus, str], None],
    ) -> None:
        self.plate_service = plate_service
        self.config = config
        self.provider_factory = provider_factory
        self.job_buffer = job_buffer
        self.stop_event = stop_event
        self.on_outcome = on_outcome
        self.on_status = on_status
        self.initialized = threading.Event()
        self.failed = threading.Event()
        self.initialization_error: Exception | None = None

    def run(self) -> None:
        processor: PlateRecognitionProcessor | None = None
        try:
            try:
                provider = self.provider_factory()
                processor = PlateRecognitionProcessor(
                    provider,
                    self.plate_service,
                    self.config,
                )
            except Exception as exc:
                self.initialization_error = exc
                self.failed.set()
                if isinstance(exc, OcrModelError):
                    LOGGER.error("OCR model initialization failed: %s", exc)
                    self.on_status(RecognitionStatus.UNAVAILABLE, str(exc))
                else:
                    LOGGER.exception("OCR initialization failed")
                    self.on_status(
                        RecognitionStatus.UNAVAILABLE,
                        f"OCR kullanılamıyor: {exc}",
                    )
                return
            finally:
                self.initialized.set()

            def emit_due_outcomes() -> bool:
                try:
                    due_outcomes = processor.finalize_due()
                except Exception as exc:
                    LOGGER.exception("Pending OCR decision finalization failed")
                    self.on_status(
                        RecognitionStatus.ERROR,
                        f"OCR karar kaydı hatası: {exc}",
                    )
                    return False
                for camera_id, due_outcome in due_outcomes:
                    self.on_outcome(camera_id, due_outcome)
                return True

            recovering_from_error = False
            while not self.stop_event.is_set():
                if not emit_due_outcomes():
                    recovering_from_error = True
                stale_before = self.job_buffer.stale_count
                job = self.job_buffer.take(
                    self.stop_event,
                    wait=True,
                    wake_at=processor.next_pending_deadline(),
                )
                stale_discarded = self.job_buffer.stale_count - stale_before
                if stale_discarded and LOGGER.isEnabledFor(logging.DEBUG):
                    LOGGER.debug(
                        "OCR buffer stale jobs discarded count=%s total=%s",
                        stale_discarded,
                        self.job_buffer.stale_count,
                    )
                if job is None:
                    if self.stop_event.is_set():
                        return
                    continue
                try:
                    outcome = processor.process_ocr_job(
                        job,
                        queue_depth=self.job_buffer.pending_count(),
                    )
                except OcrModelError as exc:
                    LOGGER.error("Fatal OCR model error: %s", exc)
                    self.failed.set()
                    self.on_status(RecognitionStatus.UNAVAILABLE, str(exc))
                    return
                except Exception as exc:
                    LOGGER.exception("Transient OCR inference error")
                    self.on_status(RecognitionStatus.ERROR, f"OCR hatası: {exc}")
                    recovering_from_error = True
                    continue
                if recovering_from_error:
                    self.on_status(RecognitionStatus.ACTIVE, "OCR yeniden aktif.")
                    recovering_from_error = False
                self.on_outcome(job.camera_id, outcome)
                if not emit_due_outcomes():
                    recovering_from_error = True
        finally:
            if processor is not None:
                processor.clear_pending_decisions()
            self.initialized.set()


class PlateRecognitionWorker(QObject):
    status_changed = Signal(object, str)
    candidate_changed = Signal(int, object)
    outcome_changed = Signal(int, object)
    detections_changed = Signal(int, object)
    record_saved = Signal(object)
    finished = Signal()

    def __init__(
        self,
        plate_service: PlateService,
        config: PlateRecognitionConfig,
        provider_factory: Callable[[], OcrProvider],
        detector_factory: Callable[[], PlateDetector] | None = None,
    ) -> None:
        super().__init__()
        self.plate_service = plate_service
        self.config = config
        self.provider_factory = provider_factory
        self.detector_factory = detector_factory
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_frames: dict[int, _PendingFrame] = {}
        self._last_processed_at: dict[int, float] = {}
        self._camera_order: deque[int] = deque()
        self._known_camera_ids: set[int] = set()
        self._last_detection_diagnostic_at: dict[int, float] = {}
        self._last_ring_diagnostic_at: dict[int, float] = {}
        self._ingest_diagnostic_started_at: dict[int, float] = {}
        self._ingest_diagnostic_counts: dict[int, int] = {}
        self._next_frame_id = 1
        self._job_buffer = OcrJobBuffer(
            config.max_pending_ocr_jobs_per_camera,
            config.ocr_job_max_age_ms,
            config.detector_crop_ocr_job_max_age_ms,
        )
        self._frame_buffer = PreDetectionFrameBuffer(config)
        self._replay_buffer = ReplayEventBuffer(config)
        self._completed_motion_events: deque[MotionEvent] = deque(
            maxlen=COMPLETED_MOTION_EVENT_LIMIT
        )
        self._processed_frame_ids: dict[int, deque[int]] = defaultdict(deque)
        self._processed_frame_id_sets: dict[int, set[int]] = defaultdict(set)
        self._live_frames_since_replay = 0
        self._ocr_worker: PlateOcrWorker | None = None
        self._ocr_thread: threading.Thread | None = None
        self._detector_processor: PlateDetectionProcessor | None = None

    def submit_frame(self, camera_id: int, direction: Direction, frame: object) -> None:
        if self._stop_event.is_set():
            return
        observed_at = time.monotonic()
        captured_at = datetime.now(timezone.utc)
        with self._lock:
            frame_id = self._next_frame_id
            self._next_frame_id += 1
        owned_frame = frame
        if isinstance(frame, np.ndarray) and frame.flags.writeable:
            owned_frame = frame.copy()
        if isinstance(owned_frame, np.ndarray):
            owned_frame.setflags(write=False)
            completed_events: tuple[MotionEvent, ...] = ()
            snapshot = FrameSnapshot(
                frame_id=frame_id,
                camera_id=camera_id,
                direction=direction,
                captured_at=captured_at,
                observed_at=observed_at,
                received_at=observed_at,
                full_frame=owned_frame,
            )
            completed_events = self._frame_buffer.ingest(snapshot)
            for event in completed_events:
                self._replay_buffer.add(event)
            stored_snapshot = self._frame_buffer.snapshots(camera_id)[-1]
            motion_event_id = (
                completed_events[-1].event_id
                if completed_events
                else self._frame_buffer.active_event_id(camera_id)
            )
            event_frames = (
                len(completed_events[-1].frames)
                if completed_events
                else self._frame_buffer.active_event_frame_count(camera_id)
            )
            ring_depth = self._frame_buffer.ring_depth(camera_id)
            self._log_ring_diagnostics(camera_id, observed_at)
            self._log_ingest_fps(camera_id, observed_at)
        else:
            completed_events = ()
            motion_event_id = None
            stored_snapshot = None
            event_frames = 0
            ring_depth = 0
        item = _PendingFrame(
            direction=direction,
            frame=owned_frame,
            received_at=observed_at,
            observed_at=observed_at,
            captured_at=captured_at,
            frame_id=frame_id,
            motion_event_id=motion_event_id,
            motion_score=(
                stored_snapshot.motion_score
                if stored_snapshot is not None
                else 0.0
            ),
            event_frames=event_frames,
            ring_depth=ring_depth,
        )
        with self._lock:
            self._completed_motion_events.extend(completed_events)
            if camera_id not in self._known_camera_ids:
                self._known_camera_ids.add(camera_id)
                self._camera_order.append(camera_id)
            self._latest_frames[camera_id] = item
        self._wake_event.set()

    @property
    def pending_frame_count(self) -> int:
        with self._lock:
            return len(self._latest_frames)

    def request_stop(self) -> None:
        self._stop_event.set()
        self._wake_event.set()
        self._job_buffer.wake_all()

    @Slot()
    def run(self) -> None:
        self.status_changed.emit(RecognitionStatus.INITIALIZING, "OCR başlatılıyor.")
        ocr_worker = PlateOcrWorker(
            self.plate_service,
            self.config,
            self.provider_factory,
            self._job_buffer,
            self._stop_event,
            self._publish_outcome,
            self.status_changed.emit,
        )
        ocr_thread = threading.Thread(
            target=ocr_worker.run,
            name="plate-ocr-worker",
        )
        self._ocr_worker = ocr_worker
        self._ocr_thread = ocr_thread
        ocr_thread.start()
        try:
            detector: PlateDetector | None = None
            detector_message = ""
            if self.config.plate_detector.enabled and self.detector_factory is not None:
                try:
                    detector = self.detector_factory()
                except Exception as exc:
                    if not self.config.plate_detector.fallback_to_roi_ocr:
                        LOGGER.error(
                            "Plate detector initialization failed error_type=%s",
                            type(exc).__name__,
                        )
                        self.status_changed.emit(
                            RecognitionStatus.UNAVAILABLE,
                            "Plate detector başlatılamadı ve ROI OCR fallback kapalı.",
                        )
                        self._stop_event.wait()
                        return
                    LOGGER.warning(
                        "Plate detector unavailable; ROI OCR fallback active error_type=%s",
                        type(exc).__name__,
                    )
                    detector_message = " Plate detector kullanılamıyor; ROI OCR fallback aktif."

            while not ocr_worker.initialized.wait(0.05):
                if self._stop_event.is_set():
                    return
            if ocr_worker.initialization_error is not None:
                self._stop_event.wait()
                return

            detector_processor = PlateDetectionProcessor(self.config, detector)
            self._detector_processor = detector_processor
            active_message = "OCR aktif."
            active_message += detector_message
            if self.config.warnings:
                active_message += " " + " ".join(self.config.warnings)
            self.status_changed.emit(RecognitionStatus.ACTIVE, active_message)
            while not self._stop_event.is_set():
                if ocr_worker.failed.is_set():
                    self._stop_event.wait(0.05)
                    continue
                pending = self._take_detector_frame(
                    replay_enabled=(
                        detector is not None and self.config.plate_detector.enabled
                    )
                )
                if pending is None:
                    completed_event = self._take_completed_motion_event(
                        replay_enabled=(
                            detector is not None
                            and self.config.plate_detector.enabled
                        )
                    )
                    if completed_event is not None:
                        result = detector_processor.prepare_event_end_fallback(
                            completed_event,
                            ring_depth=self._frame_buffer.ring_depth(
                                completed_event.camera_id
                            ),
                        )
                        buffer_result = (
                            self._job_buffer.add(result.job)
                            if result.job is not None
                            else None
                        )
                        self._log_event_end_fallback(
                            completed_event,
                            result,
                            buffer_result,
                        )
                        continue
                    self._wake_event.wait(0.05)
                    self._wake_event.clear()
                    continue
                source, camera_id, item = pending
                if not self._claim_detector_frame(camera_id, item.frame_id):
                    LOGGER.debug(
                        "Duplicate detector frame skipped camera_id=%s frame_id=%s "
                        "source=%s motion_event_id=%s motion_score=%.4f "
                        "fallback_skipped_reason=same-frame event_frames=%s "
                        "ring_depth=%s",
                        camera_id,
                        item.frame_id,
                        source,
                        item.motion_event_id,
                        item.motion_score,
                        item.event_frames,
                        item.ring_depth,
                    )
                    continue
                try:
                    result = detector_processor.prepare_job(
                        camera_id,
                        item.direction,
                        item.frame,
                        captured_at=item.captured_at,
                        observed_at=item.observed_at,
                        received_at=item.received_at,
                        frame_id=item.frame_id,
                        detector_source=source,
                        allow_zero_detection_fallback=(source == "live"),
                        zero_detection_fallback_event_id=item.motion_event_id,
                        motion_score=item.motion_score,
                        event_frames=item.event_frames,
                        ring_depth=item.ring_depth,
                    )
                except Exception as exc:
                    LOGGER.exception("Transient plate detector error")
                    self.status_changed.emit(
                        RecognitionStatus.ERROR,
                        f"Plate detector hatası: {exc}",
                    )
                    continue
                if source == "live":
                    self.detections_changed.emit(camera_id, result.detections)
                if result.job is None:
                    if source == "live":
                        self.outcome_changed.emit(
                            camera_id,
                            RecognitionOutcome(
                                candidate=None,
                                record=None,
                                state=RecognitionState.NO_OCR_TEXT,
                                detections=result.detections,
                                used_roi_fallback=result.used_roi_fallback,
                            ),
                        )
                    self._log_detection_diagnostics(
                        camera_id, item, result, None, source
                    )
                    continue
                buffer_result = self._job_buffer.add(result.job)
                self._log_detection_diagnostics(
                    camera_id,
                    item,
                    result,
                    buffer_result,
                    source,
                )
        finally:
            self._stop_event.set()
            self._job_buffer.clear()
            self._job_buffer.wake_all()
            self._frame_buffer.clear()
            self._replay_buffer.clear()
            ocr_thread.join()
            with self._lock:
                self._latest_frames.clear()
                self._camera_order.clear()
                self._known_camera_ids.clear()
                self._processed_frame_ids.clear()
                self._processed_frame_id_sets.clear()
                self._completed_motion_events.clear()
                self._ingest_diagnostic_started_at.clear()
                self._ingest_diagnostic_counts.clear()
            self.status_changed.emit(RecognitionStatus.STOPPED, "OCR durduruldu.")
            self.finished.emit()

    def _publish_outcome(self, camera_id: int, outcome: RecognitionOutcome) -> None:
        if (
            outcome.candidate is not None
            and outcome.state is not RecognitionState.LOW_CONFIDENCE
            and self._detector_processor is not None
        ):
            self._detector_processor.note_valid_candidate(
                camera_id,
                time.monotonic(),
            )
        if outcome.candidate is not None:
            self.candidate_changed.emit(camera_id, outcome.candidate)
        if outcome.record is not None:
            self.record_saved.emit(outcome.record)
        self.outcome_changed.emit(camera_id, outcome)

    def _log_detection_diagnostics(
        self,
        camera_id: int,
        item: _PendingFrame,
        result: DetectionJobResult,
        buffer_result: OcrBufferAddResult | None,
        source: str = "live",
    ) -> None:
        if not LOGGER.isEnabledFor(logging.DEBUG):
            return
        now = time.monotonic()
        last_logged_at = self._last_detection_diagnostic_at.get(camera_id)
        if (
            last_logged_at is not None
            and now - last_logged_at < OCR_DIAGNOSTIC_LOG_INTERVAL_SECONDS
        ):
            return
        self._last_detection_diagnostic_at[camera_id] = now
        LOGGER.debug(
            "Detector worker diagnostics camera_id=%s direction=%s detector_ms=%.1f "
            "detector_frame_age_ms=%.1f detections=%s ocr_queue_depth=%s "
            "camera_queue_depth=%s dropped=%s replaced=%s stale_total=%s "
            "fallback_reason=%s source=%s frame_id=%s original_frame_age_ms=%.1f "
            "replay_queue_depth=%s job_type=%s priority=%s detector_crop_count=%s "
            "accepted=%s replaced_job_type=%s drop_reason=%s coalesced=%s "
            "fallback_skipped_reason=%s motion_active=%s motion_event_id=%s "
            "fallback_attempt=%s/%s motion_score=%.4f "
            "time_since_previous_attempt_ms=%s event_frames=%s ring_depth=%s",
            camera_id,
            item.direction.value,
            result.detector_ms,
            max(0.0, (now - item.observed_at) * 1000.0),
            len(result.detections),
            self._job_buffer.pending_count(),
            0 if buffer_result is None else buffer_result.camera_depth,
            self._job_buffer.dropped_count,
            self._job_buffer.replaced_count,
            self._job_buffer.stale_count,
            result.fallback_reason or "none",
            source,
            item.frame_id,
            max(0.0, (now - item.observed_at) * 1000.0),
            self._replay_buffer.pending_event_count(),
            result.job.job_type.value if result.job is not None else "none",
            result.job.job_type.priority if result.job is not None else 0,
            (
                len(result.job.ocr_crops)
                if result.job is not None
                and result.job.job_type is OcrJobType.DETECTOR_CROP
                else 0
            ),
            (
                "no"
                if buffer_result is None
                else "yes" if buffer_result.accepted else "no"
            ),
            (
                buffer_result.replaced_job_type.value
                if buffer_result is not None
                and buffer_result.replaced_job_type is not None
                else "none"
            ),
            (
                buffer_result.drop_reason
                if buffer_result is not None and buffer_result.drop_reason is not None
                else "none"
            ),
            (
                "yes"
                if buffer_result is not None and buffer_result.coalesced
                else "no"
            ),
            result.fallback_skipped_reason or "none",
            "yes" if result.motion_event_id is not None else "no",
            result.motion_event_id,
            result.fallback_attempt,
            ZERO_DETECTION_FALLBACK_MAX_ATTEMPTS,
            result.motion_score,
            (
                "none"
                if result.time_since_previous_attempt_ms is None
                else f"{result.time_since_previous_attempt_ms:.1f}"
            ),
            result.event_frames,
            result.ring_depth,
        )

    def _log_event_end_fallback(
        self,
        event: MotionEvent,
        result: DetectionJobResult,
        buffer_result: OcrBufferAddResult | None,
    ) -> None:
        if not LOGGER.isEnabledFor(logging.DEBUG):
            return
        LOGGER.debug(
            "Event-end fallback camera_id=%s event_id=%s fallback_attempt=%s/%s "
            "frame_id=%s motion_score=%.4f fallback_reason=%s "
            "fallback_skipped_reason=%s time_since_previous_attempt_ms=%s "
            "event_frames=%s ring_depth=%s accepted=%s drop_reason=%s",
            event.camera_id,
            event.event_id,
            result.fallback_attempt,
            ZERO_DETECTION_FALLBACK_MAX_ATTEMPTS,
            result.job.frame_id if result.job is not None else None,
            result.motion_score,
            result.fallback_reason or "none",
            result.fallback_skipped_reason or "none",
            (
                "none"
                if result.time_since_previous_attempt_ms is None
                else f"{result.time_since_previous_attempt_ms:.1f}"
            ),
            result.event_frames,
            result.ring_depth,
            (
                "no"
                if buffer_result is None
                else "yes" if buffer_result.accepted else "no"
            ),
            (
                buffer_result.drop_reason
                if buffer_result is not None and buffer_result.drop_reason is not None
                else "none"
            ),
        )

    def _log_ring_diagnostics(self, camera_id: int, now: float) -> None:
        if not LOGGER.isEnabledFor(logging.DEBUG):
            return
        last_logged_at = self._last_ring_diagnostic_at.get(camera_id)
        if (
            last_logged_at is not None
            and now - last_logged_at < OCR_DIAGNOSTIC_LOG_INTERVAL_SECONDS
        ):
            return
        self._last_ring_diagnostic_at[camera_id] = now
        snapshots = self._frame_buffer.snapshots(camera_id)
        oldest_age_ms = (
            0.0
            if not snapshots
            else max(0.0, (now - snapshots[0].observed_at) * 1000.0)
        )
        newest_age_ms = (
            0.0
            if not snapshots
            else max(0.0, (now - snapshots[-1].observed_at) * 1000.0)
        )
        LOGGER.debug(
            "Pre-detection ring camera_id=%s ring_depth=%s ring_unique_frames=%s "
            "oldest_age_ms=%.1f newest_age_ms=%.1f replay_queue_depth=%s "
            "motion_active=%s",
            camera_id,
            len(snapshots),
            len({snapshot.frame_id for snapshot in snapshots}),
            oldest_age_ms,
            newest_age_ms,
            self._replay_buffer.pending_event_count(camera_id),
            "yes" if self._frame_buffer.active_event_id(camera_id) is not None else "no",
        )

    def _log_ingest_fps(self, camera_id: int, now: float) -> None:
        if not LOGGER.isEnabledFor(logging.DEBUG):
            return
        started_at = self._ingest_diagnostic_started_at.setdefault(camera_id, now)
        count = self._ingest_diagnostic_counts.get(camera_id, 0) + 1
        self._ingest_diagnostic_counts[camera_id] = count
        elapsed = now - started_at
        if elapsed < OCR_DIAGNOSTIC_LOG_INTERVAL_SECONDS:
            return
        self._ingest_diagnostic_started_at[camera_id] = now
        self._ingest_diagnostic_counts[camera_id] = 0
        LOGGER.debug(
            "Recognition ingestion camera_id=%s recognition_ingest_fps=%.2f "
            "ring_depth=%s ring_unique_frames=%s",
            camera_id,
            count / max(elapsed, 0.001),
            self._frame_buffer.ring_depth(camera_id),
            len(
                {
                    snapshot.frame_id
                    for snapshot in self._frame_buffer.snapshots(camera_id)
                }
            ),
        )

    def _take_detector_frame(
        self,
        *,
        replay_enabled: bool,
    ) -> tuple[str, int, _PendingFrame] | None:
        if (
            replay_enabled
            and self._live_frames_since_replay >= LIVE_FRAMES_PER_REPLAY_FRAME
        ):
            replay = self._take_replay_frame()
            if replay is not None:
                self._live_frames_since_replay = 0
                return replay
        live = self._take_due_frame()
        if live is not None:
            self._live_frames_since_replay += 1
            camera_id, item = live
            return "live", camera_id, item
        if replay_enabled:
            replay = self._take_replay_frame()
            if replay is not None:
                self._live_frames_since_replay = 0
                return replay
        return None

    def _take_replay_frame(self) -> tuple[str, int, _PendingFrame] | None:
        replay_item = self._replay_buffer.take_with_event()
        if replay_item is None:
            return None
        snapshot = replay_item.snapshot
        return (
            "replay",
            snapshot.camera_id,
            _PendingFrame(
                direction=snapshot.direction,
                frame=snapshot.full_frame,
                received_at=snapshot.received_at,
                observed_at=snapshot.observed_at,
                captured_at=snapshot.captured_at,
                frame_id=snapshot.frame_id,
                motion_event_id=replay_item.event_id,
                motion_score=snapshot.motion_score,
                event_frames=replay_item.event_frames,
                ring_depth=self._frame_buffer.ring_depth(snapshot.camera_id),
            ),
        )

    def _take_completed_motion_event(
        self,
        *,
        replay_enabled: bool,
    ) -> MotionEvent | None:
        if not replay_enabled or self._replay_buffer.pending_event_count() > 0:
            return None
        with self._lock:
            if self._latest_frames or not self._completed_motion_events:
                return None
            return self._completed_motion_events.popleft()

    def _claim_detector_frame(self, camera_id: int, frame_id: int) -> bool:
        recent_set = self._processed_frame_id_sets[camera_id]
        if frame_id in recent_set:
            return False
        recent = self._processed_frame_ids[camera_id]
        recent.append(frame_id)
        recent_set.add(frame_id)
        while len(recent) > RECENT_PROCESSED_FRAME_ID_LIMIT:
            recent_set.discard(recent.popleft())
        return True

    def _take_due_frame(self) -> tuple[int, _PendingFrame] | None:
        interval = self.config.recognition_interval_ms / 1000.0
        now = time.monotonic()
        with self._lock:
            for _ in range(len(self._camera_order)):
                camera_id = self._camera_order.popleft()
                self._camera_order.append(camera_id)
                if camera_id not in self._latest_frames:
                    continue
                if now - self._last_processed_at.get(camera_id, 0.0) < interval:
                    continue
                self._last_processed_at[camera_id] = now
                return camera_id, self._latest_frames.pop(camera_id)
        return None


@dataclass(slots=True)
class _RecognitionRuntime:
    worker: PlateRecognitionWorker
    thread: QThread


class PlateRecognitionService(QObject):
    status_changed = Signal(object, str)
    candidate_changed = Signal(int, object)
    outcome_changed = Signal(int, object)
    detections_changed = Signal(int, object)
    record_saved = Signal(object)
    STOP_TIMEOUT_MS = 5_000

    def __init__(
        self,
        camera_service: CameraService,
        plate_service: PlateService,
        config: PlateRecognitionConfig,
        provider_factory: Callable[[], OcrProvider] | None = None,
        detector_factory: Callable[[], PlateDetector] | None = None,
        settings_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.camera_service = camera_service
        self.plate_service = plate_service
        self.config = config
        self.settings_path = (
            settings_path or application_root() / "config" / "settings.json"
        ).resolve()
        self._custom_provider_factory = provider_factory
        self._custom_detector_factory = detector_factory
        self.provider_factory = provider_factory or self._default_provider_factory
        self.detector_factory = detector_factory or self._default_detector_factory
        self._runtime: _RecognitionRuntime | None = None
        self._directions: dict[int, Direction] = {}
        self._ingest_lock = threading.RLock()
        self._status = RecognitionStatus.STOPPED
        self.camera_service.analysis_frame_ready.connect(
            self._receive_analysis_frame,
            Qt.ConnectionType.DirectConnection,
        )

    def start(self) -> RecognitionStatus:
        with self._ingest_lock:
            if self._runtime is not None:
                return self._status
        directions = {
            camera.id: camera.direction
            for camera in self.camera_service.list_cameras()
        }
        worker = PlateRecognitionWorker(
            self.plate_service,
            self.config,
            self.provider_factory,
            self.detector_factory,
        )
        thread = QThread()
        worker.moveToThread(thread)
        with self._ingest_lock:
            self._runtime = _RecognitionRuntime(worker, thread)
            self._directions = directions
        thread.started.connect(worker.run)
        worker.status_changed.connect(self._relay_status)
        worker.candidate_changed.connect(self.candidate_changed.emit)
        worker.outcome_changed.connect(self.outcome_changed.emit)
        worker.detections_changed.connect(self.detections_changed.emit)
        worker.record_saved.connect(self.record_saved.emit)
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._set_status(RecognitionStatus.INITIALIZING, "OCR başlatılıyor.")
        thread.start()
        return RecognitionStatus.INITIALIZING

    def stop(self) -> None:
        with self._ingest_lock:
            runtime = self._runtime
        if runtime is None:
            return
        runtime.worker.request_stop()
        if runtime.thread.wait(self.STOP_TIMEOUT_MS):
            with self._ingest_lock:
                if self._runtime is runtime:
                    self._runtime = None
                    self._directions.clear()
            self._set_status(RecognitionStatus.STOPPED, "OCR durduruldu.")
        else:
            self._set_status(
                RecognitionStatus.ERROR,
                "OCR belirtilen sürede durdurulamadı.",
            )

    def get_status(self) -> RecognitionStatus:
        return self._status

    def apply_config(self, config: PlateRecognitionConfig) -> None:
        """Apply OCR settings while camera preview threads keep running."""
        was_running = self._runtime is not None
        if was_running:
            self.stop()
        self.config = config
        if self._custom_provider_factory is None:
            self.provider_factory = self._default_provider_factory
        if self._custom_detector_factory is None:
            self.detector_factory = self._default_detector_factory
        if was_running:
            self.start()

    def _default_provider_factory(self) -> OcrProvider:
        return PaddleOcrProvider(
            self.config.model_root,
            backend=self.config.ocr_backend,
            cpu_threads=self.config.ocr_cpu_threads,
        )

    def _default_detector_factory(self) -> PlateDetector:
        return OpenVinoPlateDetector(self.config.plate_detector)

    @Slot(int, object)
    def _receive_analysis_frame(self, camera_id: int, frame: object) -> None:
        # DirectConnection executes on the camera worker thread. Keep this path
        # free of UI work, DB access and inference; submit_frame owns synchronization.
        with self._ingest_lock:
            runtime = self._runtime
            direction = self._directions.get(camera_id)
        if runtime is None or direction is None:
            return
        runtime.worker.submit_frame(camera_id, direction, frame)

    @Slot(object, str)
    def _relay_status(self, status: RecognitionStatus, message: str) -> None:
        self._set_status(RecognitionStatus(status), message)

    def _set_status(self, status: RecognitionStatus, message: str) -> None:
        self._status = status
        self.status_changed.emit(status, message)


def crop_roi(frame: object, roi: NormalizedRoi) -> np.ndarray | None:
    if not isinstance(frame, np.ndarray) or frame.ndim < 2:
        return None
    height, width = frame.shape[:2]
    x1 = max(0, min(width - 1, round(roi.x * width)))
    y1 = max(0, min(height - 1, round(roi.y * height)))
    x2 = max(x1 + 1, min(width, round((roi.x + roi.width) * width)))
    y2 = max(y1 + 1, min(height, round((roi.y + roi.height) * height)))
    crop = frame[y1:y2, x1:x2]
    return crop.copy() if crop.size else None


def roi_mean_brightness(crop: np.ndarray) -> float:
    """Return mean grayscale luminance on OpenCV's 0-255 uint8 scale."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def measure_crop_quality(crop: np.ndarray) -> CropQualityMetrics:
    """Measure cheap plate-crop luminance, contrast, sharpness and clipping."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    luma_p10, luma_p90 = np.percentile(gray, (10, 90))
    gray_float = gray.astype(np.float32)
    local_background = cv2.GaussianBlur(gray_float, (0, 0), sigmaX=3.0)
    return CropQualityMetrics(
        width=int(crop.shape[1]),
        height=int(crop.shape[0]),
        luma_mean=float(np.mean(gray)),
        luma_median=float(np.median(gray)),
        luma_p10=float(luma_p10),
        luma_p90=float(luma_p90),
        dynamic_range=float(luma_p90 - luma_p10),
        grayscale_stddev=float(np.std(gray)),
        local_contrast=float(np.mean(np.abs(gray_float - local_background))),
        laplacian_sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        saturated_black_ratio=float(np.mean(gray <= SATURATED_BLACK_MAX_VALUE)),
        saturated_white_ratio=float(np.mean(gray >= SATURATED_WHITE_MIN_VALUE)),
    )


def classify_crop_quality(metrics: CropQualityMetrics) -> OcrImageProfile:
    if metrics.luma_mean < LOW_LIGHT_THRESHOLD:
        return OcrImageProfile.LOW_LIGHT
    if (
        metrics.luma_p10 >= OVEREXPOSED_P10_THRESHOLD
        and metrics.saturated_white_ratio >= OVEREXPOSED_WHITE_RATIO_THRESHOLD
    ):
        return OcrImageProfile.OVEREXPOSED
    if (
        metrics.luma_mean <= SHADOW_MAX_MEAN_LUMINANCE
        and metrics.dynamic_range < SHADOW_DYNAMIC_RANGE_THRESHOLD
        and (
            metrics.grayscale_stddev < SHADOW_GRAYSCALE_STDDEV_THRESHOLD
            or metrics.local_contrast < SHADOW_LOCAL_CONTRAST_THRESHOLD
        )
    ):
        return OcrImageProfile.SHADOW_LOW_CONTRAST
    return OcrImageProfile.NORMAL


def preprocess_shadow_comparison_variants(crop: np.ndarray) -> list[np.ndarray]:
    """Build the measured color/gray A-B pair used by the field CLI."""
    gamma_lut = _shadow_gamma_lut()
    return [
        _shadow_gamma_color(crop, gamma_lut),
        _shadow_gamma_gray(crop, gamma_lut),
        _shadow_gamma_clahe_gray(crop, gamma_lut),
    ]


def preprocess_shadow_variants(
    crop: np.ndarray,
    *,
    profile: OcrImageProfile | None = None,
) -> list[np.ndarray]:
    """Build the selected bounded recovery variant for difficult plate crops."""
    gamma_lut = _shadow_gamma_lut()
    return [_shadow_gamma_gray(crop, gamma_lut)]


def _shadow_gamma_lut() -> np.ndarray:
    gamma_lut = np.array(
        [
            ((value / 255.0) ** LOW_LIGHT_GAMMA) * 255.0
            for value in range(256)
        ],
        dtype=np.uint8,
    )
    return gamma_lut


def _shadow_gamma_color(crop: np.ndarray, gamma_lut: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    corrected_lightness = cv2.LUT(lightness, gamma_lut)
    return cv2.cvtColor(
        cv2.merge((corrected_lightness, channel_a, channel_b)),
        cv2.COLOR_LAB2BGR,
    )


def _shadow_gamma_gray(crop: np.ndarray, gamma_lut: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(cv2.LUT(gray, gamma_lut), cv2.COLOR_GRAY2BGR)


def _shadow_gamma_clahe_gray(
    crop: np.ndarray,
    gamma_lut: np.ndarray,
) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    contrasted = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    ).apply(cv2.LUT(gray, gamma_lut))
    softened = cv2.GaussianBlur(contrasted, (0, 0), sigmaX=1.0)
    contrasted = cv2.addWeighted(contrasted, 1.10, softened, -0.10, 0)
    return cv2.cvtColor(contrasted, cv2.COLOR_GRAY2BGR)


def preprocess_roi_fallback_variants(
    crop: np.ndarray,
    *,
    brightness: float | None = None,
) -> list[np.ndarray]:
    """Build two compact ROI variants for sequential safety-fallback OCR."""
    if crop.shape[1] > ROI_FALLBACK_MAX_WIDTH:
        scale = ROI_FALLBACK_MAX_WIDTH / crop.shape[1]
        compact = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    else:
        compact = crop.copy()

    gray = cv2.cvtColor(compact, cv2.COLOR_BGR2GRAY)
    measured_brightness = (
        roi_mean_brightness(compact) if brightness is None else brightness
    )
    if measured_brightness < LOW_LIGHT_THRESHOLD:
        gamma_lut = np.array(
            [
                ((value / 255.0) ** LOW_LIGHT_GAMMA) * 255.0
                for value in range(256)
            ],
            dtype=np.uint8,
        )
        gray = cv2.LUT(gray, gamma_lut)
    contrasted = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    ).apply(gray)
    return [compact, cv2.cvtColor(contrasted, cv2.COLOR_GRAY2BGR)]


def preprocess_variants(
    crop: np.ndarray,
    *,
    brightness: float | None = None,
) -> list[np.ndarray]:
    original = crop
    if crop.shape[1] < 600:
        scale = min(3.0, 600 / max(1, crop.shape[1]))
        original = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    contrasted = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    contrasted_bgr = cv2.cvtColor(contrasted, cv2.COLOR_GRAY2BGR)
    upscaled = cv2.resize(
        crop,
        None,
        fx=2.0,
        fy=2.0,
        interpolation=cv2.INTER_CUBIC,
    )
    upscaled_gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    upscaled_contrasted = cv2.createCLAHE(
        clipLimit=1.8,
        tileGridSize=(8, 8),
    ).apply(upscaled_gray)
    softened = cv2.GaussianBlur(upscaled_contrasted, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(
        upscaled_contrasted,
        1.25,
        softened,
        -0.25,
        0,
    )
    enhanced_upscaled_bgr = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    variants = [original, contrasted_bgr, enhanced_upscaled_bgr]

    measured_brightness = (
        roi_mean_brightness(crop) if brightness is None else brightness
    )
    if measured_brightness < LOW_LIGHT_THRESHOLD:
        low_light_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gamma_lut = np.array(
            [
                ((value / 255.0) ** LOW_LIGHT_GAMMA) * 255.0
                for value in range(256)
            ],
            dtype=np.uint8,
        )
        brightened = cv2.LUT(low_light_gray, gamma_lut)
        low_light_contrasted = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        ).apply(brightened)
        low_light_softened = cv2.GaussianBlur(
            low_light_contrasted,
            (0, 0),
            sigmaX=1.0,
        )
        low_light_sharpened = cv2.addWeighted(
            low_light_contrasted,
            1.10,
            low_light_softened,
            -0.10,
            0,
        )
        variants.append(
            cv2.cvtColor(low_light_sharpened, cv2.COLOR_GRAY2BGR)
        )

    return variants


def recognize_detector_crops(
    provider: OcrProvider,
    crops: Sequence[np.ndarray],
    camera_id: int,
    min_confidence: float,
    *,
    crop_detections: Sequence[PlateDetection | None] = (),
) -> DetectorCropOcrResult:
    """Run the current crop OCR first, then a bounded shadow-only recovery."""
    current_preprocess_started = time.perf_counter()
    variants: list[np.ndarray] = []
    variant_names: list[str] = []
    variant_detections: dict[int, PlateDetection] = {}
    metrics: list[CropQualityMetrics] = []
    profiles: list[OcrImageProfile] = []
    for crop_index, crop in enumerate(crops):
        crop_metrics = measure_crop_quality(crop)
        profile = classify_crop_quality(crop_metrics)
        metrics.append(crop_metrics)
        profiles.append(profile)
        current_variants = preprocess_variants(
            crop,
            brightness=crop_metrics.luma_mean,
        )
        detection = (
            crop_detections[crop_index]
            if crop_index < len(crop_detections)
            else None
        )
        for local_index, variant in enumerate(current_variants):
            variant_index = len(variants)
            variants.append(variant)
            variant_names.append(
                _crop_variant_name(
                    OCR_VARIANT_NAMES[local_index],
                    crop_index,
                    len(crops),
                )
            )
            if detection is not None:
                variant_detections[variant_index] = detection
    current_preprocess_ms = (
        time.perf_counter() - current_preprocess_started
    ) * 1000.0

    current_inference_started = time.perf_counter()
    current_segments = provider.recognize(variants)
    current_inference_ms = (
        time.perf_counter() - current_inference_started
    ) * 1000.0
    segments = list(current_segments)
    detection_box_counts = list(
        _provider_detection_box_counts(provider, len(variants), current_segments)
    )
    candidate = select_best_candidate(
        segments,
        camera_id,
        variant_detections=variant_detections,
    )
    current_variant_count = len(variants)
    shadow_variant_count = 0
    shadow_preprocess_ms = 0.0
    shadow_inference_ms = 0.0
    inference_calls = 1 if variants else 0
    needs_shadow_recovery = (
        candidate is None or candidate.confidence < min_confidence
    ) and any(
        _is_shadow_recovery_eligible(profile, crop_metrics)
        for profile, crop_metrics in zip(profiles, metrics)
    )

    if needs_shadow_recovery:
        shadow_preprocess_started = time.perf_counter()
        shadow_variants: list[np.ndarray] = []
        for crop_index, (crop, profile) in enumerate(zip(crops, profiles)):
            if not _is_shadow_recovery_eligible(profile, metrics[crop_index]):
                continue
            detection = (
                crop_detections[crop_index]
                if crop_index < len(crop_detections)
                else None
            )
            for local_index, variant in enumerate(
                preprocess_shadow_variants(crop, profile=profile)
            ):
                variant_index = len(variants) + len(shadow_variants)
                shadow_variants.append(variant)
                variant_names.append(
                    _crop_variant_name(
                        SHADOW_OCR_VARIANT_NAMES[local_index],
                        crop_index,
                        len(crops),
                    )
                )
                if detection is not None:
                    variant_detections[variant_index] = detection
        shadow_preprocess_ms = (
            time.perf_counter() - shadow_preprocess_started
        ) * 1000.0
        shadow_inference_started = time.perf_counter()
        shadow_segments = provider.recognize(shadow_variants)
        shadow_inference_ms = (
            time.perf_counter() - shadow_inference_started
        ) * 1000.0
        inference_calls += 1 if shadow_variants else 0
        offset = len(variants)
        reindexed_shadow_segments = [
            replace(segment, variant_index=segment.variant_index + offset)
            for segment in shadow_segments
        ]
        segments.extend(reindexed_shadow_segments)
        detection_box_counts.extend(
            _provider_detection_box_counts(
                provider,
                len(shadow_variants),
                shadow_segments,
            )
        )
        variants.extend(shadow_variants)
        shadow_variant_count = len(shadow_variants)
        candidate = select_best_candidate(
            segments,
            camera_id,
            variant_detections=variant_detections,
        )

    return DetectorCropOcrResult(
        variants=tuple(variants),
        variant_names=tuple(variant_names),
        segments=tuple(segments),
        candidate=candidate,
        quality_metrics=tuple(metrics),
        profiles=tuple(profiles),
        current_variant_count=current_variant_count,
        shadow_variant_count=shadow_variant_count,
        inference_calls=inference_calls,
        text_detection_box_counts=tuple(detection_box_counts),
        current_preprocess_ms=current_preprocess_ms,
        current_inference_ms=current_inference_ms,
        shadow_preprocess_ms=shadow_preprocess_ms,
        shadow_inference_ms=shadow_inference_ms,
    )


def _crop_variant_name(name: str, crop_index: int, crop_count: int) -> str:
    return name if crop_count == 1 else f"crop-{crop_index}-{name}"


def _is_shadow_recovery_eligible(
    profile: OcrImageProfile,
    metrics: CropQualityMetrics,
) -> bool:
    return profile in {
        OcrImageProfile.LOW_LIGHT,
        OcrImageProfile.SHADOW_LOW_CONTRAST,
    } and (
        metrics.dynamic_range >= SHADOW_MIN_RECOVERABLE_DYNAMIC_RANGE
        or metrics.local_contrast >= SHADOW_MIN_RECOVERABLE_LOCAL_CONTRAST
    )


def _provider_detection_box_counts(
    provider: OcrProvider,
    variant_count: int,
    segments: Sequence[OcrSegment],
) -> tuple[int | None, ...]:
    reported = getattr(provider, "last_detection_box_counts", None)
    if (
        isinstance(reported, tuple)
        and len(reported) == variant_count
        and all(isinstance(value, int) for value in reported)
    ):
        return reported
    inferred: list[int | None] = [None] * variant_count
    for segment in segments:
        if 0 <= segment.variant_index < variant_count:
            current = inferred[segment.variant_index]
            inferred[segment.variant_index] = (current or 0) + 1
    return tuple(inferred)


def _image_resolution(image: object) -> str:
    if not isinstance(image, np.ndarray) or image.ndim < 2:
        return "unknown"
    return f"{image.shape[1]}x{image.shape[0]}"


def _format_raw_ocr_segments(segments: Sequence[OcrSegment]) -> str:
    if not segments:
        return "none"
    values: list[str] = []
    for segment in segments[:12]:
        raw = segment.text.replace("|", "/").replace("\n", " ")[:48]
        corrected = correct_plate_candidate(segment.text)
        values.append(
            f"v{segment.variant_index}:{raw!r}:{segment.confidence:.3f}:"
            f"{corrected or 'invalid'}"
        )
    if len(segments) > len(values):
        values.append(f"+{len(segments) - len(values)}-more")
    return "|".join(values)


def _format_crop_quality_metrics(
    metrics: Sequence[CropQualityMetrics],
) -> str:
    if not metrics:
        return "none"
    return "|".join(
        (
            f"crop{index}={item.width}x{item.height},mean={item.luma_mean:.1f},"
            f"median={item.luma_median:.1f},p10={item.luma_p10:.1f},"
            f"p90={item.luma_p90:.1f},range={item.dynamic_range:.1f},"
            f"stddev={item.grayscale_stddev:.1f},local={item.local_contrast:.1f},"
            f"sharpness={item.laplacian_sharpness:.1f},"
            f"black={item.saturated_black_ratio:.3f},"
            f"white={item.saturated_white_ratio:.3f}"
        )
        for index, item in enumerate(metrics)
    )


def _format_detector_bboxes(
    detections: Sequence[PlateDetection | None],
) -> str:
    values = [
        f"{item.x},{item.y},{item.width},{item.height}"
        for item in detections
        if item is not None
    ]
    return "|".join(values) if values else "none"


def _format_detection_box_counts(
    variant_names: Sequence[str],
    counts: Sequence[int | None],
) -> str:
    if not variant_names:
        return "none"
    return "|".join(
        f"{name}:{'unknown' if index >= len(counts) or counts[index] is None else counts[index]}"
        for index, name in enumerate(variant_names[:12])
    )


def _format_variant_ocr_trace(
    variant_names: Sequence[str],
    segments: Sequence[OcrSegment],
    min_confidence: float,
    detection_box_counts: Sequence[int | None],
) -> str:
    if not variant_names:
        return "none"
    grouped: dict[int, list[OcrSegment]] = defaultdict(list)
    for segment in segments:
        grouped[segment.variant_index].append(segment)
    values: list[str] = []
    for variant_index, name in enumerate(variant_names[:12]):
        variant_segments = grouped.get(variant_index, [])[:2]
        box_count = (
            detection_box_counts[variant_index]
            if variant_index < len(detection_box_counts)
            else None
        )
        if not variant_segments:
            rejection = (
                "no-text-detected"
                if box_count == 0
                else "no-recognized-text"
            )
            values.append(
                f"{name}:boxes={'unknown' if box_count is None else box_count}:"
                f"raw=none:rejection={rejection}"
            )
            continue
        for segment in variant_segments:
            raw = segment.text.replace("|", "/").replace("\n", " ")[:48]
            normalized = normalize_plate_text(segment.text)
            corrected_with_cost = correct_plate_candidate_with_cost(segment.text)
            corrected = (
                corrected_with_cost[0]
                if corrected_with_cost is not None
                else None
            )
            correction_cost = (
                corrected_with_cost[1]
                if corrected_with_cost is not None
                else None
            )
            valid = bool(
                corrected is not None
                and TurkishPlateValidator.is_valid(corrected)
            )
            rejection = "none"
            if not normalized:
                rejection = "empty-normalized-text"
            elif not valid:
                rejection = "invalid-turkish-plate"
            elif segment.confidence < min_confidence:
                rejection = "below-min-confidence"
            values.append(
                f"{name}:boxes={'unknown' if box_count is None else box_count}:"
                f"raw={raw!r}:normalized={normalized!r}:corrected={corrected or 'none'}:"
                f"correction_cost={'none' if correction_cost is None else correction_cost}:"
                f"ocr_conf={segment.confidence:.3f}:valid={'yes' if valid else 'no'}:"
                f"bbox={segment.box}:rejection={rejection}"
            )
    return "|".join(values)


def select_best_candidate(
    segments: Sequence[OcrSegment],
    camera_id: int,
    *,
    variant_detections: dict[int, PlateDetection] | None = None,
) -> PlateCandidate | None:
    usable = [segment for segment in segments if segment.text.strip()]
    if not usable:
        return None
    raw_groups: list[tuple[str, float, int]] = []
    for variant_index in sorted({segment.variant_index for segment in usable}):
        ordered = sorted(
            (segment for segment in usable if segment.variant_index == variant_index),
            key=lambda segment: (segment.box[0], segment.box[1]),
        )
        for segment in ordered:
            raw_groups.append((segment.text, segment.confidence, variant_index))
        for start in range(len(ordered)):
            for end in range(start + 2, min(len(ordered), start + 4) + 1):
                group = ordered[start:end]
                raw_groups.append(
                    (
                        "".join(segment.text for segment in group),
                        sum(segment.confidence for segment in group) / len(group),
                        variant_index,
                    )
                )

    evidence: dict[str, list[tuple[str, str, int, float, int]]] = defaultdict(list)
    for raw_text, confidence, variant_index in raw_groups:
        corrected = correct_plate_candidate_with_cost(raw_text)
        if corrected is None:
            continue
        plate, correction_cost = corrected
        evidence[plate].append(
            (
                raw_text,
                normalize_plate_text(raw_text),
                correction_cost,
                _safe_score(confidence),
                variant_index,
            )
        )

    candidates: list[PlateCandidate] = []
    for plate, items in evidence.items():
        supporting_variants = tuple(sorted({item[4] for item in items}))
        best = min(items, key=lambda item: (item[2], -item[3], item[1]))
        detection = (
            variant_detections.get(best[4])
            if variant_detections is not None
            else None
        )
        candidate = PlateCandidate(
            plate=plate,
            confidence=best[3],
            raw_text=best[0],
            camera_id=camera_id,
            normalized_raw_text=best[1],
            correction_cost=best[2],
            variant_support=len(supporting_variants),
            supporting_variants=supporting_variants,
            detector_bbox=(
                (
                    float(detection.x),
                    float(detection.y),
                    float(detection.width),
                    float(detection.height),
                )
                if detection is not None
                else None
            ),
        )
        candidates.append(candidate)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda candidate: (
            candidate.variant_support,
            -candidate.correction_cost,
            candidate.confidence,
            candidate.plate,
        ),
    )


def plates_are_near_conflicts(first: str, second: str) -> bool:
    """Flag ambiguity without guessing which one-character plate is correct."""
    if first == second or len(first) != len(second):
        return False
    if not (
        TurkishPlateValidator.is_valid(first)
        and TurkishPlateValidator.is_valid(second)
    ):
        return False
    if first[:2] != second[:2] or _plate_structure(first) != _plate_structure(second):
        return False
    return sum(left != right for left, right in zip(first, second)) <= 1


def _plate_structure(plate: str) -> str:
    return "".join("D" if character.isdigit() else "L" for character in plate)


def plate_boxes_match(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    """Compare detector x/y/width/height boxes without introducing tracking."""
    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second
    intersection_width = max(
        0.0,
        min(first_x + first_width, second_x + second_width) - max(first_x, second_x),
    )
    intersection_height = max(
        0.0,
        min(first_y + first_height, second_y + second_height) - max(first_y, second_y),
    )
    intersection = intersection_width * intersection_height
    union = first_width * first_height + second_width * second_height - intersection
    if union > 0 and intersection / union >= 0.30:
        return True
    first_center = (first_x + first_width / 2.0, first_y + first_height / 2.0)
    second_center = (second_x + second_width / 2.0, second_y + second_height / 2.0)
    center_distance = (
        (first_center[0] - second_center[0]) ** 2
        + (first_center[1] - second_center[1]) ** 2
    ) ** 0.5
    reference_size = max(
        1.0,
        (first_width + first_height + second_width + second_height) / 4.0,
    )
    return center_distance <= reference_size * 0.50


def _translate(
    value: str,
    corrections: dict[str, str],
    accepts: Callable[[str], bool],
) -> tuple[str | None, int]:
    output: list[str] = []
    cost = 0
    for character in value:
        if accepts(character):
            output.append(character)
        elif character in corrections:
            output.append(corrections[character])
            cost += 1
        else:
            return None, 0
    return "".join(output), cost


def _is_ascii_letter(value: str) -> bool:
    return len(value) == 1 and "A" <= value <= "Z"


def _safe_score(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _safe_box(value: object, fallback_index: int) -> tuple[float, float, float, float]:
    try:
        x1, y1, x2, y2 = value
        return float(x1), float(y1), float(x2), float(y2)
    except (TypeError, ValueError):
        position = float(fallback_index)
        return position, 0.0, position + 1.0, 1.0
