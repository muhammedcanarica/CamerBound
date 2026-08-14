from __future__ import annotations

import re
import os
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
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

    @property
    def priority(self) -> int:
        return {
            OcrJobType.DETECTOR_CROP: 3,
            OcrJobType.DETECTOR_ERROR_FALLBACK: 2,
            OcrJobType.ZERO_DETECTION_FALLBACK: 1,
        }[self]


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
OCR_VARIANT_NAMES = (
    "adaptive-color",
    "adaptive-clahe",
    "upscaled-2x-clahe-sharpened",
    "low-light-gamma-clahe-sharpened",
)
ROI_FALLBACK_MAX_WIDTH = 960


def _log_plate_detector_diagnostics(
    camera_id: int,
    source: str,
    detector: PlateDetector,
) -> None:
    if not LOGGER.isEnabledFor(logging.DEBUG):
        return
    diagnostics = getattr(detector, "last_diagnostics", None)
    if not isinstance(diagnostics, DetectorDiagnostics):
        return
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
        "detections=%s raw_detector_ms=%.1f enhanced_detector_ms=%.1f",
        camera_id,
        source,
        diagnostics.detector_variant,
        brightness,
        brightness,
        shadow_metric,
        "yes" if diagnostics.enhanced_pass else "no",
        diagnostics.detections,
        diagnostics.raw_detector_ms,
        diagnostics.enhanced_detector_ms,
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

    def recognize(self, images: Sequence[np.ndarray]) -> list[OcrSegment]:
        segments: list[OcrSegment] = []
        if not images:
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
    return corrected[0][1]


class ConfirmationTracker:
    def __init__(self, required: int, window_seconds: float) -> None:
        self.required = max(1, required)
        self.window_seconds = max(0.1, window_seconds)
        self._observations: dict[tuple[int, str], deque[tuple[float, float]]] = defaultdict(deque)
        self._confirmed_until: dict[tuple[int, str], float] = {}

    def observe(self, candidate: PlateCandidate, observed_at: float) -> PlateCandidate | None:
        return self.observe_progress(candidate, observed_at).candidate

    def observe_progress(
        self,
        candidate: PlateCandidate,
        observed_at: float,
    ) -> ConfirmationProgress:
        key = (candidate.camera_id, candidate.plate)
        observations = self._observations[key]
        confirmed_until = self._confirmed_until.get(key, 0.0)
        if confirmed_until >= observed_at:
            return ConfirmationProgress(None, 0, self.required)

        observations.append((observed_at, candidate.confidence))
        ordered = sorted(observations, key=lambda item: item[0])
        newest_at = ordered[-1][0]
        cutoff = newest_at - self.window_seconds
        observations.clear()
        observations.extend(item for item in ordered if item[0] >= cutoff)
        if len(observations) < self.required:
            return ConfirmationProgress(None, len(observations), self.required)

        confidence = sum(value for _, value in observations) / len(observations)
        observations.clear()
        self._confirmed_until[key] = newest_at + self.window_seconds
        return ConfirmationProgress(
            PlateCandidate(
                plate=candidate.plate,
                confidence=confidence,
                raw_text=candidate.raw_text,
                camera_id=candidate.camera_id,
            ),
            self.required,
            self.required,
        )

@dataclass(frozen=True, slots=True)
class ConfirmationProgress:
    candidate: PlateCandidate | None
    observed_count: int
    required_count: int


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


class RecognitionState(StrEnum):
    NO_OCR_TEXT = "NO_OCR_TEXT"
    NO_VALID_PLATE = "NO_VALID_PLATE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
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
                return snapshot
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
            if (
                job.job_type is not OcrJobType.DETECTOR_CROP
                and any(item.job_type is job.job_type for item in camera_jobs)
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
    ) -> OcrJob | None:
        with self._condition:
            while True:
                if stop_event is not None and stop_event.is_set():
                    return None
                now = time.monotonic()
                for camera_id in tuple(self._camera_order):
                    self._discard_stale_locked(self._jobs[camera_id], now)
                for job_type in (
                    OcrJobType.DETECTOR_CROP,
                    OcrJobType.DETECTOR_ERROR_FALLBACK,
                    OcrJobType.ZERO_DETECTION_FALLBACK,
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
                self._condition.wait(0.05)

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
        self._last_zero_detection_fallback_event_id: dict[int, int] = {}
        self._last_detector_error_logged_at: dict[int, float] = {}

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
    ) -> DetectionJobResult:
        roi_crop = crop_roi(frame, self.config.roi_for(direction))
        if roi_crop is None:
            return DetectionJobResult(None, (), False, None, 0.0, 0.0)

        detector_started_at = time.perf_counter()
        detector_config = self.config.plate_detector
        detections: list[PlateDetection] = []
        selected: list[PlateDetection] = []
        used_roi_fallback = not detector_config.enabled
        fallback_reason = "detector-disabled" if used_roi_fallback else None
        ocr_crops: list[np.ndarray] = []
        fallback_skipped_reason: str | None = None

        if detector_config.enabled and self.detector is not None:
            try:
                detections = self.detector.detect(roi_crop)
                _log_plate_detector_diagnostics(
                    camera_id,
                    detector_source,
                    self.detector,
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
                ocr_crops = [
                    plate_crop
                    for detection in selected
                    if (
                        plate_crop := crop_padded_plate(
                            roi_crop,
                            detection,
                            detector_config.crop_padding_ratio,
                        )
                    )
                    is not None
                ]
                if not ocr_crops:
                    should_fallback, fallback_skipped_reason = (
                        self._should_run_zero_detection_fallback(
                            camera_id,
                            observed_at,
                            detector_config.zero_detection_roi_fallback_enabled,
                            detector_config.zero_detection_roi_fallback_interval_ms,
                            allow_zero_detection_fallback,
                            zero_detection_fallback_event_id,
                        )
                    )
                    if should_fallback:
                        used_roi_fallback = True
                        fallback_reason = "zero-detection"
                        ocr_crops = [roi_crop]
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
        )

    def _should_run_zero_detection_fallback(
        self,
        camera_id: int,
        observed_at: float,
        enabled: bool,
        interval_ms: int,
        allowed_for_source: bool,
        motion_event_id: int | None,
    ) -> tuple[bool, str | None]:
        if not enabled:
            return False, "disabled"
        if not allowed_for_source:
            return False, "replay-zero-detection"
        if motion_event_id is None:
            return False, "no-meaningful-motion"
        if self._last_zero_detection_fallback_event_id.get(camera_id) == motion_event_id:
            return False, "event-fallback-used"
        last_fallback_at = self._last_zero_detection_fallback_at.get(camera_id)
        if (
            last_fallback_at is not None
            and (observed_at - last_fallback_at) * 1000.0 < interval_ms
        ):
            return False, "interval-throttled"
        self._last_zero_detection_fallback_at[camera_id] = observed_at
        self._last_zero_detection_fallback_event_id[camera_id] = motion_event_id
        return True, None

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
        self._last_zero_detection_fallback_at: dict[int, float] = {}
        self._observation_frame_ids: dict[int, deque[int]] = defaultdict(deque)
        self._observation_frame_id_sets: dict[int, set[int]] = defaultdict(set)

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

        if detector_config.enabled and self.detector is not None:
            try:
                detections = self.detector.detect(roi_crop)
                _log_plate_detector_diagnostics(camera_id, "live", self.detector)
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
                ocr_crops = [
                    plate_crop
                    for detection in selected
                    if (
                        plate_crop := crop_padded_plate(
                            roi_crop,
                            detection,
                            detector_config.crop_padding_ratio,
                        )
                    )
                    is not None
                ]
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
        for ocr_crop in ocr_crops:
            brightness = roi_mean_brightness(ocr_crop)
            variants.extend(preprocess_variants(ocr_crop, brightness=brightness))
        inference_started_at = time.perf_counter()
        segments = self.provider.recognize(variants)
        ocr_finished_at = time.perf_counter()
        processing_duration_ms = (ocr_finished_at - ocr_started_at) * 1000.0
        inference_duration_ms = (ocr_finished_at - inference_started_at) * 1000.0
        candidate = select_best_candidate(segments, camera_id)
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
            segments=segments,
            candidate=candidate,
            observed_at=(
                monotonic_at if monotonic_at is not None else time.monotonic()
            ),
            detected_at=detected_at,
            full_frame=frame,
            frame_id=None,
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
        preprocess_started_at = ocr_started_at
        variants: list[np.ndarray] = []
        for ocr_crop in job.ocr_crops:
            brightness = roi_mean_brightness(ocr_crop)
            if job.job_type is OcrJobType.DETECTOR_CROP:
                variants.extend(preprocess_variants(ocr_crop, brightness=brightness))
            else:
                variants.extend(
                    preprocess_roi_fallback_variants(
                        ocr_crop,
                        brightness=brightness,
                    )
                )
        inference_started_at = time.perf_counter()
        preprocess_ms = (inference_started_at - preprocess_started_at) * 1000.0
        segments: list[OcrSegment] = []
        candidate: PlateCandidate | None = None
        attempted_variants: list[np.ndarray] = []
        if job.job_type is OcrJobType.DETECTOR_CROP:
            attempted_variants = variants
            segments = self.provider.recognize(variants)
            candidate = select_best_candidate(segments, job.camera_id)
        else:
            # Full ROI detection is the expensive safety path. Try the compact
            # color image first and only pay for enhanced contrast when needed.
            for variant_index, variant in enumerate(variants):
                attempted_variants.append(variant)
                batch = self.provider.recognize([variant])
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
        ocr_finished_at = time.perf_counter()
        inference_ms = (ocr_finished_at - inference_started_at) * 1000.0
        if candidate is None:
            candidate = select_best_candidate(segments, job.camera_id)
        outcome = self._complete_observation(
            camera_id=job.camera_id,
            segments=segments,
            candidate=candidate,
            observed_at=job.observed_at,
            detected_at=job.captured_at,
            full_frame=job.full_frame,
            detections=job.detections,
            used_roi_fallback=job.used_roi_fallback,
            frame_id=job.frame_id,
        )
        self._log_queued_ocr_diagnostics(
            job=job,
            variants=attempted_variants,
            queue_depth=queue_depth,
            ocr_started_at=ocr_started_at,
            preprocess_ms=preprocess_ms,
            inference_ms=inference_ms,
            queue_wait_ms=queue_wait_ms,
            candidate=candidate,
            outcome=outcome,
        )
        return outcome

    def _complete_observation(
        self,
        *,
        camera_id: int,
        segments: Sequence[OcrSegment],
        candidate: PlateCandidate | None,
        observed_at: float,
        detected_at: datetime | None,
        full_frame: object,
        detections: tuple[PlateDetection, ...],
        used_roi_fallback: bool,
        frame_id: int | None,
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
                **detection_context,
            )

        self.presences.observe(candidate, observed_at)
        progress = self.confirmations.observe_progress(candidate, observed_at)
        confirmed = progress.candidate
        if confirmed is None:
            return self._outcome(
                camera_id,
                RecognitionState.AWAITING_CONFIRMATION,
                candidate=candidate,
                confirmation_count=progress.observed_count,
                confirmation_required=progress.required_count,
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
                **detection_context,
            )
        try:
            record = self.plate_service.save_plate_detection(
                confirmed.plate,
                camera_id,
                confirmed.confidence,
                detected_at,
                full_frame,
            )
        except DuplicatePlateDetection:
            LOGGER.info("Duplicate plate detection suppressed for camera_id=%s", camera_id)
            return self._outcome(
                camera_id,
                RecognitionState.DUPLICATE_SUPPRESSED,
                candidate=confirmed,
                confirmation_count=progress.observed_count,
                confirmation_required=progress.required_count,
                duplicate=True,
                **detection_context,
            )
        except Exception:
            self.presences.release_record_claim(confirmed)
            raise
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
        queue_depth: int,
        ocr_started_at: float,
        preprocess_ms: float,
        inference_ms: float,
        queue_wait_ms: float,
        candidate: PlateCandidate | None,
        outcome: RecognitionOutcome,
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
        LOGGER.debug(
            "OCR worker diagnostics camera_id=%s direction=%s brightness=%.1f "
            "low_light=%s variants=%s queue_depth=%s queue_wait_ms=%.1f "
            "preprocess_ms=%.1f inference_ms=%.1f end_to_end_ms=%.1f "
            "job_type=%s priority=%s source=%s frame_id=%s fallback_reason=%s "
            "candidate=%s recognition_state=%s confirmation=%s/%s",
            job.camera_id,
            job.direction.value,
            brightness,
            "yes" if brightness < LOW_LIGHT_THRESHOLD else "no",
            len(variants),
            queue_depth,
            queue_wait_ms,
            preprocess_ms,
            inference_ms,
            end_to_end_ms,
            job.job_type.value,
            job.job_type.priority,
            job.detector_source,
            job.frame_id,
            job.fallback_reason or "none",
            candidate.plate if candidate is not None else "none",
            outcome.state.value,
            outcome.confirmation_count,
            outcome.confirmation_required,
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

            recovering_from_error = False
            while not self.stop_event.is_set():
                stale_before = self.job_buffer.stale_count
                job = self.job_buffer.take(self.stop_event, wait=True)
                stale_discarded = self.job_buffer.stale_count - stale_before
                if stale_discarded and LOGGER.isEnabledFor(logging.DEBUG):
                    LOGGER.debug(
                        "OCR buffer stale jobs discarded count=%s total=%s",
                        stale_discarded,
                        self.job_buffer.stale_count,
                    )
                if job is None:
                    return
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
        finally:
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
        self._processed_frame_ids: dict[int, deque[int]] = defaultdict(deque)
        self._processed_frame_id_sets: dict[int, set[int]] = defaultdict(set)
        self._live_frames_since_replay = 0
        self._ocr_worker: PlateOcrWorker | None = None
        self._ocr_thread: threading.Thread | None = None

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
            motion_event_id = (
                completed_events[-1].event_id
                if completed_events
                else self._frame_buffer.active_event_id(camera_id)
            )
            self._log_ring_diagnostics(camera_id, observed_at)
            self._log_ingest_fps(camera_id, observed_at)
        else:
            motion_event_id = None
        item = _PendingFrame(
            direction=direction,
            frame=owned_frame,
            received_at=observed_at,
            observed_at=observed_at,
            captured_at=captured_at,
            frame_id=frame_id,
            motion_event_id=motion_event_id,
        )
        with self._lock:
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
                    self._wake_event.wait(0.05)
                    self._wake_event.clear()
                    continue
                source, camera_id, item = pending
                if not self._claim_detector_frame(camera_id, item.frame_id):
                    LOGGER.debug(
                        "Duplicate detector frame skipped camera_id=%s frame_id=%s source=%s",
                        camera_id,
                        item.frame_id,
                        source,
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
                self._ingest_diagnostic_started_at.clear()
                self._ingest_diagnostic_counts.clear()
            self.status_changed.emit(RecognitionStatus.STOPPED, "OCR durduruldu.")
            self.finished.emit()

    def _publish_outcome(self, camera_id: int, outcome: RecognitionOutcome) -> None:
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
            "fallback_skipped_reason=%s motion_active=%s motion_event_id=%s",
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
        snapshot = self._replay_buffer.take()
        if snapshot is None:
            return None
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
            ),
        )

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


def _image_resolution(image: object) -> str:
    if not isinstance(image, np.ndarray) or image.ndim < 2:
        return "unknown"
    return f"{image.shape[1]}x{image.shape[0]}"


def select_best_candidate(
    segments: Sequence[OcrSegment],
    camera_id: int,
) -> PlateCandidate | None:
    usable = [segment for segment in segments if segment.text.strip()]
    if not usable:
        return None
    raw_groups: list[tuple[str, float]] = []
    for variant_index in sorted({segment.variant_index for segment in usable}):
        ordered = sorted(
            (segment for segment in usable if segment.variant_index == variant_index),
            key=lambda segment: (segment.box[0], segment.box[1]),
        )
        for segment in ordered:
            raw_groups.append((segment.text, segment.confidence))
        for start in range(len(ordered)):
            for end in range(start + 2, min(len(ordered), start + 4) + 1):
                group = ordered[start:end]
                raw_groups.append(
                    (
                        "".join(segment.text for segment in group),
                        sum(segment.confidence for segment in group) / len(group),
                    )
                )

    candidates: dict[str, PlateCandidate] = {}
    for raw_text, confidence in raw_groups:
        plate = correct_plate_candidate(raw_text)
        if plate is None:
            continue
        candidate = PlateCandidate(
            plate=plate,
            confidence=_safe_score(confidence),
            raw_text=raw_text,
            camera_id=camera_id,
        )
        previous = candidates.get(plate)
        if previous is None or candidate.confidence > previous.confidence:
            candidates[plate] = candidate
    if not candidates:
        return None
    return max(candidates.values(), key=lambda candidate: candidate.confidence)


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
