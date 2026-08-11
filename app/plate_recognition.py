from __future__ import annotations

import re
import os
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
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
    select_ocr_backend,
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


LOGGER = logging.getLogger(__name__)


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
            "text_detection_model_dir": str(detection_dir),
            "text_recognition_model_dir": str(recognition_dir),
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "device": "cpu",
            "enable_hpi": False,
        }
        if self.backend is OcrBackend.ONNX:
            options["engine"] = "onnxruntime"
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
        key = (candidate.camera_id, candidate.plate)
        cutoff = observed_at - self.window_seconds
        observations = self._observations[key]
        while observations and observations[0][0] < cutoff:
            observations.popleft()

        if self._confirmed_until.get(key, 0.0) >= observed_at:
            return None

        observations.append((observed_at, candidate.confidence))
        if len(observations) < self.required:
            return None

        confidence = sum(value for _, value in observations) / len(observations)
        observations.clear()
        self._confirmed_until[key] = observed_at + self.window_seconds
        return PlateCandidate(
            plate=candidate.plate,
            confidence=confidence,
            raw_text=candidate.raw_text,
            camera_id=candidate.camera_id,
        )


@dataclass(frozen=True, slots=True)
class RecognitionOutcome:
    candidate: PlateCandidate | None
    record: PlateRecord | None
    duplicate: bool = False


class PlateRecognitionProcessor:
    def __init__(
        self,
        provider: OcrProvider,
        plate_service: PlateService,
        config: PlateRecognitionConfig,
    ) -> None:
        self.provider = provider
        self.plate_service = plate_service
        self.config = config
        self.confirmations = ConfirmationTracker(
            config.confirmations_required,
            config.confirmation_window_seconds,
        )

    def process(
        self,
        camera_id: int,
        direction: Direction,
        frame: object,
        detected_at: datetime | None = None,
        monotonic_at: float | None = None,
    ) -> RecognitionOutcome:
        crop = crop_roi(frame, self.config.roi_for(direction))
        if crop is None:
            return RecognitionOutcome(candidate=None, record=None)
        segments = self.provider.recognize(preprocess_variants(crop))
        candidate = select_best_candidate(segments, camera_id)
        if candidate is None or candidate.confidence < self.config.min_confidence:
            return RecognitionOutcome(candidate=candidate, record=None)

        confirmed = self.confirmations.observe(
            candidate,
            monotonic_at if monotonic_at is not None else time.monotonic(),
        )
        if confirmed is None:
            return RecognitionOutcome(candidate=candidate, record=None)
        try:
            record = self.plate_service.save_plate_detection(
                confirmed.plate,
                camera_id,
                confirmed.confidence,
                detected_at,
            )
        except DuplicatePlateDetection:
            LOGGER.info("Duplicate plate detection suppressed for camera_id=%s", camera_id)
            return RecognitionOutcome(candidate=confirmed, record=None, duplicate=True)
        LOGGER.info("Plate detection saved for camera_id=%s", camera_id)
        return RecognitionOutcome(candidate=confirmed, record=record)


@dataclass(slots=True)
class _PendingFrame:
    direction: Direction
    frame: object


class PlateRecognitionWorker(QObject):
    status_changed = Signal(object, str)
    candidate_changed = Signal(int, object)
    record_saved = Signal(object)
    finished = Signal()

    def __init__(
        self,
        plate_service: PlateService,
        config: PlateRecognitionConfig,
        provider_factory: Callable[[], OcrProvider],
    ) -> None:
        super().__init__()
        self.plate_service = plate_service
        self.config = config
        self.provider_factory = provider_factory
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_frames: dict[int, _PendingFrame] = {}
        self._last_processed_at: dict[int, float] = {}

    def submit_frame(self, camera_id: int, direction: Direction, frame: object) -> None:
        with self._lock:
            self._latest_frames[camera_id] = _PendingFrame(direction, frame)
        self._wake_event.set()

    @property
    def pending_frame_count(self) -> int:
        with self._lock:
            return len(self._latest_frames)

    def request_stop(self) -> None:
        self._stop_event.set()
        self._wake_event.set()

    @Slot()
    def run(self) -> None:
        self.status_changed.emit(RecognitionStatus.INITIALIZING, "OCR başlatılıyor.")
        try:
            try:
                provider = self.provider_factory()
            except OcrModelError as exc:
                LOGGER.error("OCR model initialization failed: %s", exc)
                self.status_changed.emit(RecognitionStatus.UNAVAILABLE, str(exc))
                self._stop_event.wait()
                return
            except Exception as exc:
                LOGGER.exception("OCR initialization failed")
                self.status_changed.emit(
                    RecognitionStatus.UNAVAILABLE,
                    f"OCR kullanılamıyor: {exc}",
                )
                self._stop_event.wait()
                return

            processor = PlateRecognitionProcessor(provider, self.plate_service, self.config)
            active_message = "OCR aktif."
            if self.config.warnings:
                active_message += " " + " ".join(self.config.warnings)
            self.status_changed.emit(RecognitionStatus.ACTIVE, active_message)
            recovering_from_error = False
            while not self._stop_event.is_set():
                pending = self._take_due_frame()
                if pending is None:
                    self._wake_event.wait(0.05)
                    self._wake_event.clear()
                    continue
                camera_id, item = pending
                try:
                    outcome = processor.process(camera_id, item.direction, item.frame)
                except OcrModelError as exc:
                    LOGGER.error("Fatal OCR model error: %s", exc)
                    self.status_changed.emit(RecognitionStatus.UNAVAILABLE, str(exc))
                    self._stop_event.wait()
                    return
                except Exception as exc:
                    LOGGER.exception("Transient OCR inference error")
                    self.status_changed.emit(RecognitionStatus.ERROR, f"OCR hatası: {exc}")
                    recovering_from_error = True
                    continue
                if recovering_from_error:
                    self.status_changed.emit(RecognitionStatus.ACTIVE, "OCR yeniden aktif.")
                    recovering_from_error = False
                if outcome.candidate is not None:
                    self.candidate_changed.emit(camera_id, outcome.candidate)
                if outcome.record is not None:
                    self.record_saved.emit(outcome.record)
        finally:
            with self._lock:
                self._latest_frames.clear()
            self.status_changed.emit(RecognitionStatus.STOPPED, "OCR durduruldu.")
            self.finished.emit()

    def _take_due_frame(self) -> tuple[int, _PendingFrame] | None:
        interval = self.config.recognition_interval_ms / 1000.0
        now = time.monotonic()
        with self._lock:
            due_camera_id: int | None = None
            for camera_id, item in self._latest_frames.items():
                if now - self._last_processed_at.get(camera_id, 0.0) < interval:
                    continue
                due_camera_id = camera_id
                break
            if due_camera_id is not None:
                self._last_processed_at[due_camera_id] = now
                return due_camera_id, self._latest_frames.pop(due_camera_id)
        return None


@dataclass(slots=True)
class _RecognitionRuntime:
    worker: PlateRecognitionWorker
    thread: QThread


class PlateRecognitionService(QObject):
    status_changed = Signal(object, str)
    candidate_changed = Signal(int, object)
    record_saved = Signal(object)
    STOP_TIMEOUT_MS = 5_000

    def __init__(
        self,
        camera_service: CameraService,
        plate_service: PlateService,
        config: PlateRecognitionConfig,
        provider_factory: Callable[[], OcrProvider] | None = None,
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
        self.provider_factory = provider_factory or self._default_provider_factory
        self._runtime: _RecognitionRuntime | None = None
        self._directions: dict[int, Direction] = {}
        self._status = RecognitionStatus.STOPPED
        self.camera_service.frame_ready.connect(self._receive_frame)

    def start(self) -> RecognitionStatus:
        if self._runtime is not None:
            return self._status
        worker = PlateRecognitionWorker(
            self.plate_service,
            self.config,
            self.provider_factory,
        )
        thread = QThread()
        worker.moveToThread(thread)
        self._runtime = _RecognitionRuntime(worker, thread)
        thread.started.connect(worker.run)
        worker.status_changed.connect(self._relay_status)
        worker.candidate_changed.connect(self.candidate_changed.emit)
        worker.record_saved.connect(self.record_saved.emit)
        worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._set_status(RecognitionStatus.INITIALIZING, "OCR başlatılıyor.")
        thread.start()
        return RecognitionStatus.INITIALIZING

    def stop(self) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        runtime.worker.request_stop()
        if runtime.thread.wait(self.STOP_TIMEOUT_MS):
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
        if was_running:
            self.start()

    def _default_provider_factory(self) -> OcrProvider:
        return PaddleOcrProvider(
            self.config.model_root,
            backend=self.config.ocr_backend,
        )

    @Slot(int, object)
    def _receive_frame(self, camera_id: int, frame: object) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        direction = self._directions.get(camera_id)
        if direction is None:
            try:
                direction = self.camera_service.get_camera(camera_id).direction
            except ValueError:
                return
            self._directions[camera_id] = direction
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


def preprocess_variants(crop: np.ndarray) -> list[np.ndarray]:
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
    return [original, contrasted_bgr]


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
