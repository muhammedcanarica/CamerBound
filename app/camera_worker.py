from __future__ import annotations

import threading
import time
import logging
import sys
from enum import StrEnum
from typing import Callable, Protocol
from urllib.parse import urlsplit

import cv2
from PySide6.QtCore import QObject, Signal, Slot


LOGGER = logging.getLogger(__name__)
NETWORK_OPEN_TIMEOUT_MS = 5_000
NETWORK_READ_TIMEOUT_MS = 5_000
MAX_CONSECUTIVE_READ_FAILURES = 3
NETWORK_SCHEMES = {"rtsp", "http", "https"}


class CameraStatus(StrEnum):
    STOPPED = "STOPPED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    ERROR = "ERROR"


class VideoCaptureLike(Protocol):
    def isOpened(self) -> bool: ...

    def read(self) -> tuple[bool, object]: ...

    def release(self) -> None: ...

    def get(self, property_id: int) -> float: ...


CaptureSource = str | int
CaptureFactory = Callable[[CaptureSource], VideoCaptureLike]


def create_video_capture(source: CaptureSource) -> VideoCaptureLike:
    """Create an OpenCV capture with bounded network timeouts where supported."""
    if isinstance(source, int) and sys.platform == "win32":
        return _create_windows_webcam_capture(source)

    capture = cv2.VideoCapture()
    is_network_source = _is_network_source(source)
    timeout_parameters: list[int] = []
    if is_network_source:
        for property_name, timeout_ms in (
            ("CAP_PROP_OPEN_TIMEOUT_MSEC", NETWORK_OPEN_TIMEOUT_MS),
            ("CAP_PROP_READ_TIMEOUT_MSEC", NETWORK_READ_TIMEOUT_MS),
        ):
            property_id = getattr(cv2, property_name, None)
            if property_id is not None:
                timeout_parameters.extend((property_id, timeout_ms))

    if timeout_parameters:
        try:
            # FFmpeg timeout properties are open-only and must be supplied here.
            capture.open(source, cv2.CAP_ANY, timeout_parameters)
            return capture
        except (TypeError, cv2.error):
            # Older/different backends may not support the params overload.
            capture.release()
            capture = cv2.VideoCapture()

    capture.open(source)
    return capture


def _create_windows_webcam_capture(source: int) -> VideoCaptureLike:
    """Open a Windows webcam with deterministic backend fallback."""
    attempts = (
        (getattr(cv2, "CAP_DSHOW", None), "DSHOW"),
        (getattr(cv2, "CAP_MSMF", None), "MSMF"),
        (getattr(cv2, "CAP_ANY", 0), "CAP_ANY"),
    )
    last_capture: VideoCaptureLike | None = None
    for backend, backend_name in attempts:
        if backend is None:
            continue
        capture = cv2.VideoCapture()
        try:
            capture.open(source, backend)
            if capture.isOpened():
                LOGGER.info("Camera %s opened with %s", source, backend_name)
                return capture
        except (TypeError, cv2.error):
            pass
        capture.release()
        last_capture = capture

    return last_capture or cv2.VideoCapture()


def _is_network_source(source: CaptureSource) -> bool:
    if not isinstance(source, str):
        return False
    try:
        return urlsplit(source).scheme.lower() in NETWORK_SCHEMES
    except ValueError:
        return False


class CameraWorker(QObject):
    """Own one VideoCapture instance and run it outside the UI thread."""

    frame_ready = Signal(int, object)
    status_changed = Signal(int, object, str)
    finished = Signal(int)

    def __init__(
        self,
        camera_id: int,
        source: CaptureSource,
        capture_factory: CaptureFactory = create_video_capture,
        retry_delay_seconds: float = 2.5,
        preview_fps: float = 12.0,
    ) -> None:
        super().__init__()
        self.camera_id = camera_id
        self.source = source
        self.capture_factory = capture_factory
        self.retry_delay_seconds = max(0.1, retry_delay_seconds)
        self.preview_interval = 1.0 / max(1.0, preview_fps)
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        """Thread-safe stop request; also interrupts a reconnect wait immediately."""
        self._stop_event.set()

    @Slot()
    def run(self) -> None:
        attempt = 0
        try:
            while not self._stop_event.is_set():
                status = CameraStatus.CONNECTING if attempt == 0 else CameraStatus.RECONNECTING
                message = "Kamera kaynağına bağlanılıyor."
                self.status_changed.emit(self.camera_id, status, message)

                capture: VideoCaptureLike | None = None
                try:
                    capture = self.capture_factory(self.source)
                    if not capture.isOpened():
                        raise RuntimeError("Kamera kaynağı açılamadı.")

                    self.status_changed.emit(
                        self.camera_id,
                        CameraStatus.CONNECTED,
                        "Kamera bağlantısı kuruldu.",
                    )
                    if attempt == 0:
                        LOGGER.info("Camera connected camera_id=%s", self.camera_id)
                    else:
                        LOGGER.info("Camera reconnected camera_id=%s", self.camera_id)
                    if self._read_frames(capture):
                        break
                    error_message = "Kamera görüntüsü kesildi."
                    LOGGER.warning("Camera stream lost camera_id=%s", self.camera_id)
                except Exception:
                    if self._stop_event.is_set():
                        break
                    error_message = "Kamera kaynağı açılamadı."
                finally:
                    if capture is not None:
                        self._release_capture(capture)

                self.status_changed.emit(self.camera_id, CameraStatus.ERROR, error_message)
                LOGGER.info("Camera reconnecting camera_id=%s", self.camera_id)
                self.status_changed.emit(
                    self.camera_id,
                    CameraStatus.RECONNECTING,
                    "Kısa bir beklemeden sonra yeniden bağlanılacak.",
                )
                attempt += 1
                if self._stop_event.wait(self.retry_delay_seconds):
                    break
        finally:
            self.status_changed.emit(
                self.camera_id,
                CameraStatus.STOPPED,
                "Kamera durduruldu.",
            )
            self.finished.emit(self.camera_id)

    def _read_frames(self, capture: VideoCaptureLike) -> bool:
        last_preview_at = 0.0
        diagnostic_started_at = time.monotonic()
        diagnostic_source_read_count = 0
        diagnostic_analysis_emit_count = 0
        file_frame_interval = self._file_frame_interval(capture)
        next_file_frame_at = time.monotonic()
        is_network_source = _is_network_source(self.source)
        consecutive_read_failures = 0

        while not self._stop_event.is_set():
            if file_frame_interval is not None:
                wait_seconds = next_file_frame_at - time.monotonic()
                if wait_seconds > 0 and self._stop_event.wait(wait_seconds):
                    return True

            frame_ok, frame = capture.read()
            if not frame_ok or frame is None:
                if is_network_source:
                    consecutive_read_failures += 1
                    LOGGER.warning(
                        "Transient camera read failure camera_id=%s count=%s",
                        self.camera_id,
                        consecutive_read_failures,
                    )
                    if consecutive_read_failures < MAX_CONSECUTIVE_READ_FAILURES:
                        continue
                return self._stop_event.is_set()

            consecutive_read_failures = 0
            diagnostic_source_read_count += 1

            now = time.monotonic()
            if now - last_preview_at >= self.preview_interval:
                copied_frame = frame.copy() if hasattr(frame, "copy") else frame
                setflags = getattr(copied_frame, "setflags", None)
                if callable(setflags):
                    setflags(write=False)
                self.frame_ready.emit(self.camera_id, copied_frame)
                last_preview_at = now
                diagnostic_analysis_emit_count += 1
            diagnostic_elapsed = now - diagnostic_started_at
            if (
                LOGGER.isEnabledFor(logging.DEBUG)
                and diagnostic_elapsed >= 5.0
            ):
                LOGGER.debug(
                    "Camera worker throughput camera_id=%s source_read_fps=%.2f "
                    "analysis_emit_fps=%.2f",
                    self.camera_id,
                    diagnostic_source_read_count
                    / max(diagnostic_elapsed, 0.001),
                    diagnostic_analysis_emit_count
                    / max(diagnostic_elapsed, 0.001),
                )
                diagnostic_started_at = now
                diagnostic_source_read_count = 0
                diagnostic_analysis_emit_count = 0

            if file_frame_interval is not None:
                next_file_frame_at += file_frame_interval
                if next_file_frame_at < now - file_frame_interval:
                    next_file_frame_at = now

        return True

    def _file_frame_interval(self, capture: VideoCaptureLike) -> float | None:
        if not isinstance(self.source, str) or "://" in self.source:
            return None
        try:
            source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        except (AttributeError, TypeError, ValueError):
            source_fps = 0.0
        playback_fps = source_fps if 1.0 <= source_fps <= 60.0 else 25.0
        return 1.0 / playback_fps

    @staticmethod
    def _release_capture(capture: VideoCaptureLike) -> None:
        try:
            capture.release()
        except Exception:
            # A backend cleanup error must not keep the QThread alive.
            pass
