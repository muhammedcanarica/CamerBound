from __future__ import annotations

import threading
import time
from enum import StrEnum
from typing import Callable, Protocol

import cv2
from PySide6.QtCore import QObject, Signal, Slot


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
    capture = cv2.VideoCapture()
    is_network_source = isinstance(source, str) and "://" in source
    timeout_parameters: list[int] = []
    if is_network_source:
        for property_name in (
            "CAP_PROP_OPEN_TIMEOUT_MSEC",
            "CAP_PROP_READ_TIMEOUT_MSEC",
        ):
            property_id = getattr(cv2, property_name, None)
            if property_id is not None:
                timeout_parameters.extend((property_id, 2_000))

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
                    if self._read_frames(capture):
                        break
                    error_message = "Kamera görüntüsü kesildi."
                except Exception:
                    if self._stop_event.is_set():
                        break
                    error_message = "Kamera kaynağı açılamadı."
                finally:
                    if capture is not None:
                        self._release_capture(capture)

                self.status_changed.emit(self.camera_id, CameraStatus.ERROR, error_message)
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
        file_frame_interval = self._file_frame_interval(capture)
        next_file_frame_at = time.monotonic()

        while not self._stop_event.is_set():
            if file_frame_interval is not None:
                wait_seconds = next_file_frame_at - time.monotonic()
                if wait_seconds > 0 and self._stop_event.wait(wait_seconds):
                    return True

            frame_ok, frame = capture.read()
            if not frame_ok or frame is None:
                return self._stop_event.is_set()

            now = time.monotonic()
            if now - last_preview_at >= self.preview_interval:
                copied_frame = frame.copy() if hasattr(frame, "copy") else frame
                self.frame_ready.emit(self.camera_id, copied_frame)
                last_preview_at = now

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
