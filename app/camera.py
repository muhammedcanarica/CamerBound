from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from threading import RLock

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot

from app.auth import AuthService, SessionUser, ValidationError
from app.camera_worker import (
    CameraStatus,
    CameraWorker,
    CaptureFactory,
    create_video_capture,
)
from app.config import application_root
from app.database import Database


class Direction(StrEnum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"


@dataclass(frozen=True, slots=True)
class Camera:
    id: int
    name: str
    stream_url: str
    direction: Direction
    enabled: bool


@dataclass(slots=True)
class _CameraRuntime:
    worker: CameraWorker
    thread: QThread


class CameraService(QObject):
    """Own camera configuration and one worker thread per active camera."""

    frame_ready = Signal(int, object)
    status_changed = Signal(int, object, str)
    STOP_TIMEOUT_MS = 5_000

    def __init__(
        self,
        database: Database,
        capture_factory: CaptureFactory = create_video_capture,
        retry_delay_seconds: float = 2.5,
        preview_fps: float = 12.0,
    ) -> None:
        super().__init__()
        self.database = database
        self.capture_factory = capture_factory
        self.retry_delay_seconds = retry_delay_seconds
        self.preview_fps = preview_fps
        self._statuses: dict[int, CameraStatus] = {}
        self._runtimes: dict[int, _CameraRuntime] = {}
        self._lock = RLock()

    def list_cameras(self) -> list[Camera]:
        with self.database.connection() as connection:
            rows = connection.execute(
                "SELECT id, name, stream_url, direction, enabled FROM cameras ORDER BY id"
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def get_camera(self, camera_id: int) -> Camera:
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT id, name, stream_url, direction, enabled FROM cameras WHERE id = ?",
                (camera_id,),
            ).fetchone()
        if row is None:
            raise ValidationError("Kamera bulunamadı.")
        return self._from_row(row)

    def update_camera(
        self,
        actor: SessionUser,
        camera_id: int,
        name: str,
        stream_url: str,
        direction: Direction,
        enabled: bool,
    ) -> Camera:
        AuthService.require_admin(actor)
        normalized_name = name.strip()
        if not normalized_name:
            raise ValidationError("Kamera adı boş olamaz.")

        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE cameras
                SET name = ?, stream_url = ?, direction = ?, enabled = ?
                WHERE id = ?
                """,
                (normalized_name, stream_url.strip(), direction.value, int(enabled), camera_id),
            )
            if cursor.rowcount == 0:
                raise ValidationError("Kamera bulunamadı.")
        return self.get_camera(camera_id)

    def start_camera(self, camera_id: int) -> CameraStatus:
        camera = self.get_camera(camera_id)
        if not camera.enabled:
            raise ValidationError("Kamera aktif değil.")
        if not camera.stream_url:
            raise ValidationError("Kamera için RTSP URL tanımlanmamış.")

        with self._lock:
            if camera_id in self._runtimes:
                return self._statuses.get(camera_id, CameraStatus.CONNECTING)

            worker = CameraWorker(
                camera_id=camera_id,
                source=self._capture_source(camera.stream_url),
                capture_factory=self.capture_factory,
                retry_delay_seconds=self.retry_delay_seconds,
                preview_fps=self.preview_fps,
            )
            thread = QThread()
            worker.moveToThread(thread)
            runtime = _CameraRuntime(worker=worker, thread=thread)
            self._runtimes[camera_id] = runtime

            thread.started.connect(worker.run)
            worker.frame_ready.connect(self._on_frame_ready)
            worker.status_changed.connect(self._on_worker_status_changed)
            worker.finished.connect(
                thread.quit,
                Qt.ConnectionType.DirectConnection,
            )
            worker.finished.connect(worker.deleteLater)
            thread.finished.connect(self._on_thread_finished)
            thread.finished.connect(thread.deleteLater)

        self._set_status(camera_id, CameraStatus.CONNECTING, "Kamera başlatılıyor.")
        try:
            thread.start()
        except Exception:
            with self._lock:
                self._runtimes.pop(camera_id, None)
            self._set_status(camera_id, CameraStatus.ERROR, "Kamera thread'i başlatılamadı.")
            raise
        return CameraStatus.CONNECTING

    def stop_camera(self, camera_id: int) -> CameraStatus:
        self.get_camera(camera_id)
        with self._lock:
            runtime = self._runtimes.get(camera_id)
        if runtime is None:
            self._set_status(camera_id, CameraStatus.STOPPED, "Kamera durduruldu.")
            return CameraStatus.STOPPED

        runtime.worker.request_stop()
        if not runtime.thread.wait(self.STOP_TIMEOUT_MS):
            self._set_status(
                camera_id,
                CameraStatus.ERROR,
                "Kamera belirtilen sürede durdurulamadı.",
            )
            return CameraStatus.ERROR

        with self._lock:
            current = self._runtimes.get(camera_id)
            if current is runtime:
                self._runtimes.pop(camera_id, None)
        self._set_status(camera_id, CameraStatus.STOPPED, "Kamera durduruldu.")
        return CameraStatus.STOPPED

    def stop_all(self) -> None:
        with self._lock:
            runtimes = list(self._runtimes.items())

        for _camera_id, runtime in runtimes:
            runtime.worker.request_stop()

        for camera_id, runtime in runtimes:
            if runtime.thread.wait(self.STOP_TIMEOUT_MS):
                with self._lock:
                    current = self._runtimes.get(camera_id)
                    if current is runtime:
                        self._runtimes.pop(camera_id, None)
                self._set_status(camera_id, CameraStatus.STOPPED, "Kamera durduruldu.")
            else:
                self._set_status(
                    camera_id,
                    CameraStatus.ERROR,
                    "Kamera belirtilen sürede durdurulamadı.",
                )

    def get_status(self, camera_id: int) -> CameraStatus:
        self.get_camera(camera_id)
        with self._lock:
            return self._statuses.get(camera_id, CameraStatus.STOPPED)

    @Slot(int, object)
    def _on_frame_ready(self, camera_id: int, frame: object) -> None:
        with self._lock:
            runtime = self._runtimes.get(camera_id)
            if runtime is None or runtime.worker is not self.sender():
                return
        self.frame_ready.emit(camera_id, frame)

    @Slot(int, object, str)
    def _on_worker_status_changed(
        self,
        camera_id: int,
        status: CameraStatus,
        message: str,
    ) -> None:
        with self._lock:
            runtime = self._runtimes.get(camera_id)
            if runtime is None or runtime.worker is not self.sender():
                return
        self._set_status(camera_id, CameraStatus(status), message)

    @Slot()
    def _on_thread_finished(self) -> None:
        finished_thread = self.sender()
        if not isinstance(finished_thread, QThread):
            return

        finished_camera_id: int | None = None
        with self._lock:
            for camera_id, runtime in self._runtimes.items():
                if runtime.thread is finished_thread:
                    finished_camera_id = camera_id
                    break
            if finished_camera_id is not None:
                self._runtimes.pop(finished_camera_id, None)

        if finished_camera_id is not None:
            self._set_status(
                finished_camera_id,
                CameraStatus.STOPPED,
                "Kamera durduruldu.",
            )

    def _set_status(self, camera_id: int, status: CameraStatus, message: str) -> None:
        with self._lock:
            self._statuses[camera_id] = status
        self.status_changed.emit(camera_id, status, message)

    @staticmethod
    def _capture_source(stream_url: str) -> str | int:
        source = stream_url.strip()
        if source.isdecimal():
            return int(source)
        if "://" in source:
            return source

        local_path = Path(source).expanduser()
        if not local_path.is_absolute():
            local_path = application_root() / local_path
        return str(local_path.resolve())

    @staticmethod
    def _from_row(row: object) -> Camera:
        return Camera(
            id=row["id"],
            name=row["name"],
            stream_url=row["stream_url"],
            direction=Direction(row["direction"]),
            enabled=bool(row["enabled"]),
        )
