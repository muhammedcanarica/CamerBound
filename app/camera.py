from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from threading import RLock
import logging
import time

import numpy as np

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot

from app.audit import AuditAction, AuditService
from app.auth import AuthService, SessionUser, ValidationError
from app.camera_credentials import (
    PasswordProtector,
    build_authenticated_camera_source,
    is_authenticatable_camera_source,
    split_camera_source_credentials,
)
from app.camera_worker import (
    CameraStatus,
    CameraWorker,
    CaptureFactory,
    create_video_capture,
)
from app.config import application_root
from app.database import Database
from app.security import sanitize_camera_source_for_log, sanitize_text_for_log


LOGGER = logging.getLogger(__name__)


class Direction(StrEnum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"


@dataclass(frozen=True, slots=True)
class Camera:
    id: int
    name: str
    stream_url: str
    username: str
    protected_password: str | None
    direction: Direction
    enabled: bool

    @property
    def has_password(self) -> bool:
        return bool(self.protected_password)


@dataclass(slots=True)
class _CameraRuntime:
    worker: CameraWorker
    thread: QThread


class CameraService(QObject):
    """Own camera configuration and one worker thread per active camera."""

    frame_ready = Signal(int, object)
    analysis_frame_ready = Signal(int, object)
    status_changed = Signal(int, object, str)
    _frame_available = Signal(int)
    STOP_TIMEOUT_MS = 5_000

    def __init__(
        self,
        database: Database,
        capture_factory: CaptureFactory = create_video_capture,
        retry_delay_seconds: float = 2.5,
        preview_fps: float = 12.0,
        audit_service: AuditService | None = None,
        password_protector: PasswordProtector | None = None,
    ) -> None:
        super().__init__()
        self.database = database
        self.capture_factory = capture_factory
        self.retry_delay_seconds = retry_delay_seconds
        self.preview_fps = preview_fps
        self.audit_service = audit_service or AuditService(database)
        self.password_protector = password_protector or database.password_protector
        self._statuses: dict[int, CameraStatus] = {}
        self._runtimes: dict[int, _CameraRuntime] = {}
        self._latest_frames: dict[int, object] = {}
        self._last_delivered_frames: dict[int, object] = {}
        self._frame_notifications_pending: set[int] = set()
        self._preview_delivery_counts: dict[int, int] = {}
        self._preview_delivery_started_at: dict[int, float] = {}
        self._lock = RLock()
        self._frame_available.connect(
            self._flush_latest_frame,
            Qt.ConnectionType.QueuedConnection,
        )

    def list_cameras(self) -> list[Camera]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT id, name, stream_url, username, protected_password,
                       direction, enabled
                FROM cameras
                ORDER BY id
                """
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def get_camera(self, camera_id: int) -> Camera:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT id, name, stream_url, username, protected_password,
                       direction, enabled
                FROM cameras
                WHERE id = ?
                """,
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
        direction: Direction | str,
        enabled: bool,
        *,
        username: str | None = None,
        password: str | None = None,
        clear_credentials: bool = False,
    ) -> Camera:
        try:
            normalized_direction = (
                direction if isinstance(direction, Direction) else Direction(direction)
            )
        except (TypeError, ValueError) as exc:
            raise ValidationError("Kamera yönü ENTRY veya EXIT olmalıdır.") from exc

        AuthService.require_admin(actor)
        normalized_name = name.strip()
        if not normalized_name:
            raise ValidationError("Kamera adı boş olamaz.")

        current = self.get_camera(camera_id)
        try:
            parsed_source = split_camera_source_credentials(stream_url)
        except (UnicodeError, ValueError):
            raise ValidationError(
                "Kamera kaynağındaki kimlik bilgileri geçersiz."
            ) from None

        normalized_username = (
            current.username if username is None else username.strip()
        )
        protected_password = current.protected_password
        credential_change = "unchanged"
        if clear_credentials:
            normalized_username = ""
            protected_password = None
            credential_change = "cleared"
        else:
            if username is None and parsed_source.username is not None:
                normalized_username = parsed_source.username
            new_password = password if password else parsed_source.password
            if new_password:
                try:
                    protected_password = self.password_protector.protect(new_password)
                except Exception:
                    raise ValidationError(
                        "Kamera şifresi Windows DPAPI ile korunamadı."
                    ) from None
                credential_change = "updated"
            if protected_password and not normalized_username:
                raise ValidationError(
                    "Kayıtlı şifre için kullanıcı adı gereklidir; "
                    "kimlik bilgilerini kaldırmak için temizleme seçeneğini kullanın."
                )
            if protected_password and not is_authenticatable_camera_source(
                parsed_source.stream_url
            ):
                raise ValidationError(
                    "Kamera kimlik bilgileri yalnızca HTTP, HTTPS veya RTSP "
                    "URL kaynaklarında kullanılabilir."
                )

        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE cameras
                SET name = ?, stream_url = ?, username = ?, protected_password = ?,
                    direction = ?, enabled = ?
                WHERE id = ?
                """,
                (
                    normalized_name,
                    parsed_source.stream_url,
                    normalized_username,
                    protected_password,
                    normalized_direction.value,
                    int(enabled),
                    camera_id,
                ),
            )
            if cursor.rowcount == 0:
                raise ValidationError("Kamera bulunamadı.")
        updated = self.get_camera(camera_id)
        self.audit_service.try_log(
            AuditAction.CAMERA_SETTINGS_CHANGED,
            actor=actor,
            details=(
                f"camera_id={updated.id}; name={sanitize_text_for_log(updated.name)}; "
                f"direction={updated.direction.value}; enabled={updated.enabled}; "
                f"source={sanitize_camera_source_for_log(updated.stream_url)}; "
                f"credentials={credential_change}"
            ),
        )
        return updated

    def start_camera(self, camera_id: int) -> CameraStatus:
        camera = self.get_camera(camera_id)
        if not camera.enabled:
            raise ValidationError("Kamera aktif değil.")
        if not camera.stream_url:
            raise ValidationError("Kamera için RTSP URL tanımlanmamış.")

        with self._lock:
            if camera_id in self._runtimes:
                return self._statuses.get(camera_id, CameraStatus.CONNECTING)

            worker_source = self._runtime_capture_source(camera)
            worker = CameraWorker(
                camera_id=camera_id,
                source=worker_source,
                capture_factory=self.capture_factory,
                retry_delay_seconds=self.retry_delay_seconds,
                preview_fps=self.preview_fps,
            )
            thread = QThread()
            worker.moveToThread(thread)
            runtime = _CameraRuntime(worker=worker, thread=thread)
            self._runtimes[camera_id] = runtime

            thread.started.connect(worker.run)
            worker.frame_ready.connect(
                self._on_frame_ready,
                Qt.ConnectionType.DirectConnection,
            )
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
            self._discard_frame(camera_id)
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
                    self._discard_frame(camera_id)
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

    def is_camera_running(self, camera_id: int) -> bool:
        self.get_camera(camera_id)
        with self._lock:
            return camera_id in self._runtimes

    def get_latest_frame(self, camera_id: int) -> object | None:
        """Return a safe snapshot of the latest delivered camera frame."""
        self.get_camera(camera_id)
        with self._lock:
            frame = self._last_delivered_frames.get(camera_id)
            copier = getattr(frame, "copy", None)
            return copier() if callable(copier) else frame

    @Slot(int, object)
    def _on_frame_ready(self, camera_id: int, frame: object) -> None:
        if isinstance(frame, np.ndarray) and frame.flags.writeable:
            # Production CameraWorker emits an immutable owned copy. Preserve the
            # same ownership contract for direct/test producers without mutating
            # memory that still belongs to their capture source.
            frame = frame.copy()
            frame.setflags(write=False)
        with self._lock:
            if camera_id not in self._runtimes:
                return
        # Direct subscribers may ingest this worker-throttled frame before the UI
        # event loop coalesces preview delivery. Subscribers must remain lightweight.
        self.analysis_frame_ready.emit(camera_id, frame)
        with self._lock:
            if camera_id not in self._runtimes:
                return
            self._latest_frames[camera_id] = frame
            if camera_id in self._frame_notifications_pending:
                return
            self._frame_notifications_pending.add(camera_id)
        self._frame_available.emit(camera_id)

    @Slot(int)
    def _flush_latest_frame(self, camera_id: int) -> None:
        with self._lock:
            self._frame_notifications_pending.discard(camera_id)
            if camera_id not in self._runtimes:
                self._latest_frames.pop(camera_id, None)
                return
            frame = self._latest_frames.pop(camera_id, None)
        if frame is not None:
            with self._lock:
                copier = getattr(frame, "copy", None)
                self._last_delivered_frames[camera_id] = (
                    copier() if callable(copier) else frame
                )
            self.frame_ready.emit(camera_id, frame)
            self._log_preview_delivery_fps(camera_id)

    def _log_preview_delivery_fps(self, camera_id: int) -> None:
        if not LOGGER.isEnabledFor(logging.DEBUG):
            return
        now = time.monotonic()
        with self._lock:
            started_at = self._preview_delivery_started_at.setdefault(camera_id, now)
            count = self._preview_delivery_counts.get(camera_id, 0) + 1
            self._preview_delivery_counts[camera_id] = count
            elapsed = now - started_at
            if elapsed < 5.0:
                return
            self._preview_delivery_started_at[camera_id] = now
            self._preview_delivery_counts[camera_id] = 0
        LOGGER.debug(
            "Camera UI delivery camera_id=%s ui_preview_delivery_fps=%.2f",
            camera_id,
            count / max(elapsed, 0.001),
        )

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
                self._discard_frame(finished_camera_id)

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
        log = LOGGER.warning if status is CameraStatus.ERROR else LOGGER.info
        log("Camera status camera_id=%s status=%s message=%s", camera_id, status, message)

    def _discard_frame(self, camera_id: int) -> None:
        self._latest_frames.pop(camera_id, None)
        self._frame_notifications_pending.discard(camera_id)
        self._preview_delivery_counts.pop(camera_id, None)
        self._preview_delivery_started_at.pop(camera_id, None)

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

    def _runtime_capture_source(self, camera: Camera) -> str | int:
        source = self._capture_source(camera.stream_url)
        if not camera.protected_password:
            return source
        if not isinstance(source, str):
            raise ValidationError(
                "Kamera kimlik bilgileri yalnızca ağ URL kaynaklarında kullanılabilir."
            )
        try:
            password = self.password_protector.unprotect(camera.protected_password)
            authenticated_source = build_authenticated_camera_source(
                source,
                camera.username,
                password,
            )
            del password
            return authenticated_source
        except Exception:
            raise ValidationError(
                "Kamera kimlik bilgileri Windows DPAPI ile açılamadı."
            ) from None

    @staticmethod
    def _from_row(row: object) -> Camera:
        return Camera(
            id=row["id"],
            name=row["name"],
            stream_url=row["stream_url"],
            username=row["username"],
            protected_password=row["protected_password"],
            direction=Direction(row["direction"]),
            enabled=bool(row["enabled"]),
        )
