from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from app.auth import AuthService, SessionUser, ValidationError
from app.database import Database


class Direction(StrEnum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"


class CameraStatus(StrEnum):
    STOPPED = "STOPPED"
    STARTED = "STARTED"


@dataclass(frozen=True, slots=True)
class Camera:
    id: int
    name: str
    stream_url: str
    direction: Direction
    enabled: bool


class CameraService:
    """Owns camera configuration and the future capture lifecycle boundary."""

    def __init__(self, database: Database) -> None:
        self.database = database
        self._statuses: dict[int, CameraStatus] = {}

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

        # OpenCV VideoCapture lifecycle will be started at this boundary.
        self._statuses[camera_id] = CameraStatus.STARTED
        return CameraStatus.STARTED

    def stop_camera(self, camera_id: int) -> CameraStatus:
        self.get_camera(camera_id)
        self._statuses[camera_id] = CameraStatus.STOPPED
        return CameraStatus.STOPPED

    def get_status(self, camera_id: int) -> CameraStatus:
        self.get_camera(camera_id)
        return self._statuses.get(camera_id, CameraStatus.STOPPED)

    @staticmethod
    def _from_row(row: object) -> Camera:
        return Camera(
            id=row["id"],
            name=row["name"],
            stream_url=row["stream_url"],
            direction=Direction(row["direction"]),
            enabled=bool(row["enabled"]),
        )
