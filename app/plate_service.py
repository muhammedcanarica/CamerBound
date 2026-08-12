from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from app.audit import AuditAction, AuditService
from app.auth import AuthService, SessionUser, ValidationError
from app.camera import Direction
from app.config import (
    DEFAULT_RECORD_RETENTION_DAYS,
    SUPPORTED_RECORD_RETENTION_DAYS,
)
from app.database import Database
from app.plate_capture import PlateCaptureService
from app.time_utils import (
    as_utc,
    normalize_utc_timestamp,
    parse_utc_timestamp,
    to_utc_storage,
    utc_now,
)


PLATE_PATTERN = re.compile(r"^[A-Z0-9]{2,12}$")
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PlateRecord:
    id: int
    plate: str
    direction: Direction
    camera_id: int
    camera_name: str
    confidence: float
    timestamp: str
    image_path: str | None = None


@dataclass(frozen=True, slots=True)
class VehicleInside:
    plate: str
    entry_time: str
    camera_name: str


@dataclass(frozen=True, slots=True)
class PlateRecordStats:
    total_records: int
    oldest_timestamp: str | None


class DuplicatePlateDetection(Exception):
    """Raised when the same camera reports a plate inside the cooldown window."""

    def __init__(self, plate: str, camera_id: int) -> None:
        super().__init__("Aynı plaka cooldown süresi içinde zaten kaydedildi.")
        self.plate = plate
        self.camera_id = camera_id


class PlateService:
    def __init__(
        self,
        database: Database,
        duplicate_cooldown_seconds: int,
        record_retention_days: int = DEFAULT_RECORD_RETENTION_DAYS,
        capture_service: PlateCaptureService | None = None,
        audit_service: AuditService | None = None,
    ) -> None:
        self.database = database
        self.duplicate_cooldown_seconds = duplicate_cooldown_seconds
        self.capture_service = capture_service
        self.audit_service = audit_service or AuditService(database)
        self.set_record_retention_days(record_retention_days)

    def set_record_retention_days(
        self,
        retention_days: int,
        actor: SessionUser | None = None,
    ) -> None:
        if (
            isinstance(retention_days, bool)
            or retention_days not in SUPPORTED_RECORD_RETENTION_DAYS
        ):
            raise ValidationError("Saklama süresi 30, 90, 180 veya 0 olmalıdır.")
        previous = getattr(self, "record_retention_days", None)
        self.record_retention_days = retention_days
        if actor is not None and previous is not None:
            self.audit_service.try_log(
                AuditAction.RETENTION_CHANGED,
                actor=actor,
                details=f"old={previous}; new={retention_days}",
            )

    def save_plate_detection(
        self,
        plate: str,
        camera_id: int,
        confidence: float,
        detected_at: datetime | None = None,
        frame: object | None = None,
    ) -> PlateRecord:
        normalized_plate = "".join(plate.upper().split())
        if not PLATE_PATTERN.fullmatch(normalized_plate):
            raise ValidationError("Plaka 2-12 karakterlik harf ve rakamlardan oluşmalıdır.")
        if not 0 <= confidence <= 1:
            raise ValidationError("Güven değeri 0 ile 1 arasında olmalıdır.")

        detection_time = as_utc(detected_at) if detected_at is not None else utc_now()
        timestamp = to_utc_storage(detection_time)
        with self.database.connection() as connection:
            # Acquire the write lock before checking the latest row so another
            # recognition worker cannot insert between the SELECT and INSERT.
            connection.execute("BEGIN IMMEDIATE")
            camera = connection.execute(
                "SELECT id, name, direction FROM cameras WHERE id = ?", (camera_id,)
            ).fetchone()
            if camera is None:
                raise ValidationError("Kamera bulunamadı.")

            latest = connection.execute(
                """
                SELECT timestamp
                FROM plate_records
                WHERE camera_id = ? AND plate = ?
                ORDER BY timestamp DESC, id DESC
                LIMIT 1
                """,
                (camera_id, normalized_plate),
            ).fetchone()
            if latest is not None and self._inside_cooldown(
                latest["timestamp"], detection_time
            ):
                raise DuplicatePlateDetection(normalized_plate, camera_id)

            cursor = connection.execute(
                """
                INSERT INTO plate_records (plate, direction, camera_id, confidence, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (normalized_plate, camera["direction"], camera_id, confidence, timestamp),
            )
            record_id = int(cursor.lastrowid)

        image_path = self._save_capture(
            record_id,
            normalized_plate,
            camera["direction"],
            detection_time,
            frame,
        )

        return PlateRecord(
            id=record_id,
            plate=normalized_plate,
            direction=Direction(camera["direction"]),
            camera_id=camera_id,
            camera_name=camera["name"],
            confidence=confidence,
            timestamp=timestamp,
            image_path=image_path,
        )

    def _save_capture(
        self,
        record_id: int,
        plate: str,
        direction: str,
        detection_time: datetime,
        frame: object | None,
    ) -> str | None:
        if self.capture_service is None or frame is None:
            return None

        image_path: str | None = None
        try:
            image_path = self.capture_service.save_capture(
                frame,
                plate,
                direction,
                detection_time,
                record_id,
            )
            if image_path is None:
                return None
            with self.database.connection() as connection:
                connection.execute(
                    "UPDATE plate_records SET image_path = ? WHERE id = ?",
                    (image_path, record_id),
                )
            return image_path
        except Exception:
            if image_path is not None:
                self.capture_service.delete_captures((image_path,))
            LOGGER.exception("Plate capture save failed for %s", plate)
            return None

    def _inside_cooldown(self, previous_value: str, current: datetime) -> bool:
        if self.duplicate_cooldown_seconds <= 0:
            return False
        try:
            previous = parse_utc_timestamp(previous_value)
            current = as_utc(current)
        except (TypeError, ValueError):
            return False
        elapsed = current - previous
        return timedelta(0) <= elapsed <= timedelta(
            seconds=self.duplicate_cooldown_seconds
        )

    def get_recent_records(
        self, actor: SessionUser, limit: int = 20
    ) -> list[PlateRecord]:
        AuthService.require_authenticated(actor)
        safe_limit = max(1, min(limit, 500))
        with self.database.connection() as connection:
            rows = connection.execute(
                self._record_select() + " ORDER BY pr.timestamp DESC, pr.id DESC LIMIT ?",
                (safe_limit,),
            ).fetchall()
        return [self._record_from_row(row) for row in rows]

    def search_records(
        self,
        actor: SessionUser,
        plate_query: str = "",
        direction: Direction | str | None = None,
        limit: int = 500,
    ) -> list[PlateRecord]:
        try:
            normalized_direction = (
                direction
                if direction is None or isinstance(direction, Direction)
                else Direction(direction)
            )
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "Geçersiz kamera yönü. ENTRY veya EXIT kullanın."
            ) from exc

        AuthService.require_authenticated(actor)
        conditions: list[str] = []
        parameters: list[object] = []
        normalized_query = "".join(plate_query.upper().split())
        if normalized_query:
            conditions.append("pr.plate LIKE ?")
            parameters.append(f"%{normalized_query}%")
        if normalized_direction is not None:
            conditions.append("pr.direction = ?")
            parameters.append(normalized_direction.value)

        query = self._record_select()
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY pr.timestamp DESC, pr.id DESC LIMIT ?"
        parameters.append(max(1, min(limit, 2000)))

        with self.database.connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._record_from_row(row) for row in rows]

    def get_vehicles_inside(self, actor: SessionUser) -> list[VehicleInside]:
        AuthService.require_authenticated(actor)
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                WITH ranked_records AS (
                    SELECT
                        pr.plate,
                        pr.direction,
                        pr.timestamp,
                        pr.id,
                        c.name AS camera_name,
                        ROW_NUMBER() OVER (
                            PARTITION BY pr.plate
                            ORDER BY pr.timestamp DESC, pr.id DESC
                        ) AS row_number
                    FROM plate_records pr
                    JOIN cameras c ON c.id = pr.camera_id
                )
                SELECT plate, timestamp, camera_name
                FROM ranked_records
                WHERE row_number = 1 AND direction = 'ENTRY'
                ORDER BY timestamp DESC, id DESC
                """
            ).fetchall()
        return [
            VehicleInside(
                plate=row["plate"],
                entry_time=normalize_utc_timestamp(row["timestamp"]),
                camera_name=row["camera_name"],
            )
            for row in rows
        ]

    def delete_records_older_than(
        self,
        actor: SessionUser,
        retention_days: int | None = None,
        now: datetime | None = None,
    ) -> int:
        AuthService.require_admin(actor)
        days = self.record_retention_days if retention_days is None else retention_days
        self._validate_retention_days(days)
        deleted = self._delete_records_older_than(days, now)
        self.audit_service.try_log(
            AuditAction.OLD_RECORDS_DELETED,
            actor=actor,
            details=f"count={deleted}",
        )
        return deleted

    def apply_retention_policy(self, now: datetime | None = None) -> int:
        """Apply the configured startup policy without requiring an interactive actor."""
        deleted = self._delete_records_older_than(self.record_retention_days, now)
        if deleted:
            self.audit_service.try_log(
                AuditAction.OLD_RECORDS_DELETED,
                username="SYSTEM",
                details=f"count={deleted}; trigger=startup",
            )
        return deleted

    def delete_all_records(self, actor: SessionUser) -> int:
        AuthService.require_admin(actor)
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            image_paths = [
                row["image_path"]
                for row in connection.execute(
                    "SELECT image_path FROM plate_records WHERE image_path IS NOT NULL"
                ).fetchall()
            ]
            cursor = connection.execute("DELETE FROM plate_records")
            deleted = max(0, cursor.rowcount)
        self._delete_capture_files(image_paths)
        self.audit_service.try_log(
            AuditAction.ALL_RECORDS_DELETED,
            actor=actor,
            details=f"count={deleted}",
        )
        return deleted

    def resolve_capture_path(self, image_path: str | None) -> Path | None:
        if self.capture_service is None:
            return None
        return self.capture_service.resolve_reference(image_path)

    def get_record_stats(self, actor: SessionUser) -> PlateRecordStats:
        AuthService.require_admin(actor)
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS total_records, MIN(timestamp) AS oldest_timestamp "
                "FROM plate_records"
            ).fetchone()
        oldest = row["oldest_timestamp"]
        return PlateRecordStats(
            total_records=int(row["total_records"]),
            oldest_timestamp=(
                normalize_utc_timestamp(oldest) if oldest is not None else None
            ),
        )

    def _delete_records_older_than(
        self,
        retention_days: int,
        now: datetime | None,
    ) -> int:
        self._validate_retention_days(retention_days)
        if retention_days == 0:
            return 0
        current = as_utc(now) if now is not None else utc_now()
        cutoff = to_utc_storage(current - timedelta(days=retention_days))
        with self.database.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            image_paths = [
                row["image_path"]
                for row in connection.execute(
                    "SELECT image_path FROM plate_records "
                    "WHERE timestamp < ? AND image_path IS NOT NULL",
                    (cutoff,),
                ).fetchall()
            ]
            cursor = connection.execute(
                "DELETE FROM plate_records WHERE timestamp < ?",
                (cutoff,),
            )
            deleted = max(0, cursor.rowcount)
        self._delete_capture_files(image_paths)
        return deleted

    def _delete_capture_files(self, image_paths: list[str]) -> None:
        if self.capture_service is not None:
            self.capture_service.delete_captures(image_paths)

    @staticmethod
    def _validate_retention_days(retention_days: int) -> None:
        if (
            isinstance(retention_days, bool)
            or retention_days not in SUPPORTED_RECORD_RETENTION_DAYS
        ):
            raise ValidationError("Saklama süresi 30, 90, 180 veya 0 olmalıdır.")

    @staticmethod
    def _record_select() -> str:
        return """
            SELECT pr.id, pr.plate, pr.direction, pr.camera_id,
                   c.name AS camera_name, pr.confidence, pr.timestamp, pr.image_path
            FROM plate_records pr
            JOIN cameras c ON c.id = pr.camera_id
        """

    @staticmethod
    def _record_from_row(row: object) -> PlateRecord:
        return PlateRecord(
            id=row["id"],
            plate=row["plate"],
            direction=Direction(row["direction"]),
            camera_id=row["camera_id"],
            camera_name=row["camera_name"],
            confidence=row["confidence"],
            timestamp=normalize_utc_timestamp(row["timestamp"]),
            image_path=row["image_path"],
        )
