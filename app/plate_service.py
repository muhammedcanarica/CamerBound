from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from app.auth import AuthService, SessionUser, ValidationError
from app.camera import Direction
from app.database import Database


PLATE_PATTERN = re.compile(r"^[A-Z0-9]{2,12}$")


@dataclass(frozen=True, slots=True)
class PlateRecord:
    id: int
    plate: str
    direction: Direction
    camera_id: int
    camera_name: str
    confidence: float
    timestamp: str


@dataclass(frozen=True, slots=True)
class VehicleInside:
    plate: str
    entry_time: str
    camera_name: str


class PlateService:
    def __init__(self, database: Database, duplicate_cooldown_seconds: int) -> None:
        self.database = database
        self.duplicate_cooldown_seconds = duplicate_cooldown_seconds

    def save_plate_detection(
        self,
        plate: str,
        camera_id: int,
        confidence: float,
        detected_at: datetime | None = None,
    ) -> PlateRecord:
        normalized_plate = "".join(plate.upper().split())
        if not PLATE_PATTERN.fullmatch(normalized_plate):
            raise ValidationError("Plaka 2-12 karakterlik harf ve rakamlardan oluşmalıdır.")
        if not 0 <= confidence <= 1:
            raise ValidationError("Güven değeri 0 ile 1 arasında olmalıdır.")

        timestamp = (detected_at or datetime.now().astimezone()).isoformat(timespec="seconds")
        with self.database.connection() as connection:
            camera = connection.execute(
                "SELECT id, name, direction FROM cameras WHERE id = ?", (camera_id,)
            ).fetchone()
            if camera is None:
                raise ValidationError("Kamera bulunamadı.")

            # TODO: duplicate_cooldown_seconds kullanılarak aynı kamera/plaka için
            # kısa süreli tekrar kayıtları burada engellenecek.
            cursor = connection.execute(
                """
                INSERT INTO plate_records (plate, direction, camera_id, confidence, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (normalized_plate, camera["direction"], camera_id, confidence, timestamp),
            )
            record_id = int(cursor.lastrowid)

        return PlateRecord(
            id=record_id,
            plate=normalized_plate,
            direction=Direction(camera["direction"]),
            camera_id=camera_id,
            camera_name=camera["name"],
            confidence=confidence,
            timestamp=timestamp,
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
        direction: Direction | None = None,
        limit: int = 500,
    ) -> list[PlateRecord]:
        AuthService.require_authenticated(actor)
        conditions: list[str] = []
        parameters: list[object] = []
        normalized_query = "".join(plate_query.upper().split())
        if normalized_query:
            conditions.append("pr.plate LIKE ?")
            parameters.append(f"%{normalized_query}%")
        if direction is not None:
            conditions.append("pr.direction = ?")
            parameters.append(direction.value)

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
                plate=row["plate"], entry_time=row["timestamp"], camera_name=row["camera_name"]
            )
            for row in rows
        ]

    @staticmethod
    def _record_select() -> str:
        return """
            SELECT pr.id, pr.plate, pr.direction, pr.camera_id,
                   c.name AS camera_name, pr.confidence, pr.timestamp
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
            timestamp=row["timestamp"],
        )
