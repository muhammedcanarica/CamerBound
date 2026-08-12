from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

from app.database import Database
from app.time_utils import normalize_utc_timestamp, to_utc_storage, utc_now


LOGGER = logging.getLogger(__name__)


class AuditAction(StrEnum):
    LOGIN_SUCCESS = "LOGIN_SUCCESS"
    LOGIN_FAILURE = "LOGIN_FAILURE"
    CAMERA_SETTINGS_CHANGED = "CAMERA_SETTINGS_CHANGED"
    RETENTION_CHANGED = "RETENTION_CHANGED"
    OLD_RECORDS_DELETED = "OLD_RECORDS_DELETED"
    ALL_RECORDS_DELETED = "ALL_RECORDS_DELETED"
    USER_CREATED = "USER_CREATED"
    SYSTEM_TIME_CHANGED = "SYSTEM_TIME_CHANGED"


@dataclass(frozen=True, slots=True)
class AuditLog:
    id: int
    user_id: int | None
    username: str
    action: str
    details: str | None
    timestamp: str


class AuditService:
    def __init__(self, database: Database) -> None:
        self.database = database

    def log(
        self,
        action: AuditAction | str,
        *,
        actor: object | None = None,
        username: str | None = None,
        details: str | None = None,
    ) -> AuditLog:
        action_value = action.value if isinstance(action, AuditAction) else str(action)
        actor_id = getattr(actor, "id", None)
        actor_username = getattr(actor, "username", None)
        normalized_username = str(actor_username or username or "SYSTEM").strip()
        if not normalized_username:
            normalized_username = "<empty>"
        timestamp = to_utc_storage(utc_now())
        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO audit_logs (user_id, username, action, details, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (actor_id, normalized_username, action_value, details, timestamp),
            )
            log_id = int(cursor.lastrowid)
        return AuditLog(
            id=log_id,
            user_id=actor_id,
            username=normalized_username,
            action=action_value,
            details=details,
            timestamp=timestamp,
        )

    def try_log(self, *args: object, **kwargs: object) -> AuditLog | None:
        try:
            return self.log(*args, **kwargs)
        except Exception:
            LOGGER.exception("Security audit entry could not be stored")
            return None

    def get_recent_logs(self, actor: object, limit: int = 200) -> list[AuditLog]:
        role_value = getattr(getattr(actor, "role", None), "value", None)
        if role_value != "ADMIN":
            raise PermissionError("Güvenlik günlüğü yalnızca ADMIN tarafından görülebilir.")
        safe_limit = max(1, min(limit, 500))
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT id, user_id, username, action, details, timestamp
                FROM audit_logs
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        return [
            AuditLog(
                id=row["id"],
                user_id=row["user_id"],
                username=row["username"],
                action=row["action"],
                details=row["details"],
                timestamp=normalize_utc_timestamp(row["timestamp"]),
            )
            for row in rows
        ]
