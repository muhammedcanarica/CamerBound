from __future__ import annotations

import logging
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from app.audit import AuditAction, AuditService
from app.auth import AuthService, AuthenticationError, Role
from app.camera import CameraService
from app.database import Database
from app.logging_config import CredentialSafeFormatter
from app.plate_service import PlateService
from app.security import (
    sanitize_camera_source_for_log,
    sanitize_text_for_log,
)
from app.time_utils import parse_utc_timestamp, to_utc_storage


class AuditSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.temp_directory.name) / "audit.db")
        self.database.initialize()
        self.audit_service = AuditService(self.database)
        self.auth_service = AuthService(self.database, self.audit_service)
        self.auth_service.ensure_default_admin()
        self.admin = self.auth_service.authenticate("admin", "admin123")
        self.camera_service = CameraService(
            self.database,
            audit_service=self.audit_service,
        )
        self.plate_service = PlateService(
            self.database,
            duplicate_cooldown_seconds=10,
            audit_service=self.audit_service,
        )

    def tearDown(self) -> None:
        self.camera_service.stop_all()
        self.temp_directory.cleanup()

    def test_login_success_and_failure_are_audited_without_password(self) -> None:
        self.auth_service.authenticate(" admin ", "admin123")
        attempted_password = "NeverLogThisPassword"
        with self.assertRaises(AuthenticationError):
            self.auth_service.authenticate("unknown-user", attempted_password)

        logs = self.audit_service.get_recent_logs(self.admin)
        success = next(log for log in logs if log.action == AuditAction.LOGIN_SUCCESS)
        failure = next(log for log in logs if log.action == AuditAction.LOGIN_FAILURE)
        self.assertEqual(success.username, "admin")
        self.assertEqual(failure.username, "unknown-user")
        self.assertIsNone(failure.user_id)
        serialized = " ".join(
            f"{log.username} {log.action} {log.details or ''}" for log in logs
        )
        self.assertNotIn(attempted_password, serialized)
        self.assertNotIn("$2", serialized)

    def test_user_creation_and_utc_timestamp_are_audited(self) -> None:
        created = self.auth_service.create_user(
            self.admin,
            "security-operator",
            "safe-pass-123",
            Role.USER,
        )

        log = self._latest(AuditAction.USER_CREATED)
        self.assertEqual(log.user_id, self.admin.id)
        self.assertIn(f"created_username={created.username}", log.details)
        self.assertIn("role=USER", log.details)
        self.assertEqual(parse_utc_timestamp(log.timestamp).utcoffset(), timedelta(0))

    def test_camera_change_audit_sanitizes_credentials_and_query(self) -> None:
        camera = self.camera_service.list_cameras()[0]
        secret = "SuperSecret"
        token = "private-token"
        source = f"rtsp://admin:{secret}@192.168.1.50/live?token={token}"

        self.camera_service.update_camera(
            self.admin,
            camera.id,
            "Giriş Kamera",
            source,
            camera.direction,
            True,
        )

        log = self._latest(AuditAction.CAMERA_SETTINGS_CHANGED)
        self.assertIn("camera_id=", log.details)
        self.assertIn("rtsp://192.168.1.50/live", log.details)
        self.assertNotIn("admin", log.details)
        self.assertNotIn(secret, log.details)
        self.assertNotIn(token, log.details)
        self.assertNotIn("?", log.details)

    def test_retention_and_delete_operations_are_audited(self) -> None:
        self.plate_service.set_record_retention_days(30, actor=self.admin)
        camera = self.camera_service.list_cameras()[0]
        self.plate_service.save_plate_detection("34AUDIT1", camera.id, 0.9)
        deleted = self.plate_service.delete_all_records(self.admin)

        retention = self._latest(AuditAction.RETENTION_CHANGED)
        delete_all = self._latest(AuditAction.ALL_RECORDS_DELETED)
        self.assertEqual(retention.details, "old=90; new=30")
        self.assertEqual(delete_all.details, f"count={deleted}")

    def test_plate_cleanup_never_deletes_audit_history(self) -> None:
        camera = self.camera_service.list_cameras()[0]
        now = datetime(2026, 8, 12, 8, 0, tzinfo=timezone.utc)
        with self.database.connection() as connection:
            connection.execute(
                """
                INSERT INTO plate_records
                    (plate, direction, camera_id, confidence, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    "34OLDLOG",
                    camera.direction.value,
                    camera.id,
                    0.9,
                    to_utc_storage(now - timedelta(days=91)),
                ),
            )
        before = self._audit_count()

        self.plate_service.delete_records_older_than(self.admin, 90, now)
        after_retention = self._audit_count()
        cleanup_log = self._latest(AuditAction.OLD_RECORDS_DELETED)
        self.plate_service.delete_all_records(self.admin)

        self.assertGreater(after_retention, before)
        self.assertEqual(cleanup_log.details, "count=1")
        self.assertGreater(self._audit_count(), after_retention)

    def test_user_cannot_read_security_audit(self) -> None:
        user = self.auth_service.create_user(
            self.admin, "audit-reader", "safe-pass-123", Role.USER
        )
        with self.assertRaises(PermissionError):
            self.audit_service.get_recent_logs(user)
        self.assertTrue(self.audit_service.get_recent_logs(self.admin))

    def test_audit_insert_failure_does_not_break_main_operation(self) -> None:
        with patch.object(
            self.audit_service,
            "log",
            side_effect=sqlite3.OperationalError("audit disk error"),
        ), self.assertLogs("app.audit", level="ERROR"):
            created = self.auth_service.create_user(
                self.admin, "audit-fallback", "safe-pass-123", Role.USER
            )

        self.assertEqual(created.username, "audit-fallback")
        self.assertEqual(
            self.auth_service.authenticate("audit-fallback", "safe-pass-123").id,
            created.id,
        )

    def _latest(self, action: AuditAction):
        return next(
            log
            for log in self.audit_service.get_recent_logs(self.admin)
            if log.action == action.value
        )

    def _audit_count(self) -> int:
        with self.database.connection() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM audit_logs").fetchone()[0])


class CameraSourceSanitizationTests(unittest.TestCase):
    def test_url_credentials_and_query_secrets_are_removed(self) -> None:
        cases = (
            (
                "rtsp://admin:secret@192.168.1.50:554/live?token=abc",
                "rtsp://192.168.1.50:554/live",
            ),
            (
                "https://user:pass@example.com/snapshot?api_key=secret",
                "https://example.com/snapshot",
            ),
            (
                "http://example.com/camera?password=secret",
                "http://example.com/camera",
            ),
        )
        for source, expected in cases:
            with self.subTest(source=source):
                self.assertEqual(sanitize_camera_source_for_log(source), expected)

    def test_log_formatter_redacts_camera_urls_in_messages(self) -> None:
        secret_url = "rtsp://admin:secret@example.com/live?token=abc"
        self.assertNotIn("secret", sanitize_text_for_log(f"failed {secret_url}"))
        record = logging.LogRecord(
            "camera-test",
            logging.ERROR,
            __file__,
            1,
            "Camera failed: %s",
            (secret_url,),
            None,
        )
        rendered = CredentialSafeFormatter("%(message)s").format(record)
        self.assertEqual(rendered, "Camera failed: rtsp://example.com/live")


class AuditMigrationTests(unittest.TestCase):
    def test_existing_database_initialization_creates_audit_table(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            path = Path(temp_directory) / "legacy.db"
            connection = sqlite3.connect(path)
            connection.execute(
                "CREATE TABLE users ("
                "id INTEGER PRIMARY KEY, username TEXT, password_hash TEXT, role TEXT, "
                "created_at TEXT)"
            )
            connection.commit()
            connection.close()

            database = Database(path)
            database.initialize()

            with database.connection() as connection:
                table = connection.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name='audit_logs'"
                ).fetchone()
            self.assertIsNotNone(table)


if __name__ == "__main__":
    unittest.main()
