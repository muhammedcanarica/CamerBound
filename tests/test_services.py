from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from app.auth import (
    AuthService,
    AuthenticationError,
    AuthorizationError,
    Role,
    UserExistsError,
    ValidationError,
)
from app.camera import CameraService, Direction
from app.database import Database
from app.plate_service import PlateService
from app.plate_service import DuplicatePlateDetection
from app.time_utils import to_utc_storage
from main import apply_startup_retention_cleanup


class ServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_directory.name) / "test.db"
        self.database = Database(database_path)
        self.database.initialize()
        self.auth_service = AuthService(self.database)
        self.assertTrue(self.auth_service.ensure_default_admin())
        self.admin = self.auth_service.authenticate("admin", "admin123")
        self.camera_service = CameraService(self.database)
        self.plate_service = PlateService(self.database, duplicate_cooldown_seconds=10)

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def _insert_record(self, plate: str, timestamp: str) -> int:
        camera = self.camera_service.list_cameras()[0]
        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO plate_records
                    (plate, direction, camera_id, confidence, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (plate, camera.direction.value, camera.id, 0.90, timestamp),
            )
            return int(cursor.lastrowid)

    def test_default_admin_authentication(self) -> None:
        self.assertEqual(self.admin.role, Role.ADMIN)
        with self.assertRaises(AuthenticationError):
            self.auth_service.authenticate("admin", "wrong-password")

    def test_admin_can_create_user_and_duplicate_is_rejected(self) -> None:
        created = self.auth_service.create_user(
            self.admin, "security", "safe-pass-123", Role.USER
        )
        self.assertEqual(created.role, Role.USER)
        self.assertEqual(
            self.auth_service.authenticate("security", "safe-pass-123").id,
            created.id,
        )
        with self.assertRaises(UserExistsError):
            self.auth_service.create_user(
                self.admin, "SECURITY", "another-pass", Role.USER
            )

    def test_create_user_accepts_role_enum_and_string_values(self) -> None:
        cases = (
            ("enum-user", Role.USER, Role.USER),
            ("string-user", "USER", Role.USER),
            ("enum-admin", Role.ADMIN, Role.ADMIN),
            ("string-admin", "ADMIN", Role.ADMIN),
        )

        for username, value, expected in cases:
            with self.subTest(value=value):
                created = self.auth_service.create_user(
                    self.admin,
                    username,
                    "safe-pass-123",
                    value,
                )
                with self.database.connection() as connection:
                    stored_role = connection.execute(
                        "SELECT role FROM users WHERE id = ?",
                        (created.id,),
                    ).fetchone()["role"]

                self.assertIs(created.role, expected)
                self.assertEqual(stored_role, expected.value)

    def test_create_user_rejects_invalid_role(self) -> None:
        with self.assertRaisesRegex(ValidationError, "Geçersiz kullanıcı rolü"):
            self.auth_service.create_user(
                self.admin,
                "invalid-role-user",
                "safe-pass-123",
                "INVALID",
            )

        with self.database.connection() as connection:
            user_count = connection.execute(
                "SELECT COUNT(*) FROM users WHERE username = ?",
                ("invalid-role-user",),
            ).fetchone()[0]
        self.assertEqual(user_count, 0)

    def test_user_cannot_change_camera_or_create_user(self) -> None:
        user = self.auth_service.create_user(
            self.admin, "operator", "operator-pass", Role.USER
        )
        camera = self.camera_service.list_cameras()[0]
        with self.assertRaises(AuthorizationError):
            self.camera_service.update_camera(
                user,
                camera.id,
                camera.name,
                "rtsp://local-camera/stream",
                camera.direction,
                True,
            )
        with self.assertRaises(AuthorizationError):
            self.auth_service.create_user(user, "blocked", "blocked-pass", Role.USER)

    def test_search_and_inside_vehicle_query_use_latest_record(self) -> None:
        cameras = {camera.direction: camera for camera in self.camera_service.list_cameras()}
        now = datetime(2026, 8, 11, 10, 0, tzinfo=timezone.utc)

        self.plate_service.save_plate_detection(
            "34 ABC 123", cameras[Direction.ENTRY].id, 0.96, now
        )
        self.plate_service.save_plate_detection(
            "34ABC123",
            cameras[Direction.EXIT].id,
            0.93,
            now + timedelta(minutes=10),
        )
        self.plate_service.save_plate_detection(
            "06XYZ99",
            cameras[Direction.ENTRY].id,
            0.91,
            now + timedelta(minutes=20),
        )

        entry_records = self.plate_service.search_records(self.admin, "34", "ENTRY")
        self.assertEqual([record.plate for record in entry_records], ["34ABC123"])
        with self.assertRaisesRegex(ValidationError, "ENTRY veya EXIT"):
            self.plate_service.search_records(self.admin, direction="INVALID")
        vehicles_inside = self.plate_service.get_vehicles_inside(self.admin)
        self.assertEqual([vehicle.plate for vehicle in vehicles_inside], ["06XYZ99"])

    def test_duplicate_cooldown_is_camera_specific(self) -> None:
        cameras = {camera.direction: camera for camera in self.camera_service.list_cameras()}
        now = datetime(2026, 8, 11, 10, 0, tzinfo=timezone.utc)

        first = self.plate_service.save_plate_detection(
            "34ABC123", cameras[Direction.ENTRY].id, 0.95, now
        )
        with self.assertRaises(DuplicatePlateDetection):
            self.plate_service.save_plate_detection(
                "34ABC123",
                cameras[Direction.ENTRY].id,
                0.94,
                now + timedelta(seconds=5),
            )
        exit_record = self.plate_service.save_plate_detection(
            "34ABC123",
            cameras[Direction.EXIT].id,
            0.93,
            now + timedelta(seconds=5),
        )
        after_cooldown = self.plate_service.save_plate_detection(
            "34ABC123",
            cameras[Direction.ENTRY].id,
            0.92,
            now + timedelta(seconds=11),
        )

        self.assertEqual(first.direction, Direction.ENTRY)
        self.assertEqual(exit_record.direction, Direction.EXIT)
        self.assertEqual(after_cooldown.direction, Direction.ENTRY)

    def test_plate_timestamps_are_stored_and_returned_as_utc(self) -> None:
        camera = self.camera_service.list_cameras()[0]
        turkey_offset = timezone(timedelta(hours=3))

        aware_record = self.plate_service.save_plate_detection(
            "34UTC01",
            camera.id,
            0.95,
            datetime(2026, 8, 11, 13, 0, tzinfo=turkey_offset),
        )
        naive_record = self.plate_service.save_plate_detection(
            "34UTC02",
            camera.id,
            0.94,
            datetime(2026, 8, 11, 10, 0),
        )

        self.assertEqual(aware_record.timestamp, "2026-08-11T10:00:00+00:00")
        self.assertEqual(naive_record.timestamp, "2026-08-11T10:00:00+00:00")
        with self.database.connection() as connection:
            stored = {
                row["plate"]: row["timestamp"]
                for row in connection.execute(
                    "SELECT plate, timestamp FROM plate_records WHERE plate LIKE '34UTC%'"
                ).fetchall()
            }
        self.assertEqual(stored["34UTC01"], "2026-08-11T10:00:00+00:00")
        self.assertEqual(stored["34UTC02"], "2026-08-11T10:00:00+00:00")

    def test_initialize_normalizes_legacy_timestamps_without_losing_invalid_rows(self) -> None:
        camera = self.camera_service.list_cameras()[0]
        with self.database.connection() as connection:
            connection.execute(
                """
                INSERT INTO users (username, password_hash, role, created_at)
                VALUES (?, ?, ?, ?)
                """,
                ("legacy-user", "unused", "USER", "2026-08-11 10:00:00"),
            )
            connection.executemany(
                """
                INSERT INTO plate_records
                    (plate, direction, camera_id, confidence, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    ("34LEGACY", "ENTRY", camera.id, 0.90, "2026-08-11T13:00:00+03:00"),
                    ("34INVALID", "ENTRY", camera.id, 0.80, "legacy-value"),
                ),
            )

        self.database.initialize()

        with self.database.connection() as connection:
            created_at = connection.execute(
                "SELECT created_at FROM users WHERE username = 'legacy-user'"
            ).fetchone()["created_at"]
            timestamps = {
                row["plate"]: row["timestamp"]
                for row in connection.execute(
                    "SELECT plate, timestamp FROM plate_records WHERE plate LIKE '34L%' OR plate = '34INVALID'"
                ).fetchall()
            }
        self.assertEqual(created_at, "2026-08-11T10:00:00+00:00")
        self.assertEqual(timestamps["34LEGACY"], "2026-08-11T10:00:00+00:00")
        self.assertEqual(timestamps["34INVALID"], "legacy-value")

    def test_user_cannot_delete_or_read_admin_record_stats(self) -> None:
        user = self.auth_service.create_user(
            self.admin, "retention-user", "safe-pass-123", Role.USER
        )
        self._insert_record("34KEEP01", "2026-01-01T00:00:00+00:00")

        with self.assertRaises(AuthorizationError):
            self.plate_service.delete_records_older_than(user, 90)
        with self.assertRaises(AuthorizationError):
            self.plate_service.delete_all_records(user)
        with self.assertRaises(AuthorizationError):
            self.plate_service.get_record_stats(user)

        with self.database.connection() as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM plate_records"
            ).fetchone()[0]
        self.assertEqual(count, 1)

    def test_admin_retention_cleanup_uses_exclusive_utc_cutoff(self) -> None:
        turkey_offset = timezone(timedelta(hours=3))
        now = datetime(2026, 8, 11, 13, 0, tzinfo=turkey_offset)
        cutoff = now.astimezone(timezone.utc) - timedelta(days=90)
        old_timestamp = to_utc_storage(cutoff - timedelta(seconds=1))
        cutoff_timestamp = to_utc_storage(cutoff)
        new_timestamp = to_utc_storage(cutoff + timedelta(seconds=1))
        self._insert_record("34OLD01", old_timestamp)
        self._insert_record("34EDGE1", cutoff_timestamp)
        self._insert_record("34NEW01", new_timestamp)

        deleted = self.plate_service.delete_records_older_than(
            self.admin,
            90,
            now,
        )

        self.assertEqual(deleted, 1)
        with self.database.connection() as connection:
            remaining = {
                row["plate"]
                for row in connection.execute(
                    "SELECT plate FROM plate_records"
                ).fetchall()
            }
        self.assertEqual(remaining, {"34EDGE1", "34NEW01"})
        stats = self.plate_service.get_record_stats(self.admin)
        self.assertEqual(stats.total_records, 2)
        self.assertEqual(stats.oldest_timestamp, cutoff_timestamp)

    def test_unlimited_retention_does_not_delete_records(self) -> None:
        self._insert_record("34FOREVER", "2020-01-01T00:00:00+00:00")
        self.plate_service.set_record_retention_days(0)

        deleted = self.plate_service.apply_retention_policy(
            datetime(2026, 8, 11, 10, 0, tzinfo=timezone.utc)
        )

        self.assertEqual(deleted, 0)
        self.assertEqual(
            self.plate_service.get_record_stats(self.admin).total_records,
            1,
        )

    def test_delete_all_only_clears_plate_records_and_returns_row_count(self) -> None:
        self.auth_service.create_user(
            self.admin, "preserved-user", "safe-pass-123", Role.USER
        )
        self._insert_record("34DELETE1", "2026-08-01T00:00:00+00:00")
        self._insert_record("34DELETE2", "2026-08-02T00:00:00+00:00")
        with self.database.connection() as connection:
            users_before = connection.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            cameras_before = connection.execute("SELECT COUNT(*) FROM cameras").fetchone()[0]

        deleted = self.plate_service.delete_all_records(self.admin)

        self.assertEqual(deleted, 2)
        with self.database.connection() as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM plate_records").fetchone()[0],
                0,
            )
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM users").fetchone()[0],
                users_before,
            )
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM cameras").fetchone()[0],
                cameras_before,
            )

    def test_startup_retention_cleanup_runs_once_and_logs_removed_rows(self) -> None:
        now = datetime(2026, 8, 11, 10, 0, tzinfo=timezone.utc)
        self._insert_record(
            "34START1",
            to_utc_storage(now - timedelta(days=91)),
        )
        self._insert_record(
            "34START2",
            to_utc_storage(now - timedelta(days=10)),
        )
        with patch("app.plate_service.utc_now", return_value=now):
            with self.assertLogs("main", level="INFO") as captured:
                deleted = apply_startup_retention_cleanup(self.plate_service)

        self.assertEqual(deleted, 1)
        self.assertEqual(
            self.plate_service.get_record_stats(self.admin).total_records,
            1,
        )
        self.assertTrue(
            any("removed 1 plate records older than 90 days" in line for line in captured.output)
        )

    def test_startup_cleanup_exception_does_not_escape(self) -> None:
        with patch.object(
            self.plate_service,
            "apply_retention_policy",
            side_effect=RuntimeError("cleanup failed"),
        ):
            with self.assertLogs("main", level="ERROR") as captured:
                deleted = apply_startup_retention_cleanup(self.plate_service)

        self.assertEqual(deleted, 0)
        self.assertTrue(
            any("startup will continue" in line for line in captured.output)
        )


if __name__ == "__main__":
    unittest.main()
