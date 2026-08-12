from __future__ import annotations

import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

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
from app.plate_capture import PlateCaptureService
from app.plate_service import PlateService
from app.plate_service import DuplicatePlateDetection
from app.time_utils import parse_utc_timestamp, to_utc_storage
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
        self.capture_service = PlateCaptureService(
            Path(self.temp_directory.name) / "data" / "captures",
            Path(self.temp_directory.name),
        )
        self.plate_service = PlateService(
            self.database,
            duplicate_cooldown_seconds=10,
            capture_service=self.capture_service,
        )

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

    def test_daily_archive_uses_local_date_across_utc_midnight_boundary(self) -> None:
        cameras = {
            camera.direction: camera for camera in self.camera_service.list_cameras()
        }
        before_midnight = datetime(2026, 8, 12, 20, 59, 59, tzinfo=timezone.utc)
        after_midnight = datetime(2026, 8, 12, 21, 0, 1, tzinfo=timezone.utc)
        next_record = datetime(2026, 8, 13, 8, 0, tzinfo=timezone.utc)
        record_ids: dict[str, int] = {}
        with self.database.connection() as connection:
            for plate, direction, timestamp in (
                ("34DAY12", Direction.ENTRY, before_midnight),
                ("06DAY13", Direction.EXIT, after_midnight),
                ("35DAY13", Direction.ENTRY, next_record),
            ):
                camera = cameras[direction]
                cursor = connection.execute(
                    """
                    INSERT INTO plate_records
                        (plate, direction, camera_id, confidence, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        plate,
                        direction.value,
                        camera.id,
                        0.90,
                        to_utc_storage(timestamp),
                    ),
                )
                record_ids[plate] = int(cursor.lastrowid)

        local_timezone = timezone(timedelta(hours=3))

        def to_test_local(value: str | datetime) -> datetime:
            return parse_utc_timestamp(value).astimezone(local_timezone)

        with patch("app.plate_service.to_local_datetime", side_effect=to_test_local):
            summaries = self.plate_service.get_record_day_summaries(self.admin)
            august_12 = self.plate_service.search_records_for_local_date(
                self.admin, date(2026, 8, 12)
            )
            august_13 = self.plate_service.search_records_for_local_date(
                self.admin, date(2026, 8, 13)
            )
            filtered = self.plate_service.search_records_for_local_date(
                self.admin,
                date(2026, 8, 13),
                plate_query="06",
                direction=Direction.EXIT,
            )

        self.assertEqual(
            [summary.date for summary in summaries],
            [date(2026, 8, 13), date(2026, 8, 12)],
        )
        self.assertEqual(
            (
                summaries[0].total_count,
                summaries[0].entry_count,
                summaries[0].exit_count,
            ),
            (2, 1, 1),
        )
        self.assertEqual(
            (
                summaries[1].total_count,
                summaries[1].entry_count,
                summaries[1].exit_count,
            ),
            (1, 1, 0),
        )
        self.assertEqual(
            [record.id for record in august_12], [record_ids["34DAY12"]]
        )
        self.assertEqual(
            {record.id for record in august_13},
            {record_ids["06DAY13"], record_ids["35DAY13"]},
        )
        self.assertEqual(
            [record.id for record in filtered], [record_ids["06DAY13"]]
        )

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

    def test_confirmed_record_with_frame_stores_image_path(self) -> None:
        camera = self.camera_service.list_cameras()[0]
        frame = np.full((720, 1280, 3), 120, dtype=np.uint8)

        record = self.plate_service.save_plate_detection(
            "34PHOTO1",
            camera.id,
            0.95,
            datetime(2026, 8, 12, 8, 20, 31, tzinfo=timezone.utc),
            frame,
        )

        self.assertIsNotNone(record.image_path)
        target = self.capture_service.resolve_reference(record.image_path)
        self.assertIsNotNone(target)
        self.assertTrue(target.is_file())
        with self.database.connection() as connection:
            stored_path = connection.execute(
                "SELECT image_path FROM plate_records WHERE id = ?", (record.id,)
            ).fetchone()["image_path"]
        self.assertEqual(stored_path, record.image_path)

    def test_duplicate_record_does_not_create_another_capture(self) -> None:
        camera = self.camera_service.list_cameras()[0]
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        now = datetime(2026, 8, 12, 8, 0, tzinfo=timezone.utc)
        self.plate_service.save_plate_detection(
            "34DUP01", camera.id, 0.95, now, frame
        )

        with self.assertRaises(DuplicatePlateDetection):
            self.plate_service.save_plate_detection(
                "34DUP01", camera.id, 0.94, now + timedelta(seconds=5), frame
            )

        self.assertEqual(len(list(Path(self.temp_directory.name).rglob("*.jpg"))), 1)

    def test_capture_failure_keeps_database_record_with_null_path(self) -> None:
        camera = self.camera_service.list_cameras()[0]
        frame = np.zeros((200, 400, 3), dtype=np.uint8)

        with patch.object(
            self.capture_service,
            "save_capture",
            side_effect=OSError("disk full"),
        ), self.assertLogs("app.plate_service", level="ERROR"):
            record = self.plate_service.save_plate_detection(
                "34FAIL01", camera.id, 0.95, frame=frame
            )

        self.assertIsNone(record.image_path)
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT image_path FROM plate_records WHERE id = ?", (record.id,)
            ).fetchone()
        self.assertIsNotNone(row)
        self.assertIsNone(row["image_path"])

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

    def test_retention_cleanup_deletes_associated_capture_file(self) -> None:
        camera = self.camera_service.list_cameras()[0]
        now = datetime(2026, 8, 12, 8, 0, tzinfo=timezone.utc)
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        old_record = self.plate_service.save_plate_detection(
            "34OLDIMG",
            camera.id,
            0.9,
            now - timedelta(days=91),
            frame,
        )
        image_file = self.capture_service.resolve_reference(old_record.image_path)
        self.assertTrue(image_file.is_file())

        deleted = self.plate_service.delete_records_older_than(self.admin, 90, now)

        self.assertEqual(deleted, 1)
        self.assertFalse(image_file.exists())

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

    def test_delete_all_records_deletes_associated_capture_files(self) -> None:
        camera = self.camera_service.list_cameras()[0]
        record = self.plate_service.save_plate_detection(
            "34DELIMG",
            camera.id,
            0.9,
            frame=np.zeros((100, 200, 3), dtype=np.uint8),
        )
        image_file = self.capture_service.resolve_reference(record.image_path)
        self.assertTrue(image_file.is_file())

        deleted = self.plate_service.delete_all_records(self.admin)

        self.assertEqual(deleted, 1)
        self.assertFalse(image_file.exists())

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
