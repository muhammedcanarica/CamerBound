from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.auth import (
    AuthService,
    AuthenticationError,
    AuthorizationError,
    Role,
    UserExistsError,
)
from app.camera import CameraService, Direction
from app.database import Database
from app.plate_service import PlateService


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

        entry_records = self.plate_service.search_records(
            self.admin, "34", Direction.ENTRY
        )
        self.assertEqual([record.plate for record in entry_records], ["34ABC123"])
        vehicles_inside = self.plate_service.get_vehicles_inside(self.admin)
        self.assertEqual([vehicle.plate for vehicle in vehicles_inside], ["06XYZ99"])


if __name__ == "__main__":
    unittest.main()
