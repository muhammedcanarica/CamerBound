from __future__ import annotations

import sqlite3
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from app.auth import AuthService, ValidationError
from app.camera import Camera, CameraService
from app.camera_credentials import (
    CredentialProtectionError,
    DpapiPasswordProtector,
    build_authenticated_camera_source,
    split_camera_source_credentials,
)
from app.database import Database


class FakePasswordProtector:
    def __init__(self) -> None:
        self._values: dict[str, str] = {}

    def protect(self, plaintext: str) -> str:
        token = f"protected-camera-value-{len(self._values) + 1}"
        self._values[token] = plaintext
        return token

    def unprotect(self, protected_value: str) -> str:
        return self._values[protected_value]


class FailingPasswordProtector:
    def protect(self, plaintext: str) -> str:
        raise RuntimeError(f"must-not-leak:{plaintext}")

    def unprotect(self, protected_value: str) -> str:
        raise RuntimeError(f"must-not-leak:{protected_value}")


class SourceRecordingCapture:
    def __init__(self) -> None:
        self.released = threading.Event()

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, object]:
        self.released.wait(0.005)
        return (not self.released.is_set(), object())

    def release(self) -> None:
        self.released.set()

    def get(self, _property_id: int) -> float:
        return 0.0


class SourceRecordingFactory:
    def __init__(self) -> None:
        self.sources: list[str | int] = []
        self.captures: list[SourceRecordingCapture] = []
        self.created = threading.Event()

    def __call__(self, source: str | int) -> SourceRecordingCapture:
        self.sources.append(source)
        capture = SourceRecordingCapture()
        self.captures.append(capture)
        self.created.set()
        return capture


class CameraCredentialUrlTests(unittest.TestCase):
    def test_url_credentials_are_split_and_special_characters_are_decoded(
        self,
    ) -> None:
        source = (
            "http://test%2Duser:p%40ss%3Aword@CAMERA_IP/"
            "cgi-bin/faststream.jpg?stream=full&fps=10#preview"
        )

        parsed = split_camera_source_credentials(source)

        self.assertEqual(parsed.username, "test-user")
        self.assertEqual(parsed.password, "p@ss:word")
        self.assertEqual(
            parsed.stream_url,
            "http://CAMERA_IP/cgi-bin/faststream.jpg?stream=full&fps=10#preview",
        )

    def test_runtime_source_encodes_special_characters(self) -> None:
        source = build_authenticated_camera_source(
            "http://CAMERA_IP/cgi-bin/faststream.jpg?stream=full&fps=10",
            "test user@site",
            "test-password:@/",
        )

        self.assertEqual(
            source,
            "http://test%20user%40site:test-password%3A%40%2F@CAMERA_IP/"
            "cgi-bin/faststream.jpg?stream=full&fps=10",
        )

    @unittest.skipUnless(sys.platform == "win32", "Windows DPAPI is required")
    def test_dpapi_round_trip_does_not_return_plaintext_storage(self) -> None:
        protector = DpapiPasswordProtector()
        plaintext = "test-password-özel"

        protected = protector.protect(plaintext)

        self.assertNotEqual(protected, plaintext)
        self.assertNotIn(plaintext, protected)
        self.assertEqual(protector.unprotect(protected), plaintext)

    @patch("app.camera_credentials.sys.platform", "linux")
    def test_dpapi_has_no_plaintext_fallback_off_windows(self) -> None:
        with self.assertRaises(CredentialProtectionError):
            DpapiPasswordProtector().protect("test-password")


class CameraCredentialMigrationTests(unittest.TestCase):
    def test_legacy_camera_table_migrates_credentials_without_losing_fields(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            path = Path(temp_directory) / "legacy-camera.db"
            self._create_legacy_database(path)
            protector = FakePasswordProtector()

            database = Database(path, password_protector=protector)
            database.initialize()

            with database.connection() as connection:
                row = connection.execute(
                    """
                    SELECT name, stream_url, username, protected_password,
                           direction, enabled
                    FROM cameras
                    WHERE id = 7
                    """
                ).fetchone()
            self.assertEqual(row["name"], "Legacy Entry")
            self.assertEqual(row["direction"], "ENTRY")
            self.assertEqual(row["enabled"], 1)
            self.assertEqual(
                row["stream_url"],
                "http://CAMERA_IP/cgi-bin/faststream.jpg?stream=full&fps=10",
            )
            self.assertEqual(row["username"], "test-user")
            self.assertNotEqual(row["protected_password"], "test-password")
            self.assertEqual(
                protector.unprotect(row["protected_password"]),
                "test-password",
            )
            self.assertNotIn(b"test-password", path.read_bytes())

    def test_failed_migration_rolls_back_without_losing_legacy_credentials(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            path = Path(temp_directory) / "failed-camera.db"
            self._create_legacy_database(path)

            with self.assertRaisesRegex(
                CredentialProtectionError,
                "migration was rolled back",
            ) as raised:
                Database(
                    path,
                    password_protector=FailingPasswordProtector(),
                ).initialize()

            self.assertNotIn("test-password", str(raised.exception))
            connection = sqlite3.connect(path)
            stored_source = connection.execute(
                "SELECT stream_url FROM cameras WHERE id = 7"
            ).fetchone()[0]
            connection.close()
            self.assertIn("test-user:test-password@", stored_source)

    @staticmethod
    def _create_legacy_database(path: Path) -> None:
        connection = sqlite3.connect(path)
        connection.execute(
            """
            CREATE TABLE cameras (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                stream_url TEXT NOT NULL DEFAULT '',
                direction TEXT NOT NULL CHECK (direction IN ('ENTRY', 'EXIT')),
                enabled INTEGER NOT NULL DEFAULT 0 CHECK (enabled IN (0, 1))
            )
            """
        )
        connection.execute(
            """
            INSERT INTO cameras (id, name, stream_url, direction, enabled)
            VALUES (7, ?, ?, 'ENTRY', 1)
            """,
            (
                "Legacy Entry",
                "http://test-user:test-password@CAMERA_IP/"
                "cgi-bin/faststream.jpg?stream=full&fps=10",
            ),
        )
        connection.commit()
        connection.close()


class CameraCredentialServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.protector = FakePasswordProtector()
        self.database = Database(
            Path(self.temp_directory.name) / "credentials.db",
            password_protector=self.protector,
        )
        self.database.initialize()
        auth_service = AuthService(self.database)
        auth_service.ensure_default_admin()
        self.admin = auth_service.authenticate("admin", "admin123")
        self.capture_factory = SourceRecordingFactory()
        self.camera_service = CameraService(
            self.database,
            capture_factory=self.capture_factory,
            retry_delay_seconds=0.1,
        )

    def tearDown(self) -> None:
        self.camera_service.stop_all()
        self.temp_directory.cleanup()

    def test_password_is_protected_url_is_clean_and_audit_has_no_credentials(
        self,
    ) -> None:
        camera = self.camera_service.list_cameras()[0]

        updated = self.camera_service.update_camera(
            self.admin,
            camera.id,
            camera.name,
            "http://test-user:test-password@CAMERA_IP/live?fps=10",
            camera.direction,
            True,
        )

        self.assertEqual(updated.stream_url, "http://CAMERA_IP/live?fps=10")
        self.assertEqual(updated.username, "test-user")
        self.assertTrue(updated.has_password)
        self.assertNotEqual(updated.protected_password, "test-password")
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT stream_url, protected_password FROM cameras WHERE id = ?",
                (camera.id,),
            ).fetchone()
            audit_details = connection.execute(
                "SELECT details FROM audit_logs ORDER BY id DESC LIMIT 1"
            ).fetchone()["details"]
        self.assertNotIn("@", row["stream_url"])
        self.assertNotIn("test-password", row["protected_password"])
        self.assertNotIn("test-user", audit_details)
        self.assertNotIn("test-password", audit_details)

    def test_blank_password_preserves_existing_then_new_password_replaces_it(
        self,
    ) -> None:
        camera = self._save_credentials("test-password")
        original_protected = camera.protected_password

        preserved = self.camera_service.update_camera(
            self.admin,
            camera.id,
            camera.name,
            camera.stream_url,
            camera.direction,
            True,
            username=camera.username,
            password="",
        )
        changed = self.camera_service.update_camera(
            self.admin,
            camera.id,
            camera.name,
            camera.stream_url,
            camera.direction,
            True,
            username=camera.username,
            password="new-test-password",
        )

        self.assertEqual(preserved.protected_password, original_protected)
        self.assertNotEqual(changed.protected_password, original_protected)
        self.assertEqual(
            self.protector.unprotect(changed.protected_password),
            "new-test-password",
        )

    def test_explicit_credential_clear_removes_username_and_password(self) -> None:
        camera = self._save_credentials("test-password")

        cleared = self.camera_service.update_camera(
            self.admin,
            camera.id,
            camera.name,
            camera.stream_url,
            camera.direction,
            True,
            username=camera.username,
            clear_credentials=True,
        )

        self.assertEqual(cleared.username, "")
        self.assertIsNone(cleared.protected_password)

    def test_runtime_authenticated_source_exists_only_in_capture_factory(
        self,
    ) -> None:
        camera = self.camera_service.list_cameras()[0]
        camera = self.camera_service.update_camera(
            self.admin,
            camera.id,
            camera.name,
            "http://CAMERA_IP/live?fps=10",
            camera.direction,
            True,
            username="test user",
            password="test-password:@",
        )

        self.camera_service.start_camera(camera.id)
        self.assertTrue(self.capture_factory.created.wait(1.0))

        self.assertEqual(
            self.capture_factory.sources,
            ["http://test%20user:test-password%3A%40@CAMERA_IP/live?fps=10"],
        )
        self.assertEqual(
            self.camera_service.get_camera(camera.id).stream_url,
            "http://CAMERA_IP/live?fps=10",
        )

    def test_protection_failure_message_and_database_do_not_include_password(
        self,
    ) -> None:
        camera = self.camera_service.list_cameras()[0]
        self.camera_service.password_protector = FailingPasswordProtector()

        with self.assertRaises(ValidationError) as raised:
            self.camera_service.update_camera(
                self.admin,
                camera.id,
                camera.name,
                "http://CAMERA_IP/live",
                camera.direction,
                True,
                username="test-user",
                password="test-password",
            )

        self.assertNotIn("test-password", str(raised.exception))
        self.assertIsNone(self.camera_service.get_camera(camera.id).protected_password)

    def test_credentials_are_rejected_for_local_sources_without_database_write(
        self,
    ) -> None:
        camera = self.camera_service.list_cameras()[0]

        with self.assertRaisesRegex(ValidationError, "HTTP, HTTPS veya RTSP"):
            self.camera_service.update_camera(
                self.admin,
                camera.id,
                camera.name,
                "videos/entry.mp4",
                camera.direction,
                True,
                username="test-user",
                password="test-password",
            )

        unchanged = self.camera_service.get_camera(camera.id)
        self.assertEqual(unchanged.stream_url, "")
        self.assertIsNone(unchanged.protected_password)

    def _save_credentials(self, password: str) -> Camera:
        camera = self.camera_service.list_cameras()[0]
        return self.camera_service.update_camera(
            self.admin,
            camera.id,
            camera.name,
            "http://CAMERA_IP/live",
            camera.direction,
            True,
            username="test-user",
            password=password,
        )


if __name__ == "__main__":
    unittest.main()
