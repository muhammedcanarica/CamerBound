from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from app.auth import AuthService, ValidationError
from app.camera import CameraService, CameraStatus
from app.database import Database


class FakeCapture:
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


class RecordingCaptureFactory:
    def __init__(self) -> None:
        self.captures: list[FakeCapture] = []
        self.created = threading.Event()

    def __call__(self, _source: str | int) -> FakeCapture:
        capture = FakeCapture()
        self.captures.append(capture)
        self.created.set()
        return capture


class CameraServiceLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.temp_directory.name) / "camera-test.db")
        self.database.initialize()
        auth_service = AuthService(self.database)
        auth_service.ensure_default_admin()
        self.admin = auth_service.authenticate("admin", "admin123")
        self.capture_factory = RecordingCaptureFactory()
        self.camera_service = CameraService(
            self.database,
            capture_factory=self.capture_factory,
            retry_delay_seconds=0.1,
        )

    def tearDown(self) -> None:
        self.camera_service.stop_all()
        self.temp_directory.cleanup()

    def test_disabled_camera_does_not_start(self) -> None:
        camera = self.camera_service.list_cameras()[0]

        with self.assertRaises(ValidationError):
            self.camera_service.start_camera(camera.id)

        self.assertEqual(self.capture_factory.captures, [])
        self.assertEqual(self.camera_service.get_status(camera.id), CameraStatus.STOPPED)

    def test_empty_stream_url_does_not_start(self) -> None:
        camera = self.camera_service.list_cameras()[0]
        self.camera_service.update_camera(
            self.admin,
            camera.id,
            camera.name,
            "",
            camera.direction,
            True,
        )

        with self.assertRaises(ValidationError):
            self.camera_service.start_camera(camera.id)

        self.assertEqual(self.capture_factory.captures, [])

    def test_same_camera_is_not_started_twice(self) -> None:
        camera_id = self._configure_camera(0, "rtsp://example.invalid/entry")

        self.camera_service.start_camera(camera_id)
        self.assertTrue(self.capture_factory.created.wait(1.0))
        self.camera_service.start_camera(camera_id)

        self.assertEqual(len(self.capture_factory.captures), 1)

    def test_stop_camera_releases_capture_and_cleans_runtime(self) -> None:
        camera_id = self._configure_camera(0, "rtsp://example.invalid/entry")
        self.camera_service.start_camera(camera_id)
        self.assertTrue(self.capture_factory.created.wait(1.0))
        capture = self.capture_factory.captures[0]

        status = self.camera_service.stop_camera(camera_id)

        self.assertEqual(status, CameraStatus.STOPPED)
        self.assertTrue(capture.released.is_set())
        self.assertNotIn(camera_id, self.camera_service._runtimes)

    def test_stop_all_releases_every_capture(self) -> None:
        cameras = self.camera_service.list_cameras()
        camera_ids = [
            self._configure_camera(index, f"rtsp://example.invalid/{index}")
            for index in range(len(cameras))
        ]
        for camera_id in camera_ids:
            self.camera_service.start_camera(camera_id)

        self.assertTrue(self._wait_for_capture_count(len(camera_ids)))
        self.camera_service.stop_all()

        self.assertTrue(all(capture.released.is_set() for capture in self.capture_factory.captures))
        self.assertEqual(self.camera_service._runtimes, {})
        for camera_id in camera_ids:
            self.assertEqual(self.camera_service.get_status(camera_id), CameraStatus.STOPPED)

    def test_latest_delivered_frame_is_a_copy(self) -> None:
        camera_id = self._configure_camera(0, "rtsp://example.invalid/entry")
        self.camera_service.start_camera(camera_id)
        self.assertTrue(self.capture_factory.created.wait(1.0))
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        with self.camera_service._lock:
            self.camera_service._latest_frames[camera_id] = frame
        self.camera_service._flush_latest_frame(camera_id)

        snapshot = self.camera_service.get_latest_frame(camera_id)
        self.assertIsNot(snapshot, frame)
        snapshot[0, 0, 0] = 255
        self.assertEqual(self.camera_service.get_latest_frame(camera_id)[0, 0, 0], 0)

    def _configure_camera(self, index: int, stream_url: str) -> int:
        camera = self.camera_service.list_cameras()[index]
        updated = self.camera_service.update_camera(
            self.admin,
            camera.id,
            camera.name,
            stream_url,
            camera.direction,
            True,
        )
        return updated.id

    def _wait_for_capture_count(self, expected: int) -> bool:
        for _ in range(100):
            if len(self.capture_factory.captures) >= expected:
                return True
            threading.Event().wait(0.01)
        return False


if __name__ == "__main__":
    unittest.main()
