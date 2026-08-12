from __future__ import annotations

import unittest
from unittest.mock import patch

from app.camera_worker import (
    MAX_CONSECUTIVE_READ_FAILURES,
    NETWORK_OPEN_TIMEOUT_MS,
    NETWORK_READ_TIMEOUT_MS,
    CameraStatus,
    CameraWorker,
    create_video_capture,
)


class FakeCapture:
    def __init__(self, opened: bool) -> None:
        self.opened = opened
        self.open_calls: list[tuple[object, ...]] = []
        self.release_count = 0

    def open(self, *args) -> bool:
        self.open_calls.append(args)
        return self.opened

    def isOpened(self) -> bool:
        return self.opened

    def release(self) -> None:
        self.release_count += 1


class SequencedCapture:
    def __init__(
        self,
        results: list[tuple[bool, object | None]],
        worker: CameraWorker | None = None,
    ) -> None:
        self.results = list(results)
        self.worker = worker
        self.read_count = 0
        self.release_count = 0

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, object | None]:
        self.read_count += 1
        if self.results:
            return self.results.pop(0)
        if self.worker is not None:
            self.worker.request_stop()
        return False, None

    def release(self) -> None:
        self.release_count += 1

    def get(self, _property_id: int) -> float:
        return 25.0


class VideoCaptureBackendTests(unittest.TestCase):
    @patch("app.camera_worker.sys.platform", "win32")
    def test_windows_webcam_prefers_dshow_and_stops_after_success(self) -> None:
        dshow = FakeCapture(True)
        with patch("app.camera_worker.cv2.VideoCapture", side_effect=[dshow]) as factory:
            capture = create_video_capture(0)

        self.assertIs(capture, dshow)
        self.assertEqual(dshow.open_calls, [(0, _cv2_backend("CAP_DSHOW"))])
        self.assertEqual(dshow.release_count, 0)
        self.assertEqual(factory.call_count, 1)

    @patch("app.camera_worker.sys.platform", "win32")
    def test_windows_webcam_falls_back_from_dshow_to_msmf(self) -> None:
        dshow = FakeCapture(False)
        msmf = FakeCapture(True)
        with patch(
            "app.camera_worker.cv2.VideoCapture",
            side_effect=[dshow, msmf],
        ) as factory:
            capture = create_video_capture(0)

        self.assertIs(capture, msmf)
        self.assertEqual(dshow.release_count, 1)
        self.assertEqual(msmf.open_calls, [(0, _cv2_backend("CAP_MSMF"))])
        self.assertEqual(factory.call_count, 2)

    @patch("app.camera_worker.sys.platform", "win32")
    def test_windows_webcam_falls_back_to_cap_any_and_releases_failures(self) -> None:
        dshow = FakeCapture(False)
        msmf = FakeCapture(False)
        cap_any = FakeCapture(True)
        with patch(
            "app.camera_worker.cv2.VideoCapture",
            side_effect=[dshow, msmf, cap_any],
        ) as factory:
            capture = create_video_capture(0)

        self.assertIs(capture, cap_any)
        self.assertEqual(dshow.release_count, 1)
        self.assertEqual(msmf.release_count, 1)
        self.assertEqual(cap_any.release_count, 0)
        self.assertEqual(cap_any.open_calls, [(0, _cv2_backend("CAP_ANY"))])
        self.assertEqual(factory.call_count, 3)

    @patch("app.camera_worker.sys.platform", "win32")
    def test_rtsp_source_does_not_use_webcam_fallback(self) -> None:
        capture = FakeCapture(True)
        with patch("app.camera_worker.cv2.VideoCapture", return_value=capture) as factory:
            result = create_video_capture("rtsp://camera.invalid/live")

        self.assertIs(result, capture)
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(capture.open_calls[0][0], "rtsp://camera.invalid/live")
        self.assertNotIn(_cv2_backend("CAP_DSHOW"), capture.open_calls[0][1:])
        self.assertNotIn(_cv2_backend("CAP_MSMF"), capture.open_calls[0][1:])
        timeout_parameters = capture.open_calls[0][2]
        from app import camera_worker

        self.assertEqual(
            timeout_parameters,
            [
                camera_worker.cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                NETWORK_OPEN_TIMEOUT_MS,
                camera_worker.cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                NETWORK_READ_TIMEOUT_MS,
            ],
        )

    @patch("app.camera_worker.sys.platform", "win32")
    def test_local_video_keeps_normal_open_behavior(self) -> None:
        capture = FakeCapture(True)
        with patch("app.camera_worker.cv2.VideoCapture", return_value=capture) as factory:
            result = create_video_capture("C:\\videos\\entry.mp4")

        self.assertIs(result, capture)
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(capture.open_calls, [("C:\\videos\\entry.mp4",)])

    @patch("app.camera_worker.sys.platform", "linux")
    def test_non_windows_webcam_keeps_normal_open_behavior(self) -> None:
        capture = FakeCapture(True)
        with patch("app.camera_worker.cv2.VideoCapture", return_value=capture):
            result = create_video_capture(0)

        self.assertIs(result, capture)
        self.assertEqual(capture.open_calls, [(0,)])


class CameraWorkerReadToleranceTests(unittest.TestCase):
    def test_single_network_read_failure_does_not_end_stream(self) -> None:
        worker = CameraWorker(2, "http://CAMERA_IP/live", preview_fps=1_000)
        capture = SequencedCapture(
            [(False, None), (True, object())],
            worker=worker,
        )

        stopped_normally = worker._read_frames(capture)

        self.assertTrue(stopped_normally)
        self.assertEqual(capture.read_count, 3)

    def test_successful_network_frame_resets_failure_counter(self) -> None:
        worker = CameraWorker(2, "http://CAMERA_IP/live", preview_fps=1_000)
        capture = SequencedCapture(
            [
                (False, None),
                (False, None),
                (True, object()),
                (False, None),
                (False, None),
            ],
            worker=worker,
        )

        stopped_normally = worker._read_frames(capture)

        self.assertTrue(stopped_normally)
        self.assertEqual(capture.read_count, 6)

    def test_network_failure_threshold_ends_stream_for_reconnect(self) -> None:
        worker = CameraWorker(2, "rtsp://CAMERA_IP/live")
        capture = SequencedCapture(
            [(False, None)] * MAX_CONSECUTIVE_READ_FAILURES,
        )

        stopped_normally = worker._read_frames(capture)

        self.assertFalse(stopped_normally)
        self.assertEqual(capture.read_count, MAX_CONSECUTIVE_READ_FAILURES)

    def test_webcam_read_failure_keeps_immediate_failure_behavior(self) -> None:
        worker = CameraWorker(2, 0)
        capture = SequencedCapture([(False, None)] * 3)

        self.assertFalse(worker._read_frames(capture))
        self.assertEqual(capture.read_count, 1)

    def test_local_video_eof_keeps_immediate_failure_behavior(self) -> None:
        worker = CameraWorker(2, "videos/entry.mp4")
        capture = SequencedCapture([(False, None)] * 3)

        self.assertFalse(worker._read_frames(capture))
        self.assertEqual(capture.read_count, 1)

    def test_threshold_reconnects_and_returns_to_connected(self) -> None:
        worker = CameraWorker(
            2,
            "http://CAMERA_IP/live",
            retry_delay_seconds=0.1,
        )
        first = SequencedCapture([(False, None)] * MAX_CONSECUTIVE_READ_FAILURES)
        second = SequencedCapture([], worker=worker)
        captures = iter((first, second))
        statuses: list[CameraStatus] = []
        worker.status_changed.connect(
            lambda _camera_id, status, _message: statuses.append(CameraStatus(status))
        )
        worker.capture_factory = lambda _source: next(captures)

        worker.run()

        self.assertIn(CameraStatus.ERROR, statuses)
        self.assertIn(CameraStatus.RECONNECTING, statuses)
        connected_indexes = [
            index for index, status in enumerate(statuses) if status is CameraStatus.CONNECTED
        ]
        self.assertEqual(len(connected_indexes), 2)
        self.assertLess(connected_indexes[0], connected_indexes[1])
        self.assertEqual(statuses[-1], CameraStatus.STOPPED)


def _cv2_backend(name: str) -> int:
    from app import camera_worker

    return getattr(camera_worker.cv2, name)


if __name__ == "__main__":
    unittest.main()
