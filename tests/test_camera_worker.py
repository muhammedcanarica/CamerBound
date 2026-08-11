from __future__ import annotations

import unittest
from unittest.mock import patch

from app.camera_worker import create_video_capture


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


def _cv2_backend(name: str) -> int:
    from app import camera_worker

    return getattr(camera_worker.cv2, name)


if __name__ == "__main__":
    unittest.main()
