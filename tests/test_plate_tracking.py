from __future__ import annotations

import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.camera import Direction
from app.config import DEFAULT_ROI, PlateRecognitionConfig
from app.plate_detector import PlateDetection
from app.plate_recognition import (
    OcrJob,
    OcrJobBuffer,
    OcrJobType,
    PlateDetectionProcessor,
)
from app.plate_tracking import PlateTrackManager


def detection(x: int, y: int = 20) -> PlateDetection:
    return PlateDetection(0.90, x=x, y=y, width=100, height=30)


def manager(*, timeout_ms: int = 1_200) -> PlateTrackManager:
    return PlateTrackManager(
        max_active_tracks_per_camera=2,
        timeout_ms=timeout_ms,
        iou_threshold=0.25,
    )


class PlateTrackManagerTests(unittest.TestCase):
    def test_single_vehicle_keeps_track_id_while_bbox_moves(self) -> None:
        tracker = manager()

        first = tracker.update(1, [detection(100)], 10.0)
        second = tracker.update(1, [detection(110)], 10.2)

        self.assertEqual(len(first.active_tracks), 1)
        self.assertEqual(
            first.assignments[0].track_id,
            second.assignments[0].track_id,
        )
        self.assertFalse(second.assignments[0].created)

    def test_two_vehicle_identity_survives_detection_order_change(self) -> None:
        tracker = manager()
        first = tracker.update(1, [detection(100), detection(600)], 10.0)
        left_id, right_id = (item.track_id for item in first.assignments)

        second = tracker.update(1, [detection(590), detection(110)], 10.2)

        self.assertEqual(second.assignments[0].track_id, right_id)
        self.assertEqual(second.assignments[1].track_id, left_id)
        self.assertNotEqual(left_id, right_id)

    def test_missing_vehicle_expires_without_replacing_surviving_track(self) -> None:
        tracker = manager(timeout_ms=1_000)
        first = tracker.update(1, [detection(100), detection(600)], 10.0)
        left_id, right_id = (item.track_id for item in first.assignments)

        tracker.update(1, [detection(110)], 10.8)
        surviving = tracker.update(1, [detection(120)], 11.1)
        finalized = tracker.consume_finalized()
        with_new_vehicle = tracker.update(
            1,
            [detection(130), detection(700)],
            11.2,
        )

        self.assertEqual(surviving.assignments[0].track_id, left_id)
        self.assertEqual([item.track_id for item in finalized], [right_id])
        self.assertEqual(with_new_vehicle.assignments[0].track_id, left_id)
        self.assertGreater(with_new_vehicle.assignments[1].track_id, right_id)

    def test_third_detection_is_ignored_without_evicting_active_tracks(self) -> None:
        tracker = manager()
        first = tracker.update(1, [detection(100), detection(500)], 10.0)
        active_ids = {item.track_id for item in first.active_tracks}

        update = tracker.update(
            1,
            [detection(105), detection(505), detection(850)],
            10.2,
        )

        self.assertEqual({item.track_id for item in update.active_tracks}, active_ids)
        self.assertEqual(len(update.ignored_detections), 1)

    def test_ocr_history_and_best_candidate_are_isolated_per_track(self) -> None:
        tracker = manager()
        update = tracker.update(1, [detection(100), detection(600)], 10.0)
        left_id, right_id = (item.track_id for item in update.assignments)
        self.assertTrue(tracker.mark_ocr_scheduled(left_id))
        self.assertTrue(tracker.mark_ocr_scheduled(right_id))

        tracker.record_ocr_result(left_id, "34ABC123", 0.88, 10.1)
        tracker.record_ocr_result(right_id, "35XYZ789", 0.91, 10.1)
        snapshots = {item.track_id: item for item in tracker.active_snapshots(1)}

        self.assertEqual(snapshots[left_id].best_text, "34ABC123")
        self.assertEqual(snapshots[right_id].best_text, "35XYZ789")

    def test_retired_track_accepts_only_its_pending_result_then_becomes_stale(self) -> None:
        tracker = manager(timeout_ms=500)
        track_id = tracker.update(1, [detection(100)], 10.0).assignments[0].track_id
        self.assertTrue(tracker.mark_ocr_scheduled(track_id))

        tracker.expire_due(10.6)
        self.assertTrue(tracker.can_accept_ocr_result(track_id))
        self.assertTrue(
            tracker.record_ocr_result(track_id, "34ABC123", 0.90, 10.1)
        )
        tracker.mark_ocr_finished(track_id)
        finalized = tracker.consume_finalized()

        self.assertEqual(finalized[0].best_text, "34ABC123")
        self.assertFalse(tracker.can_accept_ocr_result(track_id))
        self.assertFalse(
            tracker.record_ocr_result(track_id, "35XYZ789", 0.99, 11.0)
        )

    def test_delayed_frame_uses_processing_activity_for_timeout(self) -> None:
        tracker = manager(timeout_ms=1_200)
        first = tracker.update(
            1,
            [detection(100)],
            observed_at=10.0,
            activity_at=100.0,
        )
        track_id = first.assignments[0].track_id

        tracker.expire_due(100.8)
        second = tracker.update(
            1,
            [detection(108)],
            observed_at=10.2,
            activity_at=101.0,
        )

        self.assertEqual(second.assignments[0].track_id, track_id)
        self.assertEqual(len(second.active_tracks), 1)
        tracker.expire_due(102.19)
        self.assertEqual(len(tracker.active_snapshots(1)), 1)
        tracker.expire_due(102.21)
        self.assertEqual(len(tracker.active_snapshots(1)), 0)

    def test_older_replay_frame_refreshes_activity_without_rolling_back_bbox(self) -> None:
        tracker = manager(timeout_ms=1_200)
        first = tracker.update(
            1,
            [detection(100)],
            observed_at=20.0,
            activity_at=100.0,
        )
        track_id = first.assignments[0].track_id

        replay = tracker.update(
            1,
            [detection(105)],
            observed_at=19.0,
            activity_at=100.8,
        )
        snapshot = replay.active_tracks[0]

        self.assertEqual(replay.assignments[0].track_id, track_id)
        self.assertEqual(snapshot.bbox, (100, 20, 100, 30))
        tracker.expire_due(101.9)
        self.assertEqual(len(tracker.active_snapshots(1)), 1)


class TrackAwareOcrSchedulingTests(unittest.TestCase):
    @staticmethod
    def _job(track_id: int, quality: float) -> OcrJob:
        now = time.monotonic()
        frame = np.zeros((80, 200, 3), dtype=np.uint8)
        crop = np.zeros((20, 90, 3), dtype=np.uint8)
        return OcrJob(
            camera_id=1,
            direction=Direction.ENTRY,
            captured_at=datetime.now(timezone.utc),
            observed_at=now,
            received_at=now,
            queued_at=now,
            full_frame=frame,
            roi_crop=frame,
            ocr_crops=(crop,),
            detections=(detection(track_id * 200),),
            used_roi_fallback=False,
            fallback_reason=None,
            detector_ms=1.0,
            quality_score=quality,
            job_type=OcrJobType.DETECTOR_CROP,
            frame_id=track_id,
            track_id=track_id,
        )

    def test_pending_job_from_one_track_does_not_block_second_track(self) -> None:
        buffer = OcrJobBuffer(max_per_camera=3, max_age_ms=2_500)

        first = buffer.add(self._job(1, 1.0))
        same_track = buffer.add(self._job(1, 0.5))
        other_track = buffer.add(self._job(2, 0.8))

        self.assertTrue(first.accepted)
        self.assertFalse(same_track.accepted)
        self.assertEqual(same_track.drop_reason, "track-pending")
        self.assertTrue(other_track.accepted)
        self.assertEqual(buffer.pending_track_count(1, 1), 1)
        self.assertEqual(buffer.pending_track_count(1, 2), 1)

    def test_detector_frame_creates_one_ocr_job_per_track(self) -> None:
        config = PlateRecognitionConfig(
            recognition_interval_ms=250,
            min_confidence=0.65,
            confirmations_required=2,
            confirmation_window_seconds=3.0,
            duplicate_cooldown_seconds=120,
            entry_roi=DEFAULT_ROI,
            exit_roi=DEFAULT_ROI,
            model_root=Path("models/ocr"),
        )

        class TwoPlateDetector:
            last_diagnostics = None

            def detect(self, _image: np.ndarray) -> list[PlateDetection]:
                return [detection(20), detection(220)]

        tracker = manager()
        processor = PlateDetectionProcessor(config, TwoPlateDetector(), tracker)
        frame = np.zeros((200, 500, 3), dtype=np.uint8)

        result = processor.prepare_job(
            1,
            Direction.ENTRY,
            frame,
            captured_at=datetime.now(timezone.utc),
            observed_at=10.0,
            received_at=10.0,
            frame_id=7,
        )

        self.assertEqual(len(result.jobs), 2)
        self.assertEqual(len({job.track_id for job in result.jobs}), 2)
        self.assertTrue(all(len(job.ocr_crops) == 1 for job in result.jobs))
        self.assertTrue(all(job.frame_id == 7 for job in result.jobs))


if __name__ == "__main__":
    unittest.main()
