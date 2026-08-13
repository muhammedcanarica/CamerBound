from __future__ import annotations

import tempfile
import unittest
import threading
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PySide6.QtCore import Qt

from app.auth import AuthService
from app.camera import CameraService, Direction
from app.config import (
    DEFAULT_ROI,
    NormalizedRoi,
    PlateDetectorConfig,
    PlateRecognitionConfig,
)
from app.database import Database
from app.plate_capture import PlateCaptureService
from app.plate_detector import PlateDetection, PlateDetectorError
from app.plate_recognition import (
    ConfirmationTracker,
    FrameSnapshot,
    LOW_LIGHT_THRESHOLD,
    MotionEvent,
    OcrModelNotFound,
    OcrSegment,
    OcrJob,
    OcrJobBuffer,
    OcrJobType,
    PreDetectionFrameBuffer,
    PaddleOcrProvider,
    PlateCandidate,
    PlateDetectionProcessor,
    PlateOcrWorker,
    PlatePresenceTracker,
    PlateRecognitionProcessor,
    PlateRecognitionService,
    PlateRecognitionWorker,
    ReplayEventBuffer,
    RecognitionState,
    TurkishPlateValidator,
    correct_plate_candidate,
    normalize_plate_text,
    crop_roi,
    preprocess_variants,
    preprocess_roi_fallback_variants,
    roi_mean_brightness,
    select_replay_frames,
    select_best_candidate,
    RecognitionStatus,
    _PendingFrame,
)
from app.ocr_models import OcrModelInvalid, OcrModelNotFound as ModelNotFound, validate_model_directory
from app.ocr_debug import save_debug_images
from app.plate_service import PlateService


class FakeOcrProvider:
    def __init__(self, text: str = "34 ABC 123", confidence: float = 0.75) -> None:
        self.text = text
        self.confidence = confidence

    def recognize(self, _images: object) -> list[OcrSegment]:
        return [OcrSegment(self.text, self.confidence, (0.0, 0.0, 100.0, 30.0))]


class RecoveringOcrProvider(FakeOcrProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def recognize(self, images: object) -> list[OcrSegment]:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary")
        return super().recognize(images)


class EmptyOcrProvider:
    def recognize(self, _images: object) -> list[OcrSegment]:
        return []


class SequencedOcrProvider:
    def __init__(self, batches: list[list[OcrSegment]]) -> None:
        self.batches = list(batches)
        self.calls: list[list[np.ndarray]] = []

    def recognize(self, images: object) -> list[OcrSegment]:
        self.calls.append(list(images))
        return self.batches.pop(0) if self.batches else []


class RecordingOcrProvider(FakeOcrProvider):
    def __init__(self, confidence: float = 0.97) -> None:
        super().__init__(confidence=confidence)
        self.images: list[np.ndarray] = []
        self.calls = 0

    def recognize(self, images: object) -> list[OcrSegment]:
        self.calls += 1
        self.images = list(images)
        return super().recognize(images)


class BlockingOcrProvider(FakeOcrProvider):
    def __init__(self, started: threading.Event, release: threading.Event) -> None:
        super().__init__(confidence=0.97)
        self.started = started
        self.release = release
        self.thread_ids: list[int] = []

    def recognize(self, images: object) -> list[OcrSegment]:
        self.thread_ids.append(threading.get_ident())
        self.started.set()
        if not self.release.wait(2.0):
            raise RuntimeError("test OCR release timeout")
        return super().recognize(images)


class FakePlateDetector:
    def __init__(
        self,
        detections: list[PlateDetection] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.detections = detections or []
        self.error = error
        self.calls = 0

    def detect(self, _image: np.ndarray) -> list[PlateDetection]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return list(self.detections)


def recognition_config(model_root: Path, confirmations: int = 2) -> PlateRecognitionConfig:
    return PlateRecognitionConfig(
        recognition_interval_ms=250,
        min_confidence=0.65,
        confirmations_required=confirmations,
        confirmation_window_seconds=3.0,
        duplicate_cooldown_seconds=10,
        entry_roi=DEFAULT_ROI,
        exit_roi=DEFAULT_ROI,
        model_root=model_root,
    )


class PlateTextTests(unittest.TestCase):
    def test_normalization(self) -> None:
        self.assertEqual(normalize_plate_text("34 abc 123"), "34ABC123")
        self.assertEqual(normalize_plate_text("34-ABC-123"), "34ABC123")

    def test_turkish_plate_validation(self) -> None:
        for plate in ("34ABC123", "06AA1234", "35A1234", "23AB123"):
            with self.subTest(plate=plate):
                self.assertTrue(TurkishPlateValidator.is_valid(plate))
        for plate in ("99ABC123", "ABC123", "34ABC", "HELLO123", "001ABC12"):
            with self.subTest(plate=plate):
                self.assertFalse(TurkishPlateValidator.is_valid(plate))

    def test_position_aware_ocr_correction(self) -> None:
        self.assertEqual(correct_plate_candidate("O6ABC123"), "06ABC123")
        self.assertEqual(correct_plate_candidate("35ABCIZ34"), "35ABC1234")
        self.assertEqual(correct_plate_candidate("34A0C123"), "34AOC123")
        self.assertIsNone(correct_plate_candidate("99ABC123"))

    def test_confirmation_requires_enough_observations_and_emits_once(self) -> None:
        tracker = ConfirmationTracker(required=2, window_seconds=3.0)
        candidate = PlateCandidate("34ABC123", 0.9, "34 ABC 123", 7)

        self.assertIsNone(tracker.observe(candidate, 10.0))
        confirmed = tracker.observe(candidate, 11.0)
        self.assertIsNotNone(confirmed)
        self.assertIsNone(tracker.observe(candidate, 12.0))

    def test_presence_record_claim_is_thread_safe(self) -> None:
        tracker = PlatePresenceTracker()
        candidate = PlateCandidate("34ABC123", 0.9, "34 ABC 123", 7)
        tracker.observe(candidate, 10.0)
        barrier = threading.Barrier(8)
        results: list[bool] = []

        def claim() -> None:
            barrier.wait()
            results.append(tracker.claim_record(candidate))

        threads = [threading.Thread(target=claim) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(results.count(True), 1)

    def test_presence_last_seen_never_moves_backwards_for_replay(self) -> None:
        tracker = PlatePresenceTracker()
        candidate = PlateCandidate("34ABC123", 0.90, "34ABC123", camera_id=1)

        tracker.observe(candidate, 20.0)
        tracker.observe(candidate, 10.0)

        self.assertEqual(tracker._presences[(1, "34ABC123")].last_seen, 20.0)

    def test_out_of_order_observation_outside_window_does_not_confirm(self) -> None:
        tracker = ConfirmationTracker(required=2, window_seconds=3.0)
        candidate = PlateCandidate("34ABC123", 0.90, "34ABC123", camera_id=1)

        newer = tracker.observe_progress(candidate, 10.0)
        too_old = tracker.observe_progress(candidate, 6.9)

        self.assertIsNone(newer.candidate)
        self.assertIsNone(too_old.candidate)
        self.assertEqual(too_old.observed_count, 1)

    def test_multiple_ocr_segments_are_combined_left_to_right(self) -> None:
        segments = [
            OcrSegment("123", 0.90, (200.0, 0.0, 260.0, 30.0)),
            OcrSegment("34", 0.94, (0.0, 0.0, 40.0, 30.0)),
            OcrSegment("ABC", 0.92, (60.0, 0.0, 160.0, 30.0)),
        ]

        candidate = select_best_candidate(segments, camera_id=5)

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.plate, "34ABC123")

    def test_same_plate_from_multiple_variants_keeps_best_confidence(self) -> None:
        segments = [
            OcrSegment("34ABC123", 0.81, (0.0, 0.0, 100.0, 30.0), 0),
            OcrSegment("34 ABC 123", 0.94, (0.0, 0.0, 200.0, 60.0), 2),
        ]

        candidate = select_best_candidate(segments, camera_id=5)

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.plate, "34ABC123")
        self.assertEqual(candidate.confidence, 0.94)


class PreprocessingTests(unittest.TestCase):
    def test_dark_roi_is_detected_as_low_light(self) -> None:
        crop = np.full((40, 120, 3), 40, dtype=np.uint8)

        brightness = roi_mean_brightness(crop)

        self.assertLess(brightness, LOW_LIGHT_THRESHOLD)

    def test_bright_roi_is_not_detected_as_low_light(self) -> None:
        crop = np.full((40, 120, 3), 160, dtype=np.uint8)

        brightness = roi_mean_brightness(crop)

        self.assertGreaterEqual(brightness, LOW_LIGHT_THRESHOLD)

    def test_low_light_variant_is_only_added_for_dark_roi(self) -> None:
        dark_crop = np.full((40, 120, 3), 40, dtype=np.uint8)
        bright_crop = np.full((40, 120, 3), 160, dtype=np.uint8)

        dark_variants = preprocess_variants(dark_crop)
        bright_variants = preprocess_variants(bright_crop)

        self.assertEqual(len(dark_variants), 4)
        self.assertEqual(len(bright_variants), 3)
        self.assertEqual(dark_variants[-1].shape, dark_crop.shape)
        self.assertEqual(dark_variants[-1].ndim, 3)

    def test_existing_variants_are_preserved_and_third_variant_is_exactly_2x(self) -> None:
        generator = np.random.default_rng(7)
        crop = generator.integers(0, 256, (120, 400, 3), dtype=np.uint8)
        expected_original = cv2.resize(
            crop,
            None,
            fx=1.5,
            fy=1.5,
            interpolation=cv2.INTER_CUBIC,
        )
        expected_gray = cv2.cvtColor(expected_original, cv2.COLOR_BGR2GRAY)
        expected_contrasted = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        ).apply(expected_gray)

        variants = preprocess_variants(crop)

        self.assertEqual(len(variants), 3)
        np.testing.assert_array_equal(variants[0], expected_original)
        np.testing.assert_array_equal(
            variants[1],
            cv2.cvtColor(expected_contrasted, cv2.COLOR_GRAY2BGR),
        )
        self.assertEqual(variants[2].shape, (240, 800, 3))

    def test_upscale_does_not_mutate_input_crop(self) -> None:
        crop = np.full((40, 80, 3), 127, dtype=np.uint8)
        original_copy = crop.copy()

        variants = preprocess_variants(crop)

        np.testing.assert_array_equal(crop, original_copy)
        self.assertFalse(np.shares_memory(crop, variants[2]))

    def test_low_light_preprocessing_does_not_mutate_input_frame(self) -> None:
        frame = np.full((100, 200, 3), 40, dtype=np.uint8)
        original_frame = frame.copy()
        roi = NormalizedRoi(x=0.1, y=0.2, width=0.6, height=0.5)

        crop = crop_roi(frame, roi)
        variants = preprocess_variants(crop)

        np.testing.assert_array_equal(frame, original_frame)
        self.assertEqual(len(variants), 4)

    def test_tiny_clamped_roi_does_not_crash_preprocessing(self) -> None:
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        roi = NormalizedRoi(x=-1.0, y=-1.0, width=0.0, height=0.0)

        crop = crop_roi(frame, roi)
        variants = preprocess_variants(crop)

        self.assertEqual(crop.shape, (1, 1, 3))
        self.assertEqual(len(variants), 4)
        self.assertEqual(variants[2].shape, (2, 2, 3))

    def test_full_roi_fallback_variants_are_compact_and_bounded(self) -> None:
        roi = np.full((672, 1_912, 3), 80, dtype=np.uint8)

        fallback_variants = preprocess_roi_fallback_variants(roi)
        detector_crop_variants = preprocess_variants(roi)

        self.assertEqual(len(fallback_variants), 2)
        self.assertTrue(all(variant.shape[1] <= 960 for variant in fallback_variants))
        fallback_pixels = sum(
            variant.shape[0] * variant.shape[1] for variant in fallback_variants
        )
        detector_pixels = sum(
            variant.shape[0] * variant.shape[1] for variant in detector_crop_variants
        )
        self.assertLess(fallback_pixels, detector_pixels * 0.20)
        np.testing.assert_array_equal(roi, np.full_like(roi, 80))


class RecognitionPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        root = Path(self.temp_directory.name)
        self.database = Database(root / "recognition.db")
        self.database.initialize()
        auth_service = AuthService(self.database)
        auth_service.ensure_default_admin()
        self.camera_service = CameraService(self.database)
        self.capture_service = PlateCaptureService(
            root / "data" / "captures",
            root,
        )
        self.plate_service = PlateService(
            self.database,
            duplicate_cooldown_seconds=10,
            capture_service=self.capture_service,
        )
        self.config = recognition_config(root)

    def tearDown(self) -> None:
        self.camera_service.stop_all()
        self.temp_directory.cleanup()

    def _make_ocr_job(
        self,
        *,
        camera_id: int = 1,
        observed_at: float | None = None,
        quality_score: float = 1.0,
        frame_value: int = 0,
        fallback_reason: str | None = None,
        frame_id: int | None = None,
        queued_at: float | None = None,
        job_type: OcrJobType | None = None,
        detector_source: str = "live",
        captured_at: datetime | None = None,
    ) -> OcrJob:
        now = time.monotonic() if observed_at is None else observed_at
        frame = np.full((200, 400, 3), frame_value, dtype=np.uint8)
        crop = np.full((40, 120, 3), max(1, frame_value), dtype=np.uint8)
        return OcrJob(
            camera_id=camera_id,
            direction=Direction.ENTRY if camera_id == 1 else Direction.EXIT,
            captured_at=(
                captured_at
                if captured_at is not None
                else datetime(2026, 8, 13, 9, 0, tzinfo=timezone.utc)
            ),
            observed_at=now,
            received_at=now,
            queued_at=time.monotonic() if queued_at is None else queued_at,
            full_frame=frame,
            roi_crop=crop,
            ocr_crops=(crop.copy(),),
            detections=(),
            used_roi_fallback=fallback_reason is not None,
            fallback_reason=fallback_reason,
            detector_ms=12.0,
            quality_score=quality_score,
            job_type=(
                job_type
                if job_type is not None
                else OcrJobType.ZERO_DETECTION_FALLBACK
                if fallback_reason == "zero-detection"
                else OcrJobType.DETECTOR_ERROR_FALLBACK
                if fallback_reason is not None
                else OcrJobType.DETECTOR_CROP
            ),
            frame_id=frame_id,
            detector_source=detector_source,
        )

    def _make_snapshot(
        self,
        frame_id: int,
        observed_at: float,
        *,
        camera_id: int = 1,
        value: int = 0,
    ) -> FrameSnapshot:
        frame = np.full((200, 400, 3), value, dtype=np.uint8)
        return FrameSnapshot(
            frame_id=frame_id,
            camera_id=camera_id,
            direction=Direction.ENTRY if camera_id == 1 else Direction.EXIT,
            captured_at=datetime(2026, 8, 13, tzinfo=timezone.utc)
            + timedelta(seconds=observed_at),
            observed_at=observed_at,
            received_at=observed_at,
            full_frame=frame,
        )

    def test_processor_saves_only_after_confirmation(self) -> None:
        camera = self.camera_service.list_cameras()[0]
        processor = PlateRecognitionProcessor(
            FakeOcrProvider(confidence=0.89), self.plate_service, self.config
        )
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        detected_at = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)

        first = processor.process(
            camera.id, camera.direction, frame, detected_at, monotonic_at=1.0
        )
        self.assertEqual(list(Path(self.temp_directory.name).rglob("*.jpg")), [])
        second = processor.process(
            camera.id, camera.direction, frame, detected_at, monotonic_at=2.0
        )

        self.assertIsNone(first.record)
        self.assertIs(first.state, RecognitionState.AWAITING_CONFIRMATION)
        self.assertEqual((first.confirmation_count, first.confirmation_required), (1, 2))
        self.assertIsNotNone(second.record)
        self.assertIs(second.state, RecognitionState.SAVED)
        self.assertEqual(second.record.plate, "34ABC123")
        self.assertEqual(second.record.direction, Direction.ENTRY)
        self.assertIsNotNone(second.record.image_path)
        self.assertTrue(
            self.capture_service.resolve_reference(second.record.image_path).is_file()
        )

    def test_very_high_confidence_still_requires_two_observations(self) -> None:
        camera = self.camera_service.list_cameras()[0]
        processor = PlateRecognitionProcessor(
            FakeOcrProvider(confidence=0.99), self.plate_service, self.config
        )

        with patch.object(
            self.plate_service,
            "save_plate_detection",
            wraps=self.plate_service.save_plate_detection,
        ) as save_plate_detection:
            first = processor.process(
                camera.id,
                camera.direction,
                np.zeros((200, 400, 3), dtype=np.uint8),
                monotonic_at=1.0,
            )
            second = processor.process(
                camera.id,
                camera.direction,
                np.zeros((200, 400, 3), dtype=np.uint8),
                monotonic_at=2.0,
            )

        self.assertIs(first.state, RecognitionState.AWAITING_CONFIRMATION)
        self.assertEqual((first.confirmation_count, first.confirmation_required), (1, 2))
        self.assertIsNone(first.record)
        save_plate_detection.assert_called_once()
        self.assertIs(second.state, RecognitionState.SAVED)
        self.assertEqual((second.confirmation_count, second.confirmation_required), (2, 2))
        self.assertIsNotNone(second.record)
        self.assertEqual(self._record_count(), 1)

    def test_low_confidence_and_invalid_plate_have_distinct_outcomes(self) -> None:
        camera = self.camera_service.list_cameras()[0]
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        low = PlateRecognitionProcessor(
            FakeOcrProvider(confidence=0.64), self.plate_service, self.config
        ).process(camera.id, camera.direction, frame, monotonic_at=1.0)
        invalid = PlateRecognitionProcessor(
            FakeOcrProvider(text="99 ABC 123", confidence=0.99),
            self.plate_service,
            self.config,
        ).process(camera.id, camera.direction, frame, monotonic_at=2.0)
        no_text = PlateRecognitionProcessor(
            EmptyOcrProvider(), self.plate_service, self.config
        ).process(camera.id, camera.direction, frame, monotonic_at=3.0)

        self.assertIs(low.state, RecognitionState.LOW_CONFIDENCE)
        self.assertIsNotNone(low.candidate)
        self.assertIs(invalid.state, RecognitionState.NO_VALID_PLATE)
        self.assertIsNone(invalid.candidate)
        self.assertIs(no_text.state, RecognitionState.NO_OCR_TEXT)
        self.assertEqual(self._record_count(), 0)

    def test_very_high_confidence_repeat_is_suppressed_by_presence(self) -> None:
        camera = self.camera_service.list_cameras()[0]
        processor = PlateRecognitionProcessor(
            FakeOcrProvider(confidence=0.97), self.plate_service, self.config
        )
        frame = np.zeros((200, 400, 3), dtype=np.uint8)

        first = processor.process(
            camera.id, camera.direction, frame, monotonic_at=1.0
        )
        saved = processor.process(
            camera.id, camera.direction, frame, monotonic_at=2.0
        )
        awaiting_again = processor.process(
            camera.id, camera.direction, frame, monotonic_at=3.0
        )
        processor.process(camera.id, camera.direction, frame, monotonic_at=6.0)
        duplicate = processor.process(
            camera.id, camera.direction, frame, monotonic_at=7.0
        )

        self.assertIs(first.state, RecognitionState.AWAITING_CONFIRMATION)
        self.assertIs(saved.state, RecognitionState.SAVED)
        self.assertIs(awaiting_again.state, RecognitionState.AWAITING_CONFIRMATION)
        self.assertIs(duplicate.state, RecognitionState.DUPLICATE_SUPPRESSED)
        self.assertTrue(duplicate.duplicate)
        self.assertEqual(self._record_count(), 1)
        self.assertEqual(len(list(Path(self.temp_directory.name).rglob("*.jpg"))), 1)

    def test_detector_disabled_preserves_existing_roi_ocr_pipeline(self) -> None:
        camera = self.camera_service.list_cameras()[0]
        provider = RecordingOcrProvider()
        config = replace(
            self.config,
            plate_detector=replace(self.config.plate_detector, enabled=False),
        )
        processor = PlateRecognitionProcessor(provider, self.plate_service, config)

        outcome = processor.process(
            camera.id,
            camera.direction,
            np.zeros((200, 400, 3), dtype=np.uint8),
            monotonic_at=1.0,
        )

        self.assertEqual(provider.calls, 1)
        self.assertEqual(len(provider.images), 4)
        self.assertTrue(outcome.used_roi_fallback)

    def test_detector_plate_crop_is_sent_to_ocr_and_full_frame_is_saved(self) -> None:
        camera = self.camera_service.list_cameras()[0]
        provider = RecordingOcrProvider()
        detector = FakePlateDetector(
            [PlateDetection(0.91, x=20, y=10, width=100, height=30)]
        )
        config = replace(
            self.config,
            plate_detector=replace(
                self.config.plate_detector,
                crop_padding_ratio=0.0,
            ),
        )
        processor = PlateRecognitionProcessor(
            provider,
            self.plate_service,
            config,
            detector=detector,
        )
        frame = np.zeros((200, 400, 3), dtype=np.uint8)

        with patch.object(
            self.plate_service,
            "save_plate_detection",
            wraps=self.plate_service.save_plate_detection,
        ) as save_plate_detection:
            first = processor.process(
                camera.id,
                camera.direction,
                frame,
                monotonic_at=1.0,
            )
            outcome = processor.process(
                camera.id,
                camera.direction,
                frame,
                monotonic_at=2.0,
            )

        self.assertIs(first.state, RecognitionState.AWAITING_CONFIRMATION)
        self.assertEqual(detector.calls, 2)
        self.assertEqual(provider.calls, 2)
        self.assertEqual(len(provider.images), 4)
        self.assertEqual(provider.images[2].shape, (60, 200, 3))
        self.assertNotEqual(provider.images[0].shape[:2], (110, 320))
        self.assertFalse(outcome.used_roi_fallback)
        self.assertEqual(outcome.detections, tuple(detector.detections))
        self.assertIs(save_plate_detection.call_args.args[4], frame)

    def test_detector_limits_ocr_to_two_ranked_plate_crops(self) -> None:
        camera = self.camera_service.list_cameras()[0]
        provider = RecordingOcrProvider()
        detector = FakePlateDetector(
            [
                PlateDetection(0.70, 0, 0, 80, 20),
                PlateDetection(0.95, 20, 10, 60, 20),
                PlateDetection(0.85, 100, 20, 70, 20),
            ]
        )
        processor = PlateRecognitionProcessor(
            provider,
            self.plate_service,
            self.config,
            detector=detector,
        )

        processor.process(
            camera.id,
            camera.direction,
            np.full((200, 400, 3), 160, dtype=np.uint8),
            monotonic_at=1.0,
        )

        self.assertEqual(provider.calls, 1)
        self.assertEqual(len(provider.images), 6)

    def test_detector_runtime_failure_uses_roi_fallback_without_reinitializing(self) -> None:
        camera = self.camera_service.list_cameras()[0]
        provider = RecordingOcrProvider()
        detector = FakePlateDetector(error=PlateDetectorError("temporary"))
        processor = PlateRecognitionProcessor(
            provider,
            self.plate_service,
            self.config,
            detector=detector,
        )
        frame = np.zeros((200, 400, 3), dtype=np.uint8)

        with self.assertLogs("app.plate_recognition", level="DEBUG") as captured:
            first = processor.process(
                camera.id,
                camera.direction,
                frame,
                monotonic_at=1.0,
            )
            second = processor.process(
                camera.id,
                camera.direction,
                frame,
                monotonic_at=2.0,
            )

        self.assertEqual(detector.calls, 2)
        self.assertEqual(provider.calls, 2)
        self.assertTrue(first.used_roi_fallback)
        self.assertTrue(second.used_roi_fallback)
        diagnostic = next(
            line for line in captured.output if "OCR diagnostics" in line
        )
        self.assertIn("fallback_reason=detector-error", diagnostic)

    def test_zero_detection_uses_roi_fallback_when_interval_allows(self) -> None:
        camera = self.camera_service.list_cameras()[0]
        provider = RecordingOcrProvider()
        detector = FakePlateDetector([])
        processor = PlateRecognitionProcessor(
            provider,
            self.plate_service,
            self.config,
            detector=detector,
        )

        with self.assertLogs("app.plate_recognition", level="DEBUG") as captured:
            outcome = processor.process(
                camera.id,
                camera.direction,
                np.zeros((200, 400, 3), dtype=np.uint8),
                monotonic_at=1.0,
            )

        self.assertEqual(provider.calls, 1)
        self.assertEqual(provider.images[0].shape, (206, 600, 3))
        self.assertEqual(provider.images[2].shape, (220, 640, 3))
        self.assertTrue(outcome.used_roi_fallback)
        diagnostic = next(
            line for line in captured.output if "OCR diagnostics" in line
        )
        self.assertIn("fallback_reason=zero-detection", diagnostic)

    def test_unusable_detector_crop_uses_zero_detection_fallback(self) -> None:
        camera = self.camera_service.list_cameras()[0]
        provider = RecordingOcrProvider()
        processor = PlateRecognitionProcessor(
            provider,
            self.plate_service,
            self.config,
            detector=FakePlateDetector(
                [PlateDetection(0.90, x=20, y=10, width=1, height=1)]
            ),
        )

        outcome = processor.process(
            camera.id,
            camera.direction,
            np.zeros((200, 400, 3), dtype=np.uint8),
            monotonic_at=1.0,
        )

        self.assertEqual(provider.calls, 1)
        self.assertTrue(outcome.used_roi_fallback)
        self.assertEqual(len(outcome.detections), 1)

    def test_zero_detection_fallback_is_throttled_and_runs_again_after_interval(self) -> None:
        camera = self.camera_service.list_cameras()[0]
        provider = RecordingOcrProvider()
        processor = PlateRecognitionProcessor(
            provider,
            self.plate_service,
            self.config,
            detector=FakePlateDetector([]),
        )
        frame = np.zeros((200, 400, 3), dtype=np.uint8)

        first = processor.process(
            camera.id, camera.direction, frame, monotonic_at=1.0
        )
        throttled = processor.process(
            camera.id, camera.direction, frame, monotonic_at=1.5
        )
        after_interval = processor.process(
            camera.id, camera.direction, frame, monotonic_at=1.75
        )

        self.assertTrue(first.used_roi_fallback)
        self.assertIs(throttled.state, RecognitionState.NO_OCR_TEXT)
        self.assertFalse(throttled.used_roi_fallback)
        self.assertTrue(after_interval.used_roi_fallback)
        self.assertEqual(provider.calls, 2)

    def test_zero_detection_fallback_throttle_is_camera_specific(self) -> None:
        cameras = self.camera_service.list_cameras()
        entry = next(camera for camera in cameras if camera.direction is Direction.ENTRY)
        exit_camera = next(camera for camera in cameras if camera.direction is Direction.EXIT)
        provider = RecordingOcrProvider()
        processor = PlateRecognitionProcessor(
            provider,
            self.plate_service,
            self.config,
            detector=FakePlateDetector([]),
        )
        frame = np.zeros((200, 400, 3), dtype=np.uint8)

        entry_outcome = processor.process(
            entry.id, entry.direction, frame, monotonic_at=1.0
        )
        exit_outcome = processor.process(
            exit_camera.id, exit_camera.direction, frame, monotonic_at=1.1
        )
        entry_throttled = processor.process(
            entry.id, entry.direction, frame, monotonic_at=1.2
        )

        self.assertTrue(entry_outcome.used_roi_fallback)
        self.assertTrue(exit_outcome.used_roi_fallback)
        self.assertFalse(entry_throttled.used_roi_fallback)
        self.assertEqual(provider.calls, 2)

    def test_zero_detection_fallback_can_be_disabled(self) -> None:
        camera = self.camera_service.list_cameras()[0]
        provider = RecordingOcrProvider()
        config = replace(
            self.config,
            plate_detector=replace(
                self.config.plate_detector,
                zero_detection_roi_fallback_enabled=False,
            ),
        )
        processor = PlateRecognitionProcessor(
            provider,
            self.plate_service,
            config,
            detector=FakePlateDetector([]),
        )

        outcome = processor.process(
            camera.id,
            camera.direction,
            np.zeros((200, 400, 3), dtype=np.uint8),
            monotonic_at=1.0,
        )

        self.assertEqual(provider.calls, 0)
        self.assertIs(outcome.state, RecognitionState.NO_OCR_TEXT)
        self.assertFalse(outcome.used_roi_fallback)

    def test_zero_detection_fallback_candidate_uses_standard_confirmation(self) -> None:
        camera = self.camera_service.list_cameras()[0]
        provider = RecordingOcrProvider(confidence=0.99)
        processor = PlateRecognitionProcessor(
            provider,
            self.plate_service,
            self.config,
            detector=FakePlateDetector([]),
        )
        frame = np.zeros((200, 400, 3), dtype=np.uint8)

        first = processor.process(
            camera.id, camera.direction, frame, monotonic_at=1.0
        )
        second = processor.process(
            camera.id, camera.direction, frame, monotonic_at=1.75
        )

        self.assertIs(first.state, RecognitionState.AWAITING_CONFIRMATION)
        self.assertEqual((first.confirmation_count, first.confirmation_required), (1, 2))
        self.assertIs(second.state, RecognitionState.SAVED)
        self.assertEqual((second.confirmation_count, second.confirmation_required), (2, 2))
        self.assertEqual(self._record_count(), 1)

    def test_detector_pipeline_does_not_change_database_schema(self) -> None:
        with self.database.connection() as connection:
            before = connection.execute(
                "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
            ).fetchall()
        camera = self.camera_service.list_cameras()[0]
        processor = PlateRecognitionProcessor(
            RecordingOcrProvider(),
            self.plate_service,
            self.config,
            detector=FakePlateDetector([]),
        )

        processor.process(
            camera.id,
            camera.direction,
            np.zeros((200, 400, 3), dtype=np.uint8),
            monotonic_at=1.0,
        )

        with self.database.connection() as connection:
            after = connection.execute(
                "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
            ).fetchall()
        self.assertEqual([tuple(row) for row in after], [tuple(row) for row in before])

    def test_ocr_diagnostics_are_throttled_and_contain_safe_dimensions(self) -> None:
        camera = self.camera_service.list_cameras()[0]
        processor = PlateRecognitionProcessor(
            FakeOcrProvider(), self.plate_service, self.config
        )
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        with patch(
            "app.plate_recognition.time.perf_counter",
            side_effect=(10.0, 10.02, 10.1, 11.0, 11.02, 11.1),
        ), self.assertLogs("app.plate_recognition", level="DEBUG") as captured:
            processor.process(
                camera.id,
                camera.direction,
                frame,
                monotonic_at=1.0,
                frame_received_at=9.95,
            )
            processor.process(
                camera.id,
                camera.direction,
                frame,
                monotonic_at=2.0,
            )

        diagnostic_lines = [
            line for line in captured.output if "OCR diagnostics" in line
        ]
        self.assertEqual(len(diagnostic_lines), 1)
        diagnostic = diagnostic_lines[0]
        self.assertIn(f"camera_id={camera.id}", diagnostic)
        self.assertIn("direction=ENTRY", diagnostic)
        self.assertIn("frame=400x200", diagnostic)
        self.assertIn("roi=320x110", diagnostic)
        self.assertIn("brightness=0.0", diagnostic)
        self.assertIn("low_light=yes", diagnostic)
        self.assertIn("variants=4", diagnostic)
        self.assertIn("detector_ms=0.0", diagnostic)
        self.assertIn("plates=0", diagnostic)
        self.assertIn("det_conf=none", diagnostic)
        self.assertIn("fallback=roi", diagnostic)
        self.assertIn("total_recognition_ms=100.0", diagnostic)
        self.assertIn("processing_ms=100.0", diagnostic)
        self.assertIn("inference_ms=80.0", diagnostic)
        self.assertIn("frame_wait_ms=50.0", diagnostic)
        self.assertIn("candidate=yes", diagnostic)

    def test_detector_diagnostics_include_latency_count_confidence_and_crop(self) -> None:
        camera = self.camera_service.list_cameras()[0]
        processor = PlateRecognitionProcessor(
            FakeOcrProvider(),
            self.plate_service,
            self.config,
            detector=FakePlateDetector(
                [PlateDetection(0.88, x=20, y=10, width=100, height=30)]
            ),
        )
        frame = np.zeros((200, 400, 3), dtype=np.uint8)

        with patch(
            "app.plate_recognition.time.perf_counter",
            side_effect=(10.0, 10.012, 10.020, 10.095),
        ), self.assertLogs("app.plate_recognition", level="DEBUG") as captured:
            processor.process(
                camera.id,
                camera.direction,
                frame,
                monotonic_at=1.0,
            )

        diagnostic = next(
            line for line in captured.output if "OCR diagnostics" in line
        )
        self.assertIn("direction=ENTRY", diagnostic)
        self.assertIn("roi=320x110", diagnostic)
        self.assertIn("detector_ms=12.0", diagnostic)
        self.assertIn("plates=1", diagnostic)
        self.assertIn("det_conf=0.880", diagnostic)
        self.assertIn("plate_crops=130x40", diagnostic)
        self.assertIn("ocr_ms=83.0", diagnostic)
        self.assertIn("total_recognition_ms=95.0", diagnostic)
        self.assertIn("candidate=yes", diagnostic)

    def test_duplicate_after_a_new_confirmation_does_not_store_a_second_image(self) -> None:
        camera = self.camera_service.list_cameras()[0]
        processor = PlateRecognitionProcessor(
            FakeOcrProvider(), self.plate_service, self.config
        )
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        detected_at = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
        processor.process(
            camera.id, camera.direction, frame, detected_at, monotonic_at=1.0
        )
        saved = processor.process(
            camera.id, camera.direction, frame, detected_at, monotonic_at=2.0
        )

        processor.process(
            camera.id,
            camera.direction,
            frame,
            detected_at + timedelta(seconds=8),
            monotonic_at=10.0,
        )
        processor.process(
            camera.id,
            camera.direction,
            frame,
            detected_at + timedelta(seconds=18),
            monotonic_at=20.0,
        )
        processor.process(
            camera.id,
            camera.direction,
            frame,
            detected_at + timedelta(seconds=28),
            monotonic_at=30.0,
        )
        duplicate = processor.process(
            camera.id,
            camera.direction,
            frame,
            detected_at + timedelta(seconds=29),
            monotonic_at=31.0,
        )

        self.assertIsNotNone(saved.record)
        self.assertTrue(duplicate.duplicate)
        self.assertIsNone(duplicate.record)
        self.assertEqual(len(list(Path(self.temp_directory.name).rglob("*.jpg"))), 1)
        self.assertEqual(self._record_count(), 1)

    def test_candidate_updates_during_presence_without_creating_a_record(self) -> None:
        camera = self.camera_service.list_cameras()[0]
        processor = PlateRecognitionProcessor(
            FakeOcrProvider(), self.plate_service, self.config
        )
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        detected_at = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
        processor.process(
            camera.id, camera.direction, frame, detected_at, monotonic_at=1.0
        )
        processor.process(
            camera.id, camera.direction, frame, detected_at, monotonic_at=2.0
        )

        outcome = processor.process(
            camera.id,
            camera.direction,
            frame,
            detected_at + timedelta(seconds=1),
            monotonic_at=3.0,
        )

        self.assertIsNotNone(outcome.candidate)
        self.assertEqual(outcome.candidate.plate, "34ABC123")
        self.assertIsNone(outcome.record)
        self.assertEqual(self._record_count(), 1)

    def test_same_plate_can_be_recorded_again_after_presence_release(self) -> None:
        camera = self.camera_service.list_cameras()[0]
        processor = PlateRecognitionProcessor(
            FakeOcrProvider(confidence=0.97), self.plate_service, self.config
        )
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        detected_at = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
        processor.process(
            camera.id, camera.direction, frame, detected_at, monotonic_at=1.0
        )
        first = processor.process(
            camera.id, camera.direction, frame, detected_at, monotonic_at=2.0
        )
        processor.process(
            camera.id,
            camera.direction,
            frame,
            detected_at + timedelta(seconds=16),
            monotonic_at=18.0,
        )
        second = processor.process(
            camera.id,
            camera.direction,
            frame,
            detected_at + timedelta(seconds=17),
            monotonic_at=19.0,
        )

        self.assertIsNotNone(first.record)
        self.assertIsNotNone(second.record)
        self.assertEqual(self._record_count(), 2)
        self.assertEqual(len(list(Path(self.temp_directory.name).rglob("*.jpg"))), 2)

    def test_same_plate_is_independent_for_entry_and_exit_cameras(self) -> None:
        cameras = {
            camera.direction: camera for camera in self.camera_service.list_cameras()
        }
        processor = PlateRecognitionProcessor(
            FakeOcrProvider(confidence=0.97),
            self.plate_service,
            recognition_config(Path(self.temp_directory.name), confirmations=1),
            detector=FakePlateDetector(
                [PlateDetection(0.9, x=20, y=10, width=100, height=30)]
            ),
        )
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        detected_at = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)

        entry = processor.process(
            cameras[Direction.ENTRY].id,
            Direction.ENTRY,
            frame,
            detected_at,
            monotonic_at=1.0,
        )
        exit_record = processor.process(
            cameras[Direction.EXIT].id,
            Direction.EXIT,
            frame,
            detected_at,
            monotonic_at=2.0,
        )

        self.assertIsNotNone(entry.record)
        self.assertIsNotNone(exit_record.record)
        self.assertEqual(self._record_count(), 2)

    def test_different_plates_do_not_share_presence(self) -> None:
        camera = self.camera_service.list_cameras()[0]
        provider = FakeOcrProvider()
        processor = PlateRecognitionProcessor(
            provider,
            self.plate_service,
            recognition_config(Path(self.temp_directory.name), confirmations=1),
        )
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        detected_at = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
        first = processor.process(
            camera.id, camera.direction, frame, detected_at, monotonic_at=1.0
        )
        provider.text = "06 XYZ 99"
        second = processor.process(
            camera.id,
            camera.direction,
            frame,
            detected_at + timedelta(seconds=1),
            monotonic_at=2.0,
        )

        self.assertIsNotNone(first.record)
        self.assertIsNotNone(second.record)
        self.assertEqual(self._record_count(), 2)

    def test_worker_keeps_only_latest_frame_per_camera(self) -> None:
        worker = PlateRecognitionWorker(
            self.plate_service,
            self.config,
            provider_factory=FakeOcrProvider,
        )
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        for _ in range(100):
            worker.submit_frame(10, Direction.ENTRY, frame)
        self.assertEqual(worker.pending_frame_count, 1)
        worker.submit_frame(20, Direction.EXIT, frame)
        self.assertEqual(worker.pending_frame_count, 2)

    def test_analysis_ingestion_keeps_twenty_frames_when_ui_preview_is_coalesced(self) -> None:
        worker = PlateRecognitionWorker(
            self.plate_service,
            self.config,
            provider_factory=FakeOcrProvider,
            detector_factory=lambda: FakePlateDetector([]),
        )
        camera_id = 1
        preview_frames: list[np.ndarray] = []
        service = PlateRecognitionService(
            self.camera_service,
            self.plate_service,
            self.config,
            provider_factory=FakeOcrProvider,
            detector_factory=lambda: FakePlateDetector([]),
        )
        service._runtime = type("Runtime", (), {"worker": worker})()
        service._directions[camera_id] = Direction.ENTRY
        self.camera_service.frame_ready.connect(
            lambda _camera_id, frame: preview_frames.append(frame)
        )
        with self.camera_service._lock:
            self.camera_service._runtimes[camera_id] = object()

        try:
            frames = [
                np.full((120, 240, 3), index, dtype=np.uint8)
                for index in range(20)
            ]
            for frame in frames:
                frame.setflags(write=False)
            with patch(
                "app.plate_recognition.time.monotonic",
                side_effect=[index / 10.0 for index in range(20)],
            ):
                for frame in frames:
                    self.camera_service._on_frame_ready(camera_id, frame)

            snapshots = worker._frame_buffer.snapshots(camera_id)
            self.assertEqual(len(snapshots), 20)
            self.assertEqual(len({snapshot.frame_id for snapshot in snapshots}), 20)
            self.assertEqual(preview_frames, [])
            self.assertIs(snapshots[-1].full_frame, frames[-1])
            self.assertFalse(snapshots[-1].full_frame.flags.writeable)

            self.camera_service._flush_latest_frame(camera_id)
            self.assertEqual(len(preview_frames), 1)
            self.assertIs(preview_frames[0], frames[-1])
        finally:
            with self.camera_service._lock:
                self.camera_service._runtimes.pop(camera_id, None)
                self.camera_service._discard_frame(camera_id)

    def test_analysis_path_replay_recovers_brief_plate_hidden_by_ui_coalescing(self) -> None:
        config = replace(
            self.config,
            motion_pre_roll_ms=200,
            motion_post_roll_ms=0,
            motion_quiet_ms=0,
            max_replay_frames_per_event=6,
        )
        worker = PlateRecognitionWorker(
            self.plate_service,
            config,
            provider_factory=FakeOcrProvider,
            detector_factory=lambda: FakePlateDetector([]),
        )
        camera_id = 1
        preview_values: list[int] = []
        service = PlateRecognitionService(
            self.camera_service,
            self.plate_service,
            config,
            provider_factory=FakeOcrProvider,
            detector_factory=lambda: FakePlateDetector([]),
        )
        service._runtime = type("Runtime", (), {"worker": worker})()
        service._directions[camera_id] = Direction.ENTRY
        self.camera_service.frame_ready.connect(
            lambda _camera_id, frame: preview_values.append(int(frame[0, 0, 0]))
        )
        with self.camera_service._lock:
            self.camera_service._runtimes[camera_id] = object()
        frames = [
            np.full((120, 240, 3), value, dtype=np.uint8)
            for value in range(1, 7)
        ]
        for frame in frames:
            frame.setflags(write=False)
        motion_scores = iter((0.0, 0.0, 0.10, 0.10, 0.10, 0.0))

        try:
            with patch.object(
                worker._frame_buffer,
                "_motion_score",
                side_effect=lambda _snapshot: next(motion_scores),
            ), patch(
                "app.plate_recognition.time.monotonic",
                side_effect=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5),
            ):
                self.camera_service._on_frame_ready(camera_id, frames[0])
                self.camera_service._flush_latest_frame(camera_id)
                for frame in frames[1:]:
                    self.camera_service._on_frame_ready(camera_id, frame)
                self.camera_service._flush_latest_frame(camera_id)

            replay_frames = []
            while (snapshot := worker._replay_buffer.take(now=0.6)) is not None:
                replay_frames.append(snapshot)
            replay_values = [int(snapshot.full_frame[0, 0, 0]) for snapshot in replay_frames]
            self.assertEqual(preview_values, [1, 6])
            self.assertEqual(worker._frame_buffer.ring_depth(camera_id), 6)
            self.assertGreaterEqual(len(set(replay_values) & {3, 4, 5}), 2)

            class BriefPlateDetector:
                def detect(self, image: np.ndarray) -> list[PlateDetection]:
                    value = int(round(float(image.mean())))
                    return (
                        [PlateDetection(0.9, x=20, y=10, width=100, height=30)]
                        if value in (3, 4, 5)
                        else []
                    )

            detector_processor = PlateDetectionProcessor(config, BriefPlateDetector())
            recognition_processor = PlateRecognitionProcessor(
                FakeOcrProvider(text="23ABC123", confidence=0.97),
                self.plate_service,
                config,
            )
            outcomes = []
            for snapshot in replay_frames:
                result = detector_processor.prepare_job(
                    snapshot.camera_id,
                    snapshot.direction,
                    snapshot.full_frame,
                    captured_at=snapshot.captured_at,
                    observed_at=snapshot.observed_at,
                    received_at=snapshot.received_at,
                    frame_id=snapshot.frame_id,
                    detector_source="replay",
                    allow_zero_detection_fallback=False,
                )
                if result.job is not None:
                    outcomes.append(
                        recognition_processor.process_ocr_job(result.job, queue_depth=0)
                    )
                if outcomes and outcomes[-1].state is RecognitionState.SAVED:
                    break

            self.assertEqual([item.confirmation_count for item in outcomes[:2]], [1, 2])
            self.assertIs(outcomes[1].state, RecognitionState.SAVED)
        finally:
            with self.camera_service._lock:
                self.camera_service._runtimes.pop(camera_id, None)
                self.camera_service._discard_frame(camera_id)

    def test_worker_uses_250_ms_interval_per_camera(self) -> None:
        worker = PlateRecognitionWorker(
            self.plate_service,
            self.config,
            provider_factory=FakeOcrProvider,
        )
        first_entry = object()
        latest_entry = object()
        exit_frame = object()

        worker.submit_frame(10, Direction.ENTRY, first_entry)
        with patch("app.plate_recognition.time.monotonic", return_value=10.0):
            camera_id, pending = worker._take_due_frame()
        self.assertEqual(camera_id, 10)
        self.assertIs(pending.frame, first_entry)

        worker.submit_frame(10, Direction.ENTRY, object())
        worker.submit_frame(10, Direction.ENTRY, latest_entry)
        worker.submit_frame(20, Direction.EXIT, exit_frame)
        with patch("app.plate_recognition.time.monotonic", return_value=10.249):
            camera_id, pending = worker._take_due_frame()

        self.assertEqual(camera_id, 20)
        self.assertIs(pending.frame, exit_frame)
        self.assertEqual(worker.pending_frame_count, 1)
        with patch("app.plate_recognition.time.monotonic", return_value=10.249):
            self.assertIsNone(worker._take_due_frame())
        with patch("app.plate_recognition.time.monotonic", return_value=10.251):
            camera_id, pending = worker._take_due_frame()

        self.assertEqual(camera_id, 10)
        self.assertIs(pending.frame, latest_entry)
        self.assertEqual(worker.pending_frame_count, 0)

    def test_worker_replaces_pending_frame_instead_of_building_backlog(self) -> None:
        worker = PlateRecognitionWorker(
            self.plate_service,
            self.config,
            provider_factory=FakeOcrProvider,
        )
        latest_frame = None
        for _ in range(100):
            latest_frame = object()
            worker.submit_frame(10, Direction.ENTRY, latest_frame)

        self.assertEqual(worker.pending_frame_count, 1)
        with patch("app.plate_recognition.time.monotonic", return_value=10.0):
            _camera_id, pending = worker._take_due_frame()
        self.assertIs(pending.frame, latest_frame)

    def test_worker_round_robin_prevents_busy_camera_starvation(self) -> None:
        worker = PlateRecognitionWorker(
            self.plate_service,
            self.config,
            provider_factory=FakeOcrProvider,
        )
        worker.submit_frame(10, Direction.ENTRY, object())
        worker.submit_frame(20, Direction.EXIT, object())

        with patch("app.plate_recognition.time.monotonic", return_value=10.0):
            first_camera_id, _ = worker._take_due_frame()
        worker.submit_frame(10, Direction.ENTRY, object())
        with patch("app.plate_recognition.time.monotonic", return_value=10.3):
            second_camera_id, _ = worker._take_due_frame()

        self.assertEqual((first_camera_id, second_camera_id), (10, 20))
        self.assertEqual(worker.pending_frame_count, 1)

    def test_pre_detection_ring_is_bounded_by_count_duration_and_camera(self) -> None:
        config = replace(
            self.config,
            pre_detection_buffer_duration_ms=500,
            pre_detection_buffer_max_frames_per_camera=3,
        )
        buffer = PreDetectionFrameBuffer(config)
        for frame_id, observed_at in enumerate((0.0, 0.2, 0.4, 0.6), start=1):
            buffer.ingest(
                self._make_snapshot(frame_id, observed_at), motion_score=0.0
            )
        buffer.ingest(
            self._make_snapshot(20, 0.6, camera_id=2), motion_score=0.0
        )

        self.assertEqual(
            [item.frame_id for item in buffer.snapshots(1)], [2, 3, 4]
        )
        self.assertEqual([item.frame_id for item in buffer.snapshots(2)], [20])

    def test_motion_event_pins_pre_and_post_roll_and_closes_after_quiet(self) -> None:
        config = replace(
            self.config,
            motion_pre_roll_ms=500,
            motion_post_roll_ms=300,
            motion_quiet_ms=200,
            motion_event_max_duration_ms=4_000,
        )
        buffer = PreDetectionFrameBuffer(config)
        sequence = (
            (1, 0.0, 0.0),
            (2, 0.2, 0.0),
            (3, 0.4, 0.1),
            (4, 0.5, 0.0),
            (5, 0.65, 0.1),
            (6, 0.8, 0.0),
            (7, 1.0, 0.0),
        )
        completed: tuple[MotionEvent, ...] = ()
        for frame_id, observed_at, motion_score in sequence:
            completed = buffer.ingest(
                self._make_snapshot(frame_id, observed_at),
                motion_score=motion_score,
            )

        self.assertEqual(len(completed), 1)
        self.assertEqual(
            [item.frame_id for item in completed[0].frames],
            [1, 2, 3, 4, 5, 6, 7],
        )

    def test_static_frames_do_not_create_motion_event(self) -> None:
        buffer = PreDetectionFrameBuffer(self.config)
        completed = ()
        for frame_id in range(1, 8):
            completed = buffer.ingest(
                self._make_snapshot(frame_id, frame_id * 0.1, value=40)
            )

        self.assertEqual(completed, ())
        self.assertEqual(buffer.ring_depth(1), 7)

    def test_replay_selection_is_temporally_distributed_and_bounded(self) -> None:
        frames = tuple(
            self._make_snapshot(index, index * 0.1, value=index)
            for index in range(1, 21)
        )

        selected = select_replay_frames(frames, 4, self.config.roi_for)

        self.assertEqual(len(selected), 4)
        selected_ids = [item.frame_id for item in selected]
        self.assertTrue(1 <= selected_ids[0] <= 5)
        self.assertTrue(6 <= selected_ids[1] <= 10)
        self.assertTrue(11 <= selected_ids[2] <= 15)
        self.assertTrue(16 <= selected_ids[3] <= 20)

    def test_replay_event_queue_is_bounded_stale_and_camera_fair(self) -> None:
        config = replace(
            self.config,
            max_pending_replay_events_per_camera=2,
            max_replay_frames_per_event=2,
            replay_event_max_age_ms=1_000,
        )
        buffer = ReplayEventBuffer(config)

        def event(event_id: int, camera_id: int, enqueued_at: float) -> MotionEvent:
            frames = (
                self._make_snapshot(event_id * 10, 1.0, camera_id=camera_id),
                self._make_snapshot(event_id * 10 + 1, 1.1, camera_id=camera_id),
            )
            return MotionEvent(
                event_id=event_id,
                camera_id=camera_id,
                direction=frames[0].direction,
                started_at=1.0,
                ended_at=1.1,
                enqueued_at=enqueued_at,
                frames=frames,
            )

        buffer.add(event(1, 1, 10.0))
        buffer.add(event(2, 1, 10.0))
        buffer.add(event(3, 1, 10.0))
        buffer.add(event(4, 2, 10.0))

        self.assertEqual(buffer.pending_event_count(1), 2)
        self.assertEqual(buffer.dropped_count, 1)
        self.assertEqual(
            (buffer.take(now=10.1).camera_id, buffer.take(now=10.1).camera_id),
            (1, 2),
        )
        stale = ReplayEventBuffer(config)
        stale.add(event(5, 1, 5.0))
        self.assertIsNone(stale.take(now=10.0))
        self.assertEqual(stale.stale_count, 1)

    def test_repeated_motion_events_remain_bounded_per_camera(self) -> None:
        config = replace(
            self.config,
            max_pending_replay_events_per_camera=2,
            max_replay_frames_per_event=1,
        )
        buffer = ReplayEventBuffer(config)

        for event_id in range(20):
            camera_id = 1 if event_id % 2 == 0 else 2
            snapshot = self._make_snapshot(
                event_id, float(event_id), camera_id=camera_id
            )
            buffer.add(
                MotionEvent(
                    event_id=event_id,
                    camera_id=camera_id,
                    direction=snapshot.direction,
                    started_at=snapshot.observed_at,
                    ended_at=snapshot.observed_at,
                    enqueued_at=10.0,
                    frames=(snapshot,),
                )
            )

        self.assertEqual(buffer.pending_event_count(1), 2)
        self.assertEqual(buffer.pending_event_count(2), 2)
        self.assertEqual(buffer.pending_event_count(), 4)
        self.assertEqual(buffer.dropped_count, 16)
        self.assertEqual(
            (buffer.take(now=10.1).camera_id, buffer.take(now=10.1).camera_id),
            (1, 2),
        )

    def test_detector_scheduler_balances_two_live_frames_with_one_replay(self) -> None:
        worker = PlateRecognitionWorker(
            self.plate_service,
            self.config,
            provider_factory=FakeOcrProvider,
        )
        event = MotionEvent(
            event_id=1,
            camera_id=1,
            direction=Direction.ENTRY,
            started_at=1.0,
            ended_at=1.1,
            enqueued_at=time.monotonic(),
            frames=(self._make_snapshot(99, 1.0),),
        )
        worker._replay_buffer.add(event)
        live_one = self._make_snapshot(1, 10.0)
        live_two = self._make_snapshot(2, 10.3)
        with worker._lock:
            worker._known_camera_ids.add(1)
            worker._camera_order.append(1)
            worker._latest_frames[1] = _PendingFrame(
                live_one.direction,
                live_one.full_frame,
                live_one.received_at,
                live_one.observed_at,
                live_one.captured_at,
                live_one.frame_id,
            )
        with patch("app.plate_recognition.time.monotonic", return_value=10.0):
            first = worker._take_detector_frame(replay_enabled=True)
        with worker._lock:
            worker._latest_frames[1] = _PendingFrame(
                live_two.direction,
                live_two.full_frame,
                live_two.received_at,
                live_two.observed_at,
                live_two.captured_at,
                live_two.frame_id,
            )
        with patch("app.plate_recognition.time.monotonic", return_value=10.3):
            second = worker._take_detector_frame(replay_enabled=True)
            third = worker._take_detector_frame(replay_enabled=True)

        self.assertEqual((first[0], second[0], third[0]), ("live", "live", "replay"))

    def test_historical_replay_detections_are_not_emitted_to_live_overlay(self) -> None:
        detector = FakePlateDetector(
            [PlateDetection(0.9, x=20, y=10, width=100, height=30)]
        )
        worker = PlateRecognitionWorker(
            self.plate_service,
            self.config,
            provider_factory=FakeOcrProvider,
            detector_factory=lambda: detector,
        )
        event = MotionEvent(
            event_id=1,
            camera_id=1,
            direction=Direction.ENTRY,
            started_at=1.0,
            ended_at=1.1,
            enqueued_at=time.monotonic(),
            frames=(self._make_snapshot(501, 1.0),),
        )
        worker._replay_buffer.add(event)
        overlay_updates: list[object] = []
        worker.detections_changed.connect(
            lambda _camera_id, detections: overlay_updates.append(detections),
            Qt.ConnectionType.DirectConnection,
        )
        thread = threading.Thread(target=worker.run)
        thread.start()
        deadline = time.monotonic() + 1.0
        while detector.calls < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        worker.request_stop()
        thread.join(2.0)

        self.assertEqual(detector.calls, 1)
        self.assertEqual(overlay_updates, [])
        self.assertFalse(thread.is_alive())

    def test_detector_continues_while_ocr_provider_is_blocked(self) -> None:
        started = threading.Event()
        release = threading.Event()
        provider = BlockingOcrProvider(started, release)
        detector = FakePlateDetector(
            [PlateDetection(0.9, x=20, y=10, width=100, height=30)]
        )
        config = replace(self.config, recognition_interval_ms=100)
        worker = PlateRecognitionWorker(
            self.plate_service,
            config,
            provider_factory=lambda: provider,
            detector_factory=lambda: detector,
        )
        thread = threading.Thread(target=worker.run)
        thread.start()
        frame = np.zeros((200, 400, 3), dtype=np.uint8)

        worker.submit_frame(1, Direction.ENTRY, frame)
        self.assertTrue(started.wait(1.0))
        for _ in range(4):
            time.sleep(0.11)
            worker.submit_frame(1, Direction.ENTRY, frame)
        deadline = time.monotonic() + 1.0
        while detector.calls < 3 and time.monotonic() < deadline:
            time.sleep(0.01)

        self.assertLessEqual(
            worker._job_buffer.pending_count(1),
            config.max_pending_ocr_jobs_per_camera,
        )
        self.assertGreaterEqual(worker._frame_buffer.ring_depth(1), 4)
        release.set()
        worker.request_stop()
        thread.join(2.0)
        self.assertGreaterEqual(detector.calls, 3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(set(provider.thread_ids)), 1)

    def test_ocr_job_buffer_is_bounded_and_replaces_only_weaker_job(self) -> None:
        buffer = OcrJobBuffer(max_per_camera=3, max_age_ms=2_500)
        for quality in (10.0, 20.0, 30.0):
            buffer.add(self._make_ocr_job(quality_score=quality))

        dropped = buffer.add(self._make_ocr_job(quality_score=5.0))
        replaced = buffer.add(self._make_ocr_job(quality_score=40.0))

        self.assertFalse(dropped.accepted)
        self.assertTrue(replaced.accepted)
        self.assertEqual(replaced.replaced, 1)
        self.assertEqual(buffer.pending_count(1), 3)

    def test_ocr_job_buffer_discards_stale_jobs(self) -> None:
        buffer = OcrJobBuffer(max_per_camera=3, max_age_ms=100)
        buffer.add(
            self._make_ocr_job(
                observed_at=time.monotonic() - 3.0,
                queued_at=time.monotonic() - 1.0,
            )
        )

        self.assertIsNone(buffer.take())
        self.assertEqual(buffer.stale_count, 1)

    def test_historical_ocr_job_is_not_stale_when_recently_queued(self) -> None:
        buffer = OcrJobBuffer(max_per_camera=3, max_age_ms=2_500)
        job = self._make_ocr_job(
            observed_at=time.monotonic() - 3.0,
            queued_at=time.monotonic(),
        )
        buffer.add(job)

        self.assertIs(buffer.take(), job)
        self.assertEqual(buffer.stale_count, 0)

    def test_replay_job_captured_at_t0_and_queued_at_t3_1_is_fresh_at_t3_2(self) -> None:
        buffer = OcrJobBuffer(max_per_camera=3, max_age_ms=2_500)
        job = self._make_ocr_job(observed_at=0.0, queued_at=3.1)
        with patch("app.plate_recognition.time.monotonic", return_value=3.1):
            buffer.add(job)
        with patch("app.plate_recognition.time.monotonic", return_value=3.2):
            taken = buffer.take()

        self.assertIs(taken, job)
        self.assertEqual(buffer.stale_count, 0)

    def test_detector_crop_survives_long_fallback_wait_while_fallback_expires(self) -> None:
        buffer = OcrJobBuffer(
            max_per_camera=3,
            max_age_ms=2_500,
            detector_crop_max_age_ms=12_000,
        )
        queued_at = 10.0
        fallback = self._make_ocr_job(
            queued_at=queued_at,
            job_type=OcrJobType.ZERO_DETECTION_FALLBACK,
        )
        crop = self._make_ocr_job(
            queued_at=queued_at,
            job_type=OcrJobType.DETECTOR_CROP,
        )
        with patch("app.plate_recognition.time.monotonic", return_value=queued_at):
            buffer.add(fallback)
            buffer.add(crop)

        with patch("app.plate_recognition.time.monotonic", return_value=13.0):
            taken = buffer.take()

        self.assertIs(taken, crop)
        self.assertEqual(buffer.stale_count, 1)

    def test_detector_crop_is_processed_next_after_busy_fallback(self) -> None:
        fallback_started = threading.Event()
        fallback_release = threading.Event()
        provider = BlockingOcrProvider(fallback_started, fallback_release)
        buffer = OcrJobBuffer(
            max_per_camera=3,
            max_age_ms=100,
            detector_crop_max_age_ms=1_000,
        )
        stop_event = threading.Event()
        outcomes = []
        outcomes_ready = threading.Event()

        def on_outcome(_camera_id: int, outcome: object) -> None:
            outcomes.append(outcome)
            if len(outcomes) == 2:
                outcomes_ready.set()

        fallback = self._make_ocr_job(
            observed_at=time.monotonic(),
            frame_id=301,
            fallback_reason="zero-detection",
        )
        buffer.add(fallback)
        worker = PlateOcrWorker(
            self.plate_service,
            self.config,
            provider_factory=lambda: provider,
            job_buffer=buffer,
            stop_event=stop_event,
            on_outcome=on_outcome,
            on_status=lambda _status, _message: None,
        )
        thread = threading.Thread(target=worker.run)
        thread.start()
        self.assertTrue(fallback_started.wait(1.0))

        crop = self._make_ocr_job(
            observed_at=fallback.observed_at + 0.1,
            frame_id=302,
            job_type=OcrJobType.DETECTOR_CROP,
        )
        buffer.add(crop)
        time.sleep(0.15)
        self.assertEqual(buffer.pending_count(), 1)
        fallback_release.set()

        try:
            self.assertTrue(outcomes_ready.wait(1.5))
            self.assertEqual(
                [outcome.state for outcome in outcomes],
                [RecognitionState.AWAITING_CONFIRMATION, RecognitionState.SAVED],
            )
            self.assertEqual(buffer.stale_count, 0)
        finally:
            stop_event.set()
            buffer.wake_all()
            thread.join(1.0)
        self.assertFalse(thread.is_alive())

    def test_ocr_job_buffer_round_robin_prevents_camera_starvation(self) -> None:
        buffer = OcrJobBuffer(max_per_camera=3, max_age_ms=2_500)
        buffer.add(self._make_ocr_job(camera_id=1, quality_score=1.0))
        buffer.add(self._make_ocr_job(camera_id=1, quality_score=2.0))
        buffer.add(self._make_ocr_job(camera_id=2, quality_score=1.0))

        first = buffer.take()
        second = buffer.take()

        self.assertEqual((first.camera_id, second.camera_id), (1, 2))

    def test_detector_crop_priority_exceeds_both_fallback_types(self) -> None:
        for fallback_type in (
            OcrJobType.DETECTOR_ERROR_FALLBACK,
            OcrJobType.ZERO_DETECTION_FALLBACK,
        ):
            with self.subTest(fallback_type=fallback_type):
                buffer = OcrJobBuffer(max_per_camera=1, max_age_ms=2_500)
                fallback = self._make_ocr_job(
                    quality_score=10_000.0,
                    job_type=fallback_type,
                )
                crop = self._make_ocr_job(
                    quality_score=1.0,
                    job_type=OcrJobType.DETECTOR_CROP,
                )

                buffer.add(fallback)
                result = buffer.add(crop)

                self.assertTrue(result.accepted)
                self.assertEqual(result.replaced_job_type, fallback_type)
                self.assertIs(buffer.take(), crop)

    def test_same_priority_replacement_uses_deterministic_quality_policy(self) -> None:
        buffer = OcrJobBuffer(max_per_camera=2, max_age_ms=2_500)
        weakest = self._make_ocr_job(quality_score=10.0, frame_id=1)
        strongest = self._make_ocr_job(quality_score=30.0, frame_id=2)
        replacement = self._make_ocr_job(quality_score=20.0, frame_id=3)
        equal_quality = self._make_ocr_job(quality_score=20.0, frame_id=4)
        buffer.add(weakest)
        buffer.add(strongest)

        accepted = buffer.add(replacement)
        dropped = buffer.add(equal_quality)

        self.assertTrue(accepted.accepted)
        self.assertEqual(accepted.replaced_job_type, OcrJobType.DETECTOR_CROP)
        self.assertFalse(dropped.accepted)
        self.assertEqual(dropped.drop_reason, "quality-not-better")
        self.assertEqual([buffer.take().frame_id, buffer.take().frame_id], [2, 3])

    def test_lower_priority_job_cannot_replace_detector_crop(self) -> None:
        buffer = OcrJobBuffer(max_per_camera=1, max_age_ms=2_500)
        crop = self._make_ocr_job(quality_score=1.0)
        fallback = self._make_ocr_job(
            quality_score=10_000.0,
            job_type=OcrJobType.ZERO_DETECTION_FALLBACK,
        )
        buffer.add(crop)

        result = buffer.add(fallback)

        self.assertFalse(result.accepted)
        self.assertEqual(result.drop_reason, "lower-priority")
        self.assertIs(buffer.take(), crop)

    def test_full_queue_accepts_detector_crop_by_evicting_weakest_fallback(self) -> None:
        buffer = OcrJobBuffer(max_per_camera=3, max_age_ms=2_500)
        zero = self._make_ocr_job(
            job_type=OcrJobType.ZERO_DETECTION_FALLBACK, frame_id=1
        )
        error = self._make_ocr_job(
            job_type=OcrJobType.DETECTOR_ERROR_FALLBACK, frame_id=2
        )
        existing_crop = self._make_ocr_job(frame_id=3)
        new_crop = self._make_ocr_job(frame_id=4)
        for job in (zero, error, existing_crop):
            buffer.add(job)

        result = buffer.add(new_crop)

        self.assertTrue(result.accepted)
        self.assertEqual(result.replaced_job_type, OcrJobType.ZERO_DETECTION_FALLBACK)
        self.assertEqual([buffer.take().frame_id, buffer.take().frame_id], [3, 4])

    def test_zero_detection_fallback_is_coalesced_and_released_after_consume(self) -> None:
        buffer = OcrJobBuffer(max_per_camera=3, max_age_ms=2_500)
        first = self._make_ocr_job(job_type=OcrJobType.ZERO_DETECTION_FALLBACK)
        second = self._make_ocr_job(job_type=OcrJobType.ZERO_DETECTION_FALLBACK)

        buffer.add(first)
        coalesced = buffer.add(second)
        consumed = buffer.take()
        accepted_again = buffer.add(second)

        self.assertFalse(coalesced.accepted)
        self.assertTrue(coalesced.coalesced)
        self.assertIs(consumed, first)
        self.assertTrue(accepted_again.accepted)

    def test_fallback_coalescing_has_no_stuck_state_after_replace_stale_or_clear(self) -> None:
        replacement_buffer = OcrJobBuffer(max_per_camera=1, max_age_ms=2_500)
        zero = self._make_ocr_job(job_type=OcrJobType.ZERO_DETECTION_FALLBACK)
        replacement_buffer.add(zero)
        replacement_buffer.add(self._make_ocr_job(job_type=OcrJobType.DETECTOR_CROP))
        replacement_buffer.take()
        self.assertTrue(replacement_buffer.add(zero).accepted)

        drop_buffer = OcrJobBuffer(max_per_camera=1, max_age_ms=2_500)
        crop = self._make_ocr_job(job_type=OcrJobType.DETECTOR_CROP)
        drop_buffer.add(crop)
        dropped = drop_buffer.add(zero)
        self.assertFalse(dropped.accepted)
        drop_buffer.take()
        self.assertTrue(drop_buffer.add(zero).accepted)

        stale_buffer = OcrJobBuffer(max_per_camera=1, max_age_ms=100)
        stale_buffer.add(
            self._make_ocr_job(
                job_type=OcrJobType.ZERO_DETECTION_FALLBACK,
                queued_at=time.monotonic() - 1.0,
            )
        )
        fresh = self._make_ocr_job(job_type=OcrJobType.ZERO_DETECTION_FALLBACK)
        stale_result = stale_buffer.add(fresh)
        self.assertTrue(stale_result.accepted)
        self.assertEqual(stale_result.stale_discarded, 1)

        stale_buffer.clear()
        self.assertTrue(stale_buffer.add(fresh).accepted)

    def test_detector_error_fallback_cannot_flood_camera_queue(self) -> None:
        buffer = OcrJobBuffer(max_per_camera=3, max_age_ms=2_500)
        first = self._make_ocr_job(job_type=OcrJobType.DETECTOR_ERROR_FALLBACK)
        buffer.add(first)

        results = [
            buffer.add(
                self._make_ocr_job(job_type=OcrJobType.DETECTOR_ERROR_FALLBACK)
            )
            for _ in range(5)
        ]

        self.assertEqual(buffer.pending_count(1), 1)
        self.assertTrue(all(result.coalesced for result in results))

    def test_buffer_debug_decision_reports_priority_and_coalescing(self) -> None:
        buffer = OcrJobBuffer(max_per_camera=3, max_age_ms=2_500)
        fallback = self._make_ocr_job(
            frame_id=41,
            job_type=OcrJobType.ZERO_DETECTION_FALLBACK,
        )
        buffer.add(fallback)

        with self.assertLogs("app.plate_recognition", level="DEBUG") as captured:
            buffer.add(replace(fallback, frame_id=42))

        diagnostic = captured.output[-1]
        self.assertIn("job_type=ZERO_DETECTION_FALLBACK", diagnostic)
        self.assertIn("priority=1", diagnostic)
        self.assertIn("accepted=no", diagnostic)
        self.assertIn("drop_reason=fallback-pending", diagnostic)
        self.assertIn("coalesced=yes", diagnostic)

    def test_ocr_consumer_prefers_detector_crop_globally(self) -> None:
        buffer = OcrJobBuffer(max_per_camera=3, max_age_ms=2_500)
        fallback = self._make_ocr_job(
            camera_id=1, job_type=OcrJobType.DETECTOR_ERROR_FALLBACK
        )
        crop = self._make_ocr_job(camera_id=2, job_type=OcrJobType.DETECTOR_CROP)
        buffer.add(fallback)
        buffer.add(crop)

        self.assertIs(buffer.take(), crop)
        self.assertIs(buffer.take(), fallback)

    def test_ocr_consumer_follows_all_priority_tiers(self) -> None:
        buffer = OcrJobBuffer(max_per_camera=3, max_age_ms=2_500)
        zero = self._make_ocr_job(
            camera_id=2,
            job_type=OcrJobType.ZERO_DETECTION_FALLBACK,
        )
        error = self._make_ocr_job(
            camera_id=1,
            job_type=OcrJobType.DETECTOR_ERROR_FALLBACK,
        )
        crop = self._make_ocr_job(
            camera_id=2,
            job_type=OcrJobType.DETECTOR_CROP,
        )
        for job in (zero, error, crop):
            buffer.add(job)

        self.assertEqual(
            [buffer.take().job_type for _ in range(3)],
            [
                OcrJobType.DETECTOR_CROP,
                OcrJobType.DETECTOR_ERROR_FALLBACK,
                OcrJobType.ZERO_DETECTION_FALLBACK,
            ],
        )

    def test_ocr_consumer_is_camera_fair_within_priority_tier(self) -> None:
        buffer = OcrJobBuffer(max_per_camera=3, max_age_ms=2_500)
        entry_jobs = [
            self._make_ocr_job(camera_id=1, frame_id=frame_id)
            for frame_id in (1, 2, 3)
        ]
        exit_jobs = [
            self._make_ocr_job(camera_id=2, frame_id=frame_id)
            for frame_id in (4, 5)
        ]
        for job in (*entry_jobs, *exit_jobs):
            buffer.add(job)

        taken = [buffer.take() for _ in range(4)]

        self.assertEqual([job.camera_id for job in taken], [1, 2, 1, 2])

    def test_waiting_ocr_consumer_wakes_and_stops_without_deadlock(self) -> None:
        buffer = OcrJobBuffer(max_per_camera=3, max_age_ms=2_500)
        stop_event = threading.Event()
        result: list[OcrJob | None] = []
        thread = threading.Thread(
            target=lambda: result.append(buffer.take(stop_event, wait=True))
        )
        thread.start()
        stop_event.set()
        buffer.clear()
        buffer.wake_all()
        thread.join(1.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(result, [None])

    def test_fallback_flood_cannot_suppress_other_camera_detector_crop(self) -> None:
        for crop_camera_id in (1, 2):
            with self.subTest(crop_camera_id=crop_camera_id):
                fallback_camera_id = 2 if crop_camera_id == 1 else 1
                buffer = OcrJobBuffer(max_per_camera=3, max_age_ms=2_500)
                for frame_id in range(10):
                    buffer.add(
                        self._make_ocr_job(
                            camera_id=fallback_camera_id,
                            frame_id=frame_id,
                            job_type=OcrJobType.ZERO_DETECTION_FALLBACK,
                        )
                    )
                crop = self._make_ocr_job(
                    camera_id=crop_camera_id,
                    frame_id=100,
                    job_type=OcrJobType.DETECTOR_CROP,
                )

                accepted = buffer.add(crop)

                self.assertTrue(accepted.accepted)
                self.assertIs(buffer.take(), crop)
                self.assertEqual(buffer.pending_count(fallback_camera_id), 1)

    def test_normal_vehicle_detector_crops_save_while_fallback_jobs_are_pending(self) -> None:
        buffer = OcrJobBuffer(max_per_camera=3, max_age_ms=2_500)
        processor = PlateRecognitionProcessor(
            FakeOcrProvider(confidence=0.97), self.plate_service, self.config
        )
        buffer.add(
            self._make_ocr_job(
                camera_id=1,
                frame_id=90,
                job_type=OcrJobType.ZERO_DETECTION_FALLBACK,
            )
        )
        buffer.add(
            self._make_ocr_job(
                camera_id=2,
                frame_id=91,
                job_type=OcrJobType.DETECTOR_ERROR_FALLBACK,
            )
        )
        first_crop = self._make_ocr_job(
            camera_id=1,
            observed_at=10.0,
            frame_id=100,
            frame_value=31,
            job_type=OcrJobType.DETECTOR_CROP,
        )
        second_crop = self._make_ocr_job(
            camera_id=1,
            observed_at=10.2,
            frame_id=104,
            frame_value=63,
            job_type=OcrJobType.DETECTOR_CROP,
        )
        self.assertTrue(buffer.add(first_crop).accepted)
        self.assertTrue(buffer.add(second_crop).accepted)

        first_taken = buffer.take()
        second_taken = buffer.take()
        awaiting = processor.process_ocr_job(
            first_taken, queue_depth=buffer.pending_count()
        )
        saved = processor.process_ocr_job(
            second_taken, queue_depth=buffer.pending_count()
        )

        self.assertEqual((first_taken.frame_id, second_taken.frame_id), (100, 104))
        self.assertIs(awaiting.state, RecognitionState.AWAITING_CONFIRMATION)
        self.assertEqual(awaiting.confirmation_count, 1)
        self.assertIs(saved.state, RecognitionState.SAVED)
        self.assertEqual(saved.confirmation_count, 2)
        self.assertEqual(self._record_count(), 1)
        saved_image = cv2.imread(
            str(self.capture_service.resolve_reference(saved.record.image_path))
        )
        self.assertAlmostEqual(float(saved_image.mean()), 63.0, delta=3.0)

    def test_ocr_worker_diagnostics_include_job_identity_and_confirmation(self) -> None:
        processor = PlateRecognitionProcessor(
            FakeOcrProvider(confidence=0.97), self.plate_service, self.config
        )
        job = self._make_ocr_job(
            observed_at=time.monotonic(),
            frame_id=55,
            detector_source="replay",
            job_type=OcrJobType.DETECTOR_CROP,
        )

        with self.assertLogs("app.plate_recognition", level="DEBUG") as captured:
            outcome = processor.process_ocr_job(job, queue_depth=2)

        diagnostic = next(
            line for line in captured.output if "OCR worker diagnostics" in line
        )
        self.assertIs(outcome.state, RecognitionState.AWAITING_CONFIRMATION)
        self.assertIn("job_type=DETECTOR_CROP", diagnostic)
        self.assertIn("priority=3", diagnostic)
        self.assertIn("source=replay", diagnostic)
        self.assertIn("frame_id=55", diagnostic)
        self.assertIn("queue_wait_ms=", diagnostic)
        self.assertIn("candidate=34ABC123", diagnostic)
        self.assertIn("recognition_state=AWAITING_CONFIRMATION", diagnostic)
        self.assertIn("confirmation=1/2", diagnostic)

    def test_full_roi_fallback_uses_single_compact_variant_on_early_success(self) -> None:
        provider = RecordingOcrProvider(confidence=0.97)
        processor = PlateRecognitionProcessor(provider, self.plate_service, self.config)
        job = self._make_ocr_job(
            frame_id=61,
            fallback_reason="zero-detection",
        )
        large_roi = np.full((672, 1_912, 3), 100, dtype=np.uint8)
        job = replace(job, roi_crop=large_roi, ocr_crops=(large_roi,))

        outcome = processor.process_ocr_job(job, queue_depth=0)

        self.assertIs(outcome.state, RecognitionState.AWAITING_CONFIRMATION)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(len(provider.images), 1)
        self.assertLessEqual(provider.images[0].shape[1], 960)

    def test_full_roi_fallback_tries_second_variant_only_when_first_has_no_plate(self) -> None:
        provider = SequencedOcrProvider(
            [
                [],
                [OcrSegment("34ABC123", 0.97, (0.0, 0.0, 100.0, 30.0))],
            ]
        )
        processor = PlateRecognitionProcessor(provider, self.plate_service, self.config)
        job = self._make_ocr_job(
            frame_id=62,
            fallback_reason="detector-error",
        )

        outcome = processor.process_ocr_job(job, queue_depth=0)

        self.assertIs(outcome.state, RecognitionState.AWAITING_CONFIRMATION)
        self.assertEqual(len(provider.calls), 2)
        self.assertTrue(all(len(batch) == 1 for batch in provider.calls))

    def test_detection_processor_queues_crop_and_preserves_full_frame(self) -> None:
        detection = PlateDetection(0.9, x=20, y=10, width=100, height=30)
        processor = PlateDetectionProcessor(
            self.config,
            FakePlateDetector([detection]),
        )
        frame = np.full((200, 400, 3), 71, dtype=np.uint8)

        result = processor.prepare_job(
            1,
            Direction.ENTRY,
            frame,
            captured_at=datetime(2026, 8, 13, tzinfo=timezone.utc),
            observed_at=10.0,
            received_at=10.0,
        )

        self.assertIsNotNone(result.job)
        self.assertIs(result.job.job_type, OcrJobType.DETECTOR_CROP)
        self.assertEqual(len(result.job.ocr_crops), 1)
        self.assertLess(result.job.ocr_crops[0].shape[0], result.job.roi_crop.shape[0])
        np.testing.assert_array_equal(result.job.full_frame, frame)
        frame[:] = 0
        self.assertEqual(int(result.job.full_frame[0, 0, 0]), 71)

    def test_detection_processor_keeps_zero_detection_throttle(self) -> None:
        processor = PlateDetectionProcessor(self.config, FakePlateDetector([]))
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        captured_at = datetime(2026, 8, 13, tzinfo=timezone.utc)

        first = processor.prepare_job(
            1, Direction.ENTRY, frame,
            captured_at=captured_at, observed_at=10.0, received_at=10.0,
            zero_detection_fallback_event_id=1,
        )
        throttled = processor.prepare_job(
            1, Direction.ENTRY, frame,
            captured_at=captured_at, observed_at=10.5, received_at=10.5,
            zero_detection_fallback_event_id=1,
        )
        after_interval = processor.prepare_job(
            1, Direction.ENTRY, frame,
            captured_at=captured_at, observed_at=10.8, received_at=10.8,
            zero_detection_fallback_event_id=2,
        )

        self.assertEqual(first.job.fallback_reason, "zero-detection")
        self.assertIs(first.job.job_type, OcrJobType.ZERO_DETECTION_FALLBACK)
        self.assertIsNone(throttled.job)
        self.assertEqual(after_interval.job.fallback_reason, "zero-detection")

    def test_static_zero_detection_does_not_generate_expensive_roi_jobs(self) -> None:
        processor = PlateDetectionProcessor(self.config, FakePlateDetector([]))
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        captured_at = datetime(2026, 8, 13, tzinfo=timezone.utc)

        results = [
            processor.prepare_job(
                1,
                Direction.ENTRY,
                frame,
                captured_at=captured_at,
                observed_at=10.0 + index,
                received_at=10.0 + index,
            )
            for index in range(20)
        ]

        self.assertTrue(all(result.job is None for result in results))
        self.assertTrue(
            all(
                result.fallback_skipped_reason == "no-meaningful-motion"
                for result in results
            )
        )

    def test_motion_event_allows_only_one_zero_detection_fallback(self) -> None:
        processor = PlateDetectionProcessor(self.config, FakePlateDetector([]))
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        captured_at = datetime(2026, 8, 13, tzinfo=timezone.utc)

        first = processor.prepare_job(
            1,
            Direction.ENTRY,
            frame,
            captured_at=captured_at,
            observed_at=10.0,
            received_at=10.0,
            zero_detection_fallback_event_id=7,
        )
        repeated = processor.prepare_job(
            1,
            Direction.ENTRY,
            frame,
            captured_at=captured_at,
            observed_at=11.0,
            received_at=11.0,
            zero_detection_fallback_event_id=7,
        )

        self.assertIs(first.job.job_type, OcrJobType.ZERO_DETECTION_FALLBACK)
        self.assertIsNone(repeated.job)
        self.assertEqual(repeated.fallback_skipped_reason, "event-fallback-used")

    def test_detection_processor_exception_queues_detector_error_fallback(self) -> None:
        processor = PlateDetectionProcessor(
            self.config,
            FakePlateDetector(error=PlateDetectorError("temporary")),
        )
        frame = np.zeros((200, 400, 3), dtype=np.uint8)

        result = processor.prepare_job(
            1,
            Direction.ENTRY,
            frame,
            captured_at=datetime(2026, 8, 13, tzinfo=timezone.utc),
            observed_at=10.0,
            received_at=10.0,
        )

        self.assertEqual(result.job.fallback_reason, "detector-error")
        self.assertIs(result.job.job_type, OcrJobType.DETECTOR_ERROR_FALLBACK)
        self.assertTrue(result.job.used_roi_fallback)
        self.assertEqual(result.job.ocr_crops[0].shape, result.job.roi_crop.shape)

    def test_replay_zero_detection_does_not_queue_roi_fallback(self) -> None:
        processor = PlateDetectionProcessor(self.config, FakePlateDetector([]))
        frame = np.zeros((200, 400, 3), dtype=np.uint8)

        result = processor.prepare_job(
            1,
            Direction.ENTRY,
            frame,
            captured_at=datetime(2026, 8, 13, tzinfo=timezone.utc),
            observed_at=10.0,
            received_at=10.0,
            frame_id=44,
            detector_source="replay",
            allow_zero_detection_fallback=False,
        )

        self.assertIsNone(result.job)
        self.assertFalse(result.used_roi_fallback)

    def test_replay_detector_crop_keeps_detector_crop_priority_and_timestamps(self) -> None:
        processor = PlateDetectionProcessor(
            self.config,
            FakePlateDetector([PlateDetection(0.9, x=20, y=10, width=100, height=30)]),
        )
        frame = np.full((200, 400, 3), 19, dtype=np.uint8)
        captured_at = datetime(2026, 8, 13, 9, 30, tzinfo=timezone.utc)

        result = processor.prepare_job(
            2,
            Direction.EXIT,
            frame,
            captured_at=captured_at,
            observed_at=12.5,
            received_at=12.5,
            frame_id=88,
            detector_source="replay",
            allow_zero_detection_fallback=False,
        )

        self.assertIsNotNone(result.job)
        self.assertIs(result.job.job_type, OcrJobType.DETECTOR_CROP)
        self.assertEqual(result.job.detector_source, "replay")
        self.assertEqual(result.job.frame_id, 88)
        self.assertEqual(result.job.observed_at, 12.5)
        self.assertEqual(result.job.captured_at, captured_at)
        np.testing.assert_array_equal(result.job.full_frame, frame)

    def test_queued_observations_use_frame_time_and_matching_full_frame(self) -> None:
        processor = PlateRecognitionProcessor(
            FakeOcrProvider(confidence=0.97),
            self.plate_service,
            self.config,
        )
        first = self._make_ocr_job(observed_at=10.0, frame_value=11)
        second = self._make_ocr_job(observed_at=12.0, frame_value=22)

        awaiting = processor.process_ocr_job(first, queue_depth=1)
        saved = processor.process_ocr_job(second, queue_depth=0)

        self.assertIs(awaiting.state, RecognitionState.AWAITING_CONFIRMATION)
        self.assertIs(saved.state, RecognitionState.SAVED)
        self.assertEqual(
            datetime.fromisoformat(saved.record.timestamp),
            second.captured_at,
        )
        saved_image = cv2.imread(str(self.capture_service.resolve_reference(saved.record.image_path)))
        self.assertIsNotNone(saved_image)
        self.assertAlmostEqual(float(saved_image.mean()), 22.0, delta=3.0)

    def test_queued_observations_outside_capture_window_do_not_confirm(self) -> None:
        processor = PlateRecognitionProcessor(
            FakeOcrProvider(confidence=0.97),
            self.plate_service,
            self.config,
        )
        first = self._make_ocr_job(observed_at=10.0)
        delayed = self._make_ocr_job(observed_at=14.0)

        processor.process_ocr_job(first, queue_depth=1)
        outcome = processor.process_ocr_job(delayed, queue_depth=0)

        self.assertIs(outcome.state, RecognitionState.AWAITING_CONFIRMATION)
        self.assertIsNone(outcome.record)

    def test_same_frame_live_and_replay_does_not_confirm_twice(self) -> None:
        processor = PlateRecognitionProcessor(
            FakeOcrProvider(confidence=0.97), self.plate_service, self.config
        )
        live = self._make_ocr_job(observed_at=10.0, frame_id=77)
        replay = replace(live, queued_at=time.monotonic(), detector_source="replay")

        first = processor.process_ocr_job(live, queue_depth=1)
        duplicate = processor.process_ocr_job(replay, queue_depth=0)

        self.assertIs(first.state, RecognitionState.AWAITING_CONFIRMATION)
        self.assertIs(duplicate.state, RecognitionState.DUPLICATE_SUPPRESSED)
        self.assertIsNone(duplicate.record)
        self.assertEqual(self._record_count(), 0)

    def test_two_distinct_historical_frames_confirm_and_save_second_frame(self) -> None:
        processor = PlateRecognitionProcessor(
            FakeOcrProvider(confidence=0.97), self.plate_service, self.config
        )
        first = self._make_ocr_job(
            observed_at=10.0, frame_value=31, frame_id=101
        )
        second = self._make_ocr_job(
            observed_at=10.2, frame_value=63, frame_id=102
        )

        awaiting = processor.process_ocr_job(first, queue_depth=1)
        saved = processor.process_ocr_job(second, queue_depth=0)

        self.assertEqual((awaiting.confirmation_count, saved.confirmation_count), (1, 2))
        self.assertIs(saved.state, RecognitionState.SAVED)
        saved_image = cv2.imread(
            str(self.capture_service.resolve_reference(saved.record.image_path))
        )
        self.assertAlmostEqual(float(saved_image.mean()), 63.0, delta=3.0)

    def test_out_of_order_replay_uses_original_observation_times(self) -> None:
        processor = PlateRecognitionProcessor(
            FakeOcrProvider(confidence=0.97), self.plate_service, self.config
        )
        newer_captured_at = datetime(2026, 8, 13, 9, 0, 1, tzinfo=timezone.utc)
        older_captured_at = datetime(2026, 8, 13, 9, 0, 0, tzinfo=timezone.utc)
        newer = self._make_ocr_job(
            observed_at=10.2,
            frame_id=202,
            captured_at=newer_captured_at,
        )
        older = self._make_ocr_job(
            observed_at=10.0,
            frame_id=201,
            captured_at=older_captured_at,
        )

        processor.process_ocr_job(newer, queue_depth=1)
        saved = processor.process_ocr_job(older, queue_depth=0)

        self.assertIs(saved.state, RecognitionState.SAVED)
        self.assertEqual(datetime.fromisoformat(saved.record.timestamp), older_captured_at)

    def test_replay_recovers_two_plate_frames_missed_by_live_detector(self) -> None:
        class PixelPlateDetector:
            def detect(self, image: np.ndarray) -> list[PlateDetection]:
                value = int(round(float(image.mean())))
                if value in (2, 3):
                    return [PlateDetection(0.9, x=20, y=10, width=100, height=30)]
                return []

        snapshots = tuple(
            self._make_snapshot(index, index * 0.1, value=index)
            for index in range(1, 5)
        )
        detector_processor = PlateDetectionProcessor(
            self.config, PixelPlateDetector()
        )
        recognition_processor = PlateRecognitionProcessor(
            FakeOcrProvider(confidence=0.97), self.plate_service, self.config
        )
        outcomes = []

        for snapshot in snapshots:
            result = detector_processor.prepare_job(
                snapshot.camera_id,
                snapshot.direction,
                snapshot.full_frame,
                captured_at=snapshot.captured_at,
                observed_at=snapshot.observed_at,
                received_at=snapshot.received_at,
                frame_id=snapshot.frame_id,
                detector_source="replay",
                allow_zero_detection_fallback=False,
            )
            if result.job is not None:
                outcomes.append(
                    recognition_processor.process_ocr_job(result.job, queue_depth=0)
                )

        self.assertEqual(
            [outcome.confirmation_count for outcome in outcomes], [1, 2]
        )
        self.assertIs(outcomes[-1].state, RecognitionState.SAVED)
        self.assertEqual(self._record_count(), 1)

    def test_worker_detector_initialization_failure_keeps_roi_fallback_active(self) -> None:
        factory_calls = 0

        def missing_detector() -> object:
            nonlocal factory_calls
            factory_calls += 1
            raise PlateDetectorError("missing model")

        worker = PlateRecognitionWorker(
            self.plate_service,
            self.config,
            provider_factory=FakeOcrProvider,
            detector_factory=missing_detector,
        )
        statuses: list[RecognitionStatus] = []
        messages: list[str] = []
        worker.status_changed.connect(
            lambda status, message: (statuses.append(status), messages.append(message)),
            Qt.ConnectionType.DirectConnection,
        )
        thread = threading.Thread(target=worker.run)
        thread.start()
        time.sleep(0.05)
        worker.request_stop()
        thread.join(1.0)

        self.assertEqual(factory_calls, 1)
        self.assertIn(RecognitionStatus.ACTIVE, statuses)
        self.assertTrue(any("fallback" in message for message in messages))

    def test_missing_model_is_reported_without_importing_paddle(self) -> None:
        with self.assertRaisesRegex(OcrModelNotFound, "Detection model"):
            PaddleOcrProvider(Path(self.temp_directory.name) / "missing-models")

    def test_empty_and_invalid_model_directories_are_rejected(self) -> None:
        empty = Path(self.temp_directory.name) / "empty"
        empty.mkdir()
        with self.assertRaises(ModelNotFound):
            validate_model_directory(empty, "Detection")

        invalid = Path(self.temp_directory.name) / "invalid"
        invalid.mkdir()
        (invalid / "inference.onnx").write_bytes(b"not-onnx")
        (invalid / "inference.yml").write_text("[]", encoding="utf-8")
        with self.assertRaises(OcrModelInvalid):
            validate_model_directory(invalid, "Detection")

    def test_worker_returns_active_after_transient_inference_error(self) -> None:
        config = recognition_config(Path(self.temp_directory.name), confirmations=1)
        config = PlateRecognitionConfig(
            recognition_interval_ms=100,
            min_confidence=config.min_confidence,
            confirmations_required=config.confirmations_required,
            confirmation_window_seconds=config.confirmation_window_seconds,
            duplicate_cooldown_seconds=config.duplicate_cooldown_seconds,
            entry_roi=config.entry_roi,
            exit_roi=config.exit_roi,
            model_root=config.model_root,
        )
        worker = PlateRecognitionWorker(
            self.plate_service, config, provider_factory=RecoveringOcrProvider
        )
        statuses: list[RecognitionStatus] = []
        worker.status_changed.connect(
            lambda status, _message: statuses.append(status),
            Qt.ConnectionType.DirectConnection,
        )
        thread = threading.Thread(target=worker.run)
        thread.start()
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        worker.submit_frame(1, Direction.ENTRY, frame)
        time.sleep(0.15)
        worker.submit_frame(1, Direction.ENTRY, frame)
        time.sleep(0.2)
        worker.request_stop()
        thread.join(1.0)

        error_index = statuses.index(RecognitionStatus.ERROR)
        self.assertIn(RecognitionStatus.ACTIVE, statuses[error_index + 1 :])

    def test_debug_output_paths_are_created(self) -> None:
        output = Path(self.temp_directory.name) / "debug"
        image = np.zeros((30, 100, 3), dtype=np.uint8)
        segments = [OcrSegment("34ABC123", 0.9, (1, 1, 80, 20))]

        paths = save_debug_images(output, image, image, [image], segments)

        self.assertGreaterEqual(len(paths), 4)
        self.assertTrue(all(path.is_file() for path in paths))

    def _record_count(self) -> int:
        with self.database.connection() as connection:
            row = connection.execute("SELECT COUNT(*) FROM plate_records").fetchone()
        return int(row[0])


if __name__ == "__main__":
    unittest.main()
