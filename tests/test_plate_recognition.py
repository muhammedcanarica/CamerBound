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
    LOW_LIGHT_THRESHOLD,
    OcrModelNotFound,
    OcrSegment,
    PaddleOcrProvider,
    PlateCandidate,
    PlatePresenceTracker,
    PlateRecognitionProcessor,
    PlateRecognitionWorker,
    RecognitionState,
    TurkishPlateValidator,
    correct_plate_candidate,
    normalize_plate_text,
    crop_roi,
    preprocess_variants,
    roi_mean_brightness,
    select_best_candidate,
    RecognitionStatus,
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


class RecordingOcrProvider(FakeOcrProvider):
    def __init__(self, confidence: float = 0.97) -> None:
        super().__init__(confidence=confidence)
        self.images: list[np.ndarray] = []
        self.calls = 0

    def recognize(self, images: object) -> list[OcrSegment]:
        self.calls += 1
        self.images = list(images)
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
