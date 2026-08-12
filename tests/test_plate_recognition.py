from __future__ import annotations

import tempfile
import unittest
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt

from app.auth import AuthService
from app.camera import CameraService, Direction
from app.config import DEFAULT_ROI, PlateRecognitionConfig
from app.database import Database
from app.plate_capture import PlateCaptureService
from app.plate_recognition import (
    ConfirmationTracker,
    OcrModelNotFound,
    OcrSegment,
    PaddleOcrProvider,
    PlateCandidate,
    PlateRecognitionProcessor,
    PlateRecognitionWorker,
    TurkishPlateValidator,
    correct_plate_candidate,
    normalize_plate_text,
    select_best_candidate,
    RecognitionStatus,
)
from app.ocr_models import OcrModelInvalid, OcrModelNotFound as ModelNotFound, validate_model_directory
from app.ocr_debug import save_debug_images
from app.plate_service import PlateService


class FakeOcrProvider:
    def __init__(self, text: str = "34 ABC 123", confidence: float = 0.91) -> None:
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


def recognition_config(model_root: Path, confirmations: int = 2) -> PlateRecognitionConfig:
    return PlateRecognitionConfig(
        recognition_interval_ms=500,
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

    def test_multiple_ocr_segments_are_combined_left_to_right(self) -> None:
        segments = [
            OcrSegment("123", 0.90, (200.0, 0.0, 260.0, 30.0)),
            OcrSegment("34", 0.94, (0.0, 0.0, 40.0, 30.0)),
            OcrSegment("ABC", 0.92, (60.0, 0.0, 160.0, 30.0)),
        ]

        candidate = select_best_candidate(segments, camera_id=5)

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.plate, "34ABC123")


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
            FakeOcrProvider(), self.plate_service, self.config
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
        self.assertIsNotNone(second.record)
        self.assertEqual(second.record.plate, "34ABC123")
        self.assertEqual(second.record.direction, Direction.ENTRY)
        self.assertIsNotNone(second.record.image_path)
        self.assertTrue(
            self.capture_service.resolve_reference(second.record.image_path).is_file()
        )

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
            detected_at,
            monotonic_at=6.0,
        )
        duplicate = processor.process(
            camera.id,
            camera.direction,
            frame,
            detected_at + timedelta(seconds=5),
            monotonic_at=7.0,
        )

        self.assertIsNotNone(saved.record)
        self.assertTrue(duplicate.duplicate)
        self.assertIsNone(duplicate.record)
        self.assertEqual(len(list(Path(self.temp_directory.name).rglob("*.jpg"))), 1)

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


if __name__ == "__main__":
    unittest.main()
