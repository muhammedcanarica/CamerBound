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
from app.plate_detector import DetectorDiagnostics, PlateDetection, PlateDetectorError
from app.plate_recognition import (
    ConfirmationTracker,
    DetectionJobResult,
    FinalizationSource,
    FrameSnapshot,
    LOW_LIGHT_THRESHOLD,
    MotionEvent,
    OcrImageProfile,
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
    RecognitionOutcome,
    RecognitionState,
    TurkishPlateValidator,
    build_ocr_search_tiles,
    correct_plate_candidate,
    correct_plate_candidate_with_cost,
    classify_crop_quality,
    normalize_plate_text,
    crop_roi,
    measure_crop_quality,
    preprocess_shadow_variants,
    preprocess_variants,
    preprocess_roi_fallback_variants,
    roi_mean_brightness,
    recognize_detector_crops,
    recognize_ocr_search_tiles,
    select_replay_frames,
    select_best_candidate,
    plates_are_near_conflicts,
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


def recognition_config(
    model_root: Path,
    confirmations: int = 2,
    *,
    stabilization_window_ms: int = 0,
    stabilization_min_hold_ms: int = 0,
) -> PlateRecognitionConfig:
    return PlateRecognitionConfig(
        recognition_interval_ms=250,
        min_confidence=0.65,
        confirmations_required=confirmations,
        confirmation_window_seconds=3.0,
        duplicate_cooldown_seconds=10,
        entry_roi=DEFAULT_ROI,
        exit_roi=DEFAULT_ROI,
        model_root=model_root,
        plate_stabilization_window_ms=stabilization_window_ms,
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

    def test_correction_cost_is_available_without_breaking_existing_api(self) -> None:
        self.assertEqual(
            correct_plate_candidate_with_cost("23ABCI23"),
            ("23ABC123", 1),
        )
        self.assertEqual(
            correct_plate_candidate_with_cost("23ABC123"),
            ("23ABC123", 0),
        )

    def test_near_conflict_requires_same_valid_structure_and_province(self) -> None:
        self.assertTrue(plates_are_near_conflicts("23ABC123", "23ABC128"))
        self.assertFalse(plates_are_near_conflicts("23ABC123", "34ABC128"))
        self.assertFalse(plates_are_near_conflicts("23ABC123", "23AB1234"))

    def test_confirmation_observations_are_isolated_by_track(self) -> None:
        tracker = ConfirmationTracker(required=2, window_seconds=3.0)
        left = PlateCandidate(
            "34ABC123", 0.90, "34ABC123", camera_id=1, track_id=11
        )
        right = PlateCandidate(
            "35XYZ789", 0.91, "35XYZ789", camera_id=1, track_id=12
        )

        left_first = tracker.observe_progress(left, 10.0, frame_id=1)
        right_first = tracker.observe_progress(right, 10.0, frame_id=1)
        left_second = tracker.observe_progress(left, 10.2, frame_id=2)

        self.assertEqual(left_first.observed_count, 1)
        self.assertEqual(right_first.observed_count, 1)
        self.assertIsNotNone(left_second.candidate)
        self.assertEqual(left_second.candidate.plate, "34ABC123")

    def test_strong_detector_consensus_requires_two_observations(self) -> None:
        tracker = ConfirmationTracker(required=2, window_seconds=3.0)
        candidate = PlateCandidate(
            "23LR640",
            0.992,
            "23LR640",
            1,
            variant_support=3,
            detector_crop_evidence=True,
        )

        outcome = tracker.observe_progress(candidate, 10.0, frame_id=1)

        self.assertEqual(outcome.required_count, 2)
        self.assertEqual(outcome.observed_count, 1)
        self.assertIsNone(outcome.candidate)

    def test_high_confidence_alone_does_not_confirm_from_one_observation(self) -> None:
        tracker = ConfirmationTracker(required=2, window_seconds=3.0)
        weak_candidate = PlateCandidate(
            "23AGF47",
            0.995,
            "23AGF47",
            1,
            variant_support=1,
            detector_crop_evidence=True,
        )
        fallback_candidate = replace(
            weak_candidate,
            plate="23LR640",
            raw_text="23LR640",
            variant_support=4,
            detector_crop_evidence=False,
            track_id=2,
        )

        weak = tracker.observe_progress(weak_candidate, 10.0, frame_id=1)
        fallback = tracker.observe_progress(fallback_candidate, 10.0, frame_id=1)

        self.assertIsNone(weak.candidate)
        self.assertEqual(weak.required_count, 2)
        self.assertIsNone(fallback.candidate)
        self.assertEqual(fallback.required_count, 2)

    def test_correction_trims_single_leading_plate_border_letter(self) -> None:
        self.assertEqual(
            correct_plate_candidate_with_cost("L23 LN 466"),
            ("23LN466", 1),
        )

    def test_correction_does_not_search_arbitrary_text_for_a_plate(self) -> None:
        self.assertIsNone(correct_plate_candidate_with_cost("XX23 LN 466"))

    def test_variant_consensus_beats_isolated_high_confidence_typo(self) -> None:
        candidate = select_best_candidate(
            (
                OcrSegment("23ABC123", 0.82, (0, 0, 10, 10), 0),
                OcrSegment("23ABC123", 0.84, (0, 0, 10, 10), 1),
                OcrSegment("23ABC128", 0.99, (0, 0, 10, 10), 2),
            ),
            camera_id=1,
        )

        self.assertEqual(candidate.plate, "23ABC123")
        self.assertEqual(candidate.variant_support, 2)

    def test_ambiguity_majority_requires_three_votes_and_two_vote_margin(self) -> None:
        tracker = ConfirmationTracker(required=2, window_seconds=3.0)
        first = PlateCandidate("23ABC123", 0.8, "23ABC123", 1)
        typo = PlateCandidate("23ABC128", 0.99, "23ABC128", 1)

        tracker.observe_progress(first, 1.0, frame_id=1)
        tracker.observe_progress(typo, 1.2, frame_id=2)
        two_vs_one = tracker.observe_progress(first, 1.4, frame_id=3)
        confirmed = tracker.observe_progress(first, 1.6, frame_id=4)

        self.assertIsNone(two_vs_one.candidate)
        self.assertEqual(
            (two_vs_one.observed_count, two_vs_one.required_count),
            (2, 3),
        )
        self.assertEqual(confirmed.candidate.plate, "23ABC123")
        self.assertEqual(confirmed.runner_up_votes, 1)

    def test_two_vs_two_and_three_vs_two_remain_ambiguous(self) -> None:
        tracker = ConfirmationTracker(required=2, window_seconds=3.0)
        first = PlateCandidate("23ABC123", 0.9, "23ABC123", 1)
        typo = PlateCandidate("23ABC128", 0.9, "23ABC128", 1)
        sequence = (first, typo, first, typo, first)
        outcomes = [
            tracker.observe_progress(item, 1.0 + index * 0.2, frame_id=index)
            for index, item in enumerate(sequence, start=1)
        ]

        self.assertIsNone(outcomes[3].candidate)
        self.assertIsNone(outcomes[4].candidate)
        self.assertEqual(outcomes[4].runner_up_votes, 2)

    def test_near_candidates_at_different_bboxes_do_not_share_votes(self) -> None:
        tracker = ConfirmationTracker(required=2, window_seconds=3.0)
        first_vehicle = PlateCandidate(
            "34ORF848",
            0.9,
            "34ORF848",
            1,
            detector_bbox=(10.0, 10.0, 80.0, 20.0),
        )
        second_vehicle = PlateCandidate(
            "34DRF848",
            0.9,
            "34DRF848",
            1,
            detector_bbox=(300.0, 10.0, 80.0, 20.0),
        )

        tracker.observe_progress(second_vehicle, 1.0, frame_id=1)
        first = tracker.observe_progress(first_vehicle, 1.2, frame_id=2)
        confirmed = tracker.observe_progress(first_vehicle, 1.4, frame_id=3)

        self.assertEqual(first.near_conflicts, ())
        self.assertEqual(first.required_count, 2)
        self.assertEqual(confirmed.candidate.plate, "34ORF848")

    def test_corrected_candidate_requires_three_distinct_frames(self) -> None:
        tracker = ConfirmationTracker(required=2, window_seconds=3.0)
        corrected = PlateCandidate(
            "23ABC123",
            0.9,
            "23ABCI23",
            1,
            correction_cost=1,
            variant_support=3,
            detector_crop_evidence=True,
        )

        first = tracker.observe_progress(corrected, 1.0, frame_id=1)
        second = tracker.observe_progress(corrected, 1.2, frame_id=2)
        third = tracker.observe_progress(corrected, 1.4, frame_id=3)

        self.assertEqual((first.required_count, second.required_count), (3, 3))
        self.assertIsNone(second.candidate)
        self.assertEqual(third.candidate.plate, "23ABC123")

    def test_post_save_near_plate_cannot_rely_on_full_roi_fallback(self) -> None:
        tracker = ConfirmationTracker(required=2, window_seconds=3.0)
        fallback = PlateCandidate("23ABC128", 0.99, "23ABC128", 1)

        outcomes = [
            tracker.observe_progress(
                fallback,
                1.0 + index * 0.2,
                frame_id=index,
                post_save_near_plate="23ABC123",
            )
            for index in range(1, 6)
        ]

        self.assertTrue(all(item.candidate is None for item in outcomes))
        self.assertTrue(all(item.required_count == 4 for item in outcomes))

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
    @staticmethod
    def _plate_crop(background: int, foreground: int) -> np.ndarray:
        crop = np.full((48, 160, 3), background, dtype=np.uint8)
        cv2.putText(
            crop,
            "34ABC123",
            (4, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (foreground, foreground, foreground),
            2,
            cv2.LINE_AA,
        )
        return crop

    def test_bright_plate_crop_selects_normal_profile(self) -> None:
        metrics = measure_crop_quality(self._plate_crop(190, 25))

        self.assertIs(classify_crop_quality(metrics), OcrImageProfile.NORMAL)

    def test_dark_plate_crop_selects_low_light_profile(self) -> None:
        metrics = measure_crop_quality(self._plate_crop(65, 20))

        self.assertIs(classify_crop_quality(metrics), OcrImageProfile.LOW_LIGHT)

    def test_normal_mean_low_local_contrast_selects_shadow_profile(self) -> None:
        metrics = measure_crop_quality(self._plate_crop(130, 116))

        self.assertGreaterEqual(metrics.luma_mean, LOW_LIGHT_THRESHOLD)
        self.assertIs(
            classify_crop_quality(metrics),
            OcrImageProfile.SHADOW_LOW_CONTRAST,
        )

    def test_clipped_bright_crop_selects_overexposed_profile(self) -> None:
        crop = np.full((48, 160, 3), 252, dtype=np.uint8)

        profile = classify_crop_quality(measure_crop_quality(crop))

        self.assertIs(profile, OcrImageProfile.OVEREXPOSED)

    def test_shadow_preprocessing_preserves_shape_dtype_and_input(self) -> None:
        crop = self._plate_crop(130, 116)
        original = crop.copy()

        variants = preprocess_shadow_variants(crop)

        self.assertEqual(len(variants), 1)
        self.assertTrue(all(item.shape == crop.shape for item in variants))
        self.assertTrue(all(item.dtype == np.uint8 for item in variants))
        self.assertTrue(all(item.ndim == 3 and item.shape[2] == 3 for item in variants))
        np.testing.assert_array_equal(crop, original)

    def test_normal_crop_does_not_add_shadow_ocr_inference(self) -> None:
        provider = SequencedOcrProvider([[]])

        result = recognize_detector_crops(
            provider,
            [self._plate_crop(190, 25)],
            camera_id=1,
            min_confidence=0.65,
        )

        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(result.inference_calls, 1)
        self.assertEqual(result.shadow_variant_count, 0)

    def test_high_confidence_current_candidate_skips_shadow_variants(self) -> None:
        provider = SequencedOcrProvider(
            [[OcrSegment("34ABC123", 0.97, (0.0, 0.0, 100.0, 30.0))]]
        )

        result = recognize_detector_crops(
            provider,
            [self._plate_crop(130, 116)],
            camera_id=1,
            min_confidence=0.65,
        )

        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(result.shadow_variant_count, 0)
        self.assertEqual(result.candidate.plate, "34ABC123")

    def test_shadow_variants_run_after_current_miss_and_recover_candidate(self) -> None:
        provider = SequencedOcrProvider(
            [
                [],
                [OcrSegment("O6 GG 9B6", 0.91, (0.0, 0.0, 100.0, 30.0))],
            ]
        )

        result = recognize_detector_crops(
            provider,
            [self._plate_crop(130, 116)],
            camera_id=1,
            min_confidence=0.65,
        )

        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(result.shadow_variant_count, 1)
        self.assertEqual(result.candidate.plate, "06GG986")
        self.assertEqual(result.candidate.correction_cost, 2)
        self.assertEqual(result.shadow_retry_reason, "missing-candidate")

    def test_tied_variant_consensus_uses_shadow_as_bounded_tie_breaker(self) -> None:
        provider = SequencedOcrProvider(
            [
                [
                    OcrSegment("01 KAC 53", 0.88, (0.0, 0.0, 100.0, 30.0), 0),
                    OcrSegment("01 KAC 53", 0.95, (0.0, 0.0, 100.0, 30.0), 1),
                    OcrSegment("31 KAC 53", 0.94, (0.0, 0.0, 100.0, 30.0), 2),
                    OcrSegment("31 KAC 53", 0.97, (0.0, 0.0, 100.0, 30.0), 3),
                ],
                [OcrSegment("01 KAC 53", 0.91, (0.0, 0.0, 100.0, 30.0))],
            ]
        )

        result = recognize_detector_crops(
            provider,
            [self._plate_crop(65, 20)],
            camera_id=1,
            min_confidence=0.65,
        )

        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(result.shadow_retry_reason, "variant-consensus-conflict")
        self.assertEqual(result.candidate.plate, "01KAC53")
        self.assertEqual(result.candidate.variant_support, 3)

    def test_normal_one_vote_margin_conflict_uses_bounded_shadow_tie_breaker(self) -> None:
        provider = SequencedOcrProvider(
            [
                [
                    OcrSegment("34 DRF 846", 0.818, (0.0, 0.0, 100.0, 30.0), 0),
                    OcrSegment("34 DRF 848", 0.917, (0.0, 0.0, 100.0, 30.0), 1),
                    OcrSegment("34 DRF 846", 0.817, (0.0, 0.0, 100.0, 30.0), 2),
                ],
                [OcrSegment("34 DRF 848", 0.950, (0.0, 0.0, 100.0, 30.0))],
            ]
        )

        result = recognize_detector_crops(
            provider,
            [self._plate_crop(190, 25)],
            camera_id=1,
            min_confidence=0.65,
        )

        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(result.shadow_retry_reason, "variant-consensus-conflict")
        self.assertEqual(result.shadow_variant_count, 1)
        self.assertEqual(result.candidate.plate, "34DRF848")
        self.assertEqual(result.candidate.variant_support, 2)

    def test_low_light_recovery_adds_only_distinct_gamma_gray_variant(self) -> None:
        provider = SequencedOcrProvider([[], []])

        result = recognize_detector_crops(
            provider,
            [self._plate_crop(65, 20)],
            camera_id=1,
            min_confidence=0.65,
        )

        self.assertEqual(result.profiles, (OcrImageProfile.LOW_LIGHT,))
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(result.current_variant_count, 4)
        self.assertEqual(result.shadow_variant_count, 1)
        self.assertEqual(len(provider.calls[1]), 1)

    def test_shadow_variants_run_after_below_threshold_current_candidate(self) -> None:
        provider = SequencedOcrProvider(
            [
                [OcrSegment("34ABC123", 0.60, (0.0, 0.0, 100.0, 30.0))],
                [OcrSegment("34ABC123", 0.93, (0.0, 0.0, 100.0, 30.0))],
            ]
        )

        result = recognize_detector_crops(
            provider,
            [self._plate_crop(130, 116)],
            camera_id=1,
            min_confidence=0.65,
        )

        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(result.shadow_variant_count, 1)
        self.assertEqual(result.candidate.plate, "34ABC123")
        self.assertEqual(result.candidate.confidence, 0.93)
        self.assertEqual(result.shadow_retry_reason, "low-confidence")

    def test_ocr_job_priority_contract_is_unchanged(self) -> None:
        self.assertEqual(
            [
                OcrJobType.DETECTOR_CROP.priority,
                OcrJobType.DETECTOR_ERROR_FALLBACK.priority,
                OcrJobType.ZERO_DETECTION_FALLBACK.priority,
                OcrJobType.STATIC_ZERO_DETECTION_RESCUE.priority,
            ],
            [3, 2, 1, 0],
        )

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
        detection: PlateDetection | None = None,
        track_id: int | None = None,
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
            detections=(detection,) if detection is not None else (),
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
            ocr_crop_detections=(detection,) if detection is not None else (),
            track_id=track_id,
        )

    def _run_single_observation_track_end(
        self,
        segments: list[OcrSegment] | None = None,
        *,
        frame_value: int = 37,
        fallback: bool = False,
        retire_before_result: bool = False,
        frame_buffer: PreDetectionFrameBuffer | None = None,
        ocr_sequences: list[list[OcrSegment]] | None = None,
    ) -> tuple[object, object, PlateRecognitionProcessor]:
        from app.plate_tracking import PlateTrackManager

        track_manager = PlateTrackManager(
            max_active_tracks_per_camera=2,
            timeout_ms=500,
            iou_threshold=0.25,
        )
        detection = PlateDetection(0.91, 20, 10, 100, 30)
        track_id = track_manager.update(
            1,
            (detection,),
            observed_at=10.0,
            activity_at=100.0,
        ).assignments[0].track_id
        provider_data = ocr_sequences if ocr_sequences is not None else [segments or []]
        processor = PlateRecognitionProcessor(
            SequencedOcrProvider(provider_data),
            self.plate_service,
            recognition_config(
                Path(self.temp_directory.name),
                stabilization_window_ms=2_000,
            ),
            track_manager=track_manager,
            frame_buffer=frame_buffer,
        )
        self.assertTrue(track_manager.mark_ocr_scheduled(track_id))
        if retire_before_result:
            track_manager.expire_due(100.6)
            self.assertTrue(track_manager.can_accept_ocr_result(track_id))
        awaiting = processor.process_ocr_job(
            self._make_ocr_job(
                frame_id=8_000 + track_id,
                observed_at=10.1,
                frame_value=frame_value,
                fallback_reason="detector-error" if fallback else None,
                detection=None if fallback else detection,
                track_id=track_id,
            ),
            queue_depth=0,
        )
        track_manager.mark_ocr_finished(track_id)
        if not retire_before_result:
            track_manager.expire_due(100.6)
        finalized_tracks = track_manager.consume_finalized()
        self.assertEqual([item.track_id for item in finalized_tracks], [track_id])
        finalized = processor.finalize_track(1, track_id)
        return awaiting, finalized, processor

    def test_track_end_single_observation_without_buffer_discards(self) -> None:
        awaiting, finalized, processor = self._run_single_observation_track_end(
            [
                OcrSegment("23ABC123", 0.93, (0, 0, 10, 10), 0),
                OcrSegment("23 ABC 123", 0.94, (0, 0, 10, 10), 1),
            ],
            frame_value=71,
        )

        self.assertIs(awaiting.state, RecognitionState.AWAITING_CONFIRMATION)
        self.assertEqual(awaiting.confirmation_count, 1)
        self.assertIs(finalized.state, RecognitionState.AMBIGUOUS_DISCARDED)
        self.assertIs(finalized.finalization_source, FinalizationSource.AMBIGUOUS_DISCARD)
        self.assertEqual(finalized.suppression_reason, "insufficient-temporal-evidence")
        self.assertIsNone(finalized.record)
        self.assertEqual(self._record_count(), 0)
        self.assertEqual(processor.pending_decision_count, 0)
        self.assertNotIn((1, finalized.candidate.track_id), processor.confirmations._observations)

    def test_track_end_waits_for_pending_ocr_then_evaluates(self) -> None:
        awaiting, finalized, _processor = self._run_single_observation_track_end(
            [
                OcrSegment("24ABC124", 0.94, (0, 0, 10, 10), 0),
                OcrSegment("24 ABC 124", 0.93, (0, 0, 10, 10), 1),
            ],
            retire_before_result=True,
        )

        self.assertIs(awaiting.state, RecognitionState.AWAITING_CONFIRMATION)
        self.assertIs(finalized.state, RecognitionState.AMBIGUOUS_DISCARDED)
        self.assertIs(finalized.finalization_source, FinalizationSource.AMBIGUOUS_DISCARD)

    def test_track_evidence_survives_cleanup_deadline_until_track_finalization(self) -> None:
        from app.plate_tracking import PlateTrackManager

        track_manager = PlateTrackManager(
            max_active_tracks_per_camera=2,
            timeout_ms=3_000,
            iou_threshold=0.25,
        )
        detection = PlateDetection(0.91, 20, 10, 100, 30)
        track_id = track_manager.update(
            1, (detection,), 10.0, activity_at=100.0
        ).assignments[0].track_id
        processor = PlateRecognitionProcessor(
            SequencedOcrProvider(
                [[
                    OcrSegment("24XYZ124", 0.94, (0, 0, 10, 10), 0),
                    OcrSegment("24 XYZ 124", 0.93, (0, 0, 10, 10), 1),
                ]]
            ),
            self.plate_service,
            recognition_config(Path(self.temp_directory.name), stabilization_window_ms=2_000),
            track_manager=track_manager,
        )
        track_manager.mark_ocr_scheduled(track_id)
        processor.process_ocr_job(
            self._make_ocr_job(
                frame_id=8_100,
                observed_at=10.1,
                detection=detection,
                track_id=track_id,
            ),
            queue_depth=0,
        )
        track_manager.mark_ocr_finished(track_id)
        cleanup_deadline = processor.next_pending_deadline()

        self.assertEqual(processor.finalize_due(cleanup_deadline + 1.0), [])
        self.assertEqual(processor.pending_decision_count, 1)
        track_manager.expire_due(103.1)
        finalized_track = track_manager.consume_finalized()[0]
        finalized = processor.finalize_track(
            finalized_track.camera_id,
            finalized_track.track_id,
        )

        self.assertIs(finalized.state, RecognitionState.AMBIGUOUS_DISCARDED)
        self.assertIs(finalized.finalization_source, FinalizationSource.AMBIGUOUS_DISCARD)
        self.assertEqual(processor.pending_decision_count, 0)

    def test_track_end_rejects_weak_corrected_and_fallback_only_observations(self) -> None:
        cases = (
            (
                "low-confidence",
                [
                    OcrSegment("25ABC125", 0.91, (0, 0, 10, 10), 0),
                    OcrSegment("25 ABC 125", 0.91, (0, 0, 10, 10), 1),
                ],
                False,
                "single-observation-low-confidence",
            ),
            (
                "one-variant",
                [OcrSegment("26ABC126", 0.96, (0, 0, 10, 10), 0)],
                False,
                "insufficient-variant-consensus",
            ),
            (
                "corrected",
                [
                    OcrSegment("27ABCI27", 0.96, (0, 0, 10, 10), 0),
                    OcrSegment("27 ABC I27", 0.95, (0, 0, 10, 10), 1),
                ],
                False,
                "corrected-candidate",
            ),
            (
                "fallback",
                [
                    OcrSegment("28ABC128", 0.97, (0, 0, 10, 10), 0),
                    OcrSegment("28 ABC 128", 0.96, (0, 0, 10, 10), 1),
                ],
                True,
                "fallback-only",
            ),
        )
        for name, segments, fallback, reason in cases:
            with self.subTest(name=name):
                _awaiting, finalized, _processor = self._run_single_observation_track_end(
                    segments,
                    fallback=fallback,
                )
                self.assertIs(finalized.state, RecognitionState.AMBIGUOUS_DISCARDED)
                self.assertEqual(finalized.suppression_reason, reason)
                self.assertIs(finalized.finalization_source, FinalizationSource.AMBIGUOUS_DISCARD)

        self.assertEqual(self._record_count(), 0)

    def test_track_end_near_conflict_is_not_rescued(self) -> None:
        from app.plate_tracking import PlateTrackManager

        track_manager = PlateTrackManager(
            max_active_tracks_per_camera=2,
            timeout_ms=500,
            iou_threshold=0.25,
        )
        detection = PlateDetection(0.91, 20, 10, 100, 30)
        track_id = track_manager.update(
            1, (detection,), 10.0, activity_at=100.0
        ).assignments[0].track_id
        processor = PlateRecognitionProcessor(
            SequencedOcrProvider(
                [
                    [OcrSegment("29ABC123", 0.95, (0, 0, 10, 10), 0)],
                    [OcrSegment("29ABC128", 0.96, (0, 0, 10, 10), 0)],
                ]
            ),
            self.plate_service,
            recognition_config(Path(self.temp_directory.name), stabilization_window_ms=2_000),
            track_manager=track_manager,
        )
        for frame_id in (1, 2):
            self.assertTrue(track_manager.mark_ocr_scheduled(track_id))
            processor.process_ocr_job(
                self._make_ocr_job(
                    frame_id=frame_id,
                    observed_at=10.0 + frame_id * 0.1,
                    detection=detection,
                    track_id=track_id,
                ),
                queue_depth=0,
            )
            track_manager.mark_ocr_finished(track_id)
        track_manager.expire_due(100.6)
        finalized = processor.finalize_track(
            1, track_manager.consume_finalized()[0].track_id
        )

        self.assertIs(finalized.state, RecognitionState.AMBIGUOUS_DISCARDED)
        self.assertEqual(finalized.suppression_reason, "near-conflict")
        self.assertEqual(self._record_count(), 0)

    def test_track_end_rescue_still_uses_duplicate_suppression(self) -> None:
        segments = [
            OcrSegment("30ABC130", 0.94, (0, 0, 10, 10), 0),
            OcrSegment("30 ABC 130", 0.93, (0, 0, 10, 10), 1),
        ]
        buffer_1 = PreDetectionFrameBuffer(self.config)
        buffer_1.ingest(self._make_snapshot(7990, 9.8), motion_score=0.0)
        _awaiting, first, _processor = self._run_single_observation_track_end(
            frame_buffer=buffer_1,
            ocr_sequences=[segments, segments],
        )
        buffer_2 = PreDetectionFrameBuffer(self.config)
        buffer_2.ingest(self._make_snapshot(7991, 9.8), motion_score=0.0)
        _awaiting, second, _processor = self._run_single_observation_track_end(
            frame_buffer=buffer_2,
            ocr_sequences=[segments, segments],
        )

        self.assertIs(first.state, RecognitionState.AMBIGUOUS_DISCARDED)
        self.assertIs(second.state, RecognitionState.AMBIGUOUS_DISCARDED)
        self.assertIs(second.finalization_source, FinalizationSource.AMBIGUOUS_DISCARD)
        self.assertEqual(self._record_count(), 0)

    def test_finalizing_single_frame_track_does_not_touch_other_track_decision(self) -> None:
        from app.plate_tracking import PlateTrackManager

        track_manager = PlateTrackManager(
            max_active_tracks_per_camera=2,
            timeout_ms=500,
            iou_threshold=0.25,
        )
        detections = (
            PlateDetection(0.91, 20, 10, 100, 30),
            PlateDetection(0.92, 240, 10, 100, 30),
        )
        update = track_manager.update(1, detections, 10.0, activity_at=100.0)
        rescue_track, normal_track = (item.track_id for item in update.assignments)
        processor = PlateRecognitionProcessor(
            SequencedOcrProvider(
                [
                    [
                        OcrSegment("31ABC131", 0.94, (0, 0, 10, 10), 0),
                        OcrSegment("31 ABC 131", 0.93, (0, 0, 10, 10), 1),
                    ],
                    [OcrSegment("32XYZ132", 0.95, (0, 0, 10, 10), 0)],
                    [OcrSegment("32XYZ132", 0.96, (0, 0, 10, 10), 0)],
                ]
            ),
            self.plate_service,
            recognition_config(Path(self.temp_directory.name), stabilization_window_ms=2_000),
            track_manager=track_manager,
        )
        for frame_id, track_id, detection in (
            (1, rescue_track, detections[0]),
            (1, normal_track, detections[1]),
            (2, normal_track, detections[1]),
        ):
            track_manager.mark_ocr_scheduled(track_id)
            processor.process_ocr_job(
                self._make_ocr_job(
                    frame_id=frame_id,
                    observed_at=10.0 + frame_id * 0.1,
                    detection=detection,
                    track_id=track_id,
                ),
                queue_depth=0,
            )
            track_manager.mark_ocr_finished(track_id)

        self.assertEqual(processor.pending_decision_count, 2)
        discarded = processor.finalize_track(1, rescue_track)
        self.assertIs(discarded.state, RecognitionState.AMBIGUOUS_DISCARDED)
        self.assertIs(discarded.finalization_source, FinalizationSource.AMBIGUOUS_DISCARD)
        self.assertEqual(processor.pending_decision_count, 1)
        remaining = processor.finalize_track(1, normal_track)
        self.assertIs(remaining.state, RecognitionState.SAVED)
        self.assertIs(remaining.finalization_source, FinalizationSource.NORMAL_CONFIRMATION)
        self.assertEqual(remaining.record.plate, "32XYZ132")

    def test_runtime_health_classifies_confirmation_and_track_end_decisions(self) -> None:
        worker = PlateRecognitionWorker(
            self.plate_service,
            self.config,
            provider_factory=FakeOcrProvider,
        )
        candidate = PlateCandidate("33ABC133", 0.94, "33ABC133", 1)
        worker._publish_outcome(
            1,
            RecognitionOutcome(
                candidate,
                None,
                RecognitionState.AWAITING_CONFIRMATION,
                confirmation_count=1,
                confirmation_required=2,
            ),
        )
        worker._publish_outcome(
            1,
            RecognitionOutcome(
                candidate,
                object(),
                RecognitionState.SAVED,
                finalization_source=FinalizationSource.NORMAL_CONFIRMATION,
            ),
        )
        worker._publish_outcome(
            1,
            RecognitionOutcome(
                candidate,
                object(),
                RecognitionState.SAVED,
                finalization_source=FinalizationSource.BUFFERED_MULTI_FRAME,
                track_ended=True,
            ),
        )
        for reason in (
            "single-observation-low-confidence",
            "near-conflict",
            "corrected-candidate",
        ):
            worker._publish_outcome(
                1,
                RecognitionOutcome(
                    candidate,
                    None,
                    RecognitionState.AMBIGUOUS_DISCARDED,
                    suppression_reason=reason,
                    finalization_source=FinalizationSource.AMBIGUOUS_DISCARD,
                    track_ended=True,
                ),
            )

        health = worker.runtime_health()
        self.assertEqual(health.awaiting_confirmation, 1)
        self.assertEqual(health.normal_confirmed, 1)
        self.assertEqual(health.confirmed_live_multiframe, 1)
        self.assertEqual(health.buffered_confirmation_succeeded, 1)
        self.assertEqual(health.buffered_confirmation_failed, 2)
        self.assertEqual(health.buffered_confirmation_conflict, 1)
        self.assertEqual(health.single_frame_discarded, 3)
        self.assertEqual(health.track_end_rescued, 1)
        self.assertEqual(health.track_end_discarded, 3)

    def test_two_tracks_on_same_frames_persist_independent_plate_results(self) -> None:
        from app.plate_tracking import PlateTrackManager

        track_manager = PlateTrackManager(
            max_active_tracks_per_camera=2,
            timeout_ms=1_200,
            iou_threshold=0.25,
        )
        detections = (
            PlateDetection(0.90, 20, 10, 100, 30),
            PlateDetection(0.90, 240, 10, 100, 30),
        )
        update = track_manager.update(1, detections, 10.0)
        left_id, right_id = (item.track_id for item in update.assignments)
        processor = PlateRecognitionProcessor(
            SequencedOcrProvider(
                [
                    [OcrSegment("34ABC123", 0.95, (0, 0, 10, 10))],
                    [OcrSegment("35XYZ789", 0.96, (0, 0, 10, 10))],
                    [OcrSegment("34ABC123", 0.97, (0, 0, 10, 10))],
                    [OcrSegment("35XYZ789", 0.98, (0, 0, 10, 10))],
                ]
            ),
            self.plate_service,
            self.config,
            track_manager=track_manager,
        )
        outcomes = []
        for frame_id, track_id, plate_detection in (
            (1, left_id, detections[0]),
            (1, right_id, detections[1]),
            (2, left_id, detections[0]),
            (2, right_id, detections[1]),
        ):
            self.assertTrue(track_manager.mark_ocr_scheduled(track_id))
            try:
                outcomes.append(
                    processor.process_ocr_job(
                        self._make_ocr_job(
                            frame_id=frame_id,
                            observed_at=10.0 + frame_id * 0.2,
                            detection=plate_detection,
                            track_id=track_id,
                        ),
                        queue_depth=0,
                    )
                )
            finally:
                track_manager.mark_ocr_finished(track_id)

        self.assertEqual(
            [outcome.state for outcome in outcomes],
            [
                RecognitionState.AWAITING_CONFIRMATION,
                RecognitionState.AWAITING_CONFIRMATION,
                RecognitionState.SAVED,
                RecognitionState.SAVED,
            ],
        )
        self.assertEqual(
            {outcome.record.plate for outcome in outcomes if outcome.record},
            {"34ABC123", "35XYZ789"},
        )

    def test_stabilization_keeps_two_track_decisions_independent(self) -> None:
        from app.plate_tracking import PlateTrackManager

        config = recognition_config(
            self.config.model_root,
            stabilization_window_ms=2_000,
            stabilization_min_hold_ms=500,
        )
        track_manager = PlateTrackManager(
            max_active_tracks_per_camera=2,
            timeout_ms=1_200,
            iou_threshold=0.25,
        )
        detections = (
            PlateDetection(0.90, 20, 10, 100, 30),
            PlateDetection(0.90, 240, 10, 100, 30),
        )
        update = track_manager.update(1, detections, 10.0)
        left_id, right_id = (item.track_id for item in update.assignments)
        processor = PlateRecognitionProcessor(
            SequencedOcrProvider(
                [
                    [OcrSegment("34ABC123", 0.95, (0, 0, 10, 10))],
                    [OcrSegment("35XYZ789", 0.96, (0, 0, 10, 10))],
                    [OcrSegment("34ABC123", 0.97, (0, 0, 10, 10))],
                    [OcrSegment("35XYZ789", 0.98, (0, 0, 10, 10))],
                ]
            ),
            self.plate_service,
            config,
            track_manager=track_manager,
        )
        for frame_id, track_id, plate_detection in (
            (1, left_id, detections[0]),
            (1, right_id, detections[1]),
            (2, left_id, detections[0]),
            (2, right_id, detections[1]),
        ):
            track_manager.mark_ocr_scheduled(track_id)
            try:
                processor.process_ocr_job(
                    self._make_ocr_job(
                        frame_id=frame_id,
                        observed_at=10.0 + frame_id * 0.2,
                        detection=plate_detection,
                        track_id=track_id,
                    ),
                    queue_depth=0,
                )
            finally:
                track_manager.mark_ocr_finished(track_id)

        self.assertEqual(processor.pending_decision_count, 2)
        finalized = processor.finalize_due(time.monotonic() + 3.0)

        self.assertEqual(len(finalized), 2)
        self.assertEqual(
            {outcome.record.plate for _, outcome in finalized if outcome.record},
            {"34ABC123", "35XYZ789"},
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
        self.assertEqual(provider.images[2].shape, (60, 270, 3))
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
        self.assertIn("plate_crops=180x40", diagnostic)
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

    def test_presence_release_cannot_create_duplicate_parked_movement(self) -> None:
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
            detected_at + timedelta(minutes=30),
            monotonic_at=1_800.0,
        )
        second = processor.process(
            camera.id,
            camera.direction,
            frame,
            detected_at + timedelta(minutes=30, seconds=1),
            monotonic_at=1_801.0,
        )

        self.assertIsNotNone(first.record)
        self.assertIsNone(second.record)
        self.assertTrue(second.duplicate)
        self.assertEqual(second.suppression_reason, "same-direction-state")
        self.assertEqual(self._record_count(), 1)
        self.assertEqual(len(list(Path(self.temp_directory.name).rglob("*.jpg"))), 1)

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
            detected_at + timedelta(seconds=1),
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
        health = worker.runtime_health()
        self.assertEqual(health.frames_ingested, 101)
        self.assertIsNotNone(health.last_frame_age_seconds)
        self.assertEqual(health.ocr_jobs_processed, 0)

    def test_analysis_ingestion_keeps_bounded_frames_when_ui_preview_is_coalesced(self) -> None:
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

    def test_two_fps_stream_keeps_five_seconds_of_unique_history_bounded(self) -> None:
        config = replace(
            self.config,
            pre_detection_buffer_duration_ms=5_000,
            pre_detection_buffer_max_frames_per_camera=16,
        )
        buffer = PreDetectionFrameBuffer(config)

        for index in range(10):
            buffer.ingest(
                self._make_snapshot(index + 1, index * 0.5),
                motion_score=0.0,
            )

        snapshots = buffer.snapshots(1)
        self.assertEqual(len(snapshots), 10)
        self.assertEqual(len({snapshot.frame_id for snapshot in snapshots}), 10)
        self.assertLessEqual(buffer.ring_depth(1), 16)
        self.assertEqual(config.recognition_interval_ms, 250)

    def test_ring_health_reports_effective_duration_fps_and_memory(self) -> None:
        config = replace(
            self.config,
            pre_detection_buffer_duration_ms=5_000,
            pre_detection_buffer_max_frames_per_camera=16,
        )
        buffer = PreDetectionFrameBuffer(config)
        for index in range(4):
            buffer.ingest(
                self._make_snapshot(index + 1, index * 0.25),
                motion_score=0.0,
            )

        health = buffer.health(1, now=1.0)

        self.assertEqual(health.ring_depth, 4)
        self.assertEqual(health.frame_cap, 16)
        self.assertEqual(health.configured_duration_ms, 5_000)
        self.assertAlmostEqual(health.oldest_frame_age_ms, 1_000.0)
        self.assertAlmostEqual(health.newest_frame_age_ms, 250.0)
        self.assertAlmostEqual(health.effective_ring_duration_ms, 750.0)
        self.assertAlmostEqual(health.recognition_ingest_fps, 4.0)
        expected_mb = 4 * 200 * 400 * 3 / (1024**2)
        self.assertAlmostEqual(health.estimated_ring_memory_mb, expected_mb)
        self.assertEqual(health.active_event_frames, 0)

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

    def test_long_motion_event_retains_bounded_early_middle_and_late_frames(self) -> None:
        config = replace(
            self.config,
            pre_detection_buffer_duration_ms=10_000,
            pre_detection_buffer_max_frames_per_camera=8,
            motion_pre_roll_ms=0,
            motion_post_roll_ms=0,
            motion_quiet_ms=0,
            motion_event_max_duration_ms=30_000,
        )
        buffer = PreDetectionFrameBuffer(config)
        for index in range(60):
            buffer.ingest(
                self._make_snapshot(index + 1, index * 0.25),
                motion_score=0.1,
            )
        completed = buffer.ingest(
            self._make_snapshot(61, 15.0),
            motion_score=0.0,
        )

        self.assertEqual(len(completed), 1)
        frame_ids = [item.frame_id for item in completed[0].frames]
        self.assertLessEqual(len(frame_ids), 8)
        self.assertEqual(frame_ids[0], 1)
        self.assertEqual(frame_ids[-1], 61)
        self.assertTrue(any(2 <= frame_id <= 20 for frame_id in frame_ids))
        self.assertTrue(any(21 <= frame_id <= 40 for frame_id in frame_ids))
        self.assertTrue(any(41 <= frame_id <= 60 for frame_id in frame_ids))

    def test_forty_frame_cap_keeps_over_three_seconds_at_twelve_fps(self) -> None:
        config = replace(
            self.config,
            pre_detection_buffer_duration_ms=5_000,
            pre_detection_buffer_max_frames_per_camera=40,
        )
        buffer = PreDetectionFrameBuffer(config)
        for index in range(72):
            buffer.ingest(
                self._make_snapshot(index + 1, index / 12.0),
                motion_score=0.0,
            )

        health = buffer.health(1, now=72 / 12.0)

        self.assertEqual(health.ring_depth, 40)
        self.assertAlmostEqual(health.recognition_ingest_fps, 12.0)
        self.assertGreaterEqual(health.effective_ring_duration_ms, 3_200.0)
        self.assertLessEqual(health.effective_ring_duration_ms, 5_000.0)

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

    def test_worker_schedules_event_end_fallback_when_detector_never_succeeds(self) -> None:
        provider = RecordingOcrProvider(confidence=0.97)
        detector = FakePlateDetector([])
        worker = PlateRecognitionWorker(
            self.plate_service,
            self.config,
            provider_factory=lambda: provider,
            detector_factory=lambda: detector,
        )
        snapshots = tuple(
            replace(
                self._make_snapshot(
                    800 + index,
                    80.0 + index * 0.5,
                    value=30 + index * 10,
                ),
                motion_score=0.05 + index * 0.01,
            )
            for index in range(3)
        )
        event = MotionEvent(
            80,
            1,
            Direction.ENTRY,
            80.0,
            81.0,
            time.monotonic(),
            snapshots,
        )
        with worker._lock:
            worker._completed_motion_events.append(event)
        outcome_ready = threading.Event()
        outcomes = []
        worker.outcome_changed.connect(
            lambda _camera_id, outcome: (
                outcomes.append(outcome),
                outcome_ready.set(),
            ),
            Qt.ConnectionType.DirectConnection,
        )
        thread = threading.Thread(target=worker.run)
        thread.start()
        try:
            self.assertTrue(outcome_ready.wait(1.5))
        finally:
            worker.request_stop()
            thread.join(2.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(provider.calls, 1)
        self.assertEqual(detector.calls, 0)
        self.assertIs(outcomes[0].state, RecognitionState.AWAITING_CONFIRMATION)

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
            OcrJobType.STATIC_ZERO_DETECTION_RESCUE,
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

    def test_zero_detection_fallback_buffer_allows_two_then_coalesces(self) -> None:
        buffer = OcrJobBuffer(max_per_camera=3, max_age_ms=2_500)
        first = self._make_ocr_job(job_type=OcrJobType.ZERO_DETECTION_FALLBACK)
        second = self._make_ocr_job(job_type=OcrJobType.ZERO_DETECTION_FALLBACK)
        third = self._make_ocr_job(job_type=OcrJobType.ZERO_DETECTION_FALLBACK)

        accepted_first = buffer.add(first)
        accepted_second = buffer.add(second)
        coalesced = buffer.add(third)
        consumed = (buffer.take(), buffer.take())
        accepted_again = buffer.add(third)

        self.assertTrue(accepted_first.accepted)
        self.assertTrue(accepted_second.accepted)
        self.assertFalse(coalesced.accepted)
        self.assertTrue(coalesced.coalesced)
        self.assertEqual(consumed, (first, second))
        self.assertTrue(accepted_again.accepted)

    def test_static_rescue_buffer_keeps_at_most_one_pending_job(self) -> None:
        buffer = OcrJobBuffer(max_per_camera=3, max_age_ms=2_500)
        first = self._make_ocr_job(
            job_type=OcrJobType.STATIC_ZERO_DETECTION_RESCUE,
            frame_id=1,
        )
        second = replace(first, frame_id=2)

        accepted = buffer.add(first)
        coalesced = buffer.add(second)

        self.assertTrue(accepted.accepted)
        self.assertFalse(coalesced.accepted)
        self.assertTrue(coalesced.coalesced)
        self.assertEqual(buffer.pending_count(1), 1)

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
        buffer.add(replace(fallback, frame_id=42))

        with self.assertLogs("app.plate_recognition", level="DEBUG") as captured:
            buffer.add(replace(fallback, frame_id=43))

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
        rescue = self._make_ocr_job(
            camera_id=1,
            job_type=OcrJobType.STATIC_ZERO_DETECTION_RESCUE,
        )
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
        for job in (rescue, zero, error, crop):
            buffer.add(job)

        self.assertEqual(
            [buffer.take().job_type for _ in range(4)],
            [
                OcrJobType.DETECTOR_CROP,
                OcrJobType.DETECTOR_ERROR_FALLBACK,
                OcrJobType.ZERO_DETECTION_FALLBACK,
                OcrJobType.STATIC_ZERO_DETECTION_RESCUE,
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
                self.assertEqual(buffer.pending_count(fallback_camera_id), 2)

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
        self.assertIn("candidate_rejection_reason=none", diagnostic)
        self.assertIn("confirmation=1/2", diagnostic)
        self.assertIn("raw_ocr_segments=v0:'34 ABC 123':0.970:34ABC123", diagnostic)
        self.assertIn("crop_quality=crop0=120x40", diagnostic)
        self.assertIn("profiles=LOW_LIGHT", diagnostic)
        self.assertIn("current_variants=4", diagnostic)
        self.assertIn("shadow_variants=0", diagnostic)
        self.assertIn("inference_calls=1", diagnostic)
        self.assertIn("text_detection_boxes=", diagnostic)
        self.assertIn("variant_trace=adaptive-color", diagnostic)

    def test_ocr_worker_diagnostics_include_candidate_rejection_reason(self) -> None:
        processor = PlateRecognitionProcessor(
            FakeOcrProvider(confidence=0.64),
            self.plate_service,
            self.config,
        )
        job = self._make_ocr_job(
            observed_at=time.monotonic(),
            frame_id=56,
            job_type=OcrJobType.DETECTOR_CROP,
        )

        with self.assertLogs("app.plate_recognition", level="DEBUG") as captured:
            outcome = processor.process_ocr_job(job, queue_depth=0)

        diagnostic = next(
            line for line in captured.output if "OCR worker diagnostics" in line
        )
        self.assertIs(outcome.state, RecognitionState.LOW_CONFIDENCE)
        self.assertIn(
            "candidate_rejection_reason=low_confidence",
            diagnostic,
        )

    def test_one_shot_capture_emits_info_trace_and_bypasses_throttles(self) -> None:
        now = time.monotonic()
        worker = PlateRecognitionWorker(
            self.plate_service,
            self.config,
            provider_factory=FakeOcrProvider,
        )
        worker._last_detection_diagnostic_at[1] = now
        item = _PendingFrame(
            direction=Direction.ENTRY,
            frame=np.zeros((200, 400, 3), dtype=np.uint8),
            received_at=now,
            observed_at=now,
            captured_at=datetime(2026, 8, 17, tzinfo=timezone.utc),
            frame_id=88,
        )
        detection_result = DetectionJobResult(
            None,
            (),
            False,
            None,
            5.0,
            100.0,
            diagnostic_capture=True,
        )
        with self.assertLogs("app.plate_recognition", level="INFO") as detector_logs:
            worker._log_detection_diagnostics(
                1,
                item,
                detection_result,
                None,
            )
        self.assertTrue(
            any(
                "Detector worker diagnostics" in line and "frame_id=88" in line
                for line in detector_logs.output
            )
        )

        processor = PlateRecognitionProcessor(
            FakeOcrProvider(confidence=0.97),
            self.plate_service,
            self.config,
        )
        processor._last_diagnostic_logged_at[1] = now
        job = replace(
            self._make_ocr_job(
                observed_at=now,
                frame_id=88,
                detection=PlateDetection(0.80, 20, 10, 60, 18),
            ),
            diagnostic_capture=True,
        )
        with self.assertLogs("app.plate_recognition", level="INFO") as ocr_logs:
            processor.process_ocr_job(job, queue_depth=0)
        self.assertTrue(
            any(
                "OCR worker diagnostics" in line and "frame_id=88" in line
                for line in ocr_logs.output
            )
        )
        self.assertTrue(any("aspect=" in line for line in ocr_logs.output))

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

    def test_ocr_search_tiles_are_bounded_overlapping_and_cover_roi(self) -> None:
        roi = np.zeros((240, 960, 3), dtype=np.uint8)

        tiles = build_ocr_search_tiles(roi)

        self.assertEqual(len(tiles), 3)
        self.assertEqual(tiles[0].roi_box[0], 0)
        self.assertEqual(
            tiles[-1].roi_box[0] + tiles[-1].roi_box[2],
            roi.shape[1],
        )
        self.assertLess(tiles[1].roi_box[0], tiles[0].roi_box[2])
        self.assertTrue(all(tile.roi_box[3] == roi.shape[0] for tile in tiles))

    def test_ocr_search_tile_maps_box_and_early_exits_on_first_valid_plate(self) -> None:
        provider = SequencedOcrProvider(
            [
                [],
                [OcrSegment("23 AAY 264", 0.97, (5.0, 6.0, 105.0, 36.0))],
                [OcrSegment("99ZZ999", 0.99, (0.0, 0.0, 1.0, 1.0))],
            ]
        )
        roi = np.zeros((240, 960, 3), dtype=np.uint8)

        result = recognize_ocr_search_tiles(provider, roi, 1, 0.65)

        self.assertEqual(result.candidate.plate, "23AAY264")
        self.assertEqual(result.inference_calls, 2)
        self.assertEqual(len(provider.calls), 2)
        second_x = result.tiles[1].roi_box[0]
        self.assertEqual(result.segments[0].box[0], second_x + 5.0)

    def test_spatial_rescue_skips_full_roi_after_tile_success(self) -> None:
        provider = RecordingOcrProvider(confidence=0.97)
        processor = PlateRecognitionProcessor(provider, self.plate_service, self.config)
        roi = np.full((240, 960, 3), 100, dtype=np.uint8)
        job = replace(
            self._make_ocr_job(frame_id=621, fallback_reason="zero-detection"),
            roi_crop=roi,
            ocr_crops=(roi,),
            spatial_search_rescue=True,
        )

        outcome = processor.process_ocr_job(job, queue_depth=0)

        self.assertIs(outcome.state, RecognitionState.AWAITING_CONFIRMATION)
        self.assertEqual(provider.calls, 1)
        self.assertLess(provider.images[0].shape[1], roi.shape[1])

    def test_spatial_rescue_has_three_tile_plus_two_roi_call_ceiling(self) -> None:
        provider = SequencedOcrProvider([[], [], [], [], []])
        processor = PlateRecognitionProcessor(provider, self.plate_service, self.config)
        roi = np.full((240, 960, 3), 100, dtype=np.uint8)
        job = replace(
            self._make_ocr_job(frame_id=622, fallback_reason="zero-detection"),
            roi_crop=roi,
            ocr_crops=(roi,),
            spatial_search_rescue=True,
        )

        outcome = processor.process_ocr_job(job, queue_depth=0)

        self.assertIs(outcome.state, RecognitionState.NO_OCR_TEXT)
        self.assertEqual(len(provider.calls), 5)

    def test_invalid_detector_crop_runs_bounded_ocr_search_rescue(self) -> None:
        provider = SequencedOcrProvider(
            [
                [OcrSegment("HEADLIGHT", 0.99, (0.0, 0.0, 20.0, 10.0))],
                [OcrSegment("06 FUP 848", 0.96, (5.0, 6.0, 105.0, 36.0))],
            ]
        )
        processor = PlateRecognitionProcessor(provider, self.plate_service, self.config)
        roi = np.full((240, 960, 3), 120, dtype=np.uint8)
        false_crop = np.full((30, 60, 3), 120, dtype=np.uint8)
        job = replace(
            self._make_ocr_job(
                frame_id=623,
                job_type=OcrJobType.DETECTOR_CROP,
                detection=PlateDetection(0.80, 600, 120, 24, 18),
            ),
            roi_crop=roi,
            ocr_crops=(false_crop,),
        )

        outcome = processor.process_ocr_job(job, queue_depth=0)

        self.assertIs(outcome.state, RecognitionState.AWAITING_CONFIRMATION)
        self.assertEqual(outcome.candidate.plate, "06FUP848")
        self.assertFalse(outcome.candidate.detector_crop_evidence)
        self.assertTrue(outcome.used_roi_fallback)
        self.assertEqual(len(provider.calls), 2)
        self.assertLess(provider.calls[1][0].shape[1], roi.shape[1])

    def test_valid_detector_crop_keeps_single_inference_fast_path(self) -> None:
        provider = SequencedOcrProvider(
            [[OcrSegment("06 FUP 848", 0.96, (0.0, 0.0, 100.0, 30.0))]]
        )
        processor = PlateRecognitionProcessor(provider, self.plate_service, self.config)
        crop = np.full((30, 90, 3), 120, dtype=np.uint8)
        job = replace(
            self._make_ocr_job(
                frame_id=624,
                job_type=OcrJobType.DETECTOR_CROP,
                detection=PlateDetection(0.90, 100, 50, 60, 18),
            ),
            roi_crop=np.full((240, 960, 3), 120, dtype=np.uint8),
            ocr_crops=(crop,),
        )

        outcome = processor.process_ocr_job(job, queue_depth=0)

        self.assertIs(outcome.state, RecognitionState.AWAITING_CONFIRMATION)
        self.assertEqual(outcome.candidate.plate, "06FUP848")
        self.assertTrue(outcome.candidate.detector_crop_evidence)
        self.assertFalse(outcome.used_roi_fallback)
        self.assertEqual(len(provider.calls), 1)

    def test_detector_crop_normal_profile_uses_one_tile_after_no_text(self) -> None:
        provider = SequencedOcrProvider([[]])
        processor = PlateRecognitionProcessor(provider, self.plate_service, self.config)
        crop = PreprocessingTests._plate_crop(190, 25)
        job = replace(
            self._make_ocr_job(frame_id=63, job_type=OcrJobType.DETECTOR_CROP),
            roi_crop=crop,
            ocr_crops=(crop.copy(),),
        )

        outcome = processor.process_ocr_job(job, queue_depth=0)

        self.assertIs(outcome.state, RecognitionState.NO_OCR_TEXT)
        self.assertEqual(len(provider.calls), 2)

    def test_detector_crop_shadow_retry_recovers_valid_candidate(self) -> None:
        provider = SequencedOcrProvider(
            [
                [],
                [OcrSegment("06 GG 986", 0.94, (0.0, 0.0, 100.0, 30.0))],
            ]
        )
        processor = PlateRecognitionProcessor(provider, self.plate_service, self.config)
        crop = PreprocessingTests._plate_crop(130, 116)
        job = replace(
            self._make_ocr_job(frame_id=64, job_type=OcrJobType.DETECTOR_CROP),
            roi_crop=crop,
            ocr_crops=(crop.copy(),),
        )

        outcome = processor.process_ocr_job(job, queue_depth=0)

        self.assertIs(outcome.state, RecognitionState.AWAITING_CONFIRMATION)
        self.assertEqual(outcome.candidate.plate, "06GG986")
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(len(provider.calls[1]), 1)

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
            captured_at=captured_at, observed_at=11.0, received_at=11.0,
            zero_detection_fallback_event_id=2,
        )

        self.assertEqual(first.job.fallback_reason, "zero-detection")
        self.assertIs(first.job.job_type, OcrJobType.ZERO_DETECTION_FALLBACK)
        self.assertIsNone(throttled.job)
        self.assertEqual(
            throttled.fallback_skipped_reason,
            "event-end-attempt-reserved",
        )
        self.assertEqual(after_interval.job.fallback_reason, "zero-detection")

    def test_tiled_recovery_uses_configured_larger_context_crop(self) -> None:
        detection = PlateDetection(0.9, x=20, y=10, width=100, height=30)
        detector = FakePlateDetector([detection])
        detector.last_diagnostics = DetectorDiagnostics(
            detector_variant="tiled",
            raw_brightness=120.0,
            shadow_metric=0.0,
            enhanced_pass=False,
            raw_detector_ms=3.0,
            enhanced_detector_ms=0.0,
            detections=1,
        )
        processor = PlateDetectionProcessor(self.config, detector)
        frame = np.zeros((200, 400, 3), dtype=np.uint8)

        result = processor.prepare_job(
            1,
            Direction.ENTRY,
            frame,
            captured_at=datetime(2026, 8, 14, tzinfo=timezone.utc),
            observed_at=10.0,
            received_at=10.0,
        )

        self.assertEqual(result.job.ocr_crops[0].shape, (55, 248, 3))

    def test_static_zero_detection_rescue_is_warmed_up_and_throttled(self) -> None:
        processor = PlateDetectionProcessor(self.config, FakePlateDetector([]))
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        captured_at = datetime(2026, 8, 13, tzinfo=timezone.utc)

        observed_times = (10.0, 11.0, 12.4, 12.5, 13.0, 15.0)
        results = [
            processor.prepare_job(
                1,
                Direction.ENTRY,
                frame,
                captured_at=captured_at,
                observed_at=observed_at,
                received_at=observed_at,
            )
            for observed_at in observed_times
        ]

        self.assertEqual(
            [result.job is not None for result in results],
            [False, False, False, True, False, True],
        )
        self.assertEqual(results[0].fallback_skipped_reason, "static-rescue-warmup")
        self.assertIs(
            results[3].job.job_type,
            OcrJobType.STATIC_ZERO_DETECTION_RESCUE,
        )
        self.assertEqual(
            results[3].job.fallback_reason,
            "static-zero-detection-rescue",
        )
        self.assertEqual(results[4].fallback_skipped_reason, "static-rescue-cooldown")

    def test_one_shot_raw_recognition_capture_is_opt_in_and_writes_once(self) -> None:
        frame = np.full((200, 400, 3), 71, dtype=np.uint8)
        detection = PlateDetection(0.9, x=20, y=10, width=100, height=30)
        captured_at = datetime(2026, 8, 14, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temp_directory, patch.dict(
            "os.environ",
            {"CAMERBOUND_CAPTURE_NEXT_RECOGNITION_FRAME": "1"},
        ), patch(
            "app.plate_recognition.application_root",
            return_value=Path(temp_directory),
        ):
            processor = PlateDetectionProcessor(
                self.config,
                FakePlateDetector([detection]),
            )
            processor.prepare_job(
                1,
                Direction.ENTRY,
                frame,
                captured_at=captured_at,
                observed_at=10.0,
                received_at=10.0,
            )
            processor.prepare_job(
                1,
                Direction.ENTRY,
                frame,
                captured_at=captured_at,
                observed_at=11.0,
                received_at=11.0,
            )
            captures = list(
                (Path(temp_directory) / "debug" / "recognition-frames").glob("*.jpg")
            )
            full_capture = cv2.imread(
                str(next(path for path in captures if path.name.endswith("-full.jpg")))
            )
            roi_capture = cv2.imread(
                str(next(path for path in captures if path.name.endswith("-roi.jpg")))
            )

        self.assertEqual(len(captures), 2)
        self.assertEqual(full_capture.shape[:2], frame.shape[:2])
        expected_roi = crop_roi(frame, self.config.roi_for(Direction.ENTRY))
        self.assertEqual(roi_capture.shape[:2], expected_roi.shape[:2])

    def test_raw_recognition_capture_can_be_armed_after_initial_frame(self) -> None:
        frame = np.full((200, 400, 3), 71, dtype=np.uint8)
        detection = PlateDetection(0.9, x=20, y=10, width=100, height=30)
        captured_at = datetime(2026, 8, 17, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temp_directory, patch.dict(
            "os.environ",
            {"CAMERBOUND_CAPTURE_NEXT_RECOGNITION_FRAME": ""},
        ), patch(
            "app.plate_recognition.application_root",
            return_value=Path(temp_directory),
        ):
            detector = FakePlateDetector([])
            processor = PlateDetectionProcessor(self.config, detector)
            processor.prepare_job(
                1,
                Direction.ENTRY,
                frame,
                captured_at=captured_at,
                observed_at=10.0,
                received_at=10.0,
            )
            processor.arm_raw_capture(Direction.ENTRY)
            processor.prepare_job(
                1,
                Direction.EXIT,
                frame,
                captured_at=captured_at,
                observed_at=11.0,
                received_at=11.0,
            )
            processor.prepare_job(
                1,
                Direction.ENTRY,
                frame,
                captured_at=captured_at,
                observed_at=12.0,
                received_at=12.0,
            )
            detector.detections = [detection]
            processor.prepare_job(
                1,
                Direction.ENTRY,
                frame,
                captured_at=captured_at,
                observed_at=13.0,
                received_at=13.0,
            )
            processor.prepare_job(
                1,
                Direction.ENTRY,
                frame,
                captured_at=captured_at,
                observed_at=14.0,
                received_at=14.0,
            )
            captures = list(
                (Path(temp_directory) / "debug" / "recognition-frames").glob("*.jpg")
            )

        self.assertEqual(len(captures), 2)
        self.assertEqual(
            sum(path.name.endswith("-full.jpg") for path in captures),
            1,
        )
        self.assertEqual(
            sum(path.name.endswith("-roi.jpg") for path in captures),
            1,
        )

    def test_armed_capture_ignores_replay_detector_hit(self) -> None:
        frame = np.full((200, 400, 3), 71, dtype=np.uint8)
        detection = PlateDetection(0.9, x=20, y=10, width=100, height=30)
        captured_at = datetime(2026, 8, 17, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temp_directory, patch.dict(
            "os.environ",
            {"CAMERBOUND_CAPTURE_NEXT_RECOGNITION_FRAME": ""},
        ), patch(
            "app.plate_recognition.application_root",
            return_value=Path(temp_directory),
        ):
            processor = PlateDetectionProcessor(
                self.config,
                FakePlateDetector([detection]),
            )
            processor.arm_raw_capture(Direction.ENTRY)
            replay = processor.prepare_job(
                1,
                Direction.ENTRY,
                frame,
                captured_at=captured_at,
                observed_at=10.0,
                received_at=10.0,
                detector_source="replay",
            )
            live = processor.prepare_job(
                1,
                Direction.ENTRY,
                frame,
                captured_at=captured_at,
                observed_at=11.0,
                received_at=11.0,
                detector_source="live",
            )

            captures = list(
                (Path(temp_directory) / "debug" / "recognition-frames").glob("*.jpg")
            )

        self.assertFalse(replay.diagnostic_capture)
        self.assertTrue(live.diagnostic_capture)
        self.assertEqual(len(captures), 2)

    def test_static_rescue_candidate_normalizes_expected_field_plate(self) -> None:
        config = recognition_config(self.config.model_root, confirmations=1)
        detection_processor = PlateDetectionProcessor(config, FakePlateDetector([]))
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        captured_at = datetime(2026, 8, 14, tzinfo=timezone.utc)
        first = detection_processor.prepare_job(
            1,
            Direction.ENTRY,
            frame,
            captured_at=captured_at,
            observed_at=10.0,
            received_at=10.0,
            frame_id=1,
        )
        rescue = detection_processor.prepare_job(
            1,
            Direction.ENTRY,
            frame,
            captured_at=captured_at,
            observed_at=12.5,
            received_at=12.5,
            frame_id=2,
        )
        provider = RecordingOcrProvider(confidence=0.98)
        provider.text = "06 GG 986"
        recognition_processor = PlateRecognitionProcessor(
            provider,
            self.plate_service,
            config,
        )

        outcome = recognition_processor.process_ocr_job(rescue.job, queue_depth=0)

        self.assertIsNone(first.job)
        self.assertIs(rescue.job.job_type, OcrJobType.STATIC_ZERO_DETECTION_RESCUE)
        self.assertEqual(outcome.candidate.plate, "06GG986")
        self.assertEqual(provider.calls, 1)
        self.assertLessEqual(provider.images[0].shape[1], 960)

    def test_recent_valid_candidate_suppresses_static_rescue_only(self) -> None:
        processor = PlateDetectionProcessor(self.config, FakePlateDetector([]))
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        captured_at = datetime(2026, 8, 14, tzinfo=timezone.utc)
        processor.prepare_job(
            1,
            Direction.ENTRY,
            frame,
            captured_at=captured_at,
            observed_at=10.0,
            received_at=10.0,
        )
        processor.note_valid_candidate(1, 12.4)

        result = processor.prepare_job(
            1,
            Direction.ENTRY,
            frame,
            captured_at=captured_at,
            observed_at=12.5,
            received_at=12.5,
        )

        self.assertIsNone(result.job)
        self.assertEqual(
            result.fallback_skipped_reason,
            "static-rescue-recent-recognition",
        )

    def test_motion_event_reserves_second_attempt_for_event_end(self) -> None:
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
            frame_id=1,
        )
        second = processor.prepare_job(
            1,
            Direction.ENTRY,
            frame,
            captured_at=captured_at,
            observed_at=11.0,
            received_at=11.0,
            zero_detection_fallback_event_id=7,
            frame_id=2,
        )
        limited = processor.prepare_job(
            1,
            Direction.ENTRY,
            frame,
            captured_at=captured_at,
            observed_at=12.0,
            received_at=12.0,
            zero_detection_fallback_event_id=7,
            frame_id=3,
        )

        self.assertIs(first.job.job_type, OcrJobType.ZERO_DETECTION_FALLBACK)
        self.assertIsNone(second.job)
        self.assertEqual(first.fallback_attempt, 1)
        self.assertEqual(
            second.fallback_skipped_reason,
            "event-end-attempt-reserved",
        )
        self.assertIsNone(limited.job)
        self.assertEqual(
            limited.fallback_skipped_reason,
            "event-end-attempt-reserved",
        )

    def test_first_fallback_miss_does_not_block_later_event_frame(self) -> None:
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
            frame_id=101,
            zero_detection_fallback_event_id=9,
            motion_score=0.05,
        )
        later = processor.prepare_job(
            1,
            Direction.ENTRY,
            frame,
            captured_at=captured_at + timedelta(seconds=1),
            observed_at=11.0,
            received_at=11.0,
            frame_id=102,
            zero_detection_fallback_event_id=9,
            motion_score=0.09,
        )

        self.assertIs(first.job.job_type, OcrJobType.ZERO_DETECTION_FALLBACK)
        self.assertIsNone(later.job)
        self.assertEqual(first.fallback_attempt, 1)
        self.assertEqual(
            later.fallback_skipped_reason,
            "event-end-attempt-reserved",
        )

    def test_second_fallback_requires_different_frame_and_one_second_spacing(self) -> None:
        processor = PlateDetectionProcessor(self.config, FakePlateDetector([]))
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        captured_at = datetime(2026, 8, 13, tzinfo=timezone.utc)
        first = processor.prepare_job(
            1,
            Direction.ENTRY,
            frame,
            captured_at=captured_at,
            observed_at=20.0,
            received_at=20.0,
            frame_id=201,
            zero_detection_fallback_event_id=10,
            motion_score=0.05,
        )
        same_frame = processor.prepare_job(
            1,
            Direction.ENTRY,
            frame,
            captured_at=captured_at,
            observed_at=21.0,
            received_at=21.0,
            frame_id=201,
            zero_detection_fallback_event_id=10,
            motion_score=0.08,
        )
        too_soon = processor.prepare_job(
            1,
            Direction.ENTRY,
            frame,
            captured_at=captured_at,
            observed_at=20.999,
            received_at=20.999,
            frame_id=202,
            zero_detection_fallback_event_id=10,
            motion_score=0.08,
        )
        spaced = processor.prepare_job(
            1,
            Direction.ENTRY,
            frame,
            captured_at=captured_at,
            observed_at=21.0,
            received_at=21.0,
            frame_id=203,
            zero_detection_fallback_event_id=10,
            motion_score=0.08,
        )

        self.assertIsNotNone(first.job)
        self.assertEqual(same_frame.fallback_skipped_reason, "same-frame")
        self.assertEqual(
            too_soon.fallback_skipped_reason,
            "event-end-attempt-reserved",
        )
        self.assertIsNone(spaced.job)
        self.assertEqual(
            spaced.fallback_skipped_reason,
            "event-end-attempt-reserved",
        )

    def test_event_end_adds_final_fallback_from_best_eligible_frame(self) -> None:
        processor = PlateDetectionProcessor(self.config, FakePlateDetector([]))
        frames = (
            replace(self._make_snapshot(301, 30.0, value=20), motion_score=0.05),
            replace(self._make_snapshot(302, 30.5, value=40), motion_score=0.20),
            replace(self._make_snapshot(303, 31.2, value=60), motion_score=0.12),
        )
        event = MotionEvent(
            event_id=11,
            camera_id=1,
            direction=Direction.ENTRY,
            started_at=30.0,
            ended_at=31.2,
            enqueued_at=31.2,
            frames=frames,
        )
        first = processor.prepare_job(
            1,
            Direction.ENTRY,
            frames[0].full_frame,
            captured_at=frames[0].captured_at,
            observed_at=frames[0].observed_at,
            received_at=frames[0].received_at,
            frame_id=frames[0].frame_id,
            zero_detection_fallback_event_id=event.event_id,
            motion_score=frames[0].motion_score,
        )

        with self.assertLogs("app.plate_recognition", level="DEBUG") as captured:
            final = processor.prepare_event_end_fallback(event, ring_depth=10)

        self.assertEqual(first.fallback_attempt, 1)
        self.assertIsNotNone(final.job)
        self.assertEqual(final.job.frame_id, 303)
        self.assertEqual(final.job.detector_source, "event-end")
        self.assertTrue(final.job.spatial_search_rescue)
        self.assertEqual(final.fallback_attempt, 2)
        self.assertEqual(final.event_frames, 3)
        self.assertEqual(final.ring_depth, 10)
        diagnostic = captured.output[-1]
        for expected in (
            "camera_id=1",
            "event_id=11",
            "fallback_attempt=2/2",
            "frame_id=303",
            "motion_score=0.1200",
            "fallback_reason=event-end",
            "time_since_previous_attempt_ms=1200.0",
            "event_frames=3",
            "ring_depth=10",
        ):
            self.assertIn(expected, diagnostic)

    def test_event_end_uses_reserved_second_attempt_after_repeated_live_misses(self) -> None:
        processor = PlateDetectionProcessor(self.config, FakePlateDetector([]))
        frames = tuple(
            replace(
                self._make_snapshot(400 + index, 40.0 + index, value=index),
                motion_score=0.10,
            )
            for index in range(3)
        )
        event = MotionEvent(12, 1, Direction.ENTRY, 40.0, 42.0, 42.0, frames)
        for snapshot in frames[:2]:
            processor.prepare_job(
                1,
                Direction.ENTRY,
                snapshot.full_frame,
                captured_at=snapshot.captured_at,
                observed_at=snapshot.observed_at,
                received_at=snapshot.received_at,
                frame_id=snapshot.frame_id,
                zero_detection_fallback_event_id=event.event_id,
                motion_score=snapshot.motion_score,
            )

        final = processor.prepare_event_end_fallback(event, ring_depth=3)

        self.assertIsNotNone(final.job)
        self.assertEqual(final.fallback_attempt, 2)
        self.assertEqual(final.job.frame_id, 402)

    def test_detector_crop_success_prevents_event_end_roi_fallback(self) -> None:
        detection = PlateDetection(0.9, x=20, y=10, width=100, height=30)
        processor = PlateDetectionProcessor(
            self.config,
            FakePlateDetector([detection]),
        )
        snapshot = replace(
            self._make_snapshot(501, 50.0, value=80),
            motion_score=0.10,
        )
        event = MotionEvent(
            13,
            1,
            Direction.ENTRY,
            50.0,
            50.0,
            50.0,
            (snapshot,),
        )
        crop = processor.prepare_job(
            1,
            Direction.ENTRY,
            snapshot.full_frame,
            captured_at=snapshot.captured_at,
            observed_at=snapshot.observed_at,
            received_at=snapshot.received_at,
            frame_id=snapshot.frame_id,
            zero_detection_fallback_event_id=event.event_id,
            motion_score=snapshot.motion_score,
        )

        final = processor.prepare_event_end_fallback(event, ring_depth=1)

        self.assertIs(crop.job.job_type, OcrJobType.DETECTOR_CROP)
        self.assertIsNone(final.job)
        self.assertEqual(final.fallback_skipped_reason, "detector-success")

    def test_replay_and_event_end_do_not_repeat_live_fallback_frame(self) -> None:
        processor = PlateDetectionProcessor(self.config, FakePlateDetector([]))
        snapshot = replace(
            self._make_snapshot(601, 60.0, value=30),
            motion_score=0.10,
        )
        event = MotionEvent(
            14,
            1,
            Direction.ENTRY,
            60.0,
            60.0,
            60.0,
            (snapshot,),
        )
        live = processor.prepare_job(
            1,
            Direction.ENTRY,
            snapshot.full_frame,
            captured_at=snapshot.captured_at,
            observed_at=snapshot.observed_at,
            received_at=snapshot.received_at,
            frame_id=snapshot.frame_id,
            zero_detection_fallback_event_id=event.event_id,
            motion_score=snapshot.motion_score,
        )
        replay = processor.prepare_job(
            1,
            Direction.ENTRY,
            snapshot.full_frame,
            captured_at=snapshot.captured_at,
            observed_at=snapshot.observed_at,
            received_at=snapshot.received_at,
            frame_id=snapshot.frame_id,
            detector_source="replay",
            allow_zero_detection_fallback=False,
            zero_detection_fallback_event_id=event.event_id,
            motion_score=snapshot.motion_score,
        )
        final = processor.prepare_event_end_fallback(event, ring_depth=1)

        self.assertEqual(live.fallback_attempt, 1)
        self.assertIsNone(replay.job)
        self.assertEqual(replay.fallback_skipped_reason, "replay-zero-detection")
        self.assertIsNone(final.job)

    def test_zero_detection_event_state_is_bounded(self) -> None:
        processor = PlateDetectionProcessor(self.config, FakePlateDetector([]))
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        captured_at = datetime(2026, 8, 13, tzinfo=timezone.utc)

        for event_id in range(40):
            processor.prepare_job(
                1,
                Direction.ENTRY,
                frame,
                captured_at=captured_at,
                observed_at=70.0 + event_id,
                received_at=70.0 + event_id,
                frame_id=700 + event_id,
                zero_detection_fallback_event_id=event_id,
                motion_score=0.10,
            )

        self.assertEqual(processor.zero_detection_event_state_count, 32)

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

    def test_detector_debug_diagnostics_include_source_and_enhanced_metrics(self) -> None:
        detector = FakePlateDetector(
            [PlateDetection(0.9, x=20, y=10, width=100, height=30)]
        )
        detector.last_diagnostics = DetectorDiagnostics(
            detector_variant="enhanced",
            raw_brightness=62.5,
            shadow_metric=181.0,
            enhanced_pass=True,
            raw_detector_ms=8.4,
            enhanced_detector_ms=9.6,
            detections=1,
            input_width=256,
            input_height=256,
            input_layout="NHWC",
            input_dtype="uint8",
            raw_candidate_count=4,
            plate_class_candidate_count=2,
            highest_plate_confidence=0.91,
            confidence_rejected_count=1,
            bbox_rejected_count=0,
        )
        processor = PlateDetectionProcessor(self.config, detector)
        frame = np.zeros((200, 400, 3), dtype=np.uint8)

        with self.assertLogs("app.plate_recognition", level="DEBUG") as captured:
            processor.prepare_job(
                2,
                Direction.EXIT,
                frame,
                captured_at=datetime(2026, 8, 13, tzinfo=timezone.utc),
                observed_at=10.0,
                received_at=10.0,
                detector_source="replay",
                allow_zero_detection_fallback=False,
            )

        diagnostic = next(
            line for line in captured.output if "Plate detector diagnostics" in line
        )
        self.assertIn("camera_id=2", diagnostic)
        self.assertIn("source=replay", diagnostic)
        self.assertIn("detector_variant=enhanced", diagnostic)
        self.assertIn("brightness=62.5", diagnostic)
        self.assertIn("raw_brightness=62.5", diagnostic)
        self.assertIn("shadow_metric=181.0", diagnostic)
        self.assertIn("enhanced_pass=yes", diagnostic)
        self.assertIn("detections=1", diagnostic)
        self.assertIn("raw_detector_ms=8.4", diagnostic)
        self.assertIn("enhanced_detector_ms=9.6", diagnostic)
        self.assertIn("raw_candidates=4", diagnostic)
        self.assertIn("plate_class_candidates=2", diagnostic)
        self.assertIn("highest_plate_confidence=0.910", diagnostic)
        self.assertIn("confidence_rejected=1", diagnostic)
        self.assertIn("detector_input=256x256", diagnostic)
        self.assertIn("roi_size=320x110", diagnostic)

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

    def test_wrong_valid_plate_never_saves_when_correct_plate_reaches_consensus(self) -> None:
        batches = [
            [OcrSegment(text, confidence, (0, 0, 10, 10), 0)]
            for text, confidence in (
                ("23ABC123", 0.82),
                ("23ABC128", 0.99),
                ("23ABC123", 0.83),
                ("23ABC123", 0.84),
            )
        ]
        processor = PlateRecognitionProcessor(
            SequencedOcrProvider(batches),
            self.plate_service,
            self.config,
        )
        base = datetime(2026, 8, 13, 10, 0, tzinfo=timezone.utc)

        outcomes = [
            processor.process_ocr_job(
                self._make_ocr_job(
                    frame_id=index,
                    observed_at=10.0 + index * 0.2,
                    captured_at=base + timedelta(milliseconds=index * 200),
                ),
                queue_depth=0,
            )
            for index in range(1, 5)
        ]

        self.assertEqual(outcomes[-1].record.plate, "23ABC123")
        self.assertEqual(self._record_count(), 1)
        with self.database.connection() as connection:
            self.assertEqual(
                connection.execute("SELECT plate FROM plate_records").fetchone()[0],
                "23ABC123",
            )

    def test_stabilization_waits_before_saving_clean_two_frame_consensus(self) -> None:
        config = recognition_config(
            Path(self.temp_directory.name),
            stabilization_window_ms=2000,
            stabilization_min_hold_ms=500,
        )
        processor = PlateRecognitionProcessor(
            SequencedOcrProvider(
                [
                    [OcrSegment("23ABC122", 0.96, (0, 0, 10, 10), 0)],
                    [OcrSegment("23ABC122", 0.97, (0, 0, 10, 10), 0)],
                ]
            ),
            self.plate_service,
            config,
        )

        first = processor.process_ocr_job(
            self._make_ocr_job(frame_id=101, observed_at=10.0, frame_value=21),
            queue_depth=0,
        )
        provisional = processor.process_ocr_job(
            self._make_ocr_job(
                frame_id=102,
                observed_at=10.2,
                frame_value=42,
                quality_score=2.0,
            ),
            queue_depth=0,
        )

        self.assertIs(first.state, RecognitionState.AWAITING_CONFIRMATION)
        self.assertIs(provisional.state, RecognitionState.STABILIZING)
        self.assertEqual(self._record_count(), 0)
        self.assertEqual(list(Path(self.temp_directory.name).rglob("*.jpg")), [])

        outcomes = processor.finalize_due(processor.next_pending_deadline() + 0.01)

        self.assertEqual(len(outcomes), 1)
        saved = outcomes[0][1]
        self.assertIs(saved.state, RecognitionState.SAVED)
        self.assertIs(saved.finalization_source, FinalizationSource.NORMAL_CONFIRMATION)
        self.assertEqual(saved.record.plate, "23ABC122")
        self.assertEqual(self._record_count(), 1)
        saved_image = cv2.imread(
            str(self.capture_service.resolve_reference(saved.record.image_path))
        )
        self.assertAlmostEqual(float(saved_image.mean()), 42.0, delta=3.0)

    def test_strong_single_observation_does_not_save_even_with_high_confidence(self) -> None:
        config = recognition_config(
            Path(self.temp_directory.name),
            stabilization_window_ms=2000,
            stabilization_min_hold_ms=500,
        )
        processor = PlateRecognitionProcessor(
            SequencedOcrProvider(
                [
                    [
                        OcrSegment("23LR640", 0.992, (0, 0, 10, 10), variant)
                        for variant in range(3)
                    ]
                ]
            ),
            self.plate_service,
            config,
        )

        outcome = processor.process_ocr_job(
            self._make_ocr_job(
                frame_id=103,
                observed_at=10.4,
                frame_value=63,
            ),
            queue_depth=0,
        )

        self.assertIs(outcome.state, RecognitionState.AWAITING_CONFIRMATION)
        self.assertEqual(outcome.confirmation_count, 1)
        self.assertEqual(outcome.confirmation_required, 2)
        self.assertEqual(self._record_count(), 0)

        deadline = processor.next_pending_deadline()
        if deadline is not None:
            outcomes = processor.finalize_due(deadline + 0.01)
            self.assertEqual(outcomes, [])
        self.assertEqual(self._record_count(), 0)

    def test_stabilization_replaces_early_wrong_consensus_with_trailing_correct_evidence(self) -> None:
        texts = (
            "23ABC127",
            "23ABC127",
            "23ABC122",
            "23ABC122",
            "23ABC122",
        )
        config = recognition_config(
            Path(self.temp_directory.name),
            stabilization_window_ms=2000,
            stabilization_min_hold_ms=500,
        )
        processor = PlateRecognitionProcessor(
            SequencedOcrProvider(
                [[OcrSegment(text, 0.97, (0, 0, 10, 10), 0)] for text in texts]
            ),
            self.plate_service,
            config,
        )
        base = datetime(2026, 8, 13, 13, 0, tzinfo=timezone.utc)

        outcomes = [
            processor.process_ocr_job(
                self._make_ocr_job(
                    frame_id=index,
                    observed_at=20.0 + index * 0.2,
                    captured_at=base + timedelta(milliseconds=index * 200),
                    frame_value=index * 10,
                    quality_score=float(index),
                ),
                queue_depth=0,
            )
            for index in range(1, 6)
        ]

        self.assertTrue(all(outcome.record is None for outcome in outcomes))
        self.assertEqual(self._record_count(), 0)
        self.assertEqual(list(Path(self.temp_directory.name).rglob("*.jpg")), [])

        saved = processor.finalize_due(processor.next_pending_deadline() + 0.01)[0][1]

        self.assertIs(saved.state, RecognitionState.SAVED)
        self.assertEqual(saved.record.plate, "23ABC122")
        with self.database.connection() as connection:
            plates = [
                row[0]
                for row in connection.execute(
                    "SELECT plate FROM plate_records ORDER BY id"
                ).fetchall()
            ]
        self.assertEqual(plates, ["23ABC122"])
        saved_image = cv2.imread(
            str(self.capture_service.resolve_reference(saved.record.image_path))
        )
        self.assertAlmostEqual(float(saved_image.mean()), 50.0, delta=3.0)

    def test_clean_three_detector_frames_wait_for_full_stabilization_window(self) -> None:
        processor = PlateRecognitionProcessor(
            SequencedOcrProvider(
                [
                    [OcrSegment("34DRF848", 0.97, (0, 0, 10, 10), 0)]
                    for _ in range(3)
                ]
            ),
            self.plate_service,
            recognition_config(
                Path(self.temp_directory.name),
                stabilization_window_ms=2000,
                stabilization_min_hold_ms=500,
            ),
        )

        processor.process_ocr_job(
            self._make_ocr_job(frame_id=701, observed_at=25.0), queue_depth=0
        )
        provisional = processor.process_ocr_job(
            self._make_ocr_job(frame_id=702, observed_at=25.2), queue_depth=0
        )
        time.sleep(0.52)
        third = processor.process_ocr_job(
            self._make_ocr_job(frame_id=703, observed_at=25.4), queue_depth=0
        )

        self.assertIs(provisional.state, RecognitionState.STABILIZING)
        self.assertIs(third.state, RecognitionState.STABILIZING)
        self.assertIsNone(third.record)
        self.assertEqual(self._record_count(), 0)
        finalized = processor.finalize_due(
            processor.next_pending_deadline() + 0.01
        )[0][1]
        self.assertEqual(finalized.record.plate, "34DRF848")
        self.assertEqual(processor.pending_decision_count, 0)

    def test_regression_early_orf_then_three_drf_frames_saves_drf(self) -> None:
        texts = ("34ORF848", "34DRF848", "34DRF848", "34DRF848")
        processor = PlateRecognitionProcessor(
            SequencedOcrProvider(
                [[OcrSegment(text, 0.97, (0, 0, 10, 10), 0)] for text in texts]
            ),
            self.plate_service,
            recognition_config(
                Path(self.temp_directory.name),
                stabilization_window_ms=2000,
                stabilization_min_hold_ms=500,
            ),
        )

        outcomes = [
            processor.process_ocr_job(
                self._make_ocr_job(
                    frame_id=710 + index,
                    observed_at=26.0 + index * 0.2,
                ),
                queue_depth=0,
            )
            for index in range(len(texts))
        ]

        self.assertTrue(all(outcome.record is None for outcome in outcomes))
        finalized = processor.finalize_due(
            processor.next_pending_deadline() + 0.01
        )[0][1]
        self.assertIs(finalized.state, RecognitionState.SAVED)
        self.assertEqual(finalized.record.plate, "34DRF848")
        self.assertEqual(self._record_count(), 1)

    def test_regression_two_orf_two_drf_frames_are_discarded_as_ambiguous(self) -> None:
        texts = ("34ORF848", "34ORF848", "34DRF848", "34DRF848")
        processor = PlateRecognitionProcessor(
            SequencedOcrProvider(
                [[OcrSegment(text, 0.97, (0, 0, 10, 10), 0)] for text in texts]
            ),
            self.plate_service,
            recognition_config(
                Path(self.temp_directory.name),
                stabilization_window_ms=2000,
                stabilization_min_hold_ms=500,
            ),
        )

        for index in range(len(texts)):
            processor.process_ocr_job(
                self._make_ocr_job(
                    frame_id=720 + index,
                    observed_at=27.0 + index * 0.2,
                ),
                queue_depth=0,
            )

        finalized = processor.finalize_due(
            processor.next_pending_deadline() + 0.01
        )[0][1]
        self.assertIs(finalized.state, RecognitionState.AMBIGUOUS_DISCARDED)
        self.assertIsNone(finalized.record)
        self.assertEqual(self._record_count(), 0)

    def test_regression_same_frame_variants_count_as_one_orf_vote(self) -> None:
        processor = PlateRecognitionProcessor(
            SequencedOcrProvider(
                [
                    [
                        OcrSegment("34 ORF 848", 0.96, (0, 0, 10, 10), 0),
                        OcrSegment("34ORF848", 0.97, (0, 0, 10, 10), 1),
                        OcrSegment("34 ORF848", 0.95, (0, 0, 10, 10), 2),
                    ]
                ]
            ),
            self.plate_service,
            recognition_config(
                Path(self.temp_directory.name),
                stabilization_window_ms=2000,
                stabilization_min_hold_ms=500,
            ),
        )

        outcome = processor.process_ocr_job(
            self._make_ocr_job(frame_id=730, observed_at=28.0), queue_depth=0
        )

        self.assertIs(outcome.state, RecognitionState.AWAITING_CONFIRMATION)
        self.assertEqual(outcome.confirmation_count, 1)
        self.assertEqual(outcome.candidate.plate, "34ORF848")
        self.assertEqual(outcome.candidate.variant_support, 3)
        self.assertEqual(self._record_count(), 0)

    def test_regression_clean_orf_is_not_rewritten_to_drf(self) -> None:
        processor = PlateRecognitionProcessor(
            SequencedOcrProvider(
                [
                    [OcrSegment("34ORF848", 0.97, (0, 0, 10, 10), 0)]
                    for _ in range(3)
                ]
            ),
            self.plate_service,
            recognition_config(
                Path(self.temp_directory.name),
                stabilization_window_ms=2000,
                stabilization_min_hold_ms=500,
            ),
        )

        outcomes = [
            processor.process_ocr_job(
                self._make_ocr_job(frame_id=740 + index, observed_at=29.0 + index * 0.2),
                queue_depth=0,
            )
            for index in range(3)
        ]

        self.assertTrue(all(outcome.record is None for outcome in outcomes))
        finalized = processor.finalize_due(
            processor.next_pending_deadline() + 0.01
        )[0][1]
        self.assertIs(finalized.state, RecognitionState.SAVED)
        self.assertEqual(finalized.record.plate, "34ORF848")

    def test_regression_late_drf_evidence_prevents_early_orf_save(self) -> None:
        texts = (
            "34ORF848",
            "34ORF848",
            "34ORF848",
            "34DRF848",
            "34DRF848",
            "34DRF848",
            "34DRF848",
        )
        processor = PlateRecognitionProcessor(
            SequencedOcrProvider(
                [[OcrSegment(text, 0.97, (0, 0, 10, 10), 0)] for text in texts]
            ),
            self.plate_service,
            recognition_config(
                Path(self.temp_directory.name),
                stabilization_window_ms=2000,
                stabilization_min_hold_ms=500,
            ),
        )

        processor.process_ocr_job(
            self._make_ocr_job(frame_id=750, observed_at=30.0), queue_depth=0
        )
        processor.process_ocr_job(
            self._make_ocr_job(frame_id=751, observed_at=30.2), queue_depth=0
        )
        time.sleep(0.52)
        early_consensus = processor.process_ocr_job(
            self._make_ocr_job(frame_id=752, observed_at=30.4), queue_depth=0
        )

        self.assertIs(early_consensus.state, RecognitionState.STABILIZING)
        self.assertIsNone(early_consensus.record)
        self.assertEqual(self._record_count(), 0)
        for index in range(3, len(texts)):
            processor.process_ocr_job(
                self._make_ocr_job(
                    frame_id=750 + index,
                    observed_at=30.0 + index * 0.2,
                ),
                queue_depth=0,
            )
        finalized = processor.finalize_due(
            processor.next_pending_deadline() + 0.01
        )[0][1]
        self.assertIs(finalized.state, RecognitionState.SAVED)
        self.assertEqual(finalized.record.plate, "34DRF848")
        self.assertEqual(self._record_count(), 1)

    def test_stabilization_applies_conflict_margin_rules(self) -> None:
        cases = (
            (
                "two_vs_two",
                ("23ABC122", "23ABC122", "23ABC127", "23ABC127"),
                False,
            ),
            (
                "three_vs_one",
                ("23ABC122", "23ABC122", "23ABC127", "23ABC122"),
                True,
            ),
            (
                "interleaved_three_vs_two",
                (
                    "23ABC122",
                    "23ABC122",
                    "23ABC127",
                    "23ABC122",
                    "23ABC127",
                ),
                False,
            ),
        )
        for name, texts, should_save in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp_directory:
                root = Path(temp_directory)
                database = Database(root / "recognition.db")
                database.initialize()
                auth_service = AuthService(database)
                auth_service.ensure_default_admin()
                camera_service = CameraService(database)
                capture_service = PlateCaptureService(root / "captures", root)
                plate_service = PlateService(
                    database,
                    duplicate_cooldown_seconds=10,
                    capture_service=capture_service,
                )
                config = recognition_config(
                    root,
                    stabilization_window_ms=2000,
                    stabilization_min_hold_ms=500,
                )
                processor = PlateRecognitionProcessor(
                    SequencedOcrProvider(
                        [
                            [OcrSegment(text, 0.97, (0, 0, 10, 10), 0)]
                            for text in texts
                        ]
                    ),
                    plate_service,
                    config,
                )
                try:
                    for index in range(len(texts)):
                        processor.process_ocr_job(
                            self._make_ocr_job(
                                frame_id=1000 + index,
                                observed_at=30.0 + index * 0.2,
                            ),
                            queue_depth=0,
                        )
                    finalized = processor.finalize_due(
                        processor.next_pending_deadline() + 0.01
                    )[0][1]
                    with database.connection() as connection:
                        record_count = connection.execute(
                            "SELECT COUNT(*) FROM plate_records"
                        ).fetchone()[0]
                    self.assertEqual(record_count, int(should_save))
                    self.assertIs(
                        finalized.state,
                        RecognitionState.SAVED
                        if should_save
                        else RecognitionState.AMBIGUOUS_DISCARDED,
                    )
                finally:
                    camera_service.stop_all()

    def test_stabilization_requires_three_votes_for_corrected_candidate(self) -> None:
        for votes, should_save in ((2, False), (3, True)):
            with self.subTest(votes=votes), tempfile.TemporaryDirectory() as temp_directory:
                root = Path(temp_directory)
                database = Database(root / "recognition.db")
                database.initialize()
                auth_service = AuthService(database)
                auth_service.ensure_default_admin()
                capture_service = PlateCaptureService(root / "captures", root)
                plate_service = PlateService(
                    database,
                    duplicate_cooldown_seconds=10,
                    capture_service=capture_service,
                )
                processor = PlateRecognitionProcessor(
                    SequencedOcrProvider(
                        [
                            [OcrSegment("23ABCI22", 0.97, (0, 0, 10, 10), 0)]
                            for _ in range(votes)
                        ]
                    ),
                    plate_service,
                    recognition_config(
                        root,
                        stabilization_window_ms=2000,
                        stabilization_min_hold_ms=500,
                    ),
                )
                for index in range(votes):
                    processor.process_ocr_job(
                        self._make_ocr_job(
                            frame_id=2000 + index,
                            observed_at=40.0 + index * 0.2,
                        ),
                        queue_depth=0,
                    )
                finalized_outcomes = processor.finalize_due(
                    processor.next_pending_deadline() + 0.01
                )
                with database.connection() as connection:
                    records = [
                        tuple(row)
                        for row in connection.execute(
                            "SELECT plate FROM plate_records"
                        ).fetchall()
                    ]
                self.assertEqual(records, [("23ABC122",)] if should_save else [])
                if should_save:
                    self.assertEqual(len(finalized_outcomes), 1)
                    self.assertIs(
                        finalized_outcomes[0][1].state,
                        RecognitionState.SAVED,
                    )
                else:
                    self.assertEqual(finalized_outcomes, [])

    def test_stabilization_counts_each_frame_only_once(self) -> None:
        processor = PlateRecognitionProcessor(
            SequencedOcrProvider(
                [
                    [OcrSegment("23ABC122", 0.97, (0, 0, 10, 10), 0)],
                    [OcrSegment("23ABC122", 0.97, (0, 0, 10, 10), 0)],
                ]
            ),
            self.plate_service,
            recognition_config(
                Path(self.temp_directory.name),
                stabilization_window_ms=2000,
                stabilization_min_hold_ms=500,
            ),
        )

        first = processor.process_ocr_job(
            self._make_ocr_job(frame_id=3001, observed_at=50.0), queue_depth=0
        )
        duplicate = processor.process_ocr_job(
            self._make_ocr_job(frame_id=3001, observed_at=50.2), queue_depth=0
        )

        self.assertEqual(first.confirmation_count, 1)
        self.assertIs(duplicate.state, RecognitionState.DUPLICATE_SUPPRESSED)
        processor.finalize_due(processor.next_pending_deadline() + 0.01)
        self.assertEqual(processor.pending_decision_count, 0)
        self.assertEqual(self._record_count(), 0)

    def test_pending_stabilization_state_is_bounded_and_expires(self) -> None:
        processor = PlateRecognitionProcessor(
            SequencedOcrProvider(
                [
                    [OcrSegment("23ABC122", 0.97, (0, 0, 10, 10), 0)]
                    for _ in range(17)
                ]
            ),
            self.plate_service,
            recognition_config(
                Path(self.temp_directory.name),
                stabilization_window_ms=2000,
                stabilization_min_hold_ms=500,
            ),
        )

        for camera_id in range(1, 18):
            processor.process_ocr_job(
                self._make_ocr_job(
                    camera_id=camera_id,
                    frame_id=4000 + camera_id,
                    observed_at=60.0 + camera_id * 0.01,
                ),
                queue_depth=0,
            )

        self.assertEqual(processor.pending_decision_count, 16)
        processor.finalize_due(time.monotonic() + 10.0)
        self.assertEqual(processor.pending_decision_count, 0)

    def test_ocr_worker_shutdown_clears_pending_decision_without_deadlock(self) -> None:
        buffer = OcrJobBuffer(max_per_camera=3, max_age_ms=2500)
        stop_event = threading.Event()
        outcomes = []
        provisional_ready = threading.Event()

        def on_outcome(_camera_id: int, outcome: object) -> None:
            outcomes.append(outcome)
            if outcome.state is RecognitionState.STABILIZING:
                provisional_ready.set()

        worker = PlateOcrWorker(
            self.plate_service,
            recognition_config(
                Path(self.temp_directory.name),
                stabilization_window_ms=2000,
                stabilization_min_hold_ms=500,
            ),
            provider_factory=lambda: SequencedOcrProvider(
                [
                    [OcrSegment("23ABC122", 0.97, (0, 0, 10, 10), 0)],
                    [OcrSegment("23ABC122", 0.97, (0, 0, 10, 10), 0)],
                ]
            ),
            job_buffer=buffer,
            stop_event=stop_event,
            on_outcome=on_outcome,
            on_status=lambda _status, _message: None,
        )
        buffer.add(self._make_ocr_job(frame_id=5001, observed_at=time.monotonic()))
        buffer.add(
            self._make_ocr_job(
                frame_id=5002,
                observed_at=time.monotonic() + 0.1,
            )
        )
        thread = threading.Thread(target=worker.run)
        thread.start()
        try:
            self.assertTrue(provisional_ready.wait(1.0))
        finally:
            stop_event.set()
            buffer.wake_all()
            thread.join(1.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(self._record_count(), 0)
        self.assertEqual(list(Path(self.temp_directory.name).rglob("*.jpg")), [])

    def test_ocr_worker_finalizes_after_vehicle_leaves_stabilization_window(self) -> None:
        buffer = OcrJobBuffer(max_per_camera=3, max_age_ms=2500)
        stop_event = threading.Event()
        outcomes = []
        saved_ready = threading.Event()

        def on_outcome(_camera_id: int, outcome: object) -> None:
            outcomes.append(outcome)
            if outcome.state is RecognitionState.SAVED:
                saved_ready.set()

        worker = PlateOcrWorker(
            self.plate_service,
            recognition_config(
                Path(self.temp_directory.name),
                stabilization_window_ms=500,
                stabilization_min_hold_ms=500,
            ),
            provider_factory=lambda: SequencedOcrProvider(
                [
                    [OcrSegment("23ABC122", 0.97, (0, 0, 10, 10), 0)],
                    [OcrSegment("23ABC122", 0.97, (0, 0, 10, 10), 0)],
                ]
            ),
            job_buffer=buffer,
            stop_event=stop_event,
            on_outcome=on_outcome,
            on_status=lambda _status, _message: None,
        )
        buffer.add(self._make_ocr_job(frame_id=6001, observed_at=time.monotonic()))
        buffer.add(
            self._make_ocr_job(
                frame_id=6002,
                observed_at=time.monotonic() + 0.1,
                frame_value=72,
                quality_score=2.0,
            )
        )
        thread = threading.Thread(target=worker.run)
        thread.start()
        try:
            self.assertTrue(saved_ready.wait(1.5))
        finally:
            stop_event.set()
            buffer.wake_all()
            thread.join(1.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(self._record_count(), 1)
        self.assertEqual(
            [outcome.state for outcome in outcomes],
            [
                RecognitionState.AWAITING_CONFIRMATION,
                RecognitionState.STABILIZING,
                RecognitionState.SAVED,
            ],
        )

    def test_corrected_candidate_saves_only_after_three_distinct_frames(self) -> None:
        batches = [
            [OcrSegment("23ABCI23", 0.95, (0, 0, 10, 10), 0)]
            for _ in range(3)
        ]
        processor = PlateRecognitionProcessor(
            SequencedOcrProvider(batches), self.plate_service, self.config
        )
        base = datetime(2026, 8, 13, 10, 30, tzinfo=timezone.utc)

        outcomes = [
            processor.process_ocr_job(
                self._make_ocr_job(
                    frame_id=index,
                    observed_at=15.0 + index * 0.2,
                    captured_at=base + timedelta(milliseconds=index * 200),
                ),
                queue_depth=0,
            )
            for index in range(1, 4)
        ]

        self.assertTrue(all(item.record is None for item in outcomes[:2]))
        self.assertEqual(outcomes[1].confirmation_required, 3)
        self.assertEqual(outcomes[2].record.plate, "23ABC123")

    def test_alternating_near_plate_tie_creates_no_record(self) -> None:
        batches = [
            [OcrSegment(text, 0.95, (0, 0, 10, 10), 0)]
            for text in ("23ABC123", "23ABC128", "23ABC123", "23ABC128")
        ]
        processor = PlateRecognitionProcessor(
            SequencedOcrProvider(batches), self.plate_service, self.config
        )
        base = datetime(2026, 8, 13, 10, 0, tzinfo=timezone.utc)

        outcomes = [
            processor.process_ocr_job(
                self._make_ocr_job(
                    frame_id=index,
                    observed_at=20.0 + index * 0.2,
                    captured_at=base + timedelta(milliseconds=index * 200),
                ),
                queue_depth=0,
            )
            for index in range(1, 5)
        ]

        self.assertTrue(all(outcome.record is None for outcome in outcomes))
        self.assertEqual(self._record_count(), 0)
        self.assertEqual(outcomes[-1].confirmation_required, 3)

    def test_post_save_spatial_ocr_alias_is_suppressed_without_jpeg(self) -> None:
        batches = [
            [OcrSegment(text, 0.95, (0, 0, 10, 10), 0)]
            for text in (
                "23ABC123",
                "23ABC123",
                "23ABC128",
                "23ABC128",
                "23ABC128",
                "23ABC128",
            )
        ]
        processor = PlateRecognitionProcessor(
            SequencedOcrProvider(batches), self.plate_service, self.config
        )
        detection = PlateDetection(0.90, x=20, y=10, width=100, height=30)
        base = datetime(2026, 8, 13, 11, 0, tzinfo=timezone.utc)

        outcomes = [
            processor.process_ocr_job(
                self._make_ocr_job(
                    frame_id=index,
                    observed_at=30.0 + index * 0.2,
                    captured_at=(
                        base + timedelta(milliseconds=index * 200)
                        if index <= 2
                        else base
                        + timedelta(minutes=30, milliseconds=index * 200)
                    ),
                    detection=detection,
                    frame_value=index * 10,
                ),
                queue_depth=0,
            )
            for index in range(1, 7)
        ]

        self.assertIs(outcomes[1].state, RecognitionState.SAVED)
        self.assertIs(outcomes[-1].state, RecognitionState.DUPLICATE_SUPPRESSED)
        self.assertEqual(outcomes[-1].suppression_reason, "near-duplicate-ambiguity")
        self.assertEqual(self._record_count(), 1)
        self.assertEqual(len(list(Path(self.temp_directory.name).rglob("*.jpg"))), 1)

    def test_different_near_plate_without_spatial_continuity_can_be_accepted(self) -> None:
        batches = [
            [OcrSegment(text, 0.95, (0, 0, 10, 10), 0)]
            for text in (
                "23ABC123",
                "23ABC123",
                "23ABC128",
                "23ABC128",
                "23ABC128",
                "23ABC128",
            )
        ]
        processor = PlateRecognitionProcessor(
            SequencedOcrProvider(batches), self.plate_service, self.config
        )
        first_detection = PlateDetection(0.90, x=20, y=10, width=80, height=25)
        second_detection = PlateDetection(0.90, x=250, y=120, width=80, height=25)
        base = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)
        outcomes = []
        for index in range(1, 7):
            outcomes.append(
                processor.process_ocr_job(
                    self._make_ocr_job(
                        frame_id=index,
                        observed_at=40.0 + index * 0.2,
                        captured_at=base + timedelta(milliseconds=index * 200),
                        detection=(first_detection if index <= 2 else second_detection),
                    ),
                    queue_depth=0,
                )
            )

        self.assertIs(outcomes[-1].state, RecognitionState.SAVED)
        self.assertEqual(outcomes[-1].confirmation_required, 4)
        self.assertEqual(self._record_count(), 2)

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
        health = worker.runtime_health()
        self.assertEqual(health.frames_ingested, 2)
        self.assertEqual(health.ocr_inference_errors, 1)
        self.assertGreaterEqual(health.ocr_jobs_processed, 1)
        self.assertIs(health.last_inference_ok, True)
        self.assertIsNotNone(health.detector_mean_ms)
        self.assertIsNotNone(health.detector_p95_ms)
        self.assertIsNotNone(health.ocr_mean_ms)
        self.assertIsNotNone(health.ocr_p95_ms)
        self.assertIsNotNone(health.queue_wait_mean_ms)
        self.assertIsNotNone(health.end_to_end_p95_ms)

    def test_debug_output_paths_are_created(self) -> None:
        output = Path(self.temp_directory.name) / "debug"
        image = np.zeros((30, 100, 3), dtype=np.uint8)
        segments = [OcrSegment("34ABC123", 0.9, (1, 1, 80, 20))]

        paths = save_debug_images(output, image, image, [image], segments)

        self.assertGreaterEqual(len(paths), 4)
        self.assertTrue(all(path.is_file() for path in paths))

    def test_two_distinct_real_frames_saves_normal(self) -> None:
        config = recognition_config(
            Path(self.temp_directory.name),
            stabilization_window_ms=2000,
            stabilization_min_hold_ms=500,
        )
        processor = PlateRecognitionProcessor(
            SequencedOcrProvider(
                [
                    [OcrSegment("23ABC123", 0.95, (0, 0, 10, 10), 0)],
                    [OcrSegment("23ABC123", 0.96, (0, 0, 10, 10), 0)],
                ]
            ),
            self.plate_service,
            config,
        )
        first = processor.process_ocr_job(
            self._make_ocr_job(frame_id=100, observed_at=10.0),
            queue_depth=1,
        )
        second = processor.process_ocr_job(
            self._make_ocr_job(frame_id=101, observed_at=10.2),
            queue_depth=0,
        )
        self.assertIs(first.state, RecognitionState.AWAITING_CONFIRMATION)
        self.assertIs(second.state, RecognitionState.STABILIZING)
        self.assertEqual(self._record_count(), 0)

        outcomes = processor.finalize_due(processor.next_pending_deadline() + 0.01)
        self.assertEqual(len(outcomes), 1)
        saved = outcomes[0][1]
        self.assertIs(saved.state, RecognitionState.SAVED)
        self.assertIs(saved.finalization_source, FinalizationSource.NORMAL_CONFIRMATION)
        self.assertEqual(saved.record.plate, "23ABC123")
        self.assertEqual(self._record_count(), 1)

    def test_track_end_does_not_use_buffered_frame_as_new_confirmation(self) -> None:
        config = replace(
            self.config,
            plate_stabilization_window_ms=2000,
            pre_detection_buffer_duration_ms=5000,
        )
        frame_buffer = PreDetectionFrameBuffer(config)
        frame_buffer.ingest(
            self._make_snapshot(95, 9.8, value=75),
            motion_score=0.0,
        )
        processor = PlateRecognitionProcessor(
            SequencedOcrProvider(
                [
                    [OcrSegment("23ABC123", 0.95, (0, 0, 10, 10), 0)],
                    [OcrSegment("23ABC123", 0.96, (0, 0, 10, 10), 0)],
                ]
            ),
            self.plate_service,
            config,
            frame_buffer=frame_buffer,
        )
        live_outcome = processor.process_ocr_job(
            self._make_ocr_job(frame_id=100, observed_at=10.0, track_id=42),
            queue_depth=0,
        )
        self.assertIs(live_outcome.state, RecognitionState.AWAITING_CONFIRMATION)
        self.assertEqual(live_outcome.confirmation_count, 1)
        self.assertEqual(self._record_count(), 0)

        finalized = processor.finalize_track(1, 42)
        self.assertIsNotNone(finalized)
        self.assertIs(finalized.state, RecognitionState.AMBIGUOUS_DISCARDED)
        self.assertIs(finalized.finalization_source, FinalizationSource.AMBIGUOUS_DISCARD)
        self.assertIsNone(finalized.record)
        self.assertEqual(self._record_count(), 0)

    def test_track_end_never_consumes_other_vehicle_buffer_evidence(self) -> None:
        from app.plate_detector import PlateDetection
        config = replace(
            self.config,
            plate_stabilization_window_ms=2000,
            pre_detection_buffer_duration_ms=5000,
        )
        frame_buffer = PreDetectionFrameBuffer(config)
        # Snapshot in buffer has Vehicle B, frame 95
        frame_buffer.ingest(
            self._make_snapshot(95, 9.8, value=75),
            motion_score=0.0,
        )
        # Snapshot in buffer has Vehicle B, frame 97
        frame_buffer.ingest(
            self._make_snapshot(97, 9.9, value=75),
            motion_score=0.0,
        )
        
        provider = SequencedOcrProvider(
            [
                [OcrSegment("34ABC123", 0.95, (0, 0, 10, 10), 0)],
                [OcrSegment("35XYZ456", 0.96, (0, 0, 10, 10), 0)],
                [OcrSegment("35XYZ456", 0.96, (0, 0, 10, 10), 0)],
            ]
        )
        processor = PlateRecognitionProcessor(
            provider,
            self.plate_service,
            config,
            frame_buffer=frame_buffer,
        )
        
        # Live frame sees Vehicle A, but it only gets 1 observation
        live_outcome = processor.process_ocr_job(
            self._make_ocr_job(frame_id=100, observed_at=10.0, track_id=43),
            queue_depth=0,
        )
        self.assertIs(live_outcome.state, RecognitionState.AWAITING_CONFIRMATION)
        
        # Finalize track A. It should NOT be rescued by Vehicle B's frame in the buffer.
        finalized = processor.finalize_track(1, 43)
        self.assertIsNotNone(finalized)
        self.assertIs(finalized.state, RecognitionState.AMBIGUOUS_DISCARDED)
        self.assertEqual(self._record_count(), 0)
        
        # Buffer OCR should NEVER have been invoked during finalize_track
        self.assertEqual(len(provider.calls), 1)

    def test_same_frame_reprocess_does_not_confirm(self) -> None:
        config = recognition_config(Path(self.temp_directory.name))
        frame_buffer = PreDetectionFrameBuffer(config)
        frame_buffer.ingest(
            self._make_snapshot(100, 10.0),
            motion_score=0.0,
        )
        processor = PlateRecognitionProcessor(
            SequencedOcrProvider([[OcrSegment("23ABC123", 0.95, (0, 0, 10, 10), 0)]]),
            self.plate_service,
            config,
            frame_buffer=frame_buffer,
        )
        processor.process_ocr_job(
            self._make_ocr_job(frame_id=100, observed_at=10.0, track_id=50),
            queue_depth=0,
        )
        finalized = processor.finalize_track(1, 50)
        self.assertIsNotNone(finalized)
        self.assertIs(finalized.state, RecognitionState.AMBIGUOUS_DISCARDED)
        self.assertIs(finalized.finalization_source, FinalizationSource.AMBIGUOUS_DISCARD)
        self.assertIsNone(finalized.record)
        self.assertEqual(self._record_count(), 0)

    def test_buffer_conflict_discards(self) -> None:
        config = recognition_config(Path(self.temp_directory.name))
        frame_buffer = PreDetectionFrameBuffer(config)
        frame_buffer.ingest(
            self._make_snapshot(95, 9.8),
            motion_score=0.0,
        )
        processor = PlateRecognitionProcessor(
            SequencedOcrProvider(
                [
                    [OcrSegment("23ABC123", 0.95, (0, 0, 10, 10), 0)],
                    [OcrSegment("23ABC128", 0.96, (0, 0, 10, 10), 0)],
                ]
            ),
            self.plate_service,
            config,
            frame_buffer=frame_buffer,
        )
        processor.process_ocr_job(
            self._make_ocr_job(frame_id=100, observed_at=10.0, track_id=60),
            queue_depth=0,
        )
        finalized = processor.finalize_track(1, 60)
        self.assertIsNotNone(finalized)
        self.assertIs(finalized.state, RecognitionState.AMBIGUOUS_DISCARDED)
        self.assertEqual(finalized.suppression_reason, "insufficient-variant-consensus")
        self.assertIsNone(finalized.record)
        self.assertEqual(self._record_count(), 0)

    def test_track_end_discards_single_observation_even_when_buffer_contains_frames(self) -> None:
        config = recognition_config(Path(self.temp_directory.name))
        frame_buffer = PreDetectionFrameBuffer(config)
        frame_buffer.ingest(self._make_snapshot(92, 9.5), motion_score=0.0)
        frame_buffer.ingest(self._make_snapshot(96, 9.8), motion_score=0.0)
        processor = PlateRecognitionProcessor(
            SequencedOcrProvider(
                [
                    [OcrSegment("23ABC123", 0.95, (0, 0, 10, 10), 0)],
                    [],
                    [OcrSegment("23ABC123", 0.96, (0, 0, 10, 10), 0)],
                ]
            ),
            self.plate_service,
            config,
            frame_buffer=frame_buffer,
        )
        processor.process_ocr_job(
            self._make_ocr_job(frame_id=100, observed_at=10.0, track_id=70),
            queue_depth=0,
        )
        finalized = processor.finalize_track(1, 70)
        self.assertIsNotNone(finalized)
        self.assertIs(finalized.state, RecognitionState.AMBIGUOUS_DISCARDED)
        self.assertIs(finalized.finalization_source, FinalizationSource.AMBIGUOUS_DISCARD)
        self.assertIsNone(finalized.record)
        self.assertEqual(self._record_count(), 0)

    def test_two_concurrent_tracks_isolate_single_observation_discard(self) -> None:
        from app.plate_detector import PlateDetection
        from app.plate_tracking import PlateTrackManager

        class DualDetector:
            def detect(self, image: np.ndarray) -> list[PlateDetection]:
                return [
                    PlateDetection(0.95, 20, 10, 80, 25),
                    PlateDetection(0.96, 240, 10, 80, 25),
                ]

        track_manager = PlateTrackManager(
            max_active_tracks_per_camera=2,
            timeout_ms=1000,
            iou_threshold=0.25,
        )
        config = recognition_config(Path(self.temp_directory.name))
        frame_buffer = PreDetectionFrameBuffer(config)
        frame_buffer.ingest(self._make_snapshot(95, 9.8), motion_score=0.0)

        update = track_manager.update(
            1,
            (
                PlateDetection(0.95, 20, 10, 80, 25),
                PlateDetection(0.96, 240, 10, 80, 25),
            ),
            observed_at=10.0,
            activity_at=10.0,
        )
        track_11, track_12 = (item.track_id for item in update.assignments)

        processor = PlateRecognitionProcessor(
            SequencedOcrProvider(
                [
                    [OcrSegment("34ABC123", 0.95, (0, 0, 10, 10), 0)],
                    [OcrSegment("35XYZ789", 0.96, (0, 0, 10, 10), 0)],
                    [OcrSegment("35XYZ789", 0.97, (0, 0, 10, 10), 0)],
                ]
            ),
            self.plate_service,
            config,
            detector=DualDetector(),
            track_manager=track_manager,
            frame_buffer=frame_buffer,
        )
        # Track 11: 1 live frame
        track_manager.mark_ocr_scheduled(track_11)
        processor.process_ocr_job(
            self._make_ocr_job(
                frame_id=100,
                observed_at=10.0,
                detection=PlateDetection(0.95, 20, 10, 80, 25),
                track_id=track_11,
            ),
            queue_depth=0,
        )
        track_manager.mark_ocr_finished(track_11)

        # Track 12: 2 live frames
        track_manager.mark_ocr_scheduled(track_12)
        processor.process_ocr_job(
            self._make_ocr_job(
                frame_id=100,
                observed_at=10.0,
                detection=PlateDetection(0.96, 240, 10, 80, 25),
                track_id=track_12,
            ),
            queue_depth=1,
        )
        track_manager.mark_ocr_finished(track_12)

        track_manager.mark_ocr_scheduled(track_12)
        outcome_12 = processor.process_ocr_job(
            self._make_ocr_job(
                frame_id=102,
                observed_at=10.2,
                detection=PlateDetection(0.96, 240, 10, 80, 25),
                track_id=track_12,
            ),
            queue_depth=0,
        )
        track_manager.mark_ocr_finished(track_12)

        self.assertIsNotNone(outcome_12)
        self.assertIs(outcome_12.state, RecognitionState.SAVED)
        self.assertIs(outcome_12.finalization_source, FinalizationSource.NORMAL_CONFIRMATION)
        self.assertEqual(outcome_12.record.plate, "35XYZ789")

        # Track 11 finalization without matching buffer frame discards
        outcome_11 = processor.finalize_track(1, track_11)
        self.assertIsNotNone(outcome_11)
        self.assertIs(outcome_11.state, RecognitionState.AMBIGUOUS_DISCARDED)

    def test_pending_ocr_completes_before_track_finalization(self) -> None:
        from app.plate_tracking import PlateTrackManager

        track_manager = PlateTrackManager(
            max_active_tracks_per_camera=2,
            timeout_ms=1000,
            iou_threshold=0.25,
        )
        detection = PlateDetection(0.91, 20, 10, 100, 30)
        track_id = track_manager.update(
            1, (detection,), 10.0, activity_at=10.0
        ).assignments[0].track_id
        processor = PlateRecognitionProcessor(
            SequencedOcrProvider(
                [
                    [OcrSegment("23ABC123", 0.95, (0, 0, 10, 10), 0)],
                    [OcrSegment("23ABC123", 0.96, (0, 0, 10, 10), 0)],
                ]
            ),
            self.plate_service,
            self.config,
            track_manager=track_manager,
        )
        track_manager.mark_ocr_scheduled(track_id)
        track_manager.mark_ocr_scheduled(track_id)
        track_manager.expire_due(12.0)
        # Track is expired but has 2 pending live OCR jobs
        self.assertEqual(track_manager.consume_finalized(), ())

        processor.process_ocr_job(
            self._make_ocr_job(
                frame_id=100,
                observed_at=10.0,
                detection=detection,
                track_id=track_id,
            ),
            queue_depth=1,
        )
        track_manager.mark_ocr_finished(track_id)
        self.assertEqual(track_manager.consume_finalized(), ())

        processor.process_ocr_job(
            self._make_ocr_job(
                frame_id=101,
                observed_at=10.2,
                detection=detection,
                track_id=track_id,
            ),
            queue_depth=0,
        )
        track_manager.mark_ocr_finished(track_id)
        finalized_tracks = track_manager.consume_finalized()
        self.assertEqual(len(finalized_tracks), 1)

        outcome = processor.finalize_track(1, track_id)
        self.assertEqual(self._record_count(), 1)

    def test_double_finalization_is_idempotent(self) -> None:
        config = recognition_config(Path(self.temp_directory.name))
        frame_buffer = PreDetectionFrameBuffer(config)
        frame_buffer.ingest(self._make_snapshot(95, 9.8), motion_score=0.0)
        processor = PlateRecognitionProcessor(
            SequencedOcrProvider(
                [
                    [OcrSegment("23ABC123", 0.95, (0, 0, 10, 10), 0)],
                    [OcrSegment("23ABC123", 0.96, (0, 0, 10, 10), 0)],
                ]
            ),
            self.plate_service,
            config,
            frame_buffer=frame_buffer,
        )
        processor.process_ocr_job(
            self._make_ocr_job(frame_id=100, observed_at=10.0, track_id=80),
            queue_depth=0,
        )
        first_call = processor.finalize_track(1, 80)
        self.assertIsNotNone(first_call)
        self.assertIs(first_call.state, RecognitionState.AMBIGUOUS_DISCARDED)
        self.assertEqual(self._record_count(), 0)

        second_call = processor.finalize_track(1, 80)
        self.assertIsNone(second_call)
        self.assertEqual(self._record_count(), 0)

    def test_corrected_candidate_requires_stricter_confirmation_even_with_rescue(self) -> None:
        config = recognition_config(Path(self.temp_directory.name))
        frame_buffer = PreDetectionFrameBuffer(config)
        frame_buffer.ingest(self._make_snapshot(95, 9.8), motion_score=0.0)
        processor = PlateRecognitionProcessor(
            SequencedOcrProvider(
                [
                    [OcrSegment("23ABCI23", 0.96, (0, 0, 10, 10), 0)],
                    [OcrSegment("23ABCI23", 0.96, (0, 0, 10, 10), 0)],
                ]
            ),
            self.plate_service,
            config,
            frame_buffer=frame_buffer,
        )
        processor.process_ocr_job(
            self._make_ocr_job(frame_id=100, observed_at=10.0, track_id=90),
            queue_depth=0,
        )
        finalized = processor.finalize_track(1, 90)
        self.assertIsNotNone(finalized)
        self.assertIs(finalized.state, RecognitionState.AMBIGUOUS_DISCARDED)
        self.assertEqual(finalized.suppression_reason, "corrected-candidate")
        self.assertIsNone(finalized.record)
        self.assertEqual(self._record_count(), 0)

    def _record_count(self) -> int:
        with self.database.connection() as connection:
            row = connection.execute("SELECT COUNT(*) FROM plate_records").fetchone()
        return int(row[0])


if __name__ == "__main__":
    unittest.main()
