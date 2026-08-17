from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_field_alpr import (
    SampleResult,
    character_accuracy,
    comparison,
    confusion_pairs,
    load_manifest,
    percentile,
    raw_ocr_trace,
    summarize,
)
from app.plate_recognition import OcrSegment


class FieldBenchmarkTests(unittest.TestCase):
    def test_manifest_requires_valid_expected_plate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "image": "frame.jpg",
                            "direction": "ENTRY",
                            "expected_plate": "34DRF848",
                            "condition": "shadow",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            samples = load_manifest(path)

        self.assertEqual(samples[0].expected_plate, "34DRF848")
        self.assertEqual(samples[0].condition, "shadow")

    def test_character_metrics_report_one_character_confusion(self) -> None:
        self.assertAlmostEqual(character_accuracy("34DRF848", "34ORF848"), 0.875)
        self.assertEqual(confusion_pairs("34DRF848", "34ORF848"), {"D/O": 1})

    def test_percentile_uses_nearest_rank(self) -> None:
        self.assertEqual(percentile([1.0, 2.0, 3.0, 4.0], 0.95), 4.0)
        self.assertIsNone(percentile([], 0.95))

    def test_raw_trace_distinguishes_no_text_from_valid_candidate(self) -> None:
        trace = raw_ocr_trace(
            [OcrSegment("23 AEP 256", 0.95, (0, 0, 20, 8), 1)],
            ("color", "clahe"),
            (0, 1),
            0.65,
        )

        self.assertEqual(trace[0]["rejection_reason"], "no-text-detected")
        self.assertEqual(trace[1]["normalized"], "23AEP256")
        self.assertEqual(trace[1]["corrected"], "23AEP256")
        self.assertEqual(trace[1]["rejection_reason"], "none")

    def test_comparison_reports_before_after_and_delta(self) -> None:
        baseline = self._summary_shape(correct=0.5, false=0.5)
        current = self._summary_shape(correct=1.0, false=0.0)

        result = comparison(baseline, current)

        self.assertEqual(result["correct_read_rate_delta"], 0.5)
        self.assertEqual(result["false_read_rate_delta"], -0.5)

    def test_summary_separates_detector_recovery_and_false_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(
                json.dumps(
                    [
                        {
                            "image": "first.jpg",
                            "expected_plate": "34DRF848",
                            "direction": "ENTRY",
                        },
                        {
                            "image": "second.jpg",
                            "expected_plate": "23K2874",
                            "direction": "ENTRY",
                        },
                    ]
                ),
                encoding="utf-8",
            )
            samples = load_manifest(manifest)
        results = (
            self._result(
                image="first.jpg",
                expected="34DRF848",
                candidate="34ORF848",
                detector_hit=True,
                detector_variant="tiled",
                detector_crop_rescue_attempted=True,
            ),
            self._result(
                image="second.jpg",
                expected="23K2874",
                candidate="23K2874",
                detector_hit=False,
                detector_variant="tiled",
                used_roi_fallback=True,
                rescue_tiles=3,
            ),
        )

        report = summarize("test", samples, results)

        self.assertEqual(report["detector"]["recall"], 0.5)
        self.assertEqual(report["detector"]["tiled_recovery_gain"], 0.5)
        self.assertEqual(report["end_to_end"]["correct_read_rate"], 0.5)
        self.assertEqual(report["end_to_end"]["false_read_rate"], 0.5)
        self.assertEqual(report["ocr"]["confusions"], {"D/O": 1})
        self.assertEqual(report["ocr_rescue"]["attempted_samples"], 1)
        self.assertEqual(report["ocr_rescue"]["successful_exact_reads"], 1)
        self.assertEqual(report["ocr_rescue"]["attempted_tiles"], 3)
        self.assertEqual(report["ocr_rescue"]["detector_crop_invalid_samples"], 1)
        self.assertEqual(report["ocr_rescue"]["detector_crop_recovered_exact"], 0)
        self.assertEqual(
            report["ocr_rescue"]["false_detection_suppressed_recovery_count"],
            1,
        )

    @staticmethod
    def _result(
        *,
        image: str,
        expected: str,
        candidate: str | None,
        detector_hit: bool,
        detector_variant: str,
        used_roi_fallback: bool = False,
        rescue_tiles: int = 0,
        detector_crop_rescue_attempted: bool = False,
    ) -> SampleResult:
        return SampleResult(
            image=image,
            direction="ENTRY",
            condition="test",
            expected_plate=expected,
            candidate=candidate,
            confidence=0.9 if candidate else None,
            detector_hit=detector_hit,
            detector_variant=detector_variant,
            detector_detections=int(detector_hit),
            detector_ms=10.0,
            ocr_ms=100.0,
            end_to_end_ms=110.0,
            used_roi_fallback=used_roi_fallback,
            raw_segment_count=int(candidate is not None),
            valid_candidate=candidate is not None,
            low_confidence=False,
            exact=candidate == expected,
            character_accuracy=character_accuracy(expected, candidate),
            crop_profiles=("NORMAL",),
            inference_calls=1,
            rescue_tiles=rescue_tiles,
            detector_crop_rescue_attempted=detector_crop_rescue_attempted,
        )

    @staticmethod
    def _summary_shape(*, correct: float, false: float) -> dict[str, object]:
        return {
            "detector": {"recall": 0.8},
            "ocr": {
                "conditional_exact_accuracy": correct,
                "character_accuracy": correct,
            },
            "end_to_end": {
                "correct_read_rate": correct,
                "no_read_rate": 1.0 - correct - false,
                "false_read_rate": false,
            },
            "performance_ms": {
                "detector_mean": 10.0,
                "ocr_mean": 100.0,
                "end_to_end_mean": 110.0,
            },
        }


if __name__ == "__main__":
    unittest.main()
