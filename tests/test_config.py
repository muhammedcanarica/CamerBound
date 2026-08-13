from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.camera import Direction
from app.config import (
    DEFAULT_ROI,
    ConfigError,
    NormalizedRoi,
    PLATE_DETECTOR_MODEL_NAME,
    load_config,
    update_plate_roi,
    update_record_retention,
)


class PlateRecognitionConfigTests(unittest.TestCase):
    def test_plate_detector_settings_are_validated_and_model_path_is_local(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            settings_path = Path(temp_directory) / "settings.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "database_path": "test.db",
                        "plate_detection": {
                            "plate_detector": {
                                "enabled": True,
                                "backend": "openvino",
                                "min_confidence": 0.50,
                                "crop_padding_ratio": 0.15,
                                "max_plate_candidates_per_frame": 2,
                                "fallback_to_roi_ocr": True,
                                "debug_overlay": False,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            detector = load_config(settings_path).plate_recognition.plate_detector

        self.assertTrue(detector.enabled)
        self.assertEqual(detector.backend, "openvino")
        self.assertEqual(detector.min_confidence, 0.50)
        self.assertEqual(detector.crop_padding_ratio, 0.15)
        self.assertEqual(detector.max_plate_candidates_per_frame, 2)
        self.assertTrue(detector.fallback_to_roi_ocr)
        self.assertFalse(detector.debug_overlay)
        self.assertEqual(detector.model_dir.name, PLATE_DETECTOR_MODEL_NAME)
        self.assertEqual(detector.model_xml.name, "model.xml")
        self.assertEqual(detector.model_bin.name, "model.bin")

    def test_invalid_plate_detector_values_use_safe_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            settings_path = Path(temp_directory) / "settings.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "database_path": "test.db",
                        "plate_detection": {
                            "plate_detector": {
                                "enabled": "yes",
                                "backend": "unknown",
                                "min_confidence": 2,
                                "crop_padding_ratio": -1,
                                "max_plate_candidates_per_frame": 0,
                                "fallback_to_roi_ocr": "yes",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = load_config(settings_path).plate_recognition

        detector = config.plate_detector
        self.assertTrue(detector.enabled)
        self.assertEqual(detector.backend, "openvino")
        self.assertEqual(detector.min_confidence, 0.50)
        self.assertEqual(detector.crop_padding_ratio, 0.15)
        self.assertEqual(detector.max_plate_candidates_per_frame, 2)
        self.assertTrue(detector.fallback_to_roi_ocr)
        self.assertGreaterEqual(len(config.warnings), 6)

    def test_ocr_sampling_interval_is_250_without_relaxing_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            settings_path = Path(temp_directory) / "settings.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "database_path": "test.db",
                        "plate_detection": {
                            "recognition_interval_ms": 250,
                            "min_confidence": 0.65,
                            "confirmations_required": 2,
                            "confirmation_window_seconds": 3,
                            "duplicate_cooldown_seconds": 10,
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = load_config(settings_path).plate_recognition

        self.assertEqual(config.recognition_interval_ms, 250)
        self.assertEqual(config.min_confidence, 0.65)
        self.assertEqual(config.confirmations_required, 2)
        self.assertEqual(config.confirmation_window_seconds, 3)
        self.assertEqual(config.duplicate_cooldown_seconds, 10)

    def test_missing_ocr_sampling_interval_defaults_to_250(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            settings_path = Path(temp_directory) / "settings.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "database_path": "test.db",
                        "plate_detection": {},
                    }
                ),
                encoding="utf-8",
            )

            config = load_config(settings_path).plate_recognition

        self.assertEqual(config.recognition_interval_ms, 250)
        self.assertEqual(config.min_confidence, 0.65)
        self.assertEqual(config.confirmations_required, 2)

    def test_plate_capture_defaults_and_invalid_values_use_safe_fallbacks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            settings_path = Path(temp_directory) / "settings.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "database_path": "test.db",
                        "plate_capture": {
                            "enabled": "yes",
                            "max_width": 10,
                            "jpeg_quality": 100,
                        },
                    }
                ),
                encoding="utf-8",
            )

            capture = load_config(settings_path).plate_capture

        self.assertTrue(capture.enabled)
        self.assertEqual(capture.max_width, 960)
        self.assertEqual(capture.jpeg_quality, 60)
        self.assertGreaterEqual(len(capture.warnings), 3)

    def test_ocr_backend_defaults_to_auto_and_invalid_value_warns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            settings_path = Path(temp_directory) / "settings.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "database_path": "test.db",
                        "plate_detection": {"ocr_backend": "invalid"},
                    }
                ),
                encoding="utf-8",
            )

            config = load_config(settings_path).plate_recognition

        self.assertEqual(config.ocr_backend, "auto")
        self.assertTrue(any("ocr_backend" in warning for warning in config.warnings))

    def test_invalid_roi_falls_back_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            settings_path = Path(temp_directory) / "settings.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "database_path": "test.db",
                        "plate_detection": {
                            "roi": {
                                "ENTRY": {"x": 0.9, "y": 0.0, "width": 0.5, "height": 1.0},
                                "EXIT": {"x": 0.1, "y": 0.2, "width": 0.7, "height": 0.6},
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = load_config(settings_path).plate_recognition

        self.assertEqual(config.entry_roi, DEFAULT_ROI)
        self.assertEqual(config.exit_roi.x, 0.1)
        self.assertTrue(config.warnings)

    def test_roi_update_is_atomic_preserves_unknown_fields_and_reloads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            settings_path = Path(temp_directory) / "settings.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "database_path": "test.db",
                        "unknown_root": {"keep": True},
                        "plate_detection": {"custom": "keep-me", "roi": {}},
                    }
                ),
                encoding="utf-8",
            )
            expected = NormalizedRoi(0.2, 0.25, 0.6, 0.5)

            config = update_plate_roi(settings_path, Direction.ENTRY, expected)
            raw = json.loads(settings_path.read_text(encoding="utf-8"))

            self.assertEqual(config.entry_roi, expected)
            self.assertEqual(load_config(settings_path).plate_recognition.entry_roi, expected)
            self.assertEqual(raw["unknown_root"], {"keep": True})
            self.assertEqual(raw["plate_detection"]["custom"], "keep-me")
            self.assertEqual(list(Path(temp_directory).glob("*.tmp")), [])

    def test_invalid_record_retention_falls_back_to_90_with_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            settings_path = Path(temp_directory) / "settings.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "database_path": "test.db",
                        "plate_detection": {"record_retention_days": 45},
                    }
                ),
                encoding="utf-8",
            )

            config = load_config(settings_path).plate_recognition

        self.assertEqual(config.record_retention_days, 90)
        self.assertTrue(
            any("record_retention_days" in warning for warning in config.warnings)
        )

    def test_record_retention_update_is_atomic_and_preserves_unknown_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            settings_path = Path(temp_directory) / "settings.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "database_path": "test.db",
                        "unknown_root": {"keep": True},
                        "plate_detection": {"custom": "keep-me"},
                    }
                ),
                encoding="utf-8",
            )

            config = update_record_retention(settings_path, 180)
            raw = json.loads(settings_path.read_text(encoding="utf-8"))

            self.assertEqual(config.record_retention_days, 180)
            self.assertEqual(raw["unknown_root"], {"keep": True})
            self.assertEqual(raw["plate_detection"]["custom"], "keep-me")
            self.assertEqual(raw["plate_detection"]["record_retention_days"], 180)
            self.assertEqual(list(Path(temp_directory).glob("*.tmp")), [])

    def test_failed_atomic_record_retention_update_keeps_original_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            settings_path = Path(temp_directory) / "settings.json"
            original = json.dumps(
                {
                    "database_path": "test.db",
                    "plate_detection": {"record_retention_days": 90},
                }
            )
            settings_path.write_text(original, encoding="utf-8")

            with patch("app.config.os.replace", side_effect=OSError("disk error")):
                with self.assertRaises(ConfigError):
                    update_record_retention(settings_path, 30)

            self.assertEqual(settings_path.read_text(encoding="utf-8"), original)
            self.assertEqual(list(Path(temp_directory).glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
