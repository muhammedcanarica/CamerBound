from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.camera import Direction
from app.config import DEFAULT_ROI, NormalizedRoi, load_config, update_plate_roi


class PlateRecognitionConfigTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
