from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.verify_ocr_models import verify_ocr_setup


class OcrVerificationTests(unittest.TestCase):
    def test_missing_models_report_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            exit_code, lines = verify_ocr_setup(Path(temporary_directory) / "ocr")

        self.assertEqual(exit_code, 1)
        self.assertEqual(
            lines,
            [
                "Detection model: MISSING",
                "Recognition model: MISSING",
                "OCR status: NOT READY",
            ],
        )


if __name__ == "__main__":
    unittest.main()
