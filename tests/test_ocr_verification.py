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

    def test_paddle_models_report_ready_with_native_backend(self) -> None:
        class Provider:
            def recognize(self, _images):
                return []

        with tempfile.TemporaryDirectory() as temporary_directory:
            model_root = Path(temporary_directory) / "ocr"
            for folder in ("detection", "recognition"):
                directory = model_root / "paddle" / folder
                directory.mkdir(parents=True)
                (directory / "inference.json").write_text("{}", encoding="utf-8")
                (directory / "inference.pdiparams").write_bytes(b"parameters")
                (directory / "inference.yml").write_text(
                    "Global:\n  model_name: fake\n",
                    encoding="utf-8",
                )

            exit_code, lines = verify_ocr_setup(
                model_root,
                backend="paddle",
                provider_factory=lambda _root, _backend: Provider(),
                paddle_runtime_checker=lambda: True,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            lines,
            [
                "Detection model: OK",
                "Recognition model: OK",
                "PaddlePaddle: OK",
                "PaddleOCR provider: OK",
                "OCR backend: PADDLE",
                "OCR status: READY",
            ],
        )


if __name__ == "__main__":
    unittest.main()
