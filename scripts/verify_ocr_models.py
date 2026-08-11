from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ocr_models import (
    OcrModelError,
    OcrModelNotFound,
    validate_model_directory,
    validate_ocr_models,
)
from app.plate_recognition import PaddleOcrProvider


EXPECTED = {
    "paddleocr": "3.7.0",
    "onnxruntime": "1.27.0",
    "opencv-contrib-python": "4.10.0.84",
}


def verify_ocr_setup(model_root: Path, provider_factory=None) -> tuple[int, list[str]]:
    """Return a short, stable readiness report without downloading models."""
    model_root = model_root.resolve()
    lines: list[str] = []
    models_ready = True

    for folder, label in (("detection", "Detection"), ("recognition", "Recognition")):
        try:
            validate_model_directory(model_root / folder, label, load_onnx=False)
        except OcrModelNotFound:
            status = "MISSING"
            models_ready = False
        except OcrModelError:
            status = "INVALID"
            models_ready = False
        else:
            status = "OK"
        lines.append(f"{label} model: {status}")

    if not models_ready:
        lines.append("OCR status: NOT READY")
        return 1, lines

    versions_ready = all(
        _installed_version(package) == expected
        for package, expected in EXPECTED.items()
    )
    if not versions_ready:
        lines.extend(
            (
                "ONNX Runtime: ERROR",
                "PaddleOCR provider: ERROR",
                "OCR status: NOT READY",
            )
        )
        return 1, lines

    try:
        validate_ocr_models(model_root, load_onnx=True)
    except OcrModelError:
        lines.extend(("ONNX Runtime: ERROR", "OCR status: NOT READY"))
        return 1, lines
    lines.append("ONNX Runtime: OK")

    factory = provider_factory or PaddleOcrProvider
    try:
        provider = factory(model_root)
        provider.recognize([np.zeros((64, 256, 3), dtype=np.uint8)])
    except Exception:
        lines.extend(("PaddleOCR provider: ERROR", "OCR status: NOT READY"))
        return 1, lines

    lines.extend(("PaddleOCR provider: OK", "OCR status: READY"))
    return 0, lines


def _installed_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lokal OCR modellerinin kullanıma hazır olup olmadığını doğrular."
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=PROJECT_ROOT / "models" / "ocr",
    )
    args = parser.parse_args()

    exit_code, lines = verify_ocr_setup(args.model_root)
    print("\n".join(lines))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
