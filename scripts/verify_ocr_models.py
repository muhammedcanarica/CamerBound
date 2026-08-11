from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ocr_models import (
    OcrBackend,
    OcrModelError,
    OcrModelNotFound,
    backend_model_root,
    normalize_ocr_backend,
    select_ocr_backend,
    validate_model_directory,
)
def verify_ocr_setup(
    model_root: Path,
    backend: str | OcrBackend = OcrBackend.AUTO,
    provider_factory=None,
    paddle_runtime_checker=None,
) -> tuple[int, list[str]]:
    """Return a stable backend-aware readiness report without downloading models."""
    model_root = model_root.resolve()
    requested = normalize_ocr_backend(backend)
    try:
        selection = select_ocr_backend(model_root, requested, load_onnx=True)
    except OcrModelError:
        diagnostic_backend = _diagnostic_backend(model_root, requested)
        diagnostic_root = backend_model_root(model_root, diagnostic_backend)
        lines = _model_status_lines(diagnostic_root, diagnostic_backend)
        lines.append("OCR status: NOT READY")
        return 1, lines

    lines = _model_status_lines(selection.model_root, selection.backend)
    if any(not line.endswith("OK") for line in lines):
        lines.append("OCR status: NOT READY")
        return 1, lines

    if selection.backend is OcrBackend.PADDLE:
        checker = paddle_runtime_checker or _paddle_runtime_available
        if not checker():
            lines.extend(("PaddlePaddle: ERROR", "OCR status: NOT READY"))
            return 1, lines
        lines.append("PaddlePaddle: OK")
    else:
        lines.append("ONNX Runtime: OK")

    try:
        if provider_factory is None:
            from app.plate_recognition import PaddleOcrProvider

            provider = PaddleOcrProvider(model_root, backend=selection.backend)
        else:
            provider = provider_factory(model_root, selection.backend)
        segments = provider.recognize([np.zeros((64, 256, 3), dtype=np.uint8)])
        if not isinstance(segments, list):
            raise TypeError("OCR provider list döndürmedi")
    except Exception:
        lines.extend(("PaddleOCR provider: ERROR", "OCR status: NOT READY"))
        return 1, lines

    lines.extend(
        (
            "PaddleOCR provider: OK",
            f"OCR backend: {selection.backend.value.upper()}",
            "OCR status: READY",
        )
    )
    return 0, lines


def _model_status_lines(model_root: Path, backend: OcrBackend) -> list[str]:
    lines: list[str] = []
    for folder, label in (("detection", "Detection"), ("recognition", "Recognition")):
        try:
            validate_model_directory(
                model_root / folder,
                label,
                backend=backend,
                load_onnx=False,
            )
        except OcrModelNotFound:
            status = "MISSING"
        except OcrModelError:
            status = "INVALID"
        else:
            status = "OK"
        lines.append(f"{label} model: {status}")
    return lines


def _diagnostic_backend(model_root: Path, requested: OcrBackend) -> OcrBackend:
    if requested is not OcrBackend.AUTO:
        return requested
    paddle_root = backend_model_root(model_root, OcrBackend.PADDLE)
    if paddle_root.exists():
        return OcrBackend.PADDLE
    return OcrBackend.ONNX


def _paddle_runtime_available() -> bool:
    try:
        import paddle

        return hasattr(paddle, "inference")
    except (ImportError, OSError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lokal OCR modellerinin kullanıma hazır olup olmadığını doğrular."
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=PROJECT_ROOT / "models" / "ocr",
    )
    parser.add_argument(
        "--backend",
        choices=tuple(item.value for item in OcrBackend),
        default=OcrBackend.AUTO.value,
    )
    args = parser.parse_args()

    exit_code, lines = verify_ocr_setup(args.model_root, backend=args.backend)
    print("\n".join(lines))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
