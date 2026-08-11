from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ocr_models import OcrModelError, validate_model_directory, validate_ocr_models


MODEL_ROOT = PROJECT_ROOT / "models" / "ocr" / "onnx"


def install_models(
    sources: dict[str, tuple[Path, str]],
    model_root: Path = MODEL_ROOT,
) -> None:
    for source, label in sources.values():
        validate_model_directory(source, label, load_onnx=True)

    for name, (source, _label) in sources.items():
        target = model_root / name
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "inference.onnx", target / "inference.onnx")
        shutil.copy2(source / "inference.yml", target / "inference.yml")

    validate_ocr_models(model_root, load_onnx=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hazır inference.onnx + inference.yml model klasörlerini uygulamaya kopyalar."
    )
    parser.add_argument("--detection-source", type=Path, required=True)
    parser.add_argument("--recognition-source", type=Path, required=True)
    args = parser.parse_args()

    sources = {
        "detection": (args.detection_source.resolve(), "Detection"),
        "recognition": (args.recognition_source.resolve(), "Recognition"),
    }
    try:
        install_models(sources)
    except OcrModelError as exc:
        parser.error(str(exc))

    for name, (source, _label) in sources.items():
        target = MODEL_ROOT / name
        print(f"{name}: {source} -> {target}")

    print("OCR modelleri models/ocr/onnx altında hazır ve ONNX Runtime ile yüklenebiliyor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
