from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ocr_models import OcrModelError, validate_model_directory


def convert(source: Path, output: Path, opset: int) -> None:
    required = (source / "inference.pdiparams",)
    if not (source / "inference.json").is_file() and not (source / "inference.pdmodel").is_file():
        raise ValueError(f"Paddle model dosyası yok: {source}/inference.json veya inference.pdmodel")
    if not all(path.is_file() for path in required):
        raise ValueError(f"Paddle parametre dosyası yok: {required[0]}")
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "paddlex",
            "--paddle2onnx",
            "--paddle_model_dir",
            str(source),
            "--onnx_model_dir",
            str(output),
            "--opset_version",
            str(opset),
        ],
        check=True,
    )
    source_config = source / "inference.yml"
    target_config = output / "inference.yml"
    if source_config.is_file() and not target_config.is_file():
        shutil.copy2(source_config, target_config)


def main() -> int:
    parser = argparse.ArgumentParser(description="İki PaddleOCR inference modelini ONNX'e dönüştürür.")
    parser.add_argument("--detection-source", type=Path, required=True)
    parser.add_argument("--recognition-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "build" / "ocr-onnx")
    parser.add_argument("--opset", type=int, default=11)
    args = parser.parse_args()

    try:
        for folder, source in (
            ("detection", args.detection_source.resolve()),
            ("recognition", args.recognition_source.resolve()),
        ):
            output = args.output_root.resolve() / folder
            convert(source, output, args.opset)
            validate_model_directory(output, folder.title(), load_onnx=True)
            print(f"[OK] {folder}: {output}")
    except (ValueError, OcrModelError, subprocess.CalledProcessError) as exc:
        print(f"[HATA] Dönüşüm tamamlanamadı: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
