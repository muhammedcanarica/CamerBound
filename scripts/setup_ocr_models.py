from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "models" / "ocr"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Önceden indirilmiş PaddleOCR model dizinlerini uygulamaya hazırlar."
    )
    parser.add_argument("--detection-source", type=Path, required=True)
    parser.add_argument("--recognition-source", type=Path, required=True)
    args = parser.parse_args()

    sources = {
        "detection": args.detection_source.resolve(),
        "recognition": args.recognition_source.resolve(),
    }
    for name, source in sources.items():
        if not source.is_dir() or not any(item.is_file() for item in source.rglob("*")):
            parser.error(f"{name} model dizini bulunamadı veya boş: {source}")

    for name, source in sources.items():
        target = MODEL_ROOT / name
        target.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target, dirs_exist_ok=True)
        print(f"{name}: {source} -> {target}")

    print("OCR modelleri hazır. Bu klasörleri production paketine dahil edin.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
