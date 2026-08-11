from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ocr_models import collect_model_diagnostics
from app.plate_recognition import PaddleOcrProvider


EXPECTED = {
    "paddleocr": "3.7.0",
    "onnxruntime": "1.27.0",
    "opencv-contrib-python": "4.10.0.84",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="OCR sürümlerini, modellerini ve provider'ı doğrular.")
    parser.add_argument("--model-root", type=Path, default=PROJECT_ROOT / "models" / "ocr")
    args = parser.parse_args()
    failed = False

    print("Runtime sürümleri")
    for package, expected in EXPECTED.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = "YÜKLÜ DEĞİL"
        ok = actual == expected
        failed |= not ok
        print(f"  [{'OK' if ok else 'HATA'}] {package}: {actual} (beklenen {expected})")

    print("Model dosyaları ve ONNX Runtime")
    checks = collect_model_diagnostics(args.model_root.resolve(), load_onnx=True)
    for check in checks:
        failed |= not check.ok
        provider_text = f" providers={','.join(check.providers)}" if check.providers else ""
        print(f"  [{'OK' if check.ok else 'HATA'}] {check.name}: {check.message}{provider_text}")

    if not failed:
        print("PaddleOCR provider başlatılıyor...")
        try:
            provider = PaddleOcrProvider(args.model_root.resolve())
            smoke_result = provider.recognize(
                [np.zeros((64, 256, 3), dtype=np.uint8)]
            )
        except Exception as exc:
            failed = True
            print(f"  [HATA] Provider başlatılamadı: {exc}")
        else:
            print(
                "  [OK] Provider ONNX Runtime CPU ile başlatıldı ve predict çalıştı "
                f"(segment={len(smoke_result)})."
            )

    print("SONUÇ: " + ("BAŞARISIZ" if failed else "BAŞARILI"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
