from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "vehicle-license-plate-detection-barrier-0123"
TARGET_DIR = PROJECT_ROOT / "models" / "plate_detector" / MODEL_NAME
WORK_DIR = PROJECT_ROOT / "models" / "plate_detector" / ".omz-work"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Open Model Zoo plate detector modelini geliştirici makinesinde "
            "indirir, OpenVINO IR formatına dönüştürür ve lokal model dizinine kopyalar."
        )
    )
    parser.add_argument(
        "--omz-tools-dir",
        type=Path,
        help="Open Model Zoo tools/model_tools dizini (opsiyonel).",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Ağ veya converter çalıştırmadan mevcut model dosyalarını doğrular.",
    )
    args = parser.parse_args()

    if args.check_only:
        return _check_target()

    downloader, converter = _resolve_tools(args.omz_tools_dir)
    download_dir = WORK_DIR / "download"
    converted_dir = WORK_DIR / "converted"
    download_dir.mkdir(parents=True, exist_ok=True)
    converted_dir.mkdir(parents=True, exist_ok=True)

    _run_tool(
        downloader,
        "--name",
        MODEL_NAME,
        "--output_dir",
        str(download_dir),
    )
    _run_tool(
        converter,
        "--name",
        MODEL_NAME,
        "--download_dir",
        str(download_dir),
        "--output_dir",
        str(converted_dir),
        "--precisions",
        "FP32",
    )

    source_xml = _find_converted_file(converted_dir, ".xml")
    source_bin = source_xml.with_suffix(".bin")
    if not source_bin.is_file():
        raise SystemExit(f"Dönüştürülmüş model.bin bulunamadı: {source_bin}")

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_xml, TARGET_DIR / "model.xml")
    shutil.copy2(source_bin, TARGET_DIR / "model.bin")
    print(f"Plate detector hazır: {TARGET_DIR}")
    return _check_target()


def _resolve_tools(tools_dir: Path | None) -> tuple[list[str], list[str]]:
    if tools_dir is not None:
        resolved = tools_dir.resolve()
        downloader = resolved / "downloader.py"
        converter = resolved / "converter.py"
        if not downloader.is_file() or not converter.is_file():
            raise SystemExit(
                "--omz-tools-dir içinde downloader.py ve converter.py bulunmalıdır."
            )
        return [sys.executable, str(downloader)], [sys.executable, str(converter)]

    downloader = shutil.which("omz_downloader")
    converter = shutil.which("omz_converter")
    if downloader is None or converter is None:
        raise SystemExit(
            "omz_downloader/omz_converter PATH üzerinde bulunamadı. "
            "Open Model Zoo tools/model_tools dizinini --omz-tools-dir ile verin."
        )
    return [downloader], [converter]


def _run_tool(command: list[str], *arguments: str) -> None:
    subprocess.run([*command, *arguments], check=True, cwd=PROJECT_ROOT)


def _find_converted_file(root: Path, suffix: str) -> Path:
    matches = [
        path
        for path in root.rglob(f"*{suffix}")
        if MODEL_NAME in path.name or MODEL_NAME in str(path.parent)
    ]
    if not matches:
        raise SystemExit(
            f"Dönüştürülmüş {suffix} model dosyası bulunamadı: {root}"
        )
    fp32_matches = [path for path in matches if "FP32" in path.parts]
    return sorted(fp32_matches or matches)[0]


def _check_target() -> int:
    missing = [
        path for path in (TARGET_DIR / "model.xml", TARGET_DIR / "model.bin")
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        print("Eksik plate detector dosyaları:")
        for path in missing:
            print(f"  {path}")
        return 1
    print(f"Plate detector model dosyaları doğrulandı: {TARGET_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
