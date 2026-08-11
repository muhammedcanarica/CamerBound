from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ocr_models import OcrModelError
from scripts.convert_paddle_models import (
    ConversionToolsError,
    check_conversion_tools,
    convert,
    validate_paddle_model_directory,
)
from scripts.setup_ocr_models import install_models
from scripts.verify_ocr_models import verify_ocr_setup


DOWNLOAD_ROOT_NAME = Path("build") / "ocr-model-downloads"
WORK_ROOT_NAME = Path("build") / "ocr-model-work"
FINAL_MODEL_ROOT_NAME = Path("models") / "ocr"
DOWNLOAD_TIMEOUT_SECONDS = 60


class ModelPreparationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ModelSpec:
    key: str
    label: str
    name: str
    url: str

    @property
    def archive_name(self) -> str:
        return f"{self.name}_infer.tar"


DEFAULT_MODELS = (
    ModelSpec(
        key="detection",
        label="Detection",
        name="PP-OCRv5_mobile_det",
        url=(
            "https://paddle-model-ecology.bj.bcebos.com/paddlex/"
            "official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_infer.tar"
        ),
    ),
    ModelSpec(
        key="recognition",
        label="Recognition",
        name="en_PP-OCRv5_mobile_rec",
        url=(
            "https://paddle-model-ecology.bj.bcebos.com/paddlex/"
            "official_inference_model/paddle3.0.0/en_PP-OCRv5_mobile_rec_infer.tar"
        ),
    ),
)


def download_file(
    url: str,
    destination: Path,
    *,
    timeout: int = DOWNLOAD_TIMEOUT_SECONDS,
    opener=None,
) -> None:
    """Download to a partial file and atomically publish only complete content."""
    opener = opener or urllib.request.urlopen
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    partial.unlink(missing_ok=True)
    bytes_written = 0
    try:
        with opener(url, timeout=timeout) as response, partial.open("wb") as output:
            expected_length = response.headers.get("Content-Length")
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                bytes_written += len(chunk)
            output.flush()
            os.fsync(output.fileno())
        if expected_length is not None and bytes_written != int(expected_length):
            raise ModelPreparationError(
                f"Download incomplete for {url}: {bytes_written}/{expected_length} bytes"
            )
        os.replace(partial, destination)
    except Exception as exc:
        partial.unlink(missing_ok=True)
        if isinstance(exc, ModelPreparationError):
            raise
        raise ModelPreparationError(f"Download failed for {url}: {exc}") from exc


def ensure_archive(
    spec: ModelSpec,
    download_root: Path,
    *,
    force: bool = False,
    downloader=download_file,
) -> bool:
    archive = download_root / spec.archive_name
    if archive.is_file() and not force:
        return False
    if force:
        archive.unlink(missing_ok=True)
    downloader(spec.url, archive)
    return True


def safe_extract_tar(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    resolved_destination = destination.resolve()
    try:
        with tarfile.open(archive, mode="r:*") as tar:
            for member in tar.getmembers():
                if member.issym() or member.islnk():
                    raise ModelPreparationError(
                        f"Unsafe archive link rejected: {member.name}"
                    )
                member_path = (resolved_destination / member.name).resolve()
                try:
                    inside_target = (
                        os.path.commonpath((resolved_destination, member_path))
                        == str(resolved_destination)
                    )
                except ValueError:
                    inside_target = False
                if not inside_target:
                    raise ModelPreparationError(
                        f"Unsafe archive path rejected: {member.name}"
                    )
            tar.extractall(resolved_destination, filter="data")
    except (tarfile.TarError, OSError) as exc:
        raise ModelPreparationError(f"Archive extraction failed: {archive}: {exc}") from exc


def find_paddle_model_directory(extracted_root: Path) -> Path:
    candidates = sorted(
        path.parent for path in extracted_root.rglob("inference.pdiparams")
    )
    for candidate in candidates:
        try:
            return validate_paddle_model_directory(candidate)
        except ValueError:
            continue
    raise ModelPreparationError(
        f"Extracted archive does not contain a valid Paddle inference model: {extracted_root}"
    )


def write_model_metadata(model_root: Path) -> None:
    metadata = {
        "detection": {"name": DEFAULT_MODELS[0].name},
        "recognition": {"name": DEFAULT_MODELS[1].name},
        "format": "onnx",
        "engine": "onnxruntime",
    }
    destination = model_root / "model-info.json"
    temporary = destination.with_name(destination.name + ".tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def dry_run_report(project_root: Path) -> list[str]:
    download_root = project_root / DOWNLOAD_ROOT_NAME
    work_root = project_root / WORK_ROOT_NAME
    model_root = project_root / FINAL_MODEL_ROOT_NAME
    return [
        f"Detection model: {DEFAULT_MODELS[0].name}",
        f"Recognition model: {DEFAULT_MODELS[1].name}",
        f"Download cache: {download_root}",
        f"Conversion output: {work_root / 'onnx'}",
        f"Final model path: {model_root}",
    ]


def prepare_default_models(
    project_root: Path = PROJECT_ROOT,
    *,
    force: bool = False,
    dry_run: bool = False,
    downloader=download_file,
    converter=convert,
    installer=install_models,
    verifier=verify_ocr_setup,
    tool_checker=check_conversion_tools,
) -> list[str]:
    project_root = project_root.resolve()
    if dry_run:
        return dry_run_report(project_root)

    tool_checker()
    download_root = project_root / DOWNLOAD_ROOT_NAME
    work_root = project_root / WORK_ROOT_NAME
    model_root = project_root / FINAL_MODEL_ROOT_NAME
    if force:
        shutil.rmtree(download_root, ignore_errors=True)
        shutil.rmtree(work_root, ignore_errors=True)
        _clear_model_outputs(model_root)

    lines: list[str] = []
    paddle_sources: dict[str, Path] = {}
    for spec in DEFAULT_MODELS:
        downloaded = ensure_archive(
            spec,
            download_root,
            force=False,
            downloader=downloader,
        )
        suffix = "OK" if downloaded else "OK (cached)"
        lines.append(f"Downloading {spec.name}... {suffix}")
        extract_root = work_root / "extracted" / spec.key
        shutil.rmtree(extract_root, ignore_errors=True)
        safe_extract_tar(download_root / spec.archive_name, extract_root)
        paddle_sources[spec.key] = find_paddle_model_directory(extract_root)

    lines.append("")
    converted_sources: dict[str, tuple[Path, str]] = {}
    for spec in DEFAULT_MODELS:
        output = work_root / "onnx" / spec.key
        shutil.rmtree(output, ignore_errors=True)
        converter(paddle_sources[spec.key], output, 11)
        converted_sources[spec.key] = (output, spec.label)
        lines.append(f"{spec.label} conversion... OK")

    installer(converted_sources, model_root)
    exit_code, verification_lines = verifier(model_root)
    if exit_code != 0:
        raise ModelPreparationError("\n".join(verification_lines))
    write_model_metadata(model_root)

    lines.extend(("", *verification_lines[:3], "", "OCR MODELS READY"))
    return lines


def _clear_model_outputs(model_root: Path) -> None:
    for folder in ("detection", "recognition"):
        directory = model_root / folder
        if not directory.is_dir():
            continue
        for item in directory.iterdir():
            if item.name == ".gitkeep":
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    (model_root / "model-info.json").unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resmî PaddleOCR modellerini indirir ve lokal ONNX modelleri hazırlar."
    )
    parser.add_argument("--force", action="store_true", help="Cache ve model çıktılarını yeniler.")
    parser.add_argument("--dry-run", action="store_true", help="Dosya değiştirmeden planı gösterir.")
    args = parser.parse_args()

    try:
        lines = prepare_default_models(force=args.force, dry_run=args.dry_run)
    except (
        ConversionToolsError,
        ModelPreparationError,
        OcrModelError,
        OSError,
        ValueError,
        subprocess.CalledProcessError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
