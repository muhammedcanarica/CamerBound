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
CONVERSION_ROOT_NAME = Path("build") / "ocr-onnx"
FINAL_MODEL_ROOT_NAME = Path("models") / "ocr"
STAGING_MODEL_ROOT_NAME = Path("models") / "ocr-staging"
BACKUP_MODEL_ROOT_NAME = Path("models") / "ocr-backup"
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
                "Model download failed: incomplete response for "
                f"{url}: {bytes_written}/{expected_length} bytes"
            )
        os.replace(partial, destination)
    except Exception as exc:
        partial.unlink(missing_ok=True)
        if isinstance(exc, ModelPreparationError):
            raise
        raise ModelPreparationError(f"Model download failed: {url}: {exc}") from exc


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
    conversion_root = project_root / CONVERSION_ROOT_NAME
    model_root = project_root / FINAL_MODEL_ROOT_NAME
    return [
        "Detection model:",
        DEFAULT_MODELS[0].name,
        "",
        "Recognition model:",
        DEFAULT_MODELS[1].name,
        "",
        "Download cache:",
        str(download_root),
        "",
        "Conversion output:",
        str(conversion_root),
        "",
        "Final models:",
        str(model_root),
        "",
        "No files changed.",
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
    conversion_root = project_root / CONVERSION_ROOT_NAME
    model_root = project_root / FINAL_MODEL_ROOT_NAME
    staging_root = project_root / STAGING_MODEL_ROOT_NAME
    backup_root = project_root / BACKUP_MODEL_ROOT_NAME
    _recover_interrupted_publish(model_root, backup_root)
    if force:
        _remove_tree(download_root, project_root / "build")
        _remove_tree(work_root, project_root / "build")
        _remove_tree(conversion_root, project_root / "build")

    _remove_tree(staging_root, project_root / "models")

    lines: list[str] = []
    try:
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
            _remove_tree(extract_root, work_root)
            safe_extract_tar(download_root / spec.archive_name, extract_root)
            paddle_sources[spec.key] = find_paddle_model_directory(extract_root)

        lines.append("")
        converted_sources: dict[str, tuple[Path, str]] = {}
        for spec in DEFAULT_MODELS:
            output = conversion_root / spec.key
            _remove_tree(output, conversion_root)
            converter(paddle_sources[spec.key], output, 11)
            converted_sources[spec.key] = (output, spec.label)
            lines.append(f"{spec.label} conversion... OK")

        installer(converted_sources, staging_root)
        _preserve_gitkeep_files(model_root, staging_root)
        write_model_metadata(staging_root)
        exit_code, verification_lines = verifier(staging_root)
        if exit_code != 0:
            raise ModelPreparationError("\n".join(verification_lines))
        _publish_staged_models(staging_root, model_root, backup_root)

        lines.extend(("", *verification_lines, "", "OCR MODELS READY"))
        return lines
    finally:
        _remove_tree(staging_root, project_root / "models")


def _remove_tree(path: Path, allowed_root: Path) -> None:
    path = path.resolve()
    allowed_root = allowed_root.resolve()
    if path == allowed_root or not path.is_relative_to(allowed_root):
        raise ModelPreparationError(f"Unsafe cleanup path rejected: {path}")
    if path.exists():
        shutil.rmtree(path)


def _recover_interrupted_publish(model_root: Path, backup_root: Path) -> None:
    if not backup_root.exists():
        return
    if model_root.exists():
        _remove_tree(backup_root, model_root.parent)
    else:
        os.replace(backup_root, model_root)


def _preserve_gitkeep_files(model_root: Path, staging_root: Path) -> None:
    for folder in ("detection", "recognition"):
        source = model_root / folder / ".gitkeep"
        if source.is_file():
            shutil.copy2(source, staging_root / folder / ".gitkeep")


def _publish_staged_models(
    staging_root: Path,
    model_root: Path,
    backup_root: Path,
) -> None:
    _recover_interrupted_publish(model_root, backup_root)
    had_existing_models = model_root.exists()
    if had_existing_models:
        os.replace(model_root, backup_root)
    try:
        os.replace(staging_root, model_root)
    except Exception:
        if had_existing_models and backup_root.exists() and not model_root.exists():
            os.replace(backup_root, model_root)
        raise
    else:
        _remove_tree(backup_root, model_root.parent)


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
