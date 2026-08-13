from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "vehicle-license-plate-detection-barrier-0123"
TARGET_DIR = PROJECT_ROOT / "models" / "plate_detector" / MODEL_NAME
WORK_DIR = PROJECT_ROOT / "models" / "plate_detector" / ".omz-work"
CONVERTER_ENV_DIR = PROJECT_ROOT / ".tools" / "openvino_model_converter"
CONVERTER_REQUIREMENTS = PROJECT_ROOT / "requirements-model-converter.txt"
SUPPORTED_CONVERTER_PYTHON = {(3, 9), (3, 10), (3, 11), (3, 12)}


class PreparationError(RuntimeError):
    """A user-actionable plate detector preparation failure."""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Open Model Zoo plate detector modelini izole bir converter ortamında "
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
        help="Ağ, venv veya pip işlemi yapmadan model.xml/model.bin dosyalarını doğrular.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Hazır model bulunsa bile download cache üzerinden dönüşümü yeniden çalıştırır.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Hata durumunda ayrıntılı Python traceback gösterir.",
    )
    args = parser.parse_args(argv)

    if args.check_only:
        return _check_target()

    if not args.force and _target_ready():
        print(f"Plate detector zaten hazır: {TARGET_DIR}", flush=True)
        return 0

    try:
        _prepare(args.omz_tools_dir, force=args.force)
    except PreparationError as exc:
        if args.debug:
            raise
        print(f"Hata: {exc}", file=sys.stderr)
        return 1
    return _check_target()


def _prepare(omz_tools_dir: Path | None, *, force: bool) -> None:
    print("[1/5] Converter environment kontrol ediliyor", flush=True)
    converter_python = _ensure_converter_environment()

    print("[2/5] Conversion dependencies kontrol ediliyor", flush=True)
    _ensure_converter_dependencies(converter_python)
    model_optimizer = _resolve_model_optimizer(converter_python)
    downloader, converter = _resolve_tools(
        omz_tools_dir,
        converter_python=converter_python,
    )

    download_dir = WORK_DIR / "download"
    converted_dir = WORK_DIR / "converted"
    download_dir.mkdir(parents=True, exist_ok=True)
    converted_dir.mkdir(parents=True, exist_ok=True)

    print("[3/5] Plate detector indiriliyor", flush=True)
    if not force and _download_cache_ready(download_dir):
        print("Tamamlanmış model download cache'i kullanılıyor.", flush=True)
    else:
        _run_command(
            [
                *downloader,
                "--name",
                MODEL_NAME,
                "--output_dir",
                str(download_dir),
            ],
            "Plate detector indirilemedi",
        )

    print("[4/5] OpenVINO IR model oluşturuluyor", flush=True)
    converted_pair = None if force else _find_converted_pair(converted_dir)
    if converted_pair is None:
        _run_command(
            [
                *converter,
                "--name",
                MODEL_NAME,
                "--download_dir",
                str(download_dir),
                "--output_dir",
                str(converted_dir),
                "--precisions",
                "FP32",
                "--python",
                str(converter_python),
                "--mo",
                str(model_optimizer),
            ],
            "Model conversion başarısız",
        )
        converted_pair = _find_converted_pair(converted_dir)
    else:
        print("Download cache içindeki dönüştürülmüş FP32 model kullanılıyor.", flush=True)

    if converted_pair is None:
        raise PreparationError(
            f"Dönüştürülmüş model.xml/model.bin bulunamadı: {converted_dir}"
        )
    source_xml, source_bin = converted_pair
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_xml, TARGET_DIR / "model.xml")
    shutil.copy2(source_bin, TARGET_DIR / "model.bin")

    print("[5/5] Model doğrulanıyor", flush=True)
    if not _target_ready():
        raise PreparationError("Model kopyalandı ancak model.xml/model.bin doğrulanamadı")


def _ensure_converter_environment() -> Path:
    _validate_converter_python_version()
    converter_python = _converter_python(CONVERTER_ENV_DIR)
    if converter_python.is_file():
        print(f"Converter environment mevcut: {CONVERTER_ENV_DIR}", flush=True)
        return converter_python

    try:
        CONVERTER_ENV_DIR.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PreparationError(
            f"Converter environment dizini oluşturulamadı: {CONVERTER_ENV_DIR}"
        ) from exc
    _run_command(
        [sys.executable, "-m", "venv", str(CONVERTER_ENV_DIR)],
        "Converter environment oluşturulamadı",
    )
    if not converter_python.is_file():
        raise PreparationError(
            f"Converter Python oluşturulamadı: {converter_python}"
        )
    return converter_python


def _validate_converter_python_version() -> None:
    version = (sys.version_info.major, sys.version_info.minor)
    if version not in SUPPORTED_CONVERTER_PYTHON:
        supported = ", ".join(f"{major}.{minor}" for major, minor in sorted(SUPPORTED_CONVERTER_PYTHON))
        raise PreparationError(
            f"Converter için Python {supported} gerekir; çalışan Python: "
            f"{sys.version_info.major}.{sys.version_info.minor}"
        )


def _converter_python(environment_dir: Path) -> Path:
    if os.name == "nt":
        return environment_dir / "Scripts" / "python.exe"
    return environment_dir / "bin" / "python"


def _ensure_converter_dependencies(converter_python: Path) -> None:
    if _converter_dependencies_ready(converter_python):
        print(
            "Converter dependency'leri hazır (OpenVINO 2024.6.0, TensorFlow 2.18.0).",
            flush=True,
        )
        return
    if not CONVERTER_REQUIREMENTS.is_file():
        raise PreparationError(
            f"Converter requirements dosyası bulunamadı: {CONVERTER_REQUIREMENTS}"
        )
    _run_command(
        [
            str(converter_python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "-r",
            str(CONVERTER_REQUIREMENTS),
        ],
        "Converter dependency'leri kurulamadı",
    )
    if not _converter_dependencies_ready(converter_python):
        raise PreparationError(
            "Converter dependency doğrulaması başarısız: TensorFlow veya Model Optimizer yüklenemedi"
        )


def _converter_dependencies_ready(converter_python: Path) -> bool:
    probe = (
        "import importlib.metadata as metadata; "
        "import tensorflow as tensorflow; "
        "from openvino.tools import mo; "
        "assert metadata.version('openvino-dev') == '2024.6.0'; "
        "assert tensorflow.__version__ == '2.18.0'; "
        "assert metadata.version('fastjsonschema') == '2.21.2'; "
        "print('TensorFlow ' + tensorflow.__version__ + '; MO OK')"
    )
    try:
        result = subprocess.run(
            [str(converter_python), "-c", probe],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    if result.returncode == 0:
        output = result.stdout.strip()
        if output:
            print(output, flush=True)
        return True
    return False


def _resolve_tools(
    tools_dir: Path | None,
    *,
    converter_python: Path,
) -> tuple[list[str], list[str]]:
    if tools_dir is not None:
        resolved = tools_dir.resolve()
        downloader = resolved / "downloader.py"
        converter = resolved / "converter.py"
        if not downloader.is_file() or not converter.is_file():
            raise PreparationError(
                "OMZ tools bulunamadı: --omz-tools-dir içinde downloader.py ve "
                "converter.py bulunmalıdır"
            )
        return (
            [str(converter_python), str(downloader)],
            [str(converter_python), str(converter)],
        )

    scripts_dir = converter_python.parent
    executable_suffix = ".exe" if os.name == "nt" else ""
    downloader = scripts_dir / f"omz_downloader{executable_suffix}"
    converter = scripts_dir / f"omz_converter{executable_suffix}"
    if not downloader.is_file() or not converter.is_file():
        raise PreparationError(
            "OMZ tools bulunamadı. Converter environment kurulumu eksik veya "
            "--omz-tools-dir geçerli bir Open Model Zoo tools/model_tools dizinine işaret etmiyor"
        )
    return [str(downloader)], [str(converter)]


def _resolve_model_optimizer(converter_python: Path) -> Path:
    probe = "from openvino.tools.mo import mo; print(mo.__file__)"
    try:
        result = subprocess.run(
            [str(converter_python), "-c", probe],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise PreparationError("Model Optimizer yolu belirlenemedi") from exc
    if result.returncode != 0:
        raise PreparationError("Model Optimizer yolu belirlenemedi")
    output_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not output_lines:
        raise PreparationError("Model Optimizer yolu belirlenemedi")
    entrypoint = Path(output_lines[-1]).resolve()
    if not entrypoint.is_file():
        raise PreparationError(f"Model Optimizer entry point bulunamadı: {entrypoint}")
    return entrypoint


def _run_command(command: Sequence[str], failure_message: str) -> None:
    try:
        subprocess.run(list(command), check=True, cwd=PROJECT_ROOT)
    except subprocess.CalledProcessError as exc:
        raise PreparationError(f"{failure_message} (çıkış kodu: {exc.returncode})") from exc
    except OSError as exc:
        raise PreparationError(f"{failure_message}: {exc}") from exc


def _find_converted_pair(root: Path) -> tuple[Path, Path] | None:
    try:
        matches = [
            path
            for path in root.rglob("*.xml")
            if MODEL_NAME in path.name or MODEL_NAME in str(path.parent)
        ]
    except OSError:
        return None
    fp32_matches = [path for path in matches if "FP32" in path.parts]
    for xml_path in sorted(fp32_matches or matches):
        bin_path = xml_path.with_suffix(".bin")
        if _nonempty_file(xml_path) and _nonempty_file(bin_path):
            return xml_path, bin_path
    return None


def _download_cache_ready(download_dir: Path) -> bool:
    model_dir = download_dir / "public" / MODEL_NAME / "model"
    return all(
        _nonempty_file(model_dir / filename)
        for filename in ("model.pb.frozen", "model.tfmo.json")
    )


def _target_ready() -> bool:
    return all(
        _nonempty_file(path)
        for path in (TARGET_DIR / "model.xml", TARGET_DIR / "model.bin")
    )


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _check_target() -> int:
    missing = [
        path
        for path in (TARGET_DIR / "model.xml", TARGET_DIR / "model.bin")
        if not _nonempty_file(path)
    ]
    if missing:
        print("Eksik plate detector dosyaları:", flush=True)
        for path in missing:
            print(f"  {path}", flush=True)
        return 1
    print(f"Plate detector model dosyaları doğrulandı: {TARGET_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
