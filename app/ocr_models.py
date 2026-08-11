from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path


LOGGER = logging.getLogger(__name__)
REQUIRED_MODEL_FILES = ("inference.onnx", "inference.yml")


class OcrModelError(RuntimeError):
    """Base error for missing or unusable OCR model artifacts."""


class OcrModelNotFound(OcrModelError):
    """Raised when required OCR model artifacts are absent."""


class OcrModelInvalid(OcrModelError):
    """Raised when model artifacts exist but cannot be loaded."""


@dataclass(frozen=True, slots=True)
class ModelCheck:
    name: str
    directory: Path
    ok: bool
    message: str
    providers: tuple[str, ...] = ()


def validate_model_directory(
    directory: Path,
    name: str,
    *,
    load_onnx: bool = False,
) -> ModelCheck:
    directory = directory.resolve()
    if not directory.is_dir():
        raise OcrModelNotFound(f"{name} model klasörü bulunamadı: {directory}")

    missing = [filename for filename in REQUIRED_MODEL_FILES if not (directory / filename).is_file()]
    if missing:
        raise OcrModelNotFound(
            f"{name} modeli eksik: {', '.join(missing)} ({directory})"
        )

    config_path = directory / "inference.yml"
    try:
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise OcrModelInvalid(f"{name} inference.yml okunamadı: {exc}") from exc
    if not isinstance(config, dict) or not isinstance(config.get("Global"), dict):
        raise OcrModelInvalid(
            f"{name} inference.yml geçersiz: 'Global' bölümü bulunamadı."
        )

    providers: tuple[str, ...] = ()
    if load_onnx:
        try:
            import onnxruntime as ort

            available = tuple(ort.get_available_providers())
            if "CPUExecutionProvider" not in available:
                raise RuntimeError(
                    f"CPUExecutionProvider yok; kullanılabilir provider'lar: {available}"
                )
            session = ort.InferenceSession(
                str(directory / "inference.onnx"),
                providers=["CPUExecutionProvider"],
            )
            if not session.get_inputs() or not session.get_outputs():
                raise RuntimeError("model giriş veya çıkış tanımlamıyor")
            providers = tuple(session.get_providers())
        except Exception as exc:
            raise OcrModelInvalid(f"{name} ONNX modeli yüklenemedi: {exc}") from exc

    return ModelCheck(
        name=name,
        directory=directory,
        ok=True,
        message="dosyalar ve yapı geçerli" + (", ONNX Runtime yükledi" if load_onnx else ""),
        providers=providers,
    )


def validate_ocr_models(model_root: Path, *, load_onnx: bool = False) -> tuple[ModelCheck, ...]:
    checks: list[ModelCheck] = []
    for folder, label in (("detection", "Detection"), ("recognition", "Recognition")):
        try:
            checks.append(
                validate_model_directory(model_root / folder, label, load_onnx=load_onnx)
            )
        except OcrModelError as exc:
            LOGGER.error("OCR model validation failed for %s: %s", label, exc)
            raise
    return tuple(checks)


def collect_model_diagnostics(model_root: Path, *, load_onnx: bool = False) -> tuple[ModelCheck, ...]:
    checks: list[ModelCheck] = []
    for folder, label in (("detection", "Detection"), ("recognition", "Recognition")):
        directory = model_root / folder
        try:
            checks.append(validate_model_directory(directory, label, load_onnx=load_onnx))
        except OcrModelError as exc:
            checks.append(ModelCheck(label, directory.resolve(), False, str(exc)))
    return tuple(checks)
