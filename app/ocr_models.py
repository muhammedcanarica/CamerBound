from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


ONNX_REQUIRED_MODEL_FILES = ("inference.onnx", "inference.yml")
PADDLE_REQUIRED_MODEL_FILES = ("inference.pdiparams", "inference.yml")


class OcrBackend(StrEnum):
    AUTO = "auto"
    ONNX = "onnx"
    PADDLE = "paddle"


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


@dataclass(frozen=True, slots=True)
class BackendSelection:
    backend: OcrBackend
    model_root: Path

    @property
    def label(self) -> str:
        return "ONNX Runtime" if self.backend is OcrBackend.ONNX else "Paddle CPU"


def normalize_ocr_backend(value: str | OcrBackend) -> OcrBackend:
    try:
        return value if isinstance(value, OcrBackend) else OcrBackend(str(value).lower())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Geçersiz OCR backend: {value}") from exc


def backend_model_root(model_root: Path, backend: str | OcrBackend) -> Path:
    """Resolve the new backend layout while retaining legacy ONNX directories."""
    model_root = model_root.resolve()
    normalized = normalize_ocr_backend(backend)
    if normalized is OcrBackend.AUTO:
        raise ValueError("AUTO backend için tek bir model klasörü çözümlenemez.")
    nested = model_root / normalized.value
    if normalized is OcrBackend.ONNX:
        nested_has_artifacts = any(
            (nested / folder / filename).is_file()
            for folder in ("detection", "recognition")
            for filename in ONNX_REQUIRED_MODEL_FILES
        )
        legacy_directories = (model_root / "detection", model_root / "recognition")
        if not nested_has_artifacts and any(
            directory.exists() for directory in legacy_directories
        ):
            return model_root
    return nested


def validate_model_directory(
    directory: Path,
    name: str,
    *,
    backend: str | OcrBackend = OcrBackend.ONNX,
    load_onnx: bool = False,
) -> ModelCheck:
    normalized_backend = normalize_ocr_backend(backend)
    if normalized_backend is OcrBackend.AUTO:
        raise ValueError("Model klasörü doğrulamasında backend AUTO olamaz.")
    directory = directory.resolve()
    if not directory.is_dir():
        raise OcrModelNotFound(f"{name} model klasörü bulunamadı: {directory}")

    if normalized_backend is OcrBackend.ONNX:
        missing = [
            filename
            for filename in ONNX_REQUIRED_MODEL_FILES
            if not (directory / filename).is_file()
        ]
    else:
        missing = [
            filename
            for filename in PADDLE_REQUIRED_MODEL_FILES
            if not (directory / filename).is_file()
        ]
        if not (directory / "inference.json").is_file() and not (
            directory / "inference.pdmodel"
        ).is_file():
            missing.insert(0, "inference.json veya inference.pdmodel")
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
        if normalized_backend is not OcrBackend.ONNX:
            raise ValueError("load_onnx yalnızca ONNX modelleri için kullanılabilir.")
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

    runtime_message = ", ONNX Runtime yükledi" if load_onnx else ""
    return ModelCheck(
        name=name,
        directory=directory,
        ok=True,
        message="dosyalar ve yapı geçerli" + runtime_message,
        providers=providers,
    )


def validate_ocr_models(
    model_root: Path,
    *,
    backend: str | OcrBackend = OcrBackend.ONNX,
    load_onnx: bool = False,
) -> tuple[ModelCheck, ...]:
    normalized_backend = normalize_ocr_backend(backend)
    resolved_root = backend_model_root(model_root, normalized_backend)
    checks: list[ModelCheck] = []
    for folder, label in (("detection", "Detection"), ("recognition", "Recognition")):
        try:
            checks.append(
                validate_model_directory(
                    resolved_root / folder,
                    label,
                    backend=normalized_backend,
                    load_onnx=load_onnx,
                )
            )
        except OcrModelError as exc:
            raise
    return tuple(checks)


def collect_model_diagnostics(
    model_root: Path,
    *,
    backend: str | OcrBackend = OcrBackend.ONNX,
    load_onnx: bool = False,
) -> tuple[ModelCheck, ...]:
    normalized_backend = normalize_ocr_backend(backend)
    resolved_root = backend_model_root(model_root, normalized_backend)
    checks: list[ModelCheck] = []
    for folder, label in (("detection", "Detection"), ("recognition", "Recognition")):
        directory = resolved_root / folder
        try:
            checks.append(
                validate_model_directory(
                    directory,
                    label,
                    backend=normalized_backend,
                    load_onnx=load_onnx,
                )
            )
        except OcrModelError as exc:
            checks.append(ModelCheck(label, directory.resolve(), False, str(exc)))
    return tuple(checks)


def select_ocr_backend(
    model_root: Path,
    requested: str | OcrBackend = OcrBackend.AUTO,
    *,
    load_onnx: bool = True,
) -> BackendSelection:
    normalized = normalize_ocr_backend(requested)
    candidates = (
        (OcrBackend.ONNX, OcrBackend.PADDLE)
        if normalized is OcrBackend.AUTO
        else (normalized,)
    )
    errors: list[str] = []
    for candidate in candidates:
        try:
            validate_ocr_models(
                model_root,
                backend=candidate,
                load_onnx=load_onnx and candidate is OcrBackend.ONNX,
            )
        except OcrModelError as exc:
            if normalized is not OcrBackend.AUTO:
                raise
            errors.append(f"{candidate.value.upper()}: {exc}")
            continue
        return BackendSelection(candidate, backend_model_root(model_root, candidate))
    raise OcrModelNotFound("Kullanılabilir OCR modeli bulunamadı. " + " | ".join(errors))
