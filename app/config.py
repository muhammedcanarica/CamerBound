from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(RuntimeError):
    """Raised when application settings are missing or invalid."""


@dataclass(frozen=True, slots=True)
class NormalizedRoi:
    x: float
    y: float
    width: float
    height: float


DEFAULT_ROI = NormalizedRoi(x=0.10, y=0.35, width=0.80, height=0.55)
DEFAULT_RECORD_RETENTION_DAYS = 90
SUPPORTED_RECORD_RETENTION_DAYS = (30, 90, 180, 0)
DEFAULT_CAPTURE_MAX_WIDTH = 960
DEFAULT_CAPTURE_JPEG_QUALITY = 60


@dataclass(frozen=True, slots=True)
class PlateCaptureConfig:
    enabled: bool
    max_width: int
    jpeg_quality: int
    capture_root: Path
    reference_root: Path
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PlateRecognitionConfig:
    recognition_interval_ms: int
    min_confidence: float
    confirmations_required: int
    confirmation_window_seconds: float
    duplicate_cooldown_seconds: int
    entry_roi: NormalizedRoi
    exit_roi: NormalizedRoi
    model_root: Path
    record_retention_days: int = DEFAULT_RECORD_RETENTION_DAYS
    ocr_backend: str = "auto"
    warnings: tuple[str, ...] = ()

    def roi_for(self, direction: object) -> NormalizedRoi:
        value = getattr(direction, "value", str(direction))
        return self.entry_roi if value == "ENTRY" else self.exit_roi


def application_root() -> Path:
    """Return a stable base directory independent from the current directory."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class AppConfig:
    database_path: Path
    duplicate_cooldown_seconds: int
    plate_recognition: PlateRecognitionConfig
    plate_capture: PlateCaptureConfig
    settings_path: Path


def load_config(settings_path: Path | None = None) -> AppConfig:
    root = application_root()
    resolved_settings_path = settings_path or root / "config" / "settings.json"

    try:
        raw_settings: dict[str, Any] = json.loads(
            resolved_settings_path.read_text(encoding="utf-8")
        )
    except FileNotFoundError as exc:
        raise ConfigError(f"Ayar dosyası bulunamadı: {resolved_settings_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Ayar dosyası geçerli JSON değil: {exc}") from exc

    database_value = raw_settings.get("database_path")
    if not isinstance(database_value, str) or not database_value.strip():
        raise ConfigError("'database_path' dolu bir metin olmalıdır.")

    database_path = Path(database_value)
    if not database_path.is_absolute():
        database_path = root / database_path

    plate_settings = raw_settings.get("plate_detection", {})
    if not isinstance(plate_settings, dict):
        plate_settings = {}
    recognition = _load_plate_recognition(plate_settings, root)
    capture_settings = raw_settings.get("plate_capture", {})
    capture = _load_plate_capture(capture_settings, root)

    return AppConfig(
        database_path=database_path.resolve(),
        duplicate_cooldown_seconds=recognition.duplicate_cooldown_seconds,
        plate_recognition=recognition,
        plate_capture=capture,
        settings_path=resolved_settings_path.resolve(),
    )


def update_plate_roi(
    settings_path: Path,
    direction: object,
    roi: NormalizedRoi,
) -> PlateRecognitionConfig:
    """Atomically update one ROI while preserving unknown settings fields."""
    if not _is_valid_roi(roi):
        raise ConfigError("ROI 0-1 aralığında ve frame sınırları içinde olmalıdır.")
    direction_value = getattr(direction, "value", str(direction))
    if direction_value not in {"ENTRY", "EXIT"}:
        raise ConfigError(f"Geçersiz kamera yönü: {direction_value}")

    settings_path = settings_path.resolve()
    raw = _read_settings_for_update(settings_path)

    plate_detection = raw.setdefault("plate_detection", {})
    if not isinstance(plate_detection, dict):
        plate_detection = {}
        raw["plate_detection"] = plate_detection
    roi_settings = plate_detection.setdefault("roi", {})
    if not isinstance(roi_settings, dict):
        roi_settings = {}
        plate_detection["roi"] = roi_settings
    roi_settings[direction_value] = {
        "x": roi.x,
        "y": roi.y,
        "width": roi.width,
        "height": roi.height,
    }

    _write_settings_atomic(settings_path, raw)
    return load_config(settings_path).plate_recognition


def update_record_retention(
    settings_path: Path,
    retention_days: int,
) -> PlateRecognitionConfig:
    """Atomically update record retention while preserving unknown settings fields."""
    if (
        isinstance(retention_days, bool)
        or retention_days not in SUPPORTED_RECORD_RETENTION_DAYS
    ):
        raise ConfigError("Saklama süresi 30, 90, 180 veya 0 olmalıdır.")

    settings_path = settings_path.resolve()
    raw = _read_settings_for_update(settings_path)
    plate_detection = raw.setdefault("plate_detection", {})
    if not isinstance(plate_detection, dict):
        plate_detection = {}
        raw["plate_detection"] = plate_detection
    plate_detection["record_retention_days"] = retention_days

    _write_settings_atomic(settings_path, raw)
    return load_config(settings_path).plate_recognition


def _read_settings_for_update(settings_path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(settings_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        raise ConfigError(f"Ayar dosyası güncellenemedi: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("Ayar dosyasının kökü JSON nesnesi olmalıdır.")
    return raw


def _write_settings_atomic(settings_path: Path, raw: dict[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=settings_path.parent, delete=False, suffix=".tmp"
        ) as handle:
            json.dump(raw, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, settings_path)
    except OSError as exc:
        raise ConfigError(f"Ayar dosyası atomik olarak yazılamadı: {exc}") from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _load_plate_recognition(
    raw: dict[str, Any],
    root: Path,
) -> PlateRecognitionConfig:
    warnings: list[str] = []
    interval = _bounded_number(
        raw.get("recognition_interval_ms"), 500, 100, 60_000, int,
        "recognition_interval_ms", warnings,
    )
    confidence = _bounded_number(
        raw.get("min_confidence"), 0.65, 0.0, 1.0, float,
        "min_confidence", warnings,
    )
    confirmations = _bounded_number(
        raw.get("confirmations_required"), 2, 1, 20, int,
        "confirmations_required", warnings,
    )
    confirmation_window = _bounded_number(
        raw.get("confirmation_window_seconds"), 3.0, 0.1, 60.0, float,
        "confirmation_window_seconds", warnings,
    )
    cooldown = _bounded_number(
        raw.get("duplicate_cooldown_seconds"), 10, 0, 86_400, int,
        "duplicate_cooldown_seconds", warnings,
    )
    retention_days = _record_retention_days(
        raw.get("record_retention_days"), warnings
    )
    backend_value = raw.get("ocr_backend", "auto")
    if not isinstance(backend_value, str) or backend_value.lower() not in {
        "auto",
        "onnx",
        "paddle",
    }:
        warnings.append("ocr_backend geçersiz; varsayılan 'auto' kullanıldı.")
        backend_value = "auto"

    roi_settings = raw.get("roi", {})
    if not isinstance(roi_settings, dict):
        roi_settings = {}
        warnings.append("roi nesne olmalıdır; varsayılan ROI kullanıldı.")

    return PlateRecognitionConfig(
        recognition_interval_ms=interval,
        min_confidence=confidence,
        confirmations_required=confirmations,
        confirmation_window_seconds=confirmation_window,
        duplicate_cooldown_seconds=cooldown,
        record_retention_days=retention_days,
        entry_roi=_parse_roi(roi_settings.get("ENTRY"), "ENTRY", warnings),
        exit_roi=_parse_roi(roi_settings.get("EXIT"), "EXIT", warnings),
        model_root=(root / "models" / "ocr").resolve(),
        ocr_backend=backend_value.lower(),
        warnings=tuple(warnings),
    )


def _load_plate_capture(raw: object, root: Path) -> PlateCaptureConfig:
    warnings: list[str] = []
    if not isinstance(raw, dict):
        raw = {}
        warnings.append(
            "plate_capture nesne olmalıdır; varsayılan değerler kullanıldı."
        )

    enabled_value = raw.get("enabled", True)
    if not isinstance(enabled_value, bool):
        enabled_value = True
        warnings.append("plate_capture.enabled geçersiz; varsayılan true kullanıldı.")
    max_width = _bounded_number(
        raw.get("max_width"),
        DEFAULT_CAPTURE_MAX_WIDTH,
        320,
        3840,
        int,
        "plate_capture.max_width",
        warnings,
    )
    jpeg_quality = _bounded_number(
        raw.get("jpeg_quality"),
        DEFAULT_CAPTURE_JPEG_QUALITY,
        20,
        95,
        int,
        "plate_capture.jpeg_quality",
        warnings,
    )
    return PlateCaptureConfig(
        enabled=enabled_value,
        max_width=max_width,
        jpeg_quality=jpeg_quality,
        capture_root=(root / "data" / "captures").resolve(),
        reference_root=root.resolve(),
        warnings=tuple(warnings),
    )


def _record_retention_days(value: object, warnings: list[str]) -> int:
    if value is None:
        return DEFAULT_RECORD_RETENTION_DAYS
    if isinstance(value, bool) or not isinstance(value, int):
        warnings.append(
            "record_retention_days geçersiz; varsayılan 90 gün kullanıldı."
        )
        return DEFAULT_RECORD_RETENTION_DAYS
    if value not in SUPPORTED_RECORD_RETENTION_DAYS:
        warnings.append(
            "record_retention_days desteklenmiyor; varsayılan 90 gün kullanıldı."
        )
        return DEFAULT_RECORD_RETENTION_DAYS
    return value


def _bounded_number(
    value: object,
    default: int | float,
    minimum: float,
    maximum: float,
    result_type: type[int] | type[float],
    name: str,
    warnings: list[str],
) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        if value is not None:
            warnings.append(f"{name} geçersiz; varsayılan değer kullanıldı.")
        return default
    converted = result_type(value)
    if not minimum <= converted <= maximum:
        warnings.append(f"{name} sınır dışında; varsayılan değer kullanıldı.")
        return default
    return converted


def _parse_roi(
    value: object,
    direction: str,
    warnings: list[str],
) -> NormalizedRoi:
    if value is None:
        return DEFAULT_ROI
    if not isinstance(value, dict):
        warnings.append(f"{direction} ROI geçersiz; varsayılan ROI kullanıldı.")
        return DEFAULT_ROI
    try:
        numbers = [float(value[key]) for key in ("x", "y", "width", "height")]
    except (KeyError, TypeError, ValueError):
        warnings.append(f"{direction} ROI eksik/geçersiz; varsayılan ROI kullanıldı.")
        return DEFAULT_ROI
    x, y, width, height = numbers
    roi = NormalizedRoi(x=x, y=y, width=width, height=height)
    if not _is_valid_roi(roi):
        warnings.append(f"{direction} ROI frame dışında; varsayılan ROI kullanıldı.")
        return DEFAULT_ROI
    return roi


def _is_valid_roi(roi: NormalizedRoi) -> bool:
    return (
        roi.x >= 0
        and roi.y >= 0
        and roi.width > 0
        and roi.height > 0
        and roi.x + roi.width <= 1
        and roi.y + roi.height <= 1
    )
