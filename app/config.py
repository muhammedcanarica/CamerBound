from __future__ import annotations

import json
import sys
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

    return AppConfig(
        database_path=database_path.resolve(),
        duplicate_cooldown_seconds=recognition.duplicate_cooldown_seconds,
        plate_recognition=recognition,
        settings_path=resolved_settings_path.resolve(),
    )


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
        entry_roi=_parse_roi(roi_settings.get("ENTRY"), "ENTRY", warnings),
        exit_roi=_parse_roi(roi_settings.get("EXIT"), "EXIT", warnings),
        model_root=(root / "models" / "ocr").resolve(),
        warnings=tuple(warnings),
    )


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
    if (
        x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > 1
        or y + height > 1
    ):
        warnings.append(f"{direction} ROI frame dışında; varsayılan ROI kullanıldı.")
        return DEFAULT_ROI
    return NormalizedRoi(x=x, y=y, width=width, height=height)
