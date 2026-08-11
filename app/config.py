from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(RuntimeError):
    """Raised when application settings are missing or invalid."""


def application_root() -> Path:
    """Return a stable base directory independent from the current directory."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class AppConfig:
    database_path: Path
    duplicate_cooldown_seconds: int
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
    cooldown = plate_settings.get("duplicate_cooldown_seconds", 10)
    if not isinstance(cooldown, int) or cooldown < 0:
        raise ConfigError("duplicate_cooldown_seconds sıfır veya pozitif olmalıdır.")

    return AppConfig(
        database_path=database_path.resolve(),
        duplicate_cooldown_seconds=cooldown,
        settings_path=resolved_settings_path.resolve(),
    )
