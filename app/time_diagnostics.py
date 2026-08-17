from __future__ import annotations

import locale
import logging
import platform
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from app.time_utils import as_utc, to_local_datetime, utc_now


LOGGER = logging.getLogger(__name__)
DEFAULT_TIMEOUT_SECONDS = 4.0


class TimeSyncStatus(str, Enum):
    SYNCED = "SYNCED"
    WARNING = "WARNING"
    UNAVAILABLE = "UNAVAILABLE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class TimeDiagnosticsResult:
    local_time: datetime
    utc_time: datetime
    timezone_name: str
    utc_offset: str
    windows_time_available: bool
    time_source: str | None
    sync_status: TimeSyncStatus
    status_message: str
    raw_status: str | None
    checked_at: datetime


def format_utc_offset(offset: timedelta | None) -> str:
    if offset is None:
        return "-"
    total_seconds = int(offset.total_seconds())
    sign = "+" if total_seconds >= 0 else "-"
    total_minutes = abs(total_seconds) // 60
    hours, minutes = divmod(total_minutes, 60)
    return f"{sign}{hours:02d}:{minutes:02d}"


class TimeDiagnosticsService:
    """Read Windows Time state without changing the system clock or service."""

    _FALLBACK_SOURCES = {
        "local cmos clock",
        "free-running system clock",
        "free running system clock",
        "yerel cmos saati",
        "serbest çalışan sistem saati",
    }

    def __init__(
        self,
        *,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        system_name_provider: Callable[[], str] = platform.system,
        now_provider: Callable[[], datetime] = utc_now,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._runner = runner
        self._system_name_provider = system_name_provider
        self._now_provider = now_provider
        self.timeout_seconds = timeout_seconds

    def check(self) -> TimeDiagnosticsResult:
        utc_time = as_utc(self._now_provider())
        local_time = to_local_datetime(utc_time)
        common = {
            "local_time": local_time,
            "utc_time": utc_time,
            "timezone_name": local_time.tzname() or "Bilinmiyor",
            "utc_offset": format_utc_offset(local_time.utcoffset()),
            "checked_at": utc_time,
        }

        if self._system_name_provider() != "Windows":
            return TimeDiagnosticsResult(
                **common,
                windows_time_available=False,
                time_source=None,
                sync_status=TimeSyncStatus.UNAVAILABLE,
                status_message=(
                    "Windows Time tanılaması yalnızca Windows'ta kullanılabilir."
                ),
                raw_status=None,
            )

        try:
            source_process = self._query("/source")
            status_process = self._query("/status")
        except Exception as exc:
            LOGGER.warning(
                "Windows Time diagnostic query failed (%s)",
                type(exc).__name__,
            )
            return self._unavailable_result(common)

        source = self._first_output_line(source_process.stdout)
        raw_status = self._clean_output(status_process.stdout)
        if source_process.returncode != 0 or status_process.returncode != 0:
            LOGGER.warning("Windows Time diagnostic query failed (non-zero exit code)")
            return self._unavailable_result(common, time_source=source)

        if not source:
            return TimeDiagnosticsResult(
                **common,
                windows_time_available=True,
                time_source=None,
                sync_status=TimeSyncStatus.UNKNOWN,
                status_message=(
                    "Windows Time çalışıyor ancak zaman kaynağı belirlenemedi."
                ),
                raw_status=raw_status,
            )

        if source.casefold() in self._FALLBACK_SOURCES:
            return TimeDiagnosticsResult(
                **common,
                windows_time_available=True,
                time_source=source,
                sync_status=TimeSyncStatus.WARNING,
                status_message=(
                    "Harici zaman kaynağı kullanılmıyor.\n"
                    "Plaka kayıt zamanları sistem saatine bağlıdır."
                ),
                raw_status=raw_status,
            )

        return TimeDiagnosticsResult(
            **common,
            windows_time_available=True,
            time_source=source,
            sync_status=TimeSyncStatus.SYNCED,
            status_message="Saat senkronizasyonu aktif.",
            raw_status=raw_status,
        )

    def _query(self, query: str) -> subprocess.CompletedProcess[str]:
        return self._runner(
            ["w32tm", "/query", query],
            capture_output=True,
            text=True,
            encoding=locale.getpreferredencoding(False) or "utf-8",
            errors="replace",
            timeout=self.timeout_seconds,
            check=False,
            shell=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    @staticmethod
    def _clean_output(output: str | None) -> str | None:
        if not output:
            return None
        cleaned = output.replace("\x00", "").lstrip("\ufeff").strip()
        return cleaned or None

    @classmethod
    def _first_output_line(cls, output: str | None) -> str | None:
        cleaned = cls._clean_output(output)
        if cleaned is None:
            return None
        return next((line.strip() for line in cleaned.splitlines() if line.strip()), None)

    @staticmethod
    def _unavailable_result(
        common: dict[str, object],
        *,
        time_source: str | None = None,
    ) -> TimeDiagnosticsResult:
        return TimeDiagnosticsResult(
            **common,
            windows_time_available=False,
            time_source=time_source,
            sync_status=TimeSyncStatus.UNAVAILABLE,
            status_message=(
                "Windows Time durumu alınamadı.\n"
                "Plaka kaydı devam ediyor ancak zaman senkronizasyonu doğrulanamadı."
            ),
            raw_status=None,
        )
