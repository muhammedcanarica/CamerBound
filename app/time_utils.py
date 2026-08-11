from __future__ import annotations

from datetime import datetime, timezone


UTC = timezone.utc


def as_utc(value: datetime) -> datetime:
    """Return an aware UTC datetime; legacy naive values are treated as UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def utc_now() -> datetime:
    return datetime.now(UTC)


def parse_utc_timestamp(value: str | datetime) -> datetime:
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    return as_utc(parsed)


def to_utc_storage(value: str | datetime) -> str:
    return parse_utc_timestamp(value).isoformat(timespec="seconds")


def normalize_utc_timestamp(value: str) -> str:
    """Normalize valid legacy values without making malformed rows unreadable."""
    try:
        return to_utc_storage(value)
    except (TypeError, ValueError):
        return value


def to_local_datetime(value: str | datetime) -> datetime:
    return parse_utc_timestamp(value).astimezone()
