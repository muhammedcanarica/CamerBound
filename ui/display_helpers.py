from __future__ import annotations

import re
from datetime import datetime

from app.time_utils import to_local_datetime


TURKISH_PLATE_PARTS = re.compile(r"^(\d{2})([A-Z]{1,3})(\d{2,4})$")


def display_timestamp(value: str | datetime) -> str:
    try:
        local_time = to_local_datetime(value)
        return local_time.strftime("%d.%m.%Y %H:%M:%S")
    except (TypeError, ValueError):
        return str(value)


def display_plate(value: str) -> str:
    match = TURKISH_PLATE_PARTS.fullmatch(value)
    return " ".join(match.groups()) if match is not None else value
