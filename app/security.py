from __future__ import annotations

import re
from urllib.parse import urlsplit, urlunsplit


SUPPORTED_CAMERA_URL_SCHEMES = {"rtsp", "http", "https"}
CAMERA_URL_PATTERN = re.compile(r"(?i)\b(?:rtsp|https?)://[^\s'\"]+")


def sanitize_camera_source_for_log(source: object) -> str:
    """Return useful camera-source context without credentials or query secrets."""
    value = str(source).strip()
    try:
        parsed = urlsplit(value)
    except ValueError:
        return "<invalid-camera-source>"
    if parsed.scheme.lower() not in SUPPORTED_CAMERA_URL_SCHEMES or not parsed.hostname:
        return value

    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    try:
        port = parsed.port
    except ValueError:
        port = None
    netloc = f"{host}:{port}" if port is not None else host
    return urlunsplit((parsed.scheme.lower(), netloc, parsed.path or "/", "", ""))


def sanitize_text_for_log(message: object) -> str:
    return CAMERA_URL_PATTERN.sub(
        lambda match: sanitize_camera_source_for_log(match.group(0)),
        str(message),
    )
