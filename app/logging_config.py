from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.config import application_root
from app.security import sanitize_text_for_log


class CredentialSafeFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return sanitize_text_for_log(super().format(record))


def configure_logging(log_path: Path | None = None) -> Path:
    resolved = (log_path or application_root() / "data" / "logs" / "app.log").resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    if not any(
        isinstance(handler, RotatingFileHandler)
        and Path(getattr(handler, "baseFilename", "")) == resolved
        for handler in root_logger.handlers
    ):
        handler = RotatingFileHandler(
            resolved,
            maxBytes=2 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(
            CredentialSafeFormatter(
                "%(asctime)s %(levelname)s %(name)s: %(message)s"
            )
        )
        root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    return resolved
