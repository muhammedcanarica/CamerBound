from __future__ import annotations

import logging
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from app.time_utils import as_utc


LOGGER = logging.getLogger(__name__)
SAFE_FILENAME_PART = re.compile(r"[^A-Z0-9]+")


class CaptureStorageError(RuntimeError):
    """Raised when a confirmed plate frame cannot be stored safely."""


class PlateCaptureService:
    def __init__(
        self,
        capture_root: Path,
        reference_root: Path,
        *,
        enabled: bool = True,
        max_width: int = 960,
        jpeg_quality: int = 60,
    ) -> None:
        self.capture_root = capture_root.resolve()
        self.reference_root = reference_root.resolve()
        if not self.capture_root.is_relative_to(self.reference_root):
            raise ValueError("Capture klasörü uygulama kökünün içinde olmalıdır.")
        self.enabled = enabled
        self.max_width = max_width
        self.jpeg_quality = jpeg_quality

    def save_capture(
        self,
        frame: object,
        plate: str,
        direction: object,
        captured_at: datetime,
        record_id: int,
    ) -> str | None:
        if not self.enabled:
            return None
        if (
            not isinstance(frame, np.ndarray)
            or frame.ndim not in (2, 3)
            or frame.size == 0
        ):
            raise CaptureStorageError("Geçerli bir OpenCV frame'i verilmedi.")

        captured_at = as_utc(captured_at)
        safe_plate = self._safe_filename_part(plate, "UNKNOWN")
        direction_value = getattr(direction, "value", str(direction))
        safe_direction = self._safe_filename_part(direction_value, "UNKNOWN")
        filename = (
            f"{safe_plate}_{captured_at.strftime('%Y%m%d_%H%M%S')}_"
            f"{safe_direction}_{record_id}.jpg"
        )
        directory = self.capture_root / captured_at.strftime("%Y") / captured_at.strftime("%m")
        target = (directory / filename).resolve()
        if not target.is_relative_to(self.capture_root):
            raise CaptureStorageError("Capture hedefi izin verilen klasörün dışında.")

        resized = self._resize(frame)
        success, encoded = cv2.imencode(
            ".jpg",
            resized,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not success:
            raise CaptureStorageError("JPEG encode başarısız oldu.")

        temporary_path: Path | None = None
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "wb", dir=directory, delete=False, suffix=".tmp"
            ) as handle:
                handle.write(encoded.tobytes())
                handle.flush()
                os.fsync(handle.fileno())
                temporary_path = Path(handle.name)
            os.replace(temporary_path, target)
        except OSError as exc:
            raise CaptureStorageError(f"JPEG dosyası yazılamadı: {exc}") from exc
        finally:
            if temporary_path is not None and temporary_path.exists():
                try:
                    temporary_path.unlink()
                except OSError:
                    LOGGER.warning("Temporary capture file could not be removed: %s", temporary_path)

        return target.relative_to(self.reference_root).as_posix()

    def resolve_reference(self, image_path: str | None) -> Path | None:
        if not image_path:
            return None
        reference = Path(image_path)
        if reference.is_absolute():
            return None
        resolved = (self.reference_root / reference).resolve()
        if not resolved.is_relative_to(self.capture_root):
            return None
        return resolved

    def delete_captures(self, image_paths: Iterable[str | None]) -> None:
        for image_path in image_paths:
            resolved = self.resolve_reference(image_path)
            if resolved is None:
                if image_path:
                    LOGGER.warning("Unsafe capture path was not deleted: %s", image_path)
                continue
            try:
                resolved.unlink(missing_ok=True)
            except OSError:
                LOGGER.exception("Plate capture could not be deleted: %s", resolved)

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if width <= self.max_width:
            return frame
        scale = self.max_width / width
        return cv2.resize(
            frame,
            (self.max_width, max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def _safe_filename_part(value: object, fallback: str) -> str:
        cleaned = SAFE_FILENAME_PART.sub("", str(value).upper())
        return cleaned or fallback
