from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
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
DEFAULT_DUPLICATE_COOLDOWN_SECONDS = 120
SUPPORTED_RECORD_RETENTION_DAYS = (30, 90, 180, 0)
DEFAULT_CAPTURE_MAX_WIDTH = 960
DEFAULT_CAPTURE_JPEG_QUALITY = 60
PLATE_DETECTOR_MODEL_NAME = "vehicle-license-plate-detection-barrier-0123"
DEFAULT_PLATE_DETECTOR_MIN_CONFIDENCE = 0.50
DEFAULT_PLATE_DETECTOR_CROP_PADDING_RATIO = 0.15
DEFAULT_MAX_PLATE_CANDIDATES_PER_FRAME = 2
DEFAULT_ZERO_DETECTION_ROI_FALLBACK_ENABLED = True
DEFAULT_ZERO_DETECTION_ROI_FALLBACK_INTERVAL_MS = 750
DEFAULT_MAX_PENDING_OCR_JOBS_PER_CAMERA = 3
DEFAULT_OCR_JOB_MAX_AGE_MS = 2_500
DEFAULT_DEBUG_DETECTION_OVERLAY_TTL_MS = 500
DEFAULT_PRE_DETECTION_BUFFER_DURATION_MS = 2_000
DEFAULT_PRE_DETECTION_BUFFER_MAX_FRAMES_PER_CAMERA = 20
DEFAULT_MOTION_PRE_ROLL_MS = 500
DEFAULT_MOTION_POST_ROLL_MS = 700
DEFAULT_MOTION_QUIET_MS = 400
DEFAULT_MOTION_CHANGED_PIXEL_RATIO = 0.03
DEFAULT_MOTION_EVENT_MAX_DURATION_MS = 4_000
DEFAULT_MAX_REPLAY_FRAMES_PER_EVENT = 8
DEFAULT_MAX_PENDING_REPLAY_EVENTS_PER_CAMERA = 2
DEFAULT_REPLAY_EVENT_MAX_AGE_MS = 8_000


@dataclass(frozen=True, slots=True)
class PlateCaptureConfig:
    enabled: bool
    max_width: int
    jpeg_quality: int
    capture_root: Path
    reference_root: Path
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PlateDetectorConfig:
    enabled: bool = True
    backend: str = "openvino"
    min_confidence: float = DEFAULT_PLATE_DETECTOR_MIN_CONFIDENCE
    crop_padding_ratio: float = DEFAULT_PLATE_DETECTOR_CROP_PADDING_RATIO
    max_plate_candidates_per_frame: int = DEFAULT_MAX_PLATE_CANDIDATES_PER_FRAME
    fallback_to_roi_ocr: bool = True
    zero_detection_roi_fallback_enabled: bool = (
        DEFAULT_ZERO_DETECTION_ROI_FALLBACK_ENABLED
    )
    zero_detection_roi_fallback_interval_ms: int = (
        DEFAULT_ZERO_DETECTION_ROI_FALLBACK_INTERVAL_MS
    )
    debug_overlay: bool = False
    debug_detection_overlay_ttl_ms: int = DEFAULT_DEBUG_DETECTION_OVERLAY_TTL_MS
    model_dir: Path = Path("models") / "plate_detector" / PLATE_DETECTOR_MODEL_NAME

    @property
    def model_xml(self) -> Path:
        return self.model_dir / "model.xml"

    @property
    def model_bin(self) -> Path:
        return self.model_dir / "model.bin"


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
    plate_detector: PlateDetectorConfig = field(default_factory=PlateDetectorConfig)
    max_pending_ocr_jobs_per_camera: int = DEFAULT_MAX_PENDING_OCR_JOBS_PER_CAMERA
    ocr_job_max_age_ms: int = DEFAULT_OCR_JOB_MAX_AGE_MS
    pre_detection_buffer_duration_ms: int = DEFAULT_PRE_DETECTION_BUFFER_DURATION_MS
    pre_detection_buffer_max_frames_per_camera: int = (
        DEFAULT_PRE_DETECTION_BUFFER_MAX_FRAMES_PER_CAMERA
    )
    motion_pre_roll_ms: int = DEFAULT_MOTION_PRE_ROLL_MS
    motion_post_roll_ms: int = DEFAULT_MOTION_POST_ROLL_MS
    motion_quiet_ms: int = DEFAULT_MOTION_QUIET_MS
    motion_changed_pixel_ratio: float = DEFAULT_MOTION_CHANGED_PIXEL_RATIO
    motion_event_max_duration_ms: int = DEFAULT_MOTION_EVENT_MAX_DURATION_MS
    max_replay_frames_per_event: int = DEFAULT_MAX_REPLAY_FRAMES_PER_EVENT
    max_pending_replay_events_per_camera: int = (
        DEFAULT_MAX_PENDING_REPLAY_EVENTS_PER_CAMERA
    )
    replay_event_max_age_ms: int = DEFAULT_REPLAY_EVENT_MAX_AGE_MS
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
        raw.get("recognition_interval_ms"), 250, 100, 60_000, int,
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
        raw.get("duplicate_cooldown_seconds"),
        DEFAULT_DUPLICATE_COOLDOWN_SECONDS,
        0,
        86_400,
        int,
        "duplicate_cooldown_seconds", warnings,
    )
    retention_days = _record_retention_days(
        raw.get("record_retention_days"), warnings
    )
    max_pending_ocr_jobs_per_camera = _bounded_number(
        raw.get("max_pending_ocr_jobs_per_camera"),
        DEFAULT_MAX_PENDING_OCR_JOBS_PER_CAMERA,
        1,
        20,
        int,
        "max_pending_ocr_jobs_per_camera",
        warnings,
    )
    ocr_job_max_age_ms = _bounded_number(
        raw.get("ocr_job_max_age_ms"),
        DEFAULT_OCR_JOB_MAX_AGE_MS,
        100,
        60_000,
        int,
        "ocr_job_max_age_ms",
        warnings,
    )
    pre_detection_buffer_duration_ms = _bounded_number(
        raw.get("pre_detection_buffer_duration_ms"),
        DEFAULT_PRE_DETECTION_BUFFER_DURATION_MS,
        100,
        10_000,
        int,
        "pre_detection_buffer_duration_ms",
        warnings,
    )
    pre_detection_buffer_max_frames_per_camera = _bounded_number(
        raw.get("pre_detection_buffer_max_frames_per_camera"),
        DEFAULT_PRE_DETECTION_BUFFER_MAX_FRAMES_PER_CAMERA,
        2,
        100,
        int,
        "pre_detection_buffer_max_frames_per_camera",
        warnings,
    )
    motion_pre_roll_ms = _bounded_number(
        raw.get("motion_pre_roll_ms"), DEFAULT_MOTION_PRE_ROLL_MS,
        0, 5_000, int, "motion_pre_roll_ms", warnings,
    )
    motion_post_roll_ms = _bounded_number(
        raw.get("motion_post_roll_ms"), DEFAULT_MOTION_POST_ROLL_MS,
        0, 5_000, int, "motion_post_roll_ms", warnings,
    )
    motion_quiet_ms = _bounded_number(
        raw.get("motion_quiet_ms"), DEFAULT_MOTION_QUIET_MS,
        0, 5_000, int, "motion_quiet_ms", warnings,
    )
    motion_changed_pixel_ratio = _bounded_number(
        raw.get("motion_changed_pixel_ratio"),
        DEFAULT_MOTION_CHANGED_PIXEL_RATIO,
        0.001,
        1.0,
        float,
        "motion_changed_pixel_ratio",
        warnings,
    )
    motion_event_max_duration_ms = _bounded_number(
        raw.get("motion_event_max_duration_ms"),
        DEFAULT_MOTION_EVENT_MAX_DURATION_MS,
        500,
        30_000,
        int,
        "motion_event_max_duration_ms",
        warnings,
    )
    max_replay_frames_per_event = _bounded_number(
        raw.get("max_replay_frames_per_event"),
        DEFAULT_MAX_REPLAY_FRAMES_PER_EVENT,
        1,
        30,
        int,
        "max_replay_frames_per_event",
        warnings,
    )
    max_pending_replay_events_per_camera = _bounded_number(
        raw.get("max_pending_replay_events_per_camera"),
        DEFAULT_MAX_PENDING_REPLAY_EVENTS_PER_CAMERA,
        1,
        10,
        int,
        "max_pending_replay_events_per_camera",
        warnings,
    )
    replay_event_max_age_ms = _bounded_number(
        raw.get("replay_event_max_age_ms"),
        DEFAULT_REPLAY_EVENT_MAX_AGE_MS,
        500,
        60_000,
        int,
        "replay_event_max_age_ms",
        warnings,
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

    detector = _load_plate_detector(raw.get("plate_detector"), root, warnings)

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
        plate_detector=detector,
        max_pending_ocr_jobs_per_camera=max_pending_ocr_jobs_per_camera,
        ocr_job_max_age_ms=ocr_job_max_age_ms,
        pre_detection_buffer_duration_ms=pre_detection_buffer_duration_ms,
        pre_detection_buffer_max_frames_per_camera=(
            pre_detection_buffer_max_frames_per_camera
        ),
        motion_pre_roll_ms=motion_pre_roll_ms,
        motion_post_roll_ms=motion_post_roll_ms,
        motion_quiet_ms=motion_quiet_ms,
        motion_changed_pixel_ratio=motion_changed_pixel_ratio,
        motion_event_max_duration_ms=motion_event_max_duration_ms,
        max_replay_frames_per_event=max_replay_frames_per_event,
        max_pending_replay_events_per_camera=max_pending_replay_events_per_camera,
        replay_event_max_age_ms=replay_event_max_age_ms,
        warnings=tuple(warnings),
    )


def _load_plate_detector(
    raw: object,
    root: Path,
    warnings: list[str],
) -> PlateDetectorConfig:
    if raw is None:
        raw = {}
    elif not isinstance(raw, dict):
        raw = {}
        warnings.append(
            "plate_detector must be an object; default values were used."
        )

    enabled = _boolean_setting(
        raw.get("enabled"), True, "plate_detector.enabled", warnings
    )
    fallback = _boolean_setting(
        raw.get("fallback_to_roi_ocr"),
        True,
        "plate_detector.fallback_to_roi_ocr",
        warnings,
    )
    zero_detection_fallback = _boolean_setting(
        raw.get("zero_detection_roi_fallback_enabled"),
        DEFAULT_ZERO_DETECTION_ROI_FALLBACK_ENABLED,
        "plate_detector.zero_detection_roi_fallback_enabled",
        warnings,
    )
    debug_overlay = _boolean_setting(
        raw.get("debug_overlay"),
        False,
        "plate_detector.debug_overlay",
        warnings,
    )
    backend = raw.get("backend", "openvino")
    if not isinstance(backend, str) or backend.lower() != "openvino":
        backend = "openvino"
        warnings.append(
            "plate_detector.backend is invalid; default 'openvino' was used."
        )

    return PlateDetectorConfig(
        enabled=enabled,
        backend=backend.lower(),
        min_confidence=_bounded_number(
            raw.get("min_confidence"),
            DEFAULT_PLATE_DETECTOR_MIN_CONFIDENCE,
            0.0,
            1.0,
            float,
            "plate_detector.min_confidence",
            warnings,
        ),
        crop_padding_ratio=_bounded_number(
            raw.get("crop_padding_ratio"),
            DEFAULT_PLATE_DETECTOR_CROP_PADDING_RATIO,
            0.0,
            1.0,
            float,
            "plate_detector.crop_padding_ratio",
            warnings,
        ),
        max_plate_candidates_per_frame=_bounded_number(
            raw.get("max_plate_candidates_per_frame"),
            DEFAULT_MAX_PLATE_CANDIDATES_PER_FRAME,
            1,
            10,
            int,
            "plate_detector.max_plate_candidates_per_frame",
            warnings,
        ),
        fallback_to_roi_ocr=fallback,
        zero_detection_roi_fallback_enabled=zero_detection_fallback,
        zero_detection_roi_fallback_interval_ms=_bounded_number(
            raw.get("zero_detection_roi_fallback_interval_ms"),
            DEFAULT_ZERO_DETECTION_ROI_FALLBACK_INTERVAL_MS,
            0,
            60_000,
            int,
            "plate_detector.zero_detection_roi_fallback_interval_ms",
            warnings,
        ),
        debug_overlay=debug_overlay,
        debug_detection_overlay_ttl_ms=_bounded_number(
            raw.get("debug_detection_overlay_ttl_ms"),
            DEFAULT_DEBUG_DETECTION_OVERLAY_TTL_MS,
            0,
            60_000,
            int,
            "plate_detector.debug_detection_overlay_ttl_ms",
            warnings,
        ),
        model_dir=(
            root / "models" / "plate_detector" / PLATE_DETECTOR_MODEL_NAME
        ).resolve(),
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


def _boolean_setting(
    value: object,
    default: bool,
    name: str,
    warnings: list[str],
) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        warnings.append(f"{name} is invalid; the default value was used.")
        return default
    return value


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
