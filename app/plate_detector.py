from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import cv2
import numpy as np

from app.config import PlateDetectorConfig


VEHICLE_CLASS_ID = 1
PLATE_CLASS_ID = 2
SSD_DETECTION_SIZE = 7
MIN_PLATE_CROP_WIDTH = 8
MIN_PLATE_CROP_HEIGHT = 4
LOW_LIGHT_MEAN_THRESHOLD = 85.0
HIGH_SHADOW_CONTRAST_THRESHOLD = 160.0
SHADOW_PERCENTILE = 0.10
HIGHLIGHT_PERCENTILE = 0.90
SHADOW_LIFT_GAMMA = 0.72
SHADOW_CLAHE_CLIP_LIMIT = 2.0
SHADOW_CLAHE_GRID_SIZE = (8, 8)
DETECTOR_RECOVERY_TILE_ASPECT_RATIO = 2.0
DETECTOR_RECOVERY_TILE_MIN_WIDTH = 384
DETECTOR_RECOVERY_MAX_TILES = 3
# Soft single-line plate references measured from verified field detector boxes.
# They affect ordering only: unusual/special-format candidates are never rejected.
PLATE_GEOMETRY_REFERENCE_ASPECT = 3.2
PLATE_GEOMETRY_MIN_USEFUL_WIDTH = 32
PLATE_GEOMETRY_MIN_USEFUL_HEIGHT = 10
PLATE_GEOMETRY_MIN_USEFUL_ROI_AREA_RATIO = 0.0025
# Detector boxes can cover only the recognized text prefix. Preserve enough
# horizontal plate context so trailing characters still reach OCR.
MIN_PLATE_CROP_ASPECT_RATIO = 4.5


class PlateDetectorError(RuntimeError):
    """Base error for detector initialization and inference failures."""


class PlateDetectorModelNotFound(PlateDetectorError):
    """Raised when the local OpenVINO IR files are missing."""


class PlateDetectorModelInvalid(PlateDetectorError):
    """Raised when model metadata does not match the supported SSD contract."""


@dataclass(frozen=True, slots=True)
class PlateDetection:
    confidence: float
    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass(frozen=True, slots=True)
class DetectorLightingMetrics:
    mean_brightness: float
    shadow_metric: float

    @property
    def is_difficult(self) -> bool:
        return (
            self.mean_brightness <= LOW_LIGHT_MEAN_THRESHOLD
            or self.shadow_metric >= HIGH_SHADOW_CONTRAST_THRESHOLD
        )


@dataclass(frozen=True, slots=True)
class DetectorDiagnostics:
    detector_variant: str
    raw_brightness: float | None
    shadow_metric: float | None
    enhanced_pass: bool
    raw_detector_ms: float
    enhanced_detector_ms: float
    detections: int
    input_width: int = 0
    input_height: int = 0
    input_layout: str = "unknown"
    input_dtype: str = "unknown"
    roi_width: int = 0
    roi_height: int = 0
    raw_candidate_count: int = 0
    plate_class_candidate_count: int = 0
    highest_plate_confidence: float | None = None
    confidence_rejected_count: int = 0
    bbox_rejected_count: int = 0
    resize_scale_x: float = 0.0
    resize_scale_y: float = 0.0
    aspect_distortion_ratio: float = 1.0
    tiled_recovery_pass: bool = False
    recovery_tile_count: int = 0
    tiled_detector_ms: float = 0.0
    raw_detector_calls: int = 0
    enhanced_detector_calls: int = 0
    tiled_detector_calls: int = 0
    raw_hit: bool = False
    enhanced_hit: bool = False
    tiled_hit: bool = False
    expensive_recovery_allowed: bool = True
    expensive_recovery_interrupted: bool = False


@dataclass(frozen=True, slots=True)
class SsdParseDiagnostics:
    raw_candidate_count: int
    plate_class_candidate_count: int
    highest_plate_confidence: float | None
    confidence_rejected_count: int
    bbox_rejected_count: int


class PlateDetector(Protocol):
    def detect(self, image: np.ndarray) -> list[PlateDetection]: ...


class OpenVinoPlateDetector:
    """Offline OpenVINO adapter for the OMZ barrier SSD detector."""

    def __init__(
        self,
        config: PlateDetectorConfig,
        *,
        core_factory: Callable[[], object] | None = None,
    ) -> None:
        self.config = config
        _validate_model_files(config.model_xml, config.model_bin)

        if core_factory is None:
            try:
                from openvino import Core
            except (ImportError, OSError) as exc:
                raise PlateDetectorError(
                    "OpenVINO Runtime yüklenemedi; plate detector kullanılamıyor."
                ) from exc
            core_factory = Core

        try:
            self._core = core_factory()
            model = self._core.read_model(model=str(config.model_xml))
            inputs = list(model.inputs)
            outputs = list(model.outputs)
            if len(inputs) != 1:
                raise PlateDetectorModelInvalid(
                    "Plate detector modeli tam olarak bir input içermelidir."
                )
            if len(outputs) != 1:
                raise PlateDetectorModelInvalid(
                    "Plate detector modeli tam olarak bir output içermelidir."
                )

            self._input_port = inputs[0]
            self._output_port = outputs[0]
            self._input_layout, self._input_height, self._input_width = (
                _resolve_input_metadata(self._input_port)
            )
            self._input_dtype = _resolve_input_dtype(self._input_port)
            self._input_key = _port_name_or_port(self._input_port)
            _validate_output_metadata(self._output_port)
            self._compiled_model = self._core.compile_model(model, "CPU")
            self._infer_request = self._compiled_model.create_infer_request()
            self.last_diagnostics: DetectorDiagnostics | None = None
        except PlateDetectorError:
            raise
        except Exception as exc:
            raise PlateDetectorError(
                f"Plate detector modeli başlatılamadı: {exc}"
            ) from exc

    def detect(self, image: np.ndarray) -> list[PlateDetection]:
        return self.detect_with_policy(
            image,
            allow_expensive_recovery=True,
        )

    def detect_with_policy(
        self,
        image: np.ndarray,
        *,
        allow_expensive_recovery: bool,
        should_continue_expensive_recovery: Callable[[], bool] | None = None,
    ) -> list[PlateDetection]:
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise PlateDetectorError("Plate detector input'u BGR renkli görüntü olmalıdır.")
        self.last_diagnostics = None

        raw_started_at = time.perf_counter()
        raw_detections, raw_parse = self._infer_detections(
            image,
            coordinate_width=image.shape[1],
            coordinate_height=image.shape[0],
        )
        raw_detector_ms = (time.perf_counter() - raw_started_at) * 1000.0
        if raw_detections:
            self.last_diagnostics = DetectorDiagnostics(
                detector_variant="raw",
                raw_brightness=None,
                shadow_metric=None,
                enhanced_pass=False,
                raw_detector_ms=raw_detector_ms,
                enhanced_detector_ms=0.0,
                detections=len(raw_detections),
                raw_detector_calls=1,
                raw_hit=True,
                expensive_recovery_allowed=allow_expensive_recovery,
                **self._diagnostic_context(image, raw_parse),
            )
            return raw_detections

        lighting = measure_detector_lighting(image)
        if not allow_expensive_recovery:
            self.last_diagnostics = DetectorDiagnostics(
                detector_variant="raw",
                raw_brightness=lighting.mean_brightness,
                shadow_metric=lighting.shadow_metric,
                enhanced_pass=False,
                raw_detector_ms=raw_detector_ms,
                enhanced_detector_ms=0.0,
                detections=0,
                raw_detector_calls=1,
                expensive_recovery_allowed=False,
                **self._diagnostic_context(image, raw_parse),
            )
            return []

        enhanced_detections: list[PlateDetection] = []
        enhanced_detector_ms = 0.0
        enhanced_pass = False
        recovery_interrupted = False
        enhanced_parse = raw_parse
        if lighting.is_difficult:
            enhanced_pass = (
                should_continue_expensive_recovery is None
                or should_continue_expensive_recovery()
            )
            recovery_interrupted = not enhanced_pass
        if enhanced_pass:
            enhanced = enhance_shadowed_detector_image(image)
            enhanced_started_at = time.perf_counter()
            enhanced_detections, enhanced_parse = self._infer_detections(
                enhanced,
                coordinate_width=image.shape[1],
                coordinate_height=image.shape[0],
            )
            enhanced_detector_ms = (
                time.perf_counter() - enhanced_started_at
            ) * 1000.0
            if enhanced_detections:
                self.last_diagnostics = DetectorDiagnostics(
                    detector_variant="enhanced",
                    raw_brightness=lighting.mean_brightness,
                    shadow_metric=lighting.shadow_metric,
                    enhanced_pass=True,
                    raw_detector_ms=raw_detector_ms,
                    enhanced_detector_ms=enhanced_detector_ms,
                    detections=len(enhanced_detections),
                    raw_detector_calls=1,
                    enhanced_detector_calls=1,
                    enhanced_hit=True,
                    expensive_recovery_allowed=True,
                    **self._diagnostic_context(image, enhanced_parse),
                )
                return enhanced_detections

        tiles = detector_recovery_tiles(image)
        tiled_started_at = time.perf_counter()
        tiled_detections: list[PlateDetection] = []
        tiled_parse = SsdParseDiagnostics(0, 0, None, 0, 0)
        tiled_detector_calls = 0
        for x_offset, tile in tiles:
            if recovery_interrupted:
                break
            if (
                should_continue_expensive_recovery is not None
                and not should_continue_expensive_recovery()
            ):
                recovery_interrupted = True
                break
            tile_detections, tile_parse = self._infer_detections(
                tile,
                coordinate_width=tile.shape[1],
                coordinate_height=tile.shape[0],
            )
            tiled_detector_calls += 1
            tiled_detections.extend(
                PlateDetection(
                    confidence=detection.confidence,
                    x=detection.x + x_offset,
                    y=detection.y,
                    width=detection.width,
                    height=detection.height,
                )
                for detection in tile_detections
            )
            tiled_parse = _merge_parse_diagnostics(tiled_parse, tile_parse)
        tiled_detector_ms = (time.perf_counter() - tiled_started_at) * 1000.0
        tiled_detections = _deduplicate_detections(tiled_detections)
        final_parse = (
            tiled_parse
            if tiled_detector_calls
            else enhanced_parse if enhanced_pass else raw_parse
        )
        self.last_diagnostics = DetectorDiagnostics(
            detector_variant="tiled" if tiled_detections else (
                "enhanced" if enhanced_pass else "raw"
            ),
            raw_brightness=lighting.mean_brightness,
            shadow_metric=lighting.shadow_metric,
            enhanced_pass=enhanced_pass,
            raw_detector_ms=raw_detector_ms,
            enhanced_detector_ms=enhanced_detector_ms,
            detections=len(tiled_detections),
            tiled_recovery_pass=tiled_detector_calls > 0,
            recovery_tile_count=tiled_detector_calls,
            tiled_detector_ms=tiled_detector_ms,
            raw_detector_calls=1,
            enhanced_detector_calls=1 if enhanced_pass else 0,
            tiled_detector_calls=tiled_detector_calls,
            tiled_hit=bool(tiled_detections),
            expensive_recovery_allowed=True,
            expensive_recovery_interrupted=recovery_interrupted,
            **self._diagnostic_context(image, final_parse),
        )
        return tiled_detections

    def _infer_detections(
        self,
        detector_image: np.ndarray,
        *,
        coordinate_width: int,
        coordinate_height: int,
    ) -> tuple[list[PlateDetection], SsdParseDiagnostics]:
        resized = cv2.resize(
            detector_image,
            (self._input_width, self._input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        tensor = np.expand_dims(resized, axis=0)
        if self._input_layout == "NCHW":
            tensor = np.transpose(tensor, (0, 3, 1, 2))
        tensor = tensor.astype(self._input_dtype, copy=False)

        try:
            result = self._infer_request.infer({self._input_key: tensor})
            output = _extract_single_output(result, self._output_port)
        except Exception as exc:
            raise PlateDetectorError(f"Plate detector inference başarısız: {exc}") from exc

        return _parse_ssd_plate_detections(
            output,
            image_width=coordinate_width,
            image_height=coordinate_height,
            min_confidence=self.config.min_confidence,
        )

    def _diagnostic_context(
        self,
        image: np.ndarray,
        parse: SsdParseDiagnostics,
    ) -> dict[str, object]:
        scale_x = self._input_width / max(1, image.shape[1])
        scale_y = self._input_height / max(1, image.shape[0])
        return {
            "input_width": self._input_width,
            "input_height": self._input_height,
            "input_layout": self._input_layout,
            "input_dtype": np.dtype(self._input_dtype).name,
            "roi_width": image.shape[1],
            "roi_height": image.shape[0],
            "raw_candidate_count": parse.raw_candidate_count,
            "plate_class_candidate_count": parse.plate_class_candidate_count,
            "highest_plate_confidence": parse.highest_plate_confidence,
            "confidence_rejected_count": parse.confidence_rejected_count,
            "bbox_rejected_count": parse.bbox_rejected_count,
            "resize_scale_x": scale_x,
            "resize_scale_y": scale_y,
            "aspect_distortion_ratio": max(scale_x, scale_y) / max(
                min(scale_x, scale_y), 1e-9
            ),
        }


def detector_recovery_tiles(
    image: np.ndarray,
    *,
    maximum: int = DETECTOR_RECOVERY_MAX_TILES,
) -> tuple[tuple[int, np.ndarray], ...]:
    """Return bounded horizontal views that preserve more small-plate pixels."""
    height, width = image.shape[:2]
    tile_width = min(
        width,
        max(
            DETECTOR_RECOVERY_TILE_MIN_WIDTH,
            round(height * DETECTOR_RECOVERY_TILE_ASPECT_RATIO),
        ),
    )
    maximum = max(0, maximum)
    if maximum == 0 or tile_width >= width:
        return ()
    tile_count = min(maximum, math.ceil(width / tile_width) + 1)
    starts = {
        round(index * (width - tile_width) / max(1, tile_count - 1))
        for index in range(tile_count)
    }
    return tuple(
        (start, image[:, start : start + tile_width]) for start in sorted(starts)
    )


def _merge_parse_diagnostics(
    first: SsdParseDiagnostics,
    second: SsdParseDiagnostics,
) -> SsdParseDiagnostics:
    confidences = tuple(
        value
        for value in (
            first.highest_plate_confidence,
            second.highest_plate_confidence,
        )
        if value is not None
    )
    return SsdParseDiagnostics(
        raw_candidate_count=first.raw_candidate_count + second.raw_candidate_count,
        plate_class_candidate_count=(
            first.plate_class_candidate_count + second.plate_class_candidate_count
        ),
        highest_plate_confidence=max(confidences) if confidences else None,
        confidence_rejected_count=(
            first.confidence_rejected_count + second.confidence_rejected_count
        ),
        bbox_rejected_count=first.bbox_rejected_count + second.bbox_rejected_count,
    )


def _deduplicate_detections(
    detections: Sequence[PlateDetection],
) -> list[PlateDetection]:
    kept: list[PlateDetection] = []
    for detection in sorted(
        detections,
        key=lambda item: (item.confidence, item.area),
        reverse=True,
    ):
        if any(_intersection_over_union(detection, other) >= 0.50 for other in kept):
            continue
        kept.append(detection)
    return kept


def _intersection_over_union(
    first: PlateDetection,
    second: PlateDetection,
) -> float:
    x1 = max(first.x, second.x)
    y1 = max(first.y, second.y)
    x2 = min(first.x + first.width, second.x + second.width)
    y2 = min(first.y + first.height, second.y + second.height)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = first.area + second.area - intersection
    return intersection / union if union > 0 else 0.0


def measure_detector_lighting(image: np.ndarray) -> DetectorLightingMetrics:
    """Return cheap, deterministic luminance metrics for raw detector misses."""
    uint8_image = _as_uint8_bgr(image)
    grayscale = cv2.cvtColor(uint8_image, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([grayscale], [0], None, [256], [0, 256]).reshape(-1)
    cumulative = np.cumsum(histogram)
    pixel_count = float(grayscale.size)
    shadow_level = int(np.searchsorted(cumulative, pixel_count * SHADOW_PERCENTILE))
    highlight_level = int(
        np.searchsorted(cumulative, pixel_count * HIGHLIGHT_PERCENTILE)
    )
    return DetectorLightingMetrics(
        mean_brightness=float(grayscale.mean()),
        shadow_metric=float(highlight_level - shadow_level),
    )


def enhance_shadowed_detector_image(image: np.ndarray) -> np.ndarray:
    """Lift shadow luminance without sharpening or replacing BGR chroma."""
    lab = cv2.cvtColor(_as_uint8_bgr(image), cv2.COLOR_BGR2LAB)
    luminance, channel_a, channel_b = cv2.split(lab)
    gamma_lut = np.clip(
        ((np.arange(256, dtype=np.float32) / 255.0) ** SHADOW_LIFT_GAMMA) * 255.0,
        0,
        255,
    ).astype(np.uint8)
    lifted = cv2.LUT(luminance, gamma_lut)
    clahe = cv2.createCLAHE(
        clipLimit=SHADOW_CLAHE_CLIP_LIMIT,
        tileGridSize=SHADOW_CLAHE_GRID_SIZE,
    )
    enhanced_luminance = clahe.apply(lifted)
    return cv2.cvtColor(
        cv2.merge((enhanced_luminance, channel_a, channel_b)),
        cv2.COLOR_LAB2BGR,
    )


def _as_uint8_bgr(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


def parse_ssd_plate_detections(
    output: object,
    *,
    image_width: int,
    image_height: int,
    min_confidence: float,
) -> list[PlateDetection]:
    plates, _diagnostics = _parse_ssd_plate_detections(
        output,
        image_width=image_width,
        image_height=image_height,
        min_confidence=min_confidence,
    )
    return plates


def _parse_ssd_plate_detections(
    output: object,
    *,
    image_width: int,
    image_height: int,
    min_confidence: float,
) -> tuple[list[PlateDetection], SsdParseDiagnostics]:
    detections = np.asarray(output, dtype=np.float32)
    if detections.size == 0:
        return [], SsdParseDiagnostics(0, 0, None, 0, 0)
    if detections.shape[-1] != SSD_DETECTION_SIZE:
        raise PlateDetectorModelInvalid(
            "Plate detector output'unun son boyutu 7 olmalıdır."
        )

    plates: list[PlateDetection] = []
    raw_candidate_count = 0
    plate_class_candidate_count = 0
    highest_plate_confidence: float | None = None
    confidence_rejected_count = 0
    bbox_rejected_count = 0
    for row in detections.reshape(-1, SSD_DETECTION_SIZE):
        image_id, label, confidence, x_min, y_min, x_max, y_max = row
        if image_id < 0:
            break
        raw_candidate_count += 1
        if not np.all(np.isfinite(row)):
            continue
        if int(label) != PLATE_CLASS_ID:
            continue
        plate_class_candidate_count += 1
        candidate_confidence = float(confidence)
        highest_plate_confidence = (
            candidate_confidence
            if highest_plate_confidence is None
            else max(highest_plate_confidence, candidate_confidence)
        )
        if candidate_confidence < min_confidence:
            confidence_rejected_count += 1
            continue

        x1 = max(0, min(image_width, round(float(x_min) * image_width)))
        y1 = max(0, min(image_height, round(float(y_min) * image_height)))
        x2 = max(0, min(image_width, round(float(x_max) * image_width)))
        y2 = max(0, min(image_height, round(float(y_max) * image_height)))
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            bbox_rejected_count += 1
            continue
        plates.append(
            PlateDetection(
                confidence=float(confidence),
                x=x1,
                y=y1,
                width=width,
                height=height,
            )
        )
    return plates, SsdParseDiagnostics(
        raw_candidate_count=raw_candidate_count,
        plate_class_candidate_count=plate_class_candidate_count,
        highest_plate_confidence=highest_plate_confidence,
        confidence_rejected_count=confidence_rejected_count,
        bbox_rejected_count=bbox_rejected_count,
    )


def select_plate_detections(
    detections: Sequence[PlateDetection],
    maximum: int,
    *,
    roi_width: int | None = None,
    roi_height: int | None = None,
) -> list[PlateDetection]:
    return sorted(
        detections,
        key=lambda detection: (
            plate_detection_ranking_score(
                detection,
                roi_width=roi_width,
                roi_height=roi_height,
            ),
            plate_detection_geometry_quality(
                detection,
                roi_width=roi_width,
                roi_height=roi_height,
            ),
            detection.confidence,
            detection.area,
        ),
        reverse=True,
    )[: max(0, maximum)]


def plate_detection_geometry_quality(
    detection: PlateDetection,
    *,
    roi_width: int | None = None,
    roi_height: int | None = None,
) -> float:
    """Return a continuous geometry hint without rejecting detector evidence."""
    aspect = detection.width / max(1, detection.height)
    aspect_quality = math.exp(
        -abs(math.log(max(aspect, 0.01) / PLATE_GEOMETRY_REFERENCE_ASPECT))
    )
    width_quality = min(1.0, detection.width / PLATE_GEOMETRY_MIN_USEFUL_WIDTH)
    height_quality = min(1.0, detection.height / PLATE_GEOMETRY_MIN_USEFUL_HEIGHT)
    scale_quality = math.sqrt(width_quality * height_quality)
    if (
        roi_width is not None
        and roi_height is not None
        and roi_width > 0
        and roi_height > 0
    ):
        area_ratio = detection.area / (roi_width * roi_height)
        area_quality = min(
            1.0,
            area_ratio / PLATE_GEOMETRY_MIN_USEFUL_ROI_AREA_RATIO,
        )
        scale_quality = math.sqrt(scale_quality * area_quality)
    return math.sqrt(aspect_quality * scale_quality)


def plate_detection_ranking_score(
    detection: PlateDetection,
    *,
    roi_width: int | None = None,
    roi_height: int | None = None,
) -> float:
    """Blend detector confidence with bounded geometry quality for selection."""
    geometry = plate_detection_geometry_quality(
        detection,
        roi_width=roi_width,
        roi_height=roi_height,
    )
    return detection.confidence * (0.5 + 0.5 * geometry)


def crop_padded_plate(
    image: np.ndarray,
    detection: PlateDetection,
    padding_ratio: float,
    *,
    minimum_aspect_ratio: float | None = None,
) -> np.ndarray | None:
    image_height, image_width = image.shape[:2]
    pad_x = detection.width * padding_ratio
    pad_y = detection.height * padding_ratio
    x1 = max(0, math.floor(detection.x - pad_x))
    y1 = max(0, math.floor(detection.y - pad_y))
    x2 = min(image_width, math.ceil(detection.x + detection.width + pad_x))
    y2 = min(image_height, math.ceil(detection.y + detection.height + pad_y))
    if minimum_aspect_ratio is not None and minimum_aspect_ratio > 0:
        required_width = math.ceil((y2 - y1) * minimum_aspect_ratio)
        if required_width > x2 - x1:
            center_x = detection.x + (detection.width / 2.0)
            x1 = math.floor(center_x - (required_width / 2.0))
            x2 = x1 + required_width
            if x1 < 0:
                x1 = 0
                x2 = min(image_width, required_width)
            elif x2 > image_width:
                x2 = image_width
                x1 = max(0, image_width - required_width)
    if x2 - x1 < MIN_PLATE_CROP_WIDTH or y2 - y1 < MIN_PLATE_CROP_HEIGHT:
        return None
    crop = image[y1:y2, x1:x2]
    return crop.copy() if crop.size else None


def _validate_model_files(model_xml: Path, model_bin: Path) -> None:
    missing = [str(path) for path in (model_xml, model_bin) if not path.is_file()]
    if missing:
        raise PlateDetectorModelNotFound(
            "Plate detector model dosyaları bulunamadı: " + ", ".join(missing)
        )


def _resolve_input_metadata(input_port: object) -> tuple[str, int, int]:
    shape = tuple(int(dimension) for dimension in input_port.shape)
    if len(shape) != 4 or shape[0] != 1:
        raise PlateDetectorModelInvalid(
            f"Plate detector input shape desteklenmiyor: {shape}"
        )
    if shape[-1] == 3:
        return "NHWC", shape[1], shape[2]
    if shape[1] == 3:
        return "NCHW", shape[2], shape[3]
    raise PlateDetectorModelInvalid(
        f"Plate detector input renk kanalı bulunamadı: {shape}"
    )


def _validate_output_metadata(output_port: object) -> None:
    shape = tuple(int(dimension) for dimension in output_port.shape)
    if len(shape) != 4 or shape[-1] != SSD_DETECTION_SIZE:
        raise PlateDetectorModelInvalid(
            f"Plate detector SSD output shape desteklenmiyor: {shape}"
        )


def _resolve_input_dtype(input_port: object) -> type[np.generic]:
    get_element_type = getattr(input_port, "get_element_type", None)
    if get_element_type is None:
        return np.uint8
    element_type = get_element_type()
    get_type_name = getattr(element_type, "get_type_name", None)
    type_name = get_type_name() if get_type_name is not None else str(element_type)
    if type_name in {"f32", "float32"}:
        return np.float32
    if type_name in {"f16", "float16"}:
        return np.float16
    if type_name in {"u8", "uint8"}:
        return np.uint8
    raise PlateDetectorModelInvalid(
        f"Plate detector input element type desteklenmiyor: {type_name}"
    )


def _port_name_or_port(port: object) -> object:
    get_any_name = getattr(port, "get_any_name", None)
    return get_any_name() if get_any_name is not None else port


def _extract_single_output(result: object, output_port: object) -> object:
    try:
        return result[output_port]
    except (KeyError, TypeError):
        values_method = getattr(result, "values", None)
        values = list(values_method()) if values_method is not None else []
        if len(values) == 1:
            return values[0]
    raise PlateDetectorModelInvalid("Plate detector tek bir output üretmelidir.")
