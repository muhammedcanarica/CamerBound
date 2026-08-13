from __future__ import annotations

import math
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
        except PlateDetectorError:
            raise
        except Exception as exc:
            raise PlateDetectorError(
                f"Plate detector modeli başlatılamadı: {exc}"
            ) from exc

    def detect(self, image: np.ndarray) -> list[PlateDetection]:
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise PlateDetectorError("Plate detector input'u BGR renkli görüntü olmalıdır.")
        resized = cv2.resize(
            image,
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

        return parse_ssd_plate_detections(
            output,
            image_width=image.shape[1],
            image_height=image.shape[0],
            min_confidence=self.config.min_confidence,
        )


def parse_ssd_plate_detections(
    output: object,
    *,
    image_width: int,
    image_height: int,
    min_confidence: float,
) -> list[PlateDetection]:
    detections = np.asarray(output, dtype=np.float32)
    if detections.size == 0:
        return []
    if detections.shape[-1] != SSD_DETECTION_SIZE:
        raise PlateDetectorModelInvalid(
            "Plate detector output'unun son boyutu 7 olmalıdır."
        )

    plates: list[PlateDetection] = []
    for row in detections.reshape(-1, SSD_DETECTION_SIZE):
        image_id, label, confidence, x_min, y_min, x_max, y_max = row
        if image_id < 0:
            break
        if not np.all(np.isfinite(row)):
            continue
        if int(label) != PLATE_CLASS_ID or float(confidence) < min_confidence:
            continue

        x1 = max(0, min(image_width, round(float(x_min) * image_width)))
        y1 = max(0, min(image_height, round(float(y_min) * image_height)))
        x2 = max(0, min(image_width, round(float(x_max) * image_width)))
        y2 = max(0, min(image_height, round(float(y_max) * image_height)))
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
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
    return plates


def select_plate_detections(
    detections: Sequence[PlateDetection],
    maximum: int,
) -> list[PlateDetection]:
    return sorted(
        detections,
        key=lambda detection: (detection.confidence, detection.area),
        reverse=True,
    )[: max(0, maximum)]


def crop_padded_plate(
    image: np.ndarray,
    detection: PlateDetection,
    padding_ratio: float,
) -> np.ndarray | None:
    image_height, image_width = image.shape[:2]
    pad_x = detection.width * padding_ratio
    pad_y = detection.height * padding_ratio
    x1 = max(0, math.floor(detection.x - pad_x))
    y1 = max(0, math.floor(detection.y - pad_y))
    x2 = min(image_width, math.ceil(detection.x + detection.width + pad_x))
    y2 = min(image_height, math.ceil(detection.y + detection.height + pad_y))
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
