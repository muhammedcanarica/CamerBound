from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.config import PlateDetectorConfig
from app.plate_detector import (
    OpenVinoPlateDetector,
    PlateDetection,
    PlateDetectorModelNotFound,
    crop_padded_plate,
    parse_ssd_plate_detections,
    select_plate_detections,
)


class _FakePort:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class _FakeModel:
    def __init__(self, input_port: object, output_port: object) -> None:
        self.inputs = [input_port]
        self.outputs = [output_port]


class _FakeInferRequest:
    def __init__(self, output_port: object, output: np.ndarray) -> None:
        self.output_port = output_port
        self.output = output
        self.calls = 0

    def infer(self, _inputs: object) -> dict[object, np.ndarray]:
        self.calls += 1
        return {self.output_port: self.output}


class _FakeCompiledModel:
    def __init__(self, request: _FakeInferRequest) -> None:
        self.request = request
        self.create_calls = 0

    def create_infer_request(self) -> _FakeInferRequest:
        self.create_calls += 1
        return self.request


class _FakeCore:
    def __init__(self, model: _FakeModel, compiled: _FakeCompiledModel) -> None:
        self.model = model
        self.compiled = compiled
        self.read_calls = 0
        self.compile_calls = 0

    def read_model(self, *, model: str) -> _FakeModel:
        self.read_calls += 1
        self.model_path = model
        return self.model

    def compile_model(self, model: object, device: str) -> _FakeCompiledModel:
        self.compile_calls += 1
        self.compiled_model = model
        self.device = device
        return self.compiled


def _ssd_row(
    label: int,
    confidence: float,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> list[float]:
    return [0.0, float(label), confidence, x_min, y_min, x_max, y_max]


class PlateDetectorTests(unittest.TestCase):
    def test_model_is_initialized_once_and_reused_for_each_detection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            model_dir = Path(temp_directory)
            (model_dir / "model.xml").write_text("<xml />", encoding="utf-8")
            (model_dir / "model.bin").write_bytes(b"weights")
            input_port = _FakePort((1, 256, 256, 3))
            output_port = _FakePort((1, 1, 200, 7))
            output = np.array(
                [[[[*_ssd_row(2, 0.9, 0.1, 0.2, 0.6, 0.5)]]]],
                dtype=np.float32,
            )
            request = _FakeInferRequest(output_port, output)
            compiled = _FakeCompiledModel(request)
            core = _FakeCore(_FakeModel(input_port, output_port), compiled)
            detector = OpenVinoPlateDetector(
                PlateDetectorConfig(model_dir=model_dir),
                core_factory=lambda: core,
            )

            image = np.zeros((100, 200, 3), dtype=np.uint8)
            detector.detect(image)
            detector.detect(image)

        self.assertEqual(core.read_calls, 1)
        self.assertEqual(core.compile_calls, 1)
        self.assertEqual(compiled.create_calls, 1)
        self.assertEqual(request.calls, 2)
        self.assertEqual(core.device, "CPU")

    def test_missing_model_files_raise_clear_error_before_runtime_import(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            config = PlateDetectorConfig(model_dir=Path(temp_directory))

            with self.assertRaisesRegex(
                PlateDetectorModelNotFound,
                "model.xml",
            ):
                OpenVinoPlateDetector(config)

    def test_bbox_is_converted_to_pixels_and_out_of_range_values_are_clamped(self) -> None:
        output = np.array(
            [
                _ssd_row(2, 0.9, 0.10, 0.20, 0.60, 0.50),
                _ssd_row(2, 0.8, -0.20, -0.10, 1.20, 1.10),
            ],
            dtype=np.float32,
        )

        detections = parse_ssd_plate_detections(
            output,
            image_width=200,
            image_height=100,
            min_confidence=0.5,
        )

        self.assertAlmostEqual(detections[0].confidence, 0.9, places=5)
        self.assertEqual(
            (detections[0].x, detections[0].y, detections[0].width, detections[0].height),
            (20, 20, 100, 30),
        )
        self.assertAlmostEqual(detections[1].confidence, 0.8, places=5)
        self.assertEqual(
            (detections[1].x, detections[1].y, detections[1].width, detections[1].height),
            (0, 0, 200, 100),
        )

    def test_low_confidence_and_vehicle_class_are_filtered_but_plate_is_kept(self) -> None:
        output = np.array(
            [
                _ssd_row(1, 0.99, 0.1, 0.1, 0.5, 0.5),
                _ssd_row(2, 0.49, 0.1, 0.1, 0.5, 0.5),
                _ssd_row(2, 0.88, 0.2, 0.2, 0.7, 0.6),
            ],
            dtype=np.float32,
        )

        detections = parse_ssd_plate_detections(
            output,
            image_width=100,
            image_height=50,
            min_confidence=0.5,
        )

        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0].confidence, 0.88, places=5)

    def test_crop_padding_is_applied_and_clamped(self) -> None:
        image = np.zeros((50, 100, 3), dtype=np.uint8)
        center = PlateDetection(0.9, 20, 10, 40, 20)
        edge = PlateDetection(0.8, 0, 0, 20, 10)

        center_crop = crop_padded_plate(image, center, 0.15)
        edge_crop = crop_padded_plate(image, edge, 0.15)

        self.assertEqual(center_crop.shape, (26, 52, 3))
        self.assertEqual(edge_crop.shape, (12, 23, 3))

    def test_candidate_limit_prefers_confidence_then_area(self) -> None:
        detections = [
            PlateDetection(0.8, 0, 0, 100, 20),
            PlateDetection(0.9, 0, 0, 20, 10),
            PlateDetection(0.9, 0, 0, 40, 10),
        ]

        selected = select_plate_detections(detections, maximum=2)

        self.assertEqual(selected, [detections[2], detections[1]])


if __name__ == "__main__":
    unittest.main()
