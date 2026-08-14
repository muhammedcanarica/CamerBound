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
    enhance_shadowed_detector_image,
    measure_detector_lighting,
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


class _SequencedInferRequest:
    def __init__(self, output_port: object, outputs: list[np.ndarray]) -> None:
        self.output_port = output_port
        self.outputs = outputs
        self.calls = 0
        self.tensors: list[np.ndarray] = []

    def infer(self, inputs: dict[object, np.ndarray]) -> dict[object, np.ndarray]:
        self.tensors.append(next(iter(inputs.values())).copy())
        output = self.outputs[self.calls]
        self.calls += 1
        return {self.output_port: output}


class _FakeCompiledModel:
    def __init__(self, request: object) -> None:
        self.request = request
        self.create_calls = 0

    def create_infer_request(self) -> object:
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


def _empty_ssd_output() -> np.ndarray:
    return np.empty((1, 1, 0, 7), dtype=np.float32)


def _plate_ssd_output(
    confidence: float = 0.90,
    *,
    x_min: float = 0.10,
    y_min: float = 0.20,
    x_max: float = 0.60,
    y_max: float = 0.50,
) -> np.ndarray:
    return np.array(
        [[[_ssd_row(2, confidence, x_min, y_min, x_max, y_max)]]],
        dtype=np.float32,
    )


def _make_sequenced_detector(
    model_dir: Path,
    outputs: list[np.ndarray],
    *,
    min_confidence: float = 0.15,
) -> tuple[OpenVinoPlateDetector, _SequencedInferRequest]:
    (model_dir / "model.xml").write_text("<xml />", encoding="utf-8")
    (model_dir / "model.bin").write_bytes(b"weights")
    input_port = _FakePort((1, 256, 256, 3))
    output_port = _FakePort((1, 1, 200, 7))
    request = _SequencedInferRequest(output_port, outputs)
    compiled = _FakeCompiledModel(request)
    core = _FakeCore(_FakeModel(input_port, output_port), compiled)
    detector = OpenVinoPlateDetector(
        PlateDetectorConfig(
            model_dir=model_dir,
            min_confidence=min_confidence,
        ),
        core_factory=lambda: core,
    )
    return detector, request


class PlateDetectorTests(unittest.TestCase):
    def test_bright_raw_miss_does_not_run_enhanced_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            detector, request = _make_sequenced_detector(
                Path(temp_directory),
                [_empty_ssd_output()],
            )

            detections = detector.detect(
                np.full((100, 200, 3), 180, dtype=np.uint8)
            )

        self.assertEqual(detections, [])
        self.assertEqual(request.calls, 1)
        self.assertEqual(detector.last_diagnostics.detector_variant, "raw")
        self.assertFalse(detector.last_diagnostics.enhanced_pass)

    def test_raw_detection_skips_enhancement_even_for_dark_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            detector, request = _make_sequenced_detector(
                Path(temp_directory),
                [_plate_ssd_output()],
            )

            detections = detector.detect(np.full((100, 200, 3), 25, dtype=np.uint8))

        self.assertEqual(len(detections), 1)
        self.assertEqual(request.calls, 1)
        self.assertEqual(detector.last_diagnostics.detector_variant, "raw")
        self.assertFalse(detector.last_diagnostics.enhanced_pass)

    def test_dark_raw_miss_runs_enhanced_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            detector, request = _make_sequenced_detector(
                Path(temp_directory),
                [_empty_ssd_output(), _plate_ssd_output()],
            )
            image = np.full((100, 200, 3), 30, dtype=np.uint8)

            detections = detector.detect(image)

        self.assertEqual(len(detections), 1)
        self.assertEqual(request.calls, 2)
        self.assertLess(
            float(request.tensors[0].mean()),
            float(request.tensors[1].mean()),
        )
        self.assertEqual(detector.last_diagnostics.detector_variant, "enhanced")
        self.assertTrue(detector.last_diagnostics.enhanced_pass)
        self.assertAlmostEqual(detector.last_diagnostics.raw_brightness, 30.0)

    def test_high_shadow_highlight_contrast_runs_enhanced_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            detector, request = _make_sequenced_detector(
                Path(temp_directory),
                [_empty_ssd_output(), _empty_ssd_output()],
            )
            image = np.full((100, 200, 3), 230, dtype=np.uint8)
            image[:, :100] = 20

            detector.detect(image)

        self.assertEqual(request.calls, 2)
        self.assertGreater(detector.last_diagnostics.shadow_metric, 160.0)
        self.assertTrue(detector.last_diagnostics.enhanced_pass)

    def test_normal_lighting_raw_miss_avoids_second_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            detector, request = _make_sequenced_detector(
                Path(temp_directory),
                [_empty_ssd_output()],
            )

            detector.detect(np.full((100, 200, 3), 120, dtype=np.uint8))

        self.assertEqual(request.calls, 1)
        self.assertFalse(detector.last_diagnostics.enhanced_pass)
        self.assertEqual(detector.last_diagnostics.shadow_metric, 0.0)

    def test_enhanced_detection_coordinates_and_crop_use_original_roi(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            detector, request = _make_sequenced_detector(
                Path(temp_directory),
                [_empty_ssd_output(), _plate_ssd_output()],
            )
            image = np.zeros((100, 200, 3), dtype=np.uint8)
            image[:, :, 0] = np.arange(200, dtype=np.uint8)
            image[:, :, 1] = 20
            image[:, :, 2] = 30

            detections = detector.detect(image)
            crop = crop_padded_plate(image, detections[0], 0.0)

        self.assertEqual(request.calls, 2)
        self.assertEqual(
            (
                detections[0].x,
                detections[0].y,
                detections[0].width,
                detections[0].height,
            ),
            (20, 20, 100, 30),
        )
        np.testing.assert_array_equal(crop, image[20:50, 20:120])
        self.assertFalse(np.array_equal(crop, request.tensors[1][0, 20:50, 20:120]))

    def test_configured_detector_threshold_is_used_for_both_passes(self) -> None:
        below_threshold = _plate_ssd_output(confidence=0.149)
        with tempfile.TemporaryDirectory() as temp_directory:
            detector, request = _make_sequenced_detector(
                Path(temp_directory),
                [below_threshold, below_threshold],
                min_confidence=0.15,
            )

            detections = detector.detect(np.full((100, 200, 3), 30, dtype=np.uint8))

        self.assertEqual(detector.config.min_confidence, 0.15)
        self.assertEqual(detections, [])
        self.assertEqual(request.calls, 2)

    def test_shadow_enhancement_deterministically_lifts_dark_detail(self) -> None:
        image = np.full((64, 128, 3), 12, dtype=np.uint8)
        image[16:48, 32:64] = 20
        image[16:48, 64:96] = 30

        enhanced = enhance_shadowed_detector_image(image)
        original_metrics = measure_detector_lighting(image)
        enhanced_metrics = measure_detector_lighting(enhanced)

        self.assertEqual(enhanced.shape, image.shape)
        self.assertEqual(enhanced.dtype, np.uint8)
        self.assertGreater(
            enhanced_metrics.mean_brightness,
            original_metrics.mean_brightness,
        )
        self.assertGreater(
            float(enhanced[16:48, 64:96].mean() - enhanced[16:48, 32:64].mean()),
            float(image[16:48, 64:96].mean() - image[16:48, 32:64].mean()),
        )

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
