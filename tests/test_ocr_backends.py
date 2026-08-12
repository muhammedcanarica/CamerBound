from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.ocr_models import (
    BackendSelection,
    OcrBackend,
    OcrModelInvalid,
    OcrModelNotFound,
    select_ocr_backend,
)
from app.plate_recognition import OcrSegment, PaddleOcrProvider


class _FakeOnnxSession:
    def __init__(self, _path: str, providers: list[str]) -> None:
        self.providers = providers

    def get_inputs(self) -> list[object]:
        return [object()]

    def get_outputs(self) -> list[object]:
        return [object()]

    def get_providers(self) -> list[str]:
        return self.providers


class _FakePipeline:
    def __init__(self, **options) -> None:
        self.options = options

    def predict(self, input):
        return [
            types.SimpleNamespace(
                json={
                    "res": {
                        "rec_texts": ["34ABC123"],
                        "rec_scores": [0.91],
                        "rec_boxes": [[1, 2, 80, 20]],
                    }
                }
            )
            for _image in input
        ]


class OcrBackendTests(unittest.TestCase):
    def test_auto_prefers_onnx_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("app.ocr_models.validate_ocr_models") as validate:
                selection = select_ocr_backend(root, "auto")

        self.assertIs(selection.backend, OcrBackend.ONNX)
        self.assertEqual(validate.call_count, 1)
        self.assertIs(validate.call_args.kwargs["backend"], OcrBackend.ONNX)

    def test_auto_falls_back_to_paddle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def validate(_root, *, backend, load_onnx):
                if backend is OcrBackend.ONNX:
                    raise OcrModelInvalid("broken ONNX runtime")
                self.assertFalse(load_onnx)
                return ()

            with patch("app.ocr_models.validate_ocr_models", side_effect=validate):
                selection = select_ocr_backend(root, "auto")

        self.assertIs(selection.backend, OcrBackend.PADDLE)

    def test_paddle_backend_rejects_missing_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(OcrModelNotFound):
                PaddleOcrProvider(
                    Path(temporary),
                    backend="paddle",
                    pipeline_factory=_FakePipeline,
                )

    def test_legacy_onnx_layout_remains_supported(self) -> None:
        fake_ort = types.SimpleNamespace(
            get_available_providers=lambda: ["CPUExecutionProvider"],
            InferenceSession=_FakeOnnxSession,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_onnx_models(root)
            (root / "onnx" / "detection").mkdir(parents=True)
            (root / "onnx" / "recognition").mkdir(parents=True)
            with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
                selection = select_ocr_backend(root, "onnx")

        self.assertIs(selection.backend, OcrBackend.ONNX)
        self.assertEqual(selection.model_root, root.resolve())

    def test_native_provider_is_offline_and_returns_ocr_segments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paddle_root = root / "paddle"
            _write_paddle_models(paddle_root)
            pipelines: list[_FakePipeline] = []

            def factory(**options):
                pipeline = _FakePipeline(**options)
                pipelines.append(pipeline)
                return pipeline

            provider = PaddleOcrProvider(
                root,
                backend="paddle",
                pipeline_factory=factory,
            )
            segments = provider.recognize(
                [
                    np.zeros((32, 128, 3), dtype=np.uint8),
                    np.zeros((48, 192, 3), dtype=np.uint8),
                    np.zeros((64, 256, 3), dtype=np.uint8),
                ]
            )

        self.assertIs(provider.backend, OcrBackend.PADDLE)
        self.assertEqual(os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"], "True")
        self.assertEqual(pipelines[0].options["device"], "cpu")
        self.assertFalse(pipelines[0].options["enable_hpi"])
        self.assertFalse(pipelines[0].options["enable_mkldnn"])
        self.assertNotIn("engine", pipelines[0].options)
        self.assertEqual(
            Path(pipelines[0].options["text_detection_model_dir"]),
            (paddle_root / "detection").resolve(),
        )
        self.assertEqual(
            Path(pipelines[0].options["text_recognition_model_dir"]),
            (paddle_root / "recognition").resolve(),
        )
        self.assertEqual(
            pipelines[0].options["text_detection_model_name"],
            "fake",
        )
        self.assertEqual(
            pipelines[0].options["text_recognition_model_name"],
            "fake",
        )
        self.assertTrue(all(isinstance(segment, OcrSegment) for segment in segments))
        self.assertEqual([segment.text for segment in segments], ["34ABC123"] * 3)
        self.assertEqual([segment.variant_index for segment in segments], [0, 1, 2])

    def test_onnx_provider_keeps_onnxruntime_engine(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            onnx_root = root / "onnx"
            _write_config(onnx_root / "detection")
            _write_config(onnx_root / "recognition")
            pipelines: list[_FakePipeline] = []

            def factory(**options):
                pipeline = _FakePipeline(**options)
                pipelines.append(pipeline)
                return pipeline

            with patch(
                "app.plate_recognition.select_ocr_backend",
                return_value=BackendSelection(OcrBackend.ONNX, onnx_root),
            ):
                provider = PaddleOcrProvider(
                    root,
                    backend="onnx",
                    pipeline_factory=factory,
                )

        self.assertIs(provider.backend, OcrBackend.ONNX)
        self.assertEqual(pipelines[0].options["engine"], "onnxruntime")
        self.assertFalse(pipelines[0].options["enable_hpi"])
        self.assertNotIn("enable_mkldnn", pipelines[0].options)


def _write_config(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "inference.yml").write_text(
        "Global:\n  model_name: fake\n",
        encoding="utf-8",
    )


def _write_onnx_models(root: Path) -> None:
    for folder in ("detection", "recognition"):
        directory = root / folder
        _write_config(directory)
        (directory / "inference.onnx").write_bytes(b"fake")


def _write_paddle_models(root: Path) -> None:
    for folder in ("detection", "recognition"):
        directory = root / folder
        _write_config(directory)
        (directory / "inference.json").write_text("{}", encoding="utf-8")
        (directory / "inference.pdiparams").write_bytes(b"fake")


if __name__ == "__main__":
    unittest.main()
