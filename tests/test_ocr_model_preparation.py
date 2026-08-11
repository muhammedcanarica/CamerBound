from __future__ import annotations

import io
import json
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.convert_paddle_models import ConversionToolsError, check_conversion_tools
from scripts.prepare_default_ocr_models import (
    DEFAULT_MODELS,
    ModelPreparationError,
    download_file,
    ensure_archive,
    prepare_default_models,
    safe_extract_tar,
)


class _FailingResponse:
    headers: dict[str, str] = {}

    def __init__(self) -> None:
        self.read_count = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _size: int) -> bytes:
        self.read_count += 1
        if self.read_count == 1:
            return b"partial"
        raise OSError("connection lost")


class OcrModelPreparationTests(unittest.TestCase):
    def test_dry_run_does_not_create_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary) / "new-project"

            lines = prepare_default_models(project_root, dry_run=True)

            self.assertFalse(project_root.exists())
            self.assertIn("PP-OCRv5_mobile_det", "\n".join(lines))
            self.assertIn("en_PP-OCRv5_mobile_rec", "\n".join(lines))

    def test_cached_archive_is_not_downloaded_again(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            download_root = Path(temporary)
            archive = download_root / DEFAULT_MODELS[0].archive_name
            archive.write_bytes(b"cached")
            calls: list[str] = []

            downloaded = ensure_archive(
                DEFAULT_MODELS[0],
                download_root,
                downloader=lambda url, _path: calls.append(url),
            )

            self.assertFalse(downloaded)
            self.assertEqual(calls, [])

    def test_force_download_replaces_cached_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            download_root = Path(temporary)
            archive = download_root / DEFAULT_MODELS[0].archive_name
            archive.write_bytes(b"old")
            calls: list[str] = []

            def fake_downloader(url: str, destination: Path) -> None:
                calls.append(url)
                destination.write_bytes(b"new")

            downloaded = ensure_archive(
                DEFAULT_MODELS[0],
                download_root,
                force=True,
                downloader=fake_downloader,
            )

            self.assertTrue(downloaded)
            self.assertEqual(len(calls), 1)
            self.assertEqual(archive.read_bytes(), b"new")

    def test_failed_download_does_not_leave_partial_or_final_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "model.tar"

            with self.assertRaisesRegex(ModelPreparationError, "Model download failed"):
                download_file(
                    "https://official.invalid/model.tar",
                    destination,
                    opener=lambda _url, timeout: _FailingResponse(),
                )

            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_name("model.tar.part").exists())

    def test_unsafe_archive_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "unsafe.tar"
            with tarfile.open(archive, "w") as tar:
                payload = b"escape"
                member = tarfile.TarInfo("../../escape.txt")
                member.size = len(payload)
                tar.addfile(member, io.BytesIO(payload))

            with self.assertRaisesRegex(ModelPreparationError, "Unsafe archive path"):
                safe_extract_tar(archive, root / "extract")

            self.assertFalse((root / "escape.txt").exists())

    def test_missing_conversion_tools_has_actionable_message(self) -> None:
        with self.assertRaisesRegex(
            ConversionToolsError,
            "python -m pip install -r requirements-model-tools.txt",
        ):
            check_conversion_tools(
                module_finder=lambda _name: None,
                executable_finder=lambda _name: None,
            )

    def test_failed_staging_preserves_existing_final_models(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            existing_model = (
                project_root / "models" / "ocr" / "onnx" / "detection" / "inference.onnx"
            )
            existing_model.parent.mkdir(parents=True)
            existing_model.write_bytes(b"working-model")

            def fail_install(_sources, _root) -> None:
                raise RuntimeError("staging failed")

            with self.assertRaisesRegex(RuntimeError, "staging failed"):
                prepare_default_models(
                    project_root,
                    backend="onnx",
                    force=True,
                    downloader=_write_fake_paddle_archive,
                    converter=_fake_converter,
                    installer=fail_install,
                    tool_checker=lambda: None,
                )

            self.assertEqual(existing_model.read_bytes(), b"working-model")
            self.assertFalse((project_root / "models" / "ocr-staging").exists())

    def test_successful_preparation_writes_model_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            verification_lines = [
                "Detection model: OK",
                "Recognition model: OK",
                "ONNX Runtime: OK",
                "PaddleOCR provider: OK",
                "OCR status: READY",
            ]

            lines = prepare_default_models(
                project_root,
                backend="onnx",
                downloader=_write_fake_paddle_archive,
                converter=_fake_converter,
                installer=_fake_installer,
                verifier=lambda _root, backend: (0, verification_lines),
                tool_checker=lambda: None,
            )

            model_root = project_root / "models" / "ocr" / "onnx"
            metadata = json.loads(
                (model_root / "model-info.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata,
                {
                    "detection": {"name": "PP-OCRv5_mobile_det"},
                    "recognition": {"name": "en_PP-OCRv5_mobile_rec"},
                    "format": "onnx",
                    "engine": "onnxruntime",
                },
            )
            for folder in ("detection", "recognition"):
                self.assertTrue((model_root / folder / "inference.onnx").is_file())
                self.assertTrue((model_root / folder / "inference.yml").is_file())
            self.assertIn("PaddleOCR provider: OK", lines)
            self.assertIn("OCR status: READY", lines)
            self.assertEqual(lines[-1], "OCR MODELS READY")

    def test_paddle_preparation_skips_conversion_and_writes_native_models(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project_root = Path(temporary)
            existing_onnx = (
                project_root / "models" / "ocr" / "onnx" / "detection" / "inference.onnx"
            )
            existing_onnx.parent.mkdir(parents=True)
            existing_onnx.write_bytes(b"keep-onnx")
            verification_lines = [
                "Detection model: OK",
                "Recognition model: OK",
                "PaddlePaddle: OK",
                "PaddleOCR provider: OK",
                "OCR backend: PADDLE",
                "OCR status: READY",
            ]

            lines = prepare_default_models(
                project_root,
                backend="paddle",
                downloader=_write_fake_paddle_archive,
                converter=lambda *_args: self.fail("Paddle preparation must not convert"),
                verifier=lambda _root, backend: (0, verification_lines),
                tool_checker=lambda: self.fail("Paddle preparation must not check conversion tools"),
            )

            model_root = project_root / "models" / "ocr" / "paddle"
            metadata = json.loads(
                (model_root / "model-info.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["format"], "paddle")
            self.assertEqual(metadata["engine"], "paddle")
            for folder in ("detection", "recognition"):
                self.assertTrue((model_root / folder / "inference.json").is_file())
                self.assertTrue((model_root / folder / "inference.pdiparams").is_file())
                self.assertTrue((model_root / folder / "inference.yml").is_file())
            self.assertIn("Detection Paddle model: OK", lines)
            self.assertIn("Recognition Paddle model: OK", lines)
            self.assertIn("OCR backend: PADDLE", lines)
            self.assertEqual(lines[-1], "OCR MODELS READY")
            self.assertEqual(existing_onnx.read_bytes(), b"keep-onnx")


def _write_fake_paddle_archive(_url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w") as tar:
        for filename, content in (
            ("inference.json", b"{}"),
            ("inference.pdiparams", b"parameters"),
            ("inference.yml", b"Global:\n  model_name: fake\n"),
        ):
            member = tarfile.TarInfo(f"fake-model/{filename}")
            member.size = len(content)
            tar.addfile(member, io.BytesIO(content))


def _fake_converter(_source: Path, output: Path, _opset: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "inference.onnx").write_bytes(b"onnx")
    (output / "inference.yml").write_text(
        "Global:\n  model_name: fake\n",
        encoding="utf-8",
    )


def _fake_installer(sources: dict[str, tuple[Path, str]], model_root: Path) -> None:
    for folder, (source, _label) in sources.items():
        target = model_root / folder
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "inference.onnx", target / "inference.onnx")
        shutil.copy2(source / "inference.yml", target / "inference.yml")


if __name__ == "__main__":
    unittest.main()
