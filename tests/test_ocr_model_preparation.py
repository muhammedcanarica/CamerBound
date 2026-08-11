from __future__ import annotations

import io
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

            with self.assertRaisesRegex(ModelPreparationError, "Download failed"):
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


if __name__ == "__main__":
    unittest.main()
