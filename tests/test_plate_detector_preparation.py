from __future__ import annotations

import io
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from scripts import prepare_plate_detector as preparation


class PlateDetectorPreparationTests(unittest.TestCase):
    def test_missing_converter_environment_is_created_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            environment = Path(temporary) / "converter"
            converter_python = preparation._converter_python(environment)

            def fake_run(command: list[str], _message: str) -> None:
                self.assertEqual(command[:3], [preparation.sys.executable, "-m", "venv"])
                converter_python.parent.mkdir(parents=True)
                converter_python.write_bytes(b"python")

            with patch.object(preparation, "CONVERTER_ENV_DIR", environment), patch.object(
                preparation, "_run_command", side_effect=fake_run
            ) as run_command:
                first = preparation._ensure_converter_environment()
                second = preparation._ensure_converter_environment()

            self.assertEqual(first, converter_python)
            self.assertEqual(second, converter_python)
            run_command.assert_called_once()

    def test_missing_tensorflow_is_detected(self) -> None:
        failed = subprocess.CompletedProcess([], 1, stdout="", stderr="missing tensorflow")
        with patch.object(preparation.subprocess, "run", return_value=failed):
            ready = preparation._converter_dependencies_ready(Path("converter-python"))
        self.assertFalse(ready)

    def test_completed_model_download_cache_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            download_dir = Path(temporary)
            model_dir = (
                download_dir
                / "public"
                / preparation.MODEL_NAME
                / "model"
            )
            model_dir.mkdir(parents=True)
            (model_dir / "model.pb.frozen").write_bytes(b"tensorflow-model")
            (model_dir / "model.tfmo.json").write_bytes(b"{}")

            self.assertTrue(preparation._download_cache_ready(download_dir))

    def test_converter_dependency_install_uses_isolated_python_and_requirements(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            requirements = Path(temporary) / "requirements-model-converter.txt"
            requirements.write_text("tensorflow==2.18.0\n", encoding="utf-8")
            converter_python = Path(temporary) / "converter-python.exe"
            with patch.object(
                preparation, "CONVERTER_REQUIREMENTS", requirements
            ), patch.object(
                preparation,
                "_converter_dependencies_ready",
                side_effect=(False, True),
            ), patch.object(preparation, "_run_command") as run_command:
                preparation._ensure_converter_dependencies(converter_python)

            command = run_command.call_args.args[0]
            self.assertEqual(command[0], str(converter_python))
            self.assertEqual(command[1:4], ["-m", "pip", "install"])
            self.assertEqual(command[-2:], ["-r", str(requirements)])

    def test_provided_omz_tools_run_with_converter_python(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tools = Path(temporary)
            (tools / "downloader.py").write_text("", encoding="utf-8")
            (tools / "converter.py").write_text("", encoding="utf-8")
            converter_python = Path(temporary) / "converter-python.exe"

            downloader, converter = preparation._resolve_tools(
                tools,
                converter_python=converter_python,
            )

        self.assertEqual(downloader, [str(converter_python), str(tools / "downloader.py")])
        self.assertEqual(converter, [str(converter_python), str(tools / "converter.py")])

    def test_model_optimizer_entrypoint_is_resolved_from_converter_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            mo_entrypoint = Path(temporary) / "mo.py"
            mo_entrypoint.write_text("", encoding="utf-8")
            completed = subprocess.CompletedProcess(
                [],
                0,
                stdout=f"{mo_entrypoint}\n",
                stderr="",
            )
            with patch.object(preparation.subprocess, "run", return_value=completed) as run:
                resolved = preparation._resolve_model_optimizer(Path("converter-python"))

        self.assertEqual(resolved, mo_entrypoint.resolve())
        self.assertEqual(run.call_args.args[0][0], "converter-python")

    def test_prepare_copies_model_and_passes_converter_python_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            work = root / "work"
            converter_python = root / "converter-python.exe"
            calls: list[list[str]] = []

            def fake_run(command: list[str], _message: str) -> None:
                calls.append(command)
                if "--precisions" in command:
                    converted = work / "converted" / "public" / preparation.MODEL_NAME / "FP32"
                    converted.mkdir(parents=True)
                    (converted / f"{preparation.MODEL_NAME}.xml").write_bytes(b"xml")
                    (converted / f"{preparation.MODEL_NAME}.bin").write_bytes(b"bin")

            with patch.object(preparation, "TARGET_DIR", target), patch.object(
                preparation, "WORK_DIR", work
            ), patch.object(
                preparation,
                "_ensure_converter_environment",
                return_value=converter_python,
            ), patch.object(
                preparation, "_ensure_converter_dependencies"
            ), patch.object(
                preparation,
                "_resolve_model_optimizer",
                return_value=root / "mo.py",
            ), patch.object(
                preparation,
                "_resolve_tools",
                return_value=(["downloader"], ["converter"]),
            ), patch.object(preparation, "_run_command", side_effect=fake_run):
                preparation._prepare(None, force=True)

            converter_call = next(call for call in calls if "--precisions" in call)
            self.assertEqual(
                converter_call[converter_call.index("--python") + 1],
                str(converter_python),
            )
            self.assertEqual(
                converter_call[converter_call.index("--mo") + 1],
                str(root / "mo.py"),
            )
            self.assertEqual((target / "model.xml").read_bytes(), b"xml")
            self.assertEqual((target / "model.bin").read_bytes(), b"bin")

    def test_check_only_does_not_call_subprocess_or_pip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            (target / "model.xml").write_bytes(b"xml")
            (target / "model.bin").write_bytes(b"bin")
            with patch.object(preparation, "TARGET_DIR", target), patch.object(
                preparation.subprocess,
                "run",
                side_effect=AssertionError("check-only subprocess çağırmamalı"),
            ):
                result = preparation.main(["--check-only"])
        self.assertEqual(result, 0)

    def test_ready_model_skips_environment_download_and_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            (target / "model.xml").write_bytes(b"xml")
            (target / "model.bin").write_bytes(b"bin")
            with patch.object(preparation, "TARGET_DIR", target), patch.object(
                preparation,
                "_prepare",
                side_effect=AssertionError("hazır model yeniden hazırlanmamalı"),
            ):
                result = preparation.main([])
        self.assertEqual(result, 0)

    def test_failure_is_reported_without_traceback(self) -> None:
        stderr = io.StringIO()
        with patch.object(preparation, "_target_ready", return_value=False), patch.object(
            preparation,
            "_prepare",
            side_effect=preparation.PreparationError("TensorFlow kurulamadı"),
        ), redirect_stderr(stderr):
            result = preparation.main([])

        self.assertEqual(result, 1)
        self.assertIn("Hata: TensorFlow kurulamadı", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_production_requirements_and_runtime_have_no_converter_dependencies(self) -> None:
        requirements = (preparation.PROJECT_ROOT / "requirements.txt").read_text(
            encoding="utf-8"
        ).lower()
        main_source = (preparation.PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
        run_source = (preparation.PROJECT_ROOT / "run.bat").read_text(encoding="utf-8")

        self.assertNotIn("tensorflow", requirements)
        self.assertNotIn("openvino-dev", requirements)
        self.assertNotIn("prepare_plate_detector", main_source)
        self.assertNotIn("prepare_plate_detector", run_source)
        self.assertNotIn("pip install", main_source)
        self.assertNotIn("pip install", run_source)


if __name__ == "__main__":
    unittest.main()
