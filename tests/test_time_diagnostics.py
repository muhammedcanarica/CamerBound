from __future__ import annotations

import locale
import os
import subprocess
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app.auth import AuthService
from app.camera import CameraService
from app.database import Database
from app.plate_service import PlateService
from app.time_diagnostics import (
    TimeDiagnosticsResult,
    TimeDiagnosticsService,
    TimeSyncStatus,
    format_utc_offset,
)
from ui.admin_widget import CameraSettingsWidget


FIXED_UTC = datetime(2026, 8, 12, 6, 10, 22, tzinfo=timezone.utc)


class RecordingRunner:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def __call__(self, args: list[str], **kwargs: object):
        self.calls.append((args, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def process(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr="")


def windows_service(runner: RecordingRunner) -> TimeDiagnosticsService:
    return TimeDiagnosticsService(
        runner=runner,
        system_name_provider=lambda: "Windows",
        now_provider=lambda: FIXED_UTC,
    )


class TimeDiagnosticsServiceTests(unittest.TestCase):
    def test_local_and_utc_datetimes_are_generated(self) -> None:
        runner = RecordingRunner([process("time.windows.com\n"), process("status")])

        result = windows_service(runner).check()

        self.assertEqual(result.utc_time, FIXED_UTC)
        self.assertIsNotNone(result.local_time.tzinfo)
        self.assertEqual(result.local_time.astimezone(timezone.utc), FIXED_UTC)
        self.assertEqual(result.checked_at, FIXED_UTC)

    def test_utc_offset_is_formatted_without_a_hardcoded_timezone(self) -> None:
        self.assertEqual(format_utc_offset(timedelta(hours=3)), "+03:00")
        self.assertEqual(format_utc_offset(timedelta(hours=-3, minutes=-30)), "-03:30")
        self.assertEqual(format_utc_offset(timedelta(0)), "+00:00")

    def test_external_source_is_reported_as_synced(self) -> None:
        runner = RecordingRunner([process("0.pool.ntp.org\r\n"), process("status")])

        result = windows_service(runner).check()

        self.assertEqual(result.time_source, "0.pool.ntp.org")
        self.assertIs(result.sync_status, TimeSyncStatus.SYNCED)
        self.assertTrue(result.windows_time_available)

    def test_local_cmos_clock_is_warning(self) -> None:
        runner = RecordingRunner([process("Local CMOS Clock"), process("status")])

        result = windows_service(runner).check()

        self.assertIs(result.sync_status, TimeSyncStatus.WARNING)
        self.assertIn("Harici zaman kaynağı", result.status_message)

    def test_free_running_system_clock_is_warning(self) -> None:
        runner = RecordingRunner(
            [process("Free-running System Clock"), process("status")]
        )

        result = windows_service(runner).check()

        self.assertIs(result.sync_status, TimeSyncStatus.WARNING)

    def test_timeout_returns_unavailable(self) -> None:
        runner = RecordingRunner(
            [subprocess.TimeoutExpired(["w32tm"], 4.0)]
        )

        result = windows_service(runner).check()

        self.assertIs(result.sync_status, TimeSyncStatus.UNAVAILABLE)
        self.assertFalse(result.windows_time_available)

    def test_missing_command_returns_unavailable(self) -> None:
        runner = RecordingRunner([FileNotFoundError("w32tm")])

        result = windows_service(runner).check()

        self.assertIs(result.sync_status, TimeSyncStatus.UNAVAILABLE)

    def test_non_zero_exit_code_returns_safe_result(self) -> None:
        runner = RecordingRunner([process("", 1), process("", 1)])

        result = windows_service(runner).check()

        self.assertIs(result.sync_status, TimeSyncStatus.UNAVAILABLE)
        self.assertIn("Plaka kaydı devam ediyor", result.status_message)

    def test_non_windows_environment_does_not_run_commands(self) -> None:
        runner = RecordingRunner([])
        service = TimeDiagnosticsService(
            runner=runner,
            system_name_provider=lambda: "Linux",
            now_provider=lambda: FIXED_UTC,
        )

        result = service.check()

        self.assertIs(result.sync_status, TimeSyncStatus.UNAVAILABLE)
        self.assertEqual(runner.calls, [])
        self.assertIn("yalnızca Windows", result.status_message)

    def test_subprocess_uses_read_only_argument_lists_without_shell(self) -> None:
        runner = RecordingRunner([process("time.windows.com"), process("status")])

        windows_service(runner).check()

        self.assertEqual(
            [call[0] for call in runner.calls],
            [
                ["w32tm", "/query", "/source"],
                ["w32tm", "/query", "/status"],
            ],
        )
        for _args, kwargs in runner.calls:
            self.assertIs(kwargs["shell"], False)
            self.assertEqual(kwargs["timeout"], 4.0)
            self.assertEqual(
                kwargs["encoding"],
                locale.getpreferredencoding(False) or "utf-8",
            )
            self.assertEqual(kwargs["errors"], "replace")
            self.assertEqual(kwargs["creationflags"], getattr(subprocess, "CREATE_NO_WINDOW", 0))


class FakeTimeDiagnosticsService:
    def __init__(
        self,
        result: TimeDiagnosticsResult,
        *,
        started: threading.Event | None = None,
        release: threading.Event | None = None,
    ) -> None:
        self.result = result
        self.started = started
        self.release = release
        self.calls = 0

    def check(self) -> TimeDiagnosticsResult:
        self.calls += 1
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            self.release.wait(timeout=2)
        return self.result


class TimeDiagnosticsUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        database = Database(Path(self.temp_directory.name) / "time-ui.db")
        database.initialize()
        auth_service = AuthService(database)
        auth_service.ensure_default_admin()
        self.admin = auth_service.authenticate("admin", "admin123")
        self.camera_service = CameraService(database)
        self.plate_service = PlateService(database, duplicate_cooldown_seconds=10)
        self.result = TimeDiagnosticsResult(
            local_time=FIXED_UTC.astimezone(timezone(timedelta(hours=3))),
            utc_time=FIXED_UTC,
            timezone_name="Turkey Standard Time",
            utc_offset="+03:00",
            windows_time_available=True,
            time_source="time.windows.com",
            sync_status=TimeSyncStatus.SYNCED,
            status_message="Saat senkronizasyonu aktif.",
            raw_status="status",
            checked_at=FIXED_UTC,
        )
        self.widget: CameraSettingsWidget | None = None

    def tearDown(self) -> None:
        if self.widget is not None:
            self.widget.close()
            self.widget.deleteLater()
        self.application.processEvents()
        self.temp_directory.cleanup()

    def _build_widget(self, service: FakeTimeDiagnosticsService) -> CameraSettingsWidget:
        self.widget = CameraSettingsWidget(
            self.camera_service,
            self.admin,
            self.plate_service,
            time_diagnostics_service=service,
        )
        self.widget.show()
        self.application.processEvents()
        return self.widget

    def _wait_for_calls(self, service: FakeTimeDiagnosticsService, expected: int) -> None:
        for _ in range(100):
            self.application.processEvents()
            if service.calls == expected and self.widget.refresh_time_button.isEnabled():
                return
            QTest.qWait(10)
        self.fail(f"Diagnostic call count did not reach {expected}")

    def test_refresh_button_runs_diagnostics_again(self) -> None:
        service = FakeTimeDiagnosticsService(self.result)
        widget = self._build_widget(service)

        QTest.mouseClick(widget.refresh_time_button, Qt.MouseButton.LeftButton)
        self._wait_for_calls(service, 1)
        QTest.mouseClick(widget.refresh_time_button, Qt.MouseButton.LeftButton)
        self._wait_for_calls(service, 2)

        self.assertEqual(widget.time_source_label.text(), "time.windows.com")
        self.assertIn("senkronizasyon", widget.time_status_label.text().casefold())

    def test_diagnostic_query_does_not_block_ui_thread(self) -> None:
        started = threading.Event()
        release = threading.Event()
        service = FakeTimeDiagnosticsService(
            self.result,
            started=started,
            release=release,
        )
        widget = self._build_widget(service)

        before = time.perf_counter()
        widget._refresh_time_diagnostics()
        elapsed = time.perf_counter() - before

        self.assertLess(elapsed, 0.2)
        self.assertTrue(started.wait(timeout=1))
        self.assertEqual(widget.time_status_label.text(), "Saat durumu kontrol ediliyor...")
        self.assertFalse(widget.refresh_time_button.isEnabled())

        release.set()
        self._wait_for_calls(service, 1)


if __name__ == "__main__":
    unittest.main()
