from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton

from app.auth import AuthService, Role
from app.camera import CameraService
from app.database import Database
from app.plate_capture import PlateCaptureService
from app.plate_service import PlateRecord, PlateService
from app.time_utils import to_local_datetime
from ui.display_helpers import display_timestamp
from ui.record_detail_dialog import MISSING_PHOTO_TEXT, RecordDetailDialog
from ui.records_widget import RecordsWidget
from ui.styles import APP_STYLESHEET


class RecordDetailDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])
        cls.application.setStyleSheet(APP_STYLESHEET)

    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_directory.name)
        self.database = Database(self.root / "detail.db")
        self.database.initialize()
        self.auth_service = AuthService(self.database)
        self.auth_service.ensure_default_admin()
        self.admin = self.auth_service.authenticate("admin", "admin123")
        self.user = self.auth_service.create_user(
            self.admin, "detail-user", "safe-pass-123", Role.USER
        )
        self.camera_service = CameraService(self.database)
        self.capture_service = PlateCaptureService(
            self.root / "data" / "captures",
            self.root,
        )
        self.plate_service = PlateService(
            self.database,
            duplicate_cooldown_seconds=10,
            capture_service=self.capture_service,
        )
        camera = self.camera_service.list_cameras()[0]
        self.record = self.plate_service.save_plate_detection(
            "34ABC123",
            camera.id,
            0.946,
            datetime(2026, 8, 12, 8, 21, 34, tzinfo=timezone.utc),
            np.full((300, 600, 3), 160, dtype=np.uint8),
        )
        self.widgets = []

    def tearDown(self) -> None:
        for widget in self.widgets:
            widget.close()
        self.application.processEvents()
        self.temp_directory.cleanup()

    def test_dialog_displays_photo_and_formatted_metadata(self) -> None:
        launcher = Mock()
        dialog = self._dialog(
            self.record,
            explorer_launcher=launcher,
            system_name_provider=lambda: "Windows",
        )

        self.assertEqual(dialog.plate_value.text(), "34 ABC 123")
        self.assertEqual(
            dialog.timestamp_value.text(), display_timestamp(self.record.timestamp)
        )
        self.assertEqual(dialog.direction_value.text(), "Giriş")
        self.assertEqual(dialog.camera_value.text(), self.record.camera_name)
        self.assertEqual(dialog.confidence_value.text(), "%94.6")
        self.assertIsNotNone(dialog.photo_label.pixmap())
        self.assertFalse(dialog.photo_label.pixmap().isNull())
        self.assertTrue(dialog.open_file_button.isEnabled())

        dialog.open_file_button.click()

        resolved = self.capture_service.resolve_reference(self.record.image_path)
        launcher.assert_called_once_with(
            ["explorer.exe", f"/select,{resolved}"],
            shell=False,
        )

        dialog.photo_label.resize(500, 300)
        self.application.processEvents()
        displayed = dialog.photo_label.pixmap().size()
        self.assertAlmostEqual(displayed.width() / displayed.height(), 2.0, places=1)

    def test_null_image_path_shows_placeholder_without_crashing(self) -> None:
        dialog = self._dialog(replace(self.record, image_path=None))

        self.assertEqual(dialog.photo_label.text(), MISSING_PHOTO_TEXT)
        self.assertFalse(dialog.open_file_button.isEnabled())

    def test_missing_capture_file_shows_placeholder_without_crashing(self) -> None:
        missing = replace(
            self.record,
            image_path="data/captures/2026/08/missing.jpg",
        )

        dialog = self._dialog(missing)

        self.assertEqual(dialog.photo_label.text(), MISSING_PHOTO_TEXT)
        self.assertFalse(dialog.open_file_button.isEnabled())

    def test_file_removed_after_preview_shows_warning_without_launching(self) -> None:
        launcher = Mock()
        dialog = self._dialog(
            self.record,
            explorer_launcher=launcher,
            system_name_provider=lambda: "Windows",
        )
        resolved = self.capture_service.resolve_reference(self.record.image_path)
        resolved.unlink()

        with patch("ui.record_detail_dialog.QMessageBox.warning") as warning:
            dialog.open_file_button.click()

        launcher.assert_not_called()
        self.assertFalse(dialog.open_file_button.isEnabled())
        warning.assert_called_once()
        self.assertEqual(warning.call_args.args[2], "Fotoğraf dosyası bulunamadı.")

    def test_unsafe_capture_path_is_not_loaded_or_opened(self) -> None:
        unsafe = replace(self.record, image_path="../../outside.jpg")
        launcher = Mock()

        with patch("ui.record_detail_dialog.QPixmap") as pixmap:
            dialog = self._dialog(
                unsafe,
                explorer_launcher=launcher,
                system_name_provider=lambda: "Windows",
            )

        pixmap.assert_not_called()
        self.assertEqual(dialog.photo_label.text(), MISSING_PHOTO_TEXT)
        self.assertFalse(dialog.open_file_button.isEnabled())

        with patch("ui.record_detail_dialog.QMessageBox.warning"):
            dialog._open_file()
        launcher.assert_not_called()

    def test_absolute_path_outside_capture_root_is_not_opened(self) -> None:
        outside = self.root / "outside.jpg"
        outside.write_bytes(b"not-an-image")
        unsafe = replace(self.record, image_path=str(outside))
        launcher = Mock()
        dialog = self._dialog(
            unsafe,
            explorer_launcher=launcher,
            system_name_provider=lambda: "Windows",
        )

        with patch("ui.record_detail_dialog.QMessageBox.warning") as warning:
            dialog._open_file()

        launcher.assert_not_called()
        self.assertFalse(dialog.open_file_button.isEnabled())
        self.assertEqual(warning.call_args.args[2], "Fotoğraf dosyası bulunamadı.")

    def test_path_with_spaces_is_passed_as_one_safe_explorer_argument(self) -> None:
        source = self.capture_service.resolve_reference(self.record.image_path)
        spaced_directory = self.capture_service.capture_root / "folder with spaces"
        spaced_directory.mkdir()
        spaced_path = spaced_directory / "vehicle photo.jpg"
        shutil.copyfile(source, spaced_path)
        spaced_record = replace(
            self.record,
            image_path=spaced_path.relative_to(self.root).as_posix(),
        )
        launcher = Mock()
        dialog = self._dialog(
            spaced_record,
            explorer_launcher=launcher,
            system_name_provider=lambda: "Windows",
        )

        dialog.open_file_button.click()

        launcher.assert_called_once_with(
            ["explorer.exe", f"/select,{spaced_path}"],
            shell=False,
        )

    def test_non_windows_environment_does_not_launch_process(self) -> None:
        launcher = Mock()
        dialog = self._dialog(
            self.record,
            explorer_launcher=launcher,
            system_name_provider=lambda: "Linux",
        )

        with patch("ui.record_detail_dialog.QMessageBox.warning") as warning:
            dialog.open_file_button.click()

        launcher.assert_not_called()
        self.assertIn("yalnızca Windows", warning.call_args.args[2])

    def test_daily_archive_and_records_table_do_not_eager_load_images(self) -> None:
        widget = self._records_widget(self.admin)

        with patch("ui.record_detail_dialog.QPixmap") as pixmap:
            widget.refresh()
            self.assertEqual(widget.table.columnCount(), 5)
            self.assertEqual(widget.table.cellWidget(0, 4).text(), "Aç")
            self._open_record_day(widget, self.record)

        pixmap.assert_not_called()
        row = self._row_for_record(widget, self.record.id)
        open_button = widget.table.cellWidget(row, 5)
        self.assertIsInstance(open_button, QPushButton)
        self.assertEqual(open_button.text(), "Aç")

    def test_record_without_image_shows_dash_in_photo_column(self) -> None:
        camera = self.camera_service.list_cameras()[0]
        record_without_image = self.plate_service.save_plate_detection(
            "06NOPHOTO",
            camera.id,
            0.91,
            datetime(2026, 8, 12, 8, 22, 0, tzinfo=timezone.utc),
        )
        widget = self._records_widget(self.admin)

        widget.refresh()
        self._open_record_day(widget, record_without_image)

        row = self._row_for_record(widget, record_without_image.id)
        self.assertIsNone(widget.table.cellWidget(row, 5))
        self.assertEqual(widget.table.item(row, 5).text(), "-")

    def test_photo_button_opens_existing_dialog_for_correct_record(self) -> None:
        widget = self._records_widget(self.user)
        widget.refresh()
        self._open_record_day(widget, self.record)
        row = self._row_for_record(widget, self.record.id)

        with patch("ui.records_widget.RecordDetailDialog") as dialog_factory:
            widget.table.cellWidget(row, 5).click()

        dialog_factory.assert_called_once()
        self.assertEqual(dialog_factory.call_args.args[0].id, self.record.id)
        dialog_factory.return_value.exec.assert_called_once()

    def test_photo_buttons_keep_record_mapping_after_sort_and_refresh(self) -> None:
        camera = self.camera_service.list_cameras()[0]
        second_record = self.plate_service.save_plate_detection(
            "06PHOTO2",
            camera.id,
            0.92,
            datetime(2026, 8, 12, 8, 23, 0, tzinfo=timezone.utc),
            np.full((300, 600, 3), 120, dtype=np.uint8),
        )
        widget = self._records_widget(self.user)
        widget.table.setSortingEnabled(True)
        widget.refresh()
        self._open_record_day(widget, self.record)
        widget.table.sortItems(0, Qt.SortOrder.DescendingOrder)

        for expected_record in (self.record, second_record):
            with self.subTest(record_id=expected_record.id), patch(
                "ui.records_widget.RecordDetailDialog"
            ) as dialog_factory:
                row = self._row_for_record(widget, expected_record.id)
                widget.table.cellWidget(row, 5).click()

                dialog_factory.assert_called_once()
                self.assertEqual(dialog_factory.call_args.args[0].id, expected_record.id)

        widget.search_input.setText(second_record.plate)
        with patch("ui.records_widget.RecordDetailDialog") as dialog_factory:
            widget.refresh()
            row = self._row_for_record(widget, second_record.id)
            widget.table.cellWidget(row, 5).click()

        self.assertEqual(widget.table.rowCount(), 1)
        self.assertEqual(dialog_factory.call_args.args[0].id, second_record.id)

    def test_user_can_open_correct_record_by_double_clicking_row(self) -> None:
        widget = self._records_widget(self.user)
        widget.refresh()
        self._open_record_day(widget, self.record)
        widget.resize(900, 500)
        widget.show()
        self.application.processEvents()
        first_item = widget.table.item(0, 0)
        click_position = widget.table.visualItemRect(first_item).center()

        with patch("ui.records_widget.RecordDetailDialog") as dialog_factory:
            QTest.mouseClick(
                widget.table.viewport(),
                Qt.MouseButton.LeftButton,
                pos=click_position,
            )
            QTest.mouseDClick(
                widget.table.viewport(),
                Qt.MouseButton.LeftButton,
                pos=click_position,
            )
            self.application.processEvents()

        dialog_factory.assert_called_once()
        opened_record = dialog_factory.call_args.args[0]
        self.assertEqual(opened_record.id, self.record.id)
        dialog_factory.return_value.exec.assert_called_once()

    def test_day_detail_only_shows_selected_day_and_back_returns_to_archive(self) -> None:
        camera = self.camera_service.list_cameras()[0]
        previous_day = self.plate_service.save_plate_detection(
            "06OLD123",
            camera.id,
            0.91,
            datetime(2026, 8, 11, 8, 21, 34, tzinfo=timezone.utc),
        )
        widget = self._records_widget(self.user)

        widget.refresh()

        self.assertEqual(widget.table.rowCount(), 2)
        self._open_record_day(widget, self.record)
        self.assertEqual(widget.table.rowCount(), 1)
        record_id = widget.table.item(0, 0).data(Qt.ItemDataRole.UserRole)
        self.assertEqual(record_id, self.record.id)
        self.assertNotEqual(record_id, previous_day.id)
        self.assertFalse(widget.back_button.isHidden())

        widget._show_archive()

        self.assertEqual(widget.table.columnCount(), 5)
        self.assertEqual(widget.table.rowCount(), 2)
        self.assertTrue(widget.back_button.isHidden())

    def test_archive_search_and_direction_filter_apply_to_day_detail(self) -> None:
        exit_camera = next(
            camera
            for camera in self.camera_service.list_cameras()
            if camera.direction.value == "EXIT"
        )
        exit_record = self.plate_service.save_plate_detection(
            "06EXIT99",
            exit_camera.id,
            0.92,
            datetime(2026, 8, 12, 8, 25, 0, tzinfo=timezone.utc),
        )
        widget = self._records_widget(self.user)
        widget.search_input.setText("06 EXIT")

        widget.refresh()

        self.assertEqual(widget.table.rowCount(), 1)
        self.assertEqual(widget.table.item(0, 1).text(), "1")
        widget.search_input.clear()
        widget.direction_filter.setCurrentIndex(
            widget.direction_filter.findData(exit_camera.direction)
        )
        self.assertEqual(widget.table.item(0, 2).text(), "0")
        self.assertEqual(widget.table.item(0, 3).text(), "1")

        self._open_record_day(widget, exit_record)

        self.assertEqual(widget.table.rowCount(), 1)
        self.assertEqual(
            widget.table.item(0, 0).data(Qt.ItemDataRole.UserRole),
            exit_record.id,
        )

    def _dialog(self, record: PlateRecord, **kwargs: object) -> RecordDetailDialog:
        dialog = RecordDetailDialog(record, self.plate_service, **kwargs)
        self.widgets.append(dialog)
        dialog.show()
        self.application.processEvents()
        return dialog

    def _records_widget(self, actor: object) -> RecordsWidget:
        widget = RecordsWidget(self.plate_service, actor)
        self.widgets.append(widget)
        return widget

    @staticmethod
    def _open_record_day(widget: RecordsWidget, record: PlateRecord) -> None:
        widget._open_day(to_local_datetime(record.timestamp).date())

    @staticmethod
    def _row_for_record(widget: RecordsWidget, record_id: int) -> int:
        for row in range(widget.table.rowCount()):
            item = widget.table.item(row, 0)
            if item.data(Qt.ItemDataRole.UserRole) == record_id:
                return row
        raise AssertionError(f"Record row not found: {record_id}")


if __name__ == "__main__":
    unittest.main()
