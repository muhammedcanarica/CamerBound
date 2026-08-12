from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app.auth import AuthService, Role
from app.camera import CameraService
from app.database import Database
from app.plate_capture import PlateCaptureService
from app.plate_service import PlateRecord, PlateService
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
        dialog = self._dialog(self.record)

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

        with patch(
            "ui.record_detail_dialog.QDesktopServices.openUrl", return_value=True
        ) as open_url:
            dialog.open_file_button.click()
        open_url.assert_called_once()

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

    def test_unsafe_capture_path_is_not_loaded_or_opened(self) -> None:
        unsafe = replace(self.record, image_path="../../outside.jpg")

        with patch("ui.record_detail_dialog.QPixmap") as pixmap:
            dialog = self._dialog(unsafe)

        pixmap.assert_not_called()
        self.assertEqual(dialog.photo_label.text(), MISSING_PHOTO_TEXT)
        self.assertFalse(dialog.open_file_button.isEnabled())

    def test_records_table_does_not_eager_load_images(self) -> None:
        widget = self._records_widget(self.admin)

        with patch("ui.record_detail_dialog.QPixmap") as pixmap:
            widget.refresh()

        pixmap.assert_not_called()
        self.assertEqual(widget.table.item(0, 5).text(), "Var")

    def test_user_can_open_correct_record_by_double_clicking_row(self) -> None:
        widget = self._records_widget(self.user)
        widget.refresh()
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

    def _dialog(self, record: PlateRecord) -> RecordDetailDialog:
        dialog = RecordDetailDialog(record, self.plate_service)
        self.widgets.append(dialog)
        dialog.show()
        self.application.processEvents()
        return dialog

    def _records_widget(self, actor: object) -> RecordsWidget:
        widget = RecordsWidget(self.plate_service, actor)
        self.widgets.append(widget)
        return widget


if __name__ == "__main__":
    unittest.main()
