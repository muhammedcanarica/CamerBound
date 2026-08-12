from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGroupBox,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QWidget,
)

from app.auth import AuthService, Role
from app.camera import CameraService
from app.config import load_config
from app.database import Database
from app.plate_capture import PlateCaptureService
from app.plate_recognition import PlateRecognitionService, RecognitionStatus
from app.plate_service import PlateService
from main import ApplicationController
from ui.admin_widget import CameraCredentialsDialog
from ui.styles import APP_STYLESHEET


class LoginFlowSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])
        cls.application.setStyleSheet(APP_STYLESHEET)

    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        database = Database(Path(self.temp_directory.name) / "ui-test.db")
        database.initialize()
        auth_service = AuthService(database)
        auth_service.ensure_default_admin()
        capture_service = PlateCaptureService(
            Path(self.temp_directory.name) / "data" / "captures",
            Path(self.temp_directory.name),
        )
        plate_service = PlateService(
            database,
            duplicate_cooldown_seconds=10,
            capture_service=capture_service,
        )
        camera_service = CameraService(database)
        recognition_config = replace(
            load_config().plate_recognition,
            model_root=Path(self.temp_directory.name) / "missing-ocr-models",
        )
        settings_path = Path(self.temp_directory.name) / "settings.json"
        settings_path.write_text(
            json.dumps(
                {
                    "database_path": "ui-test.db",
                    "plate_detection": {"record_retention_days": 90},
                }
            ),
            encoding="utf-8",
        )
        recognition_service = PlateRecognitionService(
            camera_service,
            plate_service,
            recognition_config,
            settings_path=settings_path,
        )
        self.controller = ApplicationController(
            auth_service,
            plate_service,
            camera_service,
            recognition_service,
        )

    def tearDown(self) -> None:
        if self.controller.dashboard_window is not None:
            self.controller.dashboard_window.close()
        if self.controller.login_window is not None:
            self.controller.login_window.close()
        self.application.processEvents()
        self.temp_directory.cleanup()

    def test_admin_login_opens_dashboard(self) -> None:
        self.controller.show_login()
        login = self.controller.login_window
        self.assertIsNotNone(login)
        login.username_input.setText("admin")
        login.password_input.setText("admin123")

        QTest.mouseClick(login.login_button, Qt.MouseButton.LeftButton)
        self.application.processEvents()

        dashboard = self.controller.dashboard_window
        self.assertIsNotNone(dashboard)
        dashboard.resize(1280, 720)
        self.application.processEvents()
        self.assertTrue(dashboard.isVisible())
        self.assertEqual(dashboard.user.username, "admin")
        self.assertEqual(dashboard.stack.count(), 5)
        self.assertTrue(dashboard.dashboard_home.recent_table.isVisible())
        for card in dashboard.dashboard_home.camera_cards.values():
            self.assertGreaterEqual(card.preview.minimumHeight(), 280)
            self.assertGreaterEqual(card.preview.height(), 280)
            self.assertEqual(
                card.preview.sizePolicy().horizontalPolicy(),
                QSizePolicy.Policy.Expanding,
            )
            self.assertEqual(
                card.preview.sizePolicy().verticalPolicy(),
                QSizePolicy.Policy.Expanding,
            )

        for _ in range(100):
            self.application.processEvents()
            if self.controller.recognition_service.get_status() is RecognitionStatus.UNAVAILABLE:
                break
            QTest.qWait(10)
        self.assertEqual(
            self.controller.recognition_service.get_status(),
            RecognitionStatus.UNAVAILABLE,
        )
        self.assertIn(
            "OCR: Kullanılamıyor",
            dashboard.dashboard_home.camera_cards[next(iter(dashboard.dashboard_home.camera_cards))]
            .ocr_status.text(),
        )

        users_page = dashboard.pages[3]
        for username, role in (("ui-user", Role.USER), ("ui-admin", Role.ADMIN)):
            with self.subTest(role=role):
                role_index = users_page.role_input.findData(role)
                users_page.role_input.setItemData(role_index, role.value)
                users_page.role_input.setCurrentIndex(role_index)
                users_page.username_input.setText(username)
                users_page.password_input.setText("safe-pass-123")

                with patch("ui.admin_widget.QMessageBox.information"):
                    users_page._create_user()

                created = self.controller.auth_service.authenticate(
                    username,
                    "safe-pass-123",
                )
                self.assertIs(created.role, role)
                table_usernames = {
                    users_page.table.item(row, 1).text()
                    for row in range(users_page.table.rowCount())
                }
                self.assertIn(username, table_usernames)

        camera_settings = dashboard.pages[4]
        dashboard.stack.setCurrentWidget(camera_settings)
        camera_settings.refresh()
        self.application.processEvents()
        self.assertIsInstance(camera_settings.scroll_area, QScrollArea)
        self.assertTrue(camera_settings.scroll_area.widgetResizable())
        self.assertEqual(
            camera_settings.scroll_area.frameShape(),
            QFrame.Shape.NoFrame,
        )
        self.assertIs(camera_settings.scroll_area.widget(), camera_settings.content_widget)
        self.assertEqual(
            camera_settings.scroll_area.viewport().grab().toImage().pixelColor(1, 1).name(),
            "#f4f7fb",
        )
        self.assertEqual(
            camera_settings.content_widget.grab().toImage().pixelColor(1, 1).name(),
            "#f4f7fb",
        )
        self.assertEqual(len(camera_settings._camera_buttons), 2)
        for editor in camera_settings._editors.values():
            self.assertEqual(len(editor), 4)
        for camera in self.controller.camera_service.list_cameras():
            credential_status = camera_settings.findChild(
                QLabel,
                f"cameraCredentialStatus-{camera.id}",
            )
            self.assertIsNotNone(credential_status)
            self.assertEqual(
                credential_status.text(),
                "⚠ Erişim bilgileri ayarlanmamış",
            )
            credential_button = camera_settings.findChild(
                QPushButton,
                f"cameraCredentialButton-{camera.id}",
            )
            self.assertIsNotNone(credential_button)
            self.assertEqual(credential_button.text(), "Erişim Bilgilerini Ayarla")
            credential_description = camera_settings.findChild(
                QLabel,
                f"cameraCredentialDescription-{camera.id}",
            )
            self.assertIsNotNone(credential_description)
            self.assertEqual(
                credential_description.text(),
                "CamerBound, kameraya bağlanmak için bu kullanıcı adı ve şifreyi "
                "kullanır.",
            )
            self.assertIsNone(
                camera_settings.findChild(QWidget, f"cameraUsernameInput-{camera.id}")
            )
            self.assertIsNone(
                camera_settings.findChild(QWidget, f"cameraPasswordInput-{camera.id}")
            )
            self.assertIsNone(
                camera_settings.findChild(
                    QWidget,
                    f"cameraClearCredentials-{camera.id}",
                )
            )
        for save_button, roi_button in camera_settings._camera_buttons.values():
            self.assertGreaterEqual(save_button.minimumHeight(), 36)
            self.assertGreaterEqual(roi_button.minimumHeight(), 36)
            self.assertEqual(
                save_button.sizePolicy().horizontalPolicy(),
                QSizePolicy.Policy.Expanding,
            )
            self.assertEqual(
                roi_button.sizePolicy().horizontalPolicy(),
                QSizePolicy.Policy.Expanding,
            )
        self.assertIsInstance(camera_settings.retention_group, QGroupBox)
        self.assertIsInstance(camera_settings.time_group, QGroupBox)
        self.assertEqual(camera_settings.time_group.title(), "Saat Durumu")
        self.assertIsInstance(camera_settings.audit_group, QGroupBox)
        self.assertEqual(camera_settings.audit_group.title(), "Güvenlik Günlüğü")
        self.assertGreater(camera_settings.audit_table.rowCount(), 0)
        self.assertEqual(
            [
                camera_settings.retention_combo.itemText(index)
                for index in range(camera_settings.retention_combo.count())
            ],
            ["30 gün", "90 gün", "180 gün", "Süresiz"],
        )
        self.assertEqual(
            [
                camera_settings.retention_combo.itemData(index)
                for index in range(camera_settings.retention_combo.count())
            ],
            [30, 90, 180, 0],
        )
        self.assertEqual(
            camera_settings.delete_all_records_button.text(),
            "Tüm Plaka Kayıtlarını Temizle",
        )

        camera_settings.retention_combo.setCurrentIndex(
            camera_settings.retention_combo.findData(180)
        )
        with patch("ui.admin_widget.QMessageBox.information"):
            camera_settings._save_retention_policy()
        self.assertEqual(self.controller.plate_service.record_retention_days, 180)
        self.assertEqual(
            load_config(camera_settings.settings_path).plate_recognition.record_retention_days,
            180,
        )

        self.controller.plate_service.set_record_retention_days(0)
        camera_settings.refresh()
        self.assertFalse(camera_settings.cleanup_old_button.isEnabled())
        self.assertIn("Süresiz", camera_settings.cleanup_old_button.toolTip())

    def test_camera_credentials_dialog_is_clear_secure_and_preserves_blank_password(self) -> None:
        admin = self.controller.auth_service.authenticate("admin", "admin123")
        camera = self.controller.camera_service.list_cameras()[0]
        configured = self.controller.camera_service.update_camera(
            admin,
            camera.id,
            camera.name,
            "http://CAMERA_IP/live",
            camera.direction,
            False,
            username="test-user",
            password="test-password",
        )
        original_protected_password = configured.protected_password

        self.controller.show_dashboard(admin)
        settings_page = self.controller.dashboard_window.pages[4]
        self.controller.dashboard_window.stack.setCurrentWidget(settings_page)
        settings_page.refresh()
        self.application.processEvents()

        status = settings_page.findChild(
            QLabel,
            f"cameraCredentialStatus-{camera.id}",
        )
        self.assertIsNotNone(status)
        self.assertEqual(status.text(), "✓ Erişim bilgileri kayıtlı")
        credential_button = settings_page.findChild(
            QPushButton,
            f"cameraCredentialButton-{camera.id}",
        )
        self.assertIsNotNone(credential_button)
        self.assertEqual(credential_button.text(), "Erişim Bilgilerini Güncelle")
        self.assertIsNone(
            settings_page.findChild(QWidget, f"cameraUsernameInput-{camera.id}")
        )
        self.assertIsNone(
            settings_page.findChild(QWidget, f"cameraPasswordInput-{camera.id}")
        )
        self.assertIsNone(
            settings_page.findChild(
                QWidget,
                f"cameraClearCredentials-{camera.id}",
            )
        )

        visible_text: list[str] = []
        for widget in settings_page.findChildren(QWidget):
            text_method = getattr(widget, "text", None)
            if callable(text_method):
                try:
                    visible_text.append(str(text_method()))
                except TypeError:
                    pass
            placeholder_method = getattr(widget, "placeholderText", None)
            if callable(placeholder_method):
                visible_text.append(str(placeholder_method()))
        rendered_text = "\n".join(visible_text)
        self.assertNotIn("test-user", rendered_text)
        self.assertNotIn("test-password", rendered_text)

        dialog = CameraCredentialsDialog(configured)
        self.assertEqual(dialog.windowTitle(), "Kamera Erişim Bilgileri")
        self.assertEqual(dialog.username_input.text(), "test-user")
        self.assertEqual(dialog.password_input.text(), "")
        self.assertIs(
            dialog.password_input.echoMode(),
            QLineEdit.EchoMode.Password,
        )
        dialog_text = "\n".join(
            label.text() for label in dialog.findChildren(QLabel)
        )
        self.assertIn(
            "Bu işlem kameranın kendi kullanıcı adı veya şifresini değiştirmez. "
            "Yalnızca CamerBound'un kameraya bağlanırken kullandığı bilgileri günceller.",
            dialog_text,
        )
        self.assertIn("Mevcut şifreyi korumak için boş bırakın.", dialog_text)
        self.assertIn("Kamera Kullanıcı Adı", dialog_text)
        self.assertIn("Bağlantı Şifresi", dialog_text)
        self.assertNotIn("test-password", dialog_text)
        for forbidden_text in (
            "Şifre Değiştir",
            "Kamera Şifresini Değiştir",
            "Kimlik Bilgilerini Temizle",
        ):
            self.assertNotIn(forbidden_text, rendered_text + dialog_text)
        self.assertEqual(
            dialog.buttons.button(QDialogButtonBox.StandardButton.Cancel).text(),
            "İptal",
        )
        self.assertEqual(
            dialog.buttons.button(QDialogButtonBox.StandardButton.Save).text(),
            "Kaydet",
        )
        dialog.close()

        with patch("ui.admin_widget.CameraCredentialsDialog") as dialog_factory, patch(
            "ui.admin_widget.QMessageBox.information"
        ), patch("ui.admin_widget.QMessageBox.warning") as warning:
            credential_dialog = dialog_factory.return_value
            credential_dialog.exec.return_value = QDialog.DialogCode.Accepted
            credential_dialog.username_input.text.return_value = "test-user"
            credential_dialog.password_input.text.return_value = ""
            settings_page._open_camera_credentials(camera.id)

        warning.assert_not_called()
        credential_dialog.password_input.clear.assert_called_once()
        preserved = self.controller.camera_service.get_camera(camera.id)
        self.assertEqual(preserved.username, configured.username)
        self.assertEqual(
            preserved.protected_password,
            original_protected_password,
        )
        self.assertEqual(preserved.stream_url, configured.stream_url)

        with patch("ui.admin_widget.QMessageBox.information"), patch(
            "ui.admin_widget.QMessageBox.warning"
        ) as warning:
            settings_page._save_camera(camera.id)

        warning.assert_not_called()
        preserved = self.controller.camera_service.get_camera(camera.id)
        self.assertEqual(preserved.username, configured.username)
        self.assertEqual(
            preserved.protected_password,
            original_protected_password,
        )
        self.assertEqual(preserved.stream_url, configured.stream_url)

        _name, source, _direction, _enabled = settings_page._editors[camera.id]
        source.setText(
            "http://other-user:other-password@CAMERA_IP/live"
        )
        with patch("ui.admin_widget.QMessageBox.warning") as warning:
            settings_page._save_camera(camera.id)

        warning.assert_called_once()
        self.assertEqual(source.text(), configured.stream_url)
        rejected = self.controller.camera_service.get_camera(camera.id)
        self.assertEqual(rejected.username, configured.username)
        self.assertEqual(
            rejected.protected_password,
            original_protected_password,
        )
        self.assertEqual(rejected.stream_url, configured.stream_url)

    def test_camera_credentials_dialog_saves_new_password_with_existing_service(self) -> None:
        admin = self.controller.auth_service.authenticate("admin", "admin123")
        camera = self.controller.camera_service.list_cameras()[0]
        configured = self.controller.camera_service.update_camera(
            admin,
            camera.id,
            camera.name,
            "http://CAMERA_IP/live",
            camera.direction,
            False,
            username="test-user",
            password="test-password",
        )
        self.controller.show_dashboard(admin)
        settings_page = self.controller.dashboard_window.pages[4]
        settings_page.refresh()

        with patch("ui.admin_widget.CameraCredentialsDialog") as dialog_factory, patch(
            "ui.admin_widget.QMessageBox.information"
        ), patch("ui.admin_widget.QMessageBox.warning") as warning:
            dialog = dialog_factory.return_value
            dialog.exec.return_value = QDialog.DialogCode.Accepted
            dialog.username_input.text.return_value = "test-user"
            dialog.password_input.text.return_value = "new-test-password"
            settings_page._open_camera_credentials(camera.id)

        warning.assert_not_called()
        updated = self.controller.camera_service.get_camera(camera.id)
        self.assertNotEqual(updated.protected_password, configured.protected_password)
        self.assertNotEqual(updated.protected_password, "new-test-password")
        self.assertNotIn("new-test-password", updated.protected_password)

    def test_user_dashboard_has_no_admin_retention_controls(self) -> None:
        admin = self.controller.auth_service.authenticate("admin", "admin123")
        user = self.controller.auth_service.create_user(
            admin,
            "ui-operator",
            "safe-pass-123",
            Role.USER,
        )

        self.controller.show_dashboard(user)
        self.application.processEvents()

        dashboard = self.controller.dashboard_window
        self.assertEqual(dashboard.stack.count(), 3)
        self.assertFalse(
            any(
                button.objectName().startswith("cameraCredentialButton-")
                for button in dashboard.findChildren(QPushButton)
            )
        )
        self.assertIsNone(dashboard.findChild(QGroupBox, "dataRetentionGroup"))
        self.assertIsNone(
            dashboard.findChild(QPushButton, "deleteAllPlateRecordsButton")
        )
        self.assertIsNone(dashboard.findChild(QGroupBox, "securityAuditGroup"))
        self.assertIsNone(dashboard.findChild(QGroupBox, "timeDiagnosticsGroup"))

    def test_records_page_opens_detail_on_double_click(self) -> None:
        admin = self.controller.auth_service.authenticate("admin", "admin123")
        camera = self.controller.camera_service.list_cameras()[0]
        record = self.controller.plate_service.save_plate_detection(
            "34UIIMG1",
            camera.id,
            0.94,
            frame=np.zeros((100, 200, 3), dtype=np.uint8),
        )
        self.controller.show_dashboard(admin)
        records_page = self.controller.dashboard_window.pages[1]
        records_page.refresh()

        self.assertEqual(records_page.table.columnCount(), 5)
        day_button = records_page.table.cellWidget(0, 4)
        self.assertIsInstance(day_button, QPushButton)
        self.assertEqual(day_button.text(), "Aç")
        day_button.click()
        self.assertEqual(records_page.table.columnCount(), 6)
        self.assertEqual(records_page.table.horizontalHeaderItem(5).text(), "Fotoğraf")
        photo_button = records_page.table.cellWidget(0, 5)
        self.assertIsInstance(photo_button, QPushButton)
        self.assertEqual(photo_button.text(), "Aç")
        self.controller.dashboard_window.stack.setCurrentWidget(records_page)
        self.application.processEvents()
        click_position = records_page.table.visualItemRect(
            records_page.table.item(0, 0)
        ).center()

        with patch("ui.records_widget.RecordDetailDialog") as dialog_factory:
            QTest.mouseClick(
                records_page.table.viewport(),
                Qt.MouseButton.LeftButton,
                pos=click_position,
            )
            QTest.mouseDClick(
                records_page.table.viewport(),
                Qt.MouseButton.LeftButton,
                pos=click_position,
            )
            self.application.processEvents()

        dialog_factory.assert_called_once()
        self.assertEqual(dialog_factory.call_args.args[0].id, record.id)
        dialog_factory.return_value.exec.assert_called_once()

    def test_delete_all_requires_exact_confirmation_text(self) -> None:
        admin = self.controller.auth_service.authenticate("admin", "admin123")
        self.controller.show_dashboard(admin)
        settings_page = self.controller.dashboard_window.pages[4]

        with patch.object(
            self.controller.plate_service,
            "delete_all_records",
        ) as delete_all, patch(
            "ui.admin_widget.QMessageBox.warning",
            return_value=QMessageBox.StandardButton.Ok,
        ), patch(
            "ui.admin_widget.QInputDialog.getText",
            return_value=("temizle", True),
        ):
            settings_page._delete_all_records()

        delete_all.assert_not_called()

    def test_combo_box_popup_has_readable_theme_colors(self) -> None:
        self.application.setStyleSheet(APP_STYLESHEET)
        self.controller.show_login()
        login = self.controller.login_window
        login.username_input.setText("admin")
        login.password_input.setText("admin123")

        QTest.mouseClick(login.login_button, Qt.MouseButton.LeftButton)
        self.application.processEvents()

        role_input = self.controller.dashboard_window.pages[3].role_input
        role_input.ensurePolished()
        role_input.view().ensurePolished()
        popup_palette = role_input.view().palette()

        self.assertEqual(
            popup_palette.color(popup_palette.ColorRole.Base).name(),
            "#ffffff",
        )
        self.assertEqual(
            popup_palette.color(popup_palette.ColorRole.Text).name(),
            "#172033",
        )
        self.assertEqual(
            popup_palette.color(popup_palette.ColorRole.Highlight).name(),
            "#3468d4",
        )
        self.assertEqual(
            popup_palette.color(popup_palette.ColorRole.HighlightedText).name(),
            "#ffffff",
        )

    def test_confirmation_dialogs_have_readable_light_theme(self) -> None:
        self.application.setStyleSheet(APP_STYLESHEET)
        dialogs = (
            QMessageBox(
                QMessageBox.Icon.Warning,
                "Tüm kayıtları temizle",
                "Tüm plaka hareket kayıtları kalıcı olarak silinecek.",
            ),
            QInputDialog(),
        )
        dialogs[1].setLabelText("Devam etmek için TEMIZLE yazın:")

        for dialog in dialogs:
            with self.subTest(dialog=type(dialog).__name__):
                dialog.ensurePolished()
                dialog.show()
                self.application.processEvents()
                label = next(
                    child
                    for child in dialog.findChildren(QLabel)
                    if child.text()
                )
                self.assertEqual(
                    dialog.palette().color(dialog.palette().ColorRole.Window).name(),
                    "#f4f7fb",
                )
                self.assertEqual(
                    label.palette().color(label.palette().ColorRole.WindowText).name(),
                    "#172033",
                )
                dialog.close()


if __name__ == "__main__":
    unittest.main()
