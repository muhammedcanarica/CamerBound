from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QSizePolicy

from app.auth import AuthService
from app.camera import CameraService
from app.config import load_config
from app.database import Database
from app.plate_recognition import PlateRecognitionService, RecognitionStatus
from app.plate_service import PlateService
from main import ApplicationController


class LoginFlowSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        database = Database(Path(self.temp_directory.name) / "ui-test.db")
        database.initialize()
        auth_service = AuthService(database)
        auth_service.ensure_default_admin()
        plate_service = PlateService(database, duplicate_cooldown_seconds=10)
        camera_service = CameraService(database)
        recognition_service = PlateRecognitionService(
            camera_service,
            plate_service,
            load_config().plate_recognition,
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


if __name__ == "__main__":
    unittest.main()
