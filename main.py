from __future__ import annotations

import sys
import logging

from PySide6.QtWidgets import QApplication, QMessageBox

from app.auth import AuthService, SessionUser
from app.camera import CameraService
from app.config import load_config
from app.database import Database
from app.logging_config import configure_logging
from app.plate_recognition import PlateRecognitionService
from app.plate_service import PlateService
from ui.dashboard_window import DashboardWindow
from ui.login_window import LoginWindow
from ui.styles import APP_STYLESHEET


LOGGER = logging.getLogger(__name__)


class ApplicationController:
    def __init__(
        self,
        auth_service: AuthService,
        plate_service: PlateService,
        camera_service: CameraService,
        recognition_service: PlateRecognitionService | None = None,
    ) -> None:
        self.auth_service = auth_service
        self.plate_service = plate_service
        self.camera_service = camera_service
        self.recognition_service = recognition_service
        self.login_window: LoginWindow | None = None
        self.dashboard_window: DashboardWindow | None = None

    def show_login(self) -> None:
        if self.dashboard_window is not None:
            self.dashboard_window.close()
            self.dashboard_window.deleteLater()
            self.dashboard_window = None
        if self.login_window is None:
            self.login_window = LoginWindow(self.auth_service)
            self.login_window.authenticated.connect(self.show_dashboard)
        self.login_window.reset()
        self.login_window.show()

    def show_dashboard(self, user: SessionUser) -> None:
        if self.login_window is not None:
            self.login_window.hide()
        if self.recognition_service is not None:
            self.recognition_service.start()
        self.dashboard_window = DashboardWindow(
            user,
            self.auth_service,
            self.plate_service,
            self.camera_service,
            self.recognition_service,
        )
        self.dashboard_window.logout_requested.connect(self.show_login)
        self.dashboard_window.show()


def build_services() -> tuple[
    AuthService,
    PlateService,
    CameraService,
    PlateRecognitionService,
]:
    config = load_config()
    database = Database(config.database_path)
    database.initialize()
    auth_service = AuthService(database)
    auth_service.ensure_default_admin()
    plate_service = PlateService(
        database,
        config.duplicate_cooldown_seconds,
        config.plate_recognition.record_retention_days,
    )
    apply_startup_retention_cleanup(plate_service)
    camera_service = CameraService(database)
    recognition_service = PlateRecognitionService(
        camera_service,
        plate_service,
        config.plate_recognition,
        settings_path=config.settings_path,
    )
    return auth_service, plate_service, camera_service, recognition_service


def apply_startup_retention_cleanup(plate_service: PlateService) -> int:
    try:
        removed = plate_service.apply_retention_policy()
    except Exception:
        LOGGER.exception("Retention cleanup failed; application startup will continue.")
        return 0
    if removed:
        LOGGER.info(
            "Retention cleanup removed %s plate records older than %s days.",
            removed,
            plate_service.record_retention_days,
        )
    return removed


def main() -> int:
    log_path = configure_logging()
    LOGGER.info("Application startup log_path=%s", log_path)
    application = QApplication(sys.argv)
    application.setApplicationName("Plaka Takip Sistemi")
    application.setStyle("Fusion")
    application.setStyleSheet(APP_STYLESHEET)

    try:
        services = build_services()
    except Exception as exc:
        QMessageBox.critical(
            None,
            "Başlatma hatası",
            f"Uygulama başlatılamadı:\n{exc}",
        )
        return 1

    controller = ApplicationController(*services)
    application.aboutToQuit.connect(services[2].stop_all)
    application.aboutToQuit.connect(services[3].stop)
    application.aboutToQuit.connect(
        lambda: LOGGER.info("Application shutdown")
    )
    controller.show_login()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
