from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication, QMessageBox

from app.auth import AuthService, SessionUser
from app.camera import CameraService
from app.config import load_config
from app.database import Database
from app.plate_service import PlateService
from ui.dashboard_window import DashboardWindow
from ui.login_window import LoginWindow
from ui.styles import APP_STYLESHEET


class ApplicationController:
    def __init__(
        self,
        auth_service: AuthService,
        plate_service: PlateService,
        camera_service: CameraService,
    ) -> None:
        self.auth_service = auth_service
        self.plate_service = plate_service
        self.camera_service = camera_service
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
        self.dashboard_window = DashboardWindow(
            user,
            self.auth_service,
            self.plate_service,
            self.camera_service,
        )
        self.dashboard_window.logout_requested.connect(self.show_login)
        self.dashboard_window.show()


def build_services() -> tuple[AuthService, PlateService, CameraService]:
    config = load_config()
    database = Database(config.database_path)
    database.initialize()
    auth_service = AuthService(database)
    auth_service.ensure_default_admin()
    return (
        auth_service,
        PlateService(database, config.duplicate_cooldown_seconds),
        CameraService(database),
    )


def main() -> int:
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
    controller.show_login()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
