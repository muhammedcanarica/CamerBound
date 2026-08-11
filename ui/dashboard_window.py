from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.auth import Role, SessionUser
from app.camera import Camera, CameraService, Direction
from app.plate_service import PlateRecord, PlateService
from ui.admin_widget import CameraSettingsWidget, UsersAdminWidget
from ui.records_widget import (
    InsideVehiclesWidget,
    RecordsWidget,
    display_timestamp,
    prepare_table,
)


class DashboardHome(QWidget):
    def __init__(
        self, plate_service: PlateService, camera_service: CameraService, user: SessionUser
    ) -> None:
        super().__init__()
        self.plate_service = plate_service
        self.camera_service = camera_service
        self.user = user
        self.camera_details: dict[Direction, tuple[QLabel, QLabel]] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(16)

        header = QHBoxLayout()
        header.addWidget(QLabel("Dashboard", objectName="pageTitle"))
        header.addStretch()
        refresh_button = QPushButton("Yenile")
        refresh_button.clicked.connect(self.refresh)
        header.addWidget(refresh_button)
        layout.addLayout(header)

        camera_layout = QHBoxLayout()
        camera_layout.setSpacing(16)
        camera_layout.addWidget(self._camera_card(Direction.ENTRY), 1)
        camera_layout.addWidget(self._camera_card(Direction.EXIT), 1)
        layout.addLayout(camera_layout, 2)

        layout.addWidget(QLabel("Son Hareketler", objectName="sectionTitle"))
        self.recent_table = QTableWidget()
        prepare_table(
            self.recent_table,
            ["Plaka", "İşlem", "Kamera", "Güven", "Tarih / Saat"],
        )
        layout.addWidget(self.recent_table, 3)

    def _camera_card(self, direction: Direction) -> QFrame:
        card = QFrame(objectName="cameraCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(18, 16, 18, 16)
        card_layout.setSpacing(9)
        title_text = "GİRİŞ KAMERASI" if direction is Direction.ENTRY else "ÇIKIŞ KAMERASI"
        card_layout.addWidget(QLabel(title_text, objectName="sectionTitle"))

        preview = QLabel("Kamera görüntüsü\nRTSP entegrasyonu bekleniyor")
        preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview.setMinimumHeight(150)
        preview.setStyleSheet(
            "background:#111827; color:#94a3b8; border-radius:7px; font-size:14px;"
        )
        status = QLabel("● Bağlı değil")
        status.setStyleSheet("color:#b42318; font-weight:600;")
        details = QLabel("Kamera yapılandırması yükleniyor", objectName="mutedLabel")
        last_plate = QLabel("Son okunan plaka: -")
        last_plate.setStyleSheet("font-weight:600;")
        card_layout.addWidget(preview, 1)
        card_layout.addWidget(status)
        card_layout.addWidget(details)
        card_layout.addWidget(last_plate)
        self.camera_details[direction] = (details, last_plate)
        return card

    def refresh(self) -> None:
        try:
            cameras = self.camera_service.list_cameras()
            records = self.plate_service.get_recent_records(self.user, 20)
        except (ValueError, PermissionError) as exc:
            QMessageBox.warning(self, "Dashboard yenilenemedi", str(exc))
            return

        for direction, (details_label, plate_label) in self.camera_details.items():
            camera = next((item for item in cameras if item.direction is direction), None)
            latest = next((item for item in records if item.direction is direction), None)
            details_label.setText(self._camera_description(camera))
            plate_label.setText(
                f"Son okunan plaka: {latest.plate}" if latest else "Son okunan plaka: -"
            )
        self._populate_recent(records)

    @staticmethod
    def _camera_description(camera: Camera | None) -> str:
        if camera is None:
            return "Kamera kaydı bulunamadı"
        state = "Aktif" if camera.enabled else "Pasif"
        url_state = "URL tanımlı" if camera.stream_url else "URL tanımlı değil"
        return f"{camera.name} · {state} · {url_state}"

    def _populate_recent(self, records: list[PlateRecord]) -> None:
        self.recent_table.setRowCount(len(records))
        for row_index, record in enumerate(records):
            values = (
                record.plate,
                "Giriş" if record.direction is Direction.ENTRY else "Çıkış",
                record.camera_name,
                f"%{record.confidence * 100:.1f}",
                display_timestamp(record.timestamp),
            )
            for column, value in enumerate(values):
                self.recent_table.setItem(row_index, column, QTableWidgetItem(value))


class DashboardWindow(QMainWindow):
    logout_requested = Signal()

    def __init__(
        self,
        user: SessionUser,
        auth_service: object,
        plate_service: PlateService,
        camera_service: CameraService,
    ) -> None:
        super().__init__()
        self.user = user
        self.auth_service = auth_service
        self.plate_service = plate_service
        self.camera_service = camera_service
        self.pages: list[QWidget] = []
        self.nav_buttons: list[QPushButton] = []
        self.setWindowTitle("Plaka Takip Sistemi")
        self.setMinimumSize(1080, 700)
        self.resize(1280, 820)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget(objectName="appRoot")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        top_bar = QFrame()
        top_bar.setStyleSheet("background:white; border-bottom:1px solid #dce3ee;")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(22, 12, 22, 12)
        top_layout.addWidget(QLabel("Plaka Takip Sistemi", objectName="appTitle"))
        top_layout.addStretch()
        user_label = QLabel(f"{self.user.username}  ·  {self.user.role.value}")
        user_label.setStyleSheet("font-weight:600;")
        logout_button = QPushButton("Çıkış")
        logout_button.setObjectName("secondaryButton")
        logout_button.clicked.connect(self.logout_requested.emit)
        top_layout.addWidget(user_label)
        top_layout.addSpacing(10)
        top_layout.addWidget(logout_button)
        root_layout.addWidget(top_bar)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        sidebar = QFrame(objectName="sidebar")
        sidebar.setFixedWidth(220)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(12, 18, 12, 18)
        sidebar_layout.setSpacing(5)
        sidebar_layout.addWidget(QLabel("MENÜ", objectName="sidebarTitle"))
        sidebar_layout.addSpacing(12)

        self.stack = QStackedWidget()
        self._add_page(
            sidebar_layout,
            "Dashboard",
            DashboardHome(self.plate_service, self.camera_service, self.user),
        )
        self._add_page(
            sidebar_layout,
            "Kayıtlar",
            RecordsWidget(self.plate_service, self.user),
        )
        self._add_page(
            sidebar_layout,
            "İçerideki Araçlar",
            InsideVehiclesWidget(self.plate_service, self.user),
        )

        if self.user.role is Role.ADMIN:
            self._add_page(
                sidebar_layout,
                "Kullanıcılar",
                UsersAdminWidget(self.auth_service, self.user),
            )
            self._add_page(
                sidebar_layout,
                "Ayarlar",
                CameraSettingsWidget(self.camera_service, self.user),
            )

        sidebar_layout.addStretch()
        body.addWidget(sidebar)
        body.addWidget(self.stack, 1)
        root_layout.addLayout(body, 1)
        self.setCentralWidget(root)
        self._activate_page(0)

    def _add_page(self, layout: QVBoxLayout, title: str, page: QWidget) -> None:
        page_index = self.stack.addWidget(page)
        self.pages.append(page)
        button = QPushButton(title)
        button.setObjectName("navButton")
        button.setCheckable(True)
        button.clicked.connect(lambda _checked=False, index=page_index: self._activate_page(index))
        self.nav_buttons.append(button)
        layout.addWidget(button)

    def _activate_page(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        for button_index, button in enumerate(self.nav_buttons):
            button.setChecked(button_index == index)
        page = self.stack.widget(index)
        refresh = getattr(page, "refresh", None)
        if callable(refresh):
            refresh()
