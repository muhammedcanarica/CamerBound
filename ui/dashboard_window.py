from __future__ import annotations

from dataclasses import dataclass

import cv2
from PySide6.QtCore import QTimer, Qt, Signal, Slot
from PySide6.QtGui import QColor, QCloseEvent, QImage, QPainter, QPen, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.auth import Role, SessionUser, ValidationError
from app.camera import Camera, CameraService, CameraStatus, Direction
from app.plate_recognition import (
    PlateCandidate,
    PlateRecognitionService,
    RecognitionStatus,
)
from app.plate_service import PlateRecord, PlateService
from ui.admin_widget import CameraSettingsWidget, UsersAdminWidget
from ui.records_widget import (
    InsideVehiclesWidget,
    RecordsWidget,
    display_timestamp,
    prepare_table,
)


@dataclass(slots=True)
class CameraCardWidgets:
    preview: QLabel
    status: QLabel
    details: QLabel
    last_plate: QLabel
    ocr_status: QLabel


class DashboardHome(QWidget):
    def __init__(
        self,
        plate_service: PlateService,
        camera_service: CameraService,
        user: SessionUser,
        recognition_service: PlateRecognitionService | None = None,
    ) -> None:
        super().__init__()
        self.plate_service = plate_service
        self.camera_service = camera_service
        self.user = user
        self.recognition_service = recognition_service
        self.camera_cards: dict[Direction, CameraCardWidgets] = {}
        self.camera_directions: dict[int, Direction] = {}
        self._latest_images: dict[Direction, QImage] = {}
        self._build_ui()
        self.camera_service.frame_ready.connect(self._show_frame)
        self.camera_service.status_changed.connect(self._show_status)
        if self.recognition_service is not None:
            self.recognition_service.status_changed.connect(self._show_ocr_status)
            self.recognition_service.candidate_changed.connect(self._show_candidate)
            self.recognition_service.record_saved.connect(self._record_saved)
            QTimer.singleShot(0, self._sync_ocr_status)
        QTimer.singleShot(0, self._start_enabled_cameras)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

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
        layout.addLayout(camera_layout, 7)

        layout.addWidget(QLabel("Son Hareketler", objectName="sectionTitle"))
        self.recent_table = QTableWidget()
        prepare_table(
            self.recent_table,
            ["Plaka", "İşlem", "Kamera", "Güven", "Tarih / Saat"],
        )
        self.recent_table.setMinimumHeight(120)
        self.recent_table.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        layout.addWidget(self.recent_table, 3)

    def _camera_card(self, direction: Direction) -> QFrame:
        card = QFrame(objectName="cameraCard")
        card.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 12, 14, 12)
        card_layout.setSpacing(5)
        title_text = "GİRİŞ KAMERASI" if direction is Direction.ENTRY else "ÇIKIŞ KAMERASI"
        card_layout.addWidget(QLabel(title_text, objectName="sectionTitle"))

        preview = QLabel("Kamera görüntüsü\nRTSP entegrasyonu bekleniyor")
        preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview.setMinimumHeight(290)
        preview.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        preview.setScaledContents(False)
        preview.setStyleSheet(
            "background:#111827; color:#94a3b8; border-radius:7px; font-size:14px;"
        )
        status = QLabel("● Bağlı değil")
        status.setStyleSheet("color:#b42318; font-weight:600;")
        details = QLabel("Kamera yapılandırması yükleniyor", objectName="mutedLabel")
        last_plate = QLabel("Son okunan plaka: -")
        last_plate.setStyleSheet("font-weight:600;")
        ocr_status = QLabel("OCR: Başlatılmadı", objectName="mutedLabel")
        card_layout.addWidget(preview, 1)
        card_layout.addWidget(status)
        card_layout.addWidget(details)
        card_layout.addWidget(last_plate)
        card_layout.addWidget(ocr_status)
        self.camera_cards[direction] = CameraCardWidgets(
            preview=preview,
            status=status,
            details=details,
            last_plate=last_plate,
            ocr_status=ocr_status,
        )
        return card

    def refresh(self) -> None:
        try:
            cameras = self.camera_service.list_cameras()
            records = self.plate_service.get_recent_records(self.user, 20)
        except (ValueError, PermissionError) as exc:
            QMessageBox.warning(self, "Dashboard yenilenemedi", str(exc))
            return

        self.camera_directions = {camera.id: camera.direction for camera in cameras}
        for direction, card in self.camera_cards.items():
            camera = next((item for item in cameras if item.direction is direction), None)
            latest = next((item for item in records if item.direction is direction), None)
            card.details.setText(self._camera_description(camera))
            card.last_plate.setText(
                f"Son okunan plaka: {latest.plate}" if latest else "Son okunan plaka: -"
            )
            if camera is None:
                self._set_card_status(direction, CameraStatus.ERROR, "Kamera kaydı bulunamadı")
            elif not camera.enabled:
                self._set_card_status(direction, CameraStatus.STOPPED, "Kamera pasif")
            elif not camera.stream_url:
                self._set_card_status(direction, CameraStatus.ERROR, "Kamera URL'si tanımlı değil")
            else:
                self._set_card_status(
                    direction,
                    self.camera_service.get_status(camera.id),
                )
        self._populate_recent(records)

    @Slot()
    def _start_enabled_cameras(self) -> None:
        try:
            cameras = self.camera_service.list_cameras()
        except ValueError:
            return

        self.camera_directions = {camera.id: camera.direction for camera in cameras}
        for camera in cameras:
            if not camera.enabled or not camera.stream_url:
                continue
            try:
                self.camera_service.start_camera(camera.id)
            except ValidationError as exc:
                self._set_card_status(camera.direction, CameraStatus.ERROR, str(exc))

    @Slot(int, object)
    def _show_frame(self, camera_id: int, frame: object) -> None:
        direction = self.camera_directions.get(camera_id)
        if direction is None or direction not in self.camera_cards:
            return

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb_frame.shape
            if channels != 3:
                return
            image = QImage(
                rgb_frame.data,
                width,
                height,
                channels * width,
                QImage.Format.Format_RGB888,
            ).copy()
        except (AttributeError, TypeError, ValueError, cv2.error):
            return

        self._draw_roi(image, direction)
        self._latest_images[direction] = image
        self._display_image(direction, image)

    @Slot()
    def _sync_ocr_status(self) -> None:
        if self.recognition_service is None:
            return
        status = self.recognition_service.get_status()
        messages = {
            RecognitionStatus.STOPPED: "OCR başlatılmadı.",
            RecognitionStatus.INITIALIZING: "OCR başlatılıyor.",
            RecognitionStatus.ACTIVE: "OCR aktif.",
            RecognitionStatus.UNAVAILABLE: "OCR kullanılamıyor.",
            RecognitionStatus.ERROR: "OCR hatası.",
        }
        self._show_ocr_status(status, messages[status])

    @Slot(object, str)
    def _show_ocr_status(self, status: RecognitionStatus, message: str) -> None:
        status = RecognitionStatus(status)
        prefixes = {
            RecognitionStatus.STOPPED: "OCR: Kapalı",
            RecognitionStatus.INITIALIZING: "OCR: Başlatılıyor",
            RecognitionStatus.ACTIVE: "OCR: Aktif",
            RecognitionStatus.UNAVAILABLE: "OCR: Kullanılamıyor",
            RecognitionStatus.ERROR: "OCR: Hata",
        }
        for card in self.camera_cards.values():
            card.ocr_status.setText(prefixes[status])
            card.ocr_status.setToolTip(message)

    @Slot(int, object)
    def _show_candidate(self, camera_id: int, candidate: PlateCandidate) -> None:
        direction = self.camera_directions.get(camera_id)
        if direction is None:
            return
        self.camera_cards[direction].ocr_status.setText(
            f"OCR: Aktif · Son: {candidate.plate} · %{candidate.confidence * 100:.0f}"
        )
        self.camera_cards[direction].ocr_status.setToolTip(
            f"Ham OCR: {candidate.raw_text}"
        )

    @Slot(object)
    def _record_saved(self, record: PlateRecord) -> None:
        card = self.camera_cards.get(record.direction)
        if card is not None:
            card.last_plate.setText(f"Son okunan plaka: {record.plate}")
        self.refresh()

    @Slot(int, object, str)
    def _show_status(
        self,
        camera_id: int,
        status: CameraStatus,
        message: str,
    ) -> None:
        direction = self.camera_directions.get(camera_id)
        if direction is None:
            return
        self._set_card_status(direction, CameraStatus(status), tooltip=message)

    def _set_card_status(
        self,
        direction: Direction,
        status: CameraStatus,
        text: str | None = None,
        tooltip: str = "",
    ) -> None:
        card = self.camera_cards[direction]
        status_texts = {
            CameraStatus.STOPPED: "● Bağlı değil",
            CameraStatus.CONNECTING: "Bağlanıyor...",
            CameraStatus.CONNECTED: "● Bağlı",
            CameraStatus.RECONNECTING: "Yeniden bağlanılıyor...",
            CameraStatus.ERROR: "● Bağlantı kesildi",
        }
        status_colors = {
            CameraStatus.STOPPED: "#b42318",
            CameraStatus.CONNECTING: "#b76e00",
            CameraStatus.CONNECTED: "#16803a",
            CameraStatus.RECONNECTING: "#b76e00",
            CameraStatus.ERROR: "#b42318",
        }
        card.status.setText(text or status_texts[status])
        card.status.setToolTip(tooltip or text or "")
        card.status.setStyleSheet(f"color:{status_colors[status]}; font-weight:600;")

        if status is CameraStatus.STOPPED:
            self._clear_preview(direction, text or "Kamera görüntüsü bekleniyor")
        elif status is CameraStatus.ERROR and direction not in self._latest_images:
            self._clear_preview(direction, text or "Kamera bağlantısı kurulamadı")

    def _display_image(self, direction: Direction, image: QImage) -> None:
        preview = self.camera_cards[direction].preview
        target_size = preview.contentsRect().size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            return
        pixmap = QPixmap.fromImage(image).scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        preview.setText("")
        preview.setPixmap(pixmap)

    def _draw_roi(self, image: QImage, direction: Direction) -> None:
        if self.recognition_service is None:
            return
        roi = self.recognition_service.config.roi_for(direction)
        painter = QPainter(image)
        painter.setPen(QPen(QColor("#22c55e"), 2))
        painter.drawRect(
            round(roi.x * image.width()),
            round(roi.y * image.height()),
            max(1, round(roi.width * image.width())),
            max(1, round(roi.height * image.height())),
        )
        painter.end()

    def _clear_preview(self, direction: Direction, message: str) -> None:
        self._latest_images.pop(direction, None)
        preview = self.camera_cards[direction].preview
        preview.clear()
        preview.setText(message)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        for direction, image in self._latest_images.items():
            self._display_image(direction, image)

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
        recognition_service: PlateRecognitionService | None = None,
    ) -> None:
        super().__init__()
        self.user = user
        self.auth_service = auth_service
        self.plate_service = plate_service
        self.camera_service = camera_service
        self.recognition_service = recognition_service
        self.pages: list[QWidget] = []
        self.nav_buttons: list[QPushButton] = []
        self._camera_shutdown_started = False
        self.setWindowTitle("Plaka Takip Sistemi")
        self.setMinimumSize(1080, 700)
        self.resize(1280, 820)
        self._build_ui()
        if self.recognition_service is not None:
            self.recognition_service.record_saved.connect(self._refresh_record_pages)

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
        self.dashboard_home = DashboardHome(
            self.plate_service,
            self.camera_service,
            self.user,
            self.recognition_service,
        )
        self._add_page(
            sidebar_layout,
            "Dashboard",
            self.dashboard_home,
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
                CameraSettingsWidget(
                    self.camera_service,
                    self.user,
                    self.recognition_service,
                ),
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

    @Slot(object)
    def _refresh_record_pages(self, _record: PlateRecord) -> None:
        for page in self.pages[1:3]:
            refresh = getattr(page, "refresh", None)
            if callable(refresh):
                refresh()

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self._camera_shutdown_started:
            self._camera_shutdown_started = True
            self.camera_service.stop_all()
            if self.recognition_service is not None:
                self.recognition_service.stop()
        super().closeEvent(event)
