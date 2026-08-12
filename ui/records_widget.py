from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.auth import SessionUser
from app.camera import Direction
from app.plate_service import PlateRecord, PlateService, VehicleInside
from app.time_utils import to_local_datetime


def display_timestamp(value: str | datetime) -> str:
    try:
        local_time = to_local_datetime(value)
        return local_time.strftime("%d.%m.%Y %H:%M:%S")
    except (TypeError, ValueError):
        return str(value)


def prepare_table(table: QTableWidget, headers: list[str]) -> None:
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    table.setAlternatingRowColors(True)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)


class RecordsWidget(QWidget):
    def __init__(self, plate_service: PlateService, user: SessionUser) -> None:
        super().__init__()
        self.plate_service = plate_service
        self.user = user
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(14)
        layout.addWidget(QLabel("Plaka Kayıtları", objectName="pageTitle"))

        filters = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Plakaya göre ara...")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.returnPressed.connect(self.refresh)
        self.direction_filter = QComboBox()
        self.direction_filter.addItem("Tümü", None)
        self.direction_filter.addItem("Giriş", Direction.ENTRY)
        self.direction_filter.addItem("Çıkış", Direction.EXIT)
        self.direction_filter.currentIndexChanged.connect(self.refresh)
        refresh_button = QPushButton("Ara / Yenile")
        refresh_button.clicked.connect(self.refresh)
        filters.addWidget(self.search_input, 1)
        filters.addWidget(self.direction_filter)
        filters.addWidget(refresh_button)
        layout.addLayout(filters)

        self.table = QTableWidget()
        prepare_table(
            self.table,
            ["Plaka", "İşlem", "Kamera", "Güven", "Tarih / Saat", "Fotoğraf"],
        )
        layout.addWidget(self.table, 1)

    def refresh(self) -> None:
        try:
            records = self.plate_service.search_records(
                self.user,
                self.search_input.text(),
                self.direction_filter.currentData(),
            )
        except (ValueError, PermissionError) as exc:
            QMessageBox.warning(self, "Kayıtlar alınamadı", str(exc))
            return
        self._populate(records)

    def _populate(self, records: list[PlateRecord]) -> None:
        self.table.setRowCount(len(records))
        for row_index, record in enumerate(records):
            values = (
                record.plate,
                "Giriş" if record.direction is Direction.ENTRY else "Çıkış",
                record.camera_name,
                f"%{record.confidence * 100:.1f}",
                display_timestamp(record.timestamp),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column in (1, 3):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row_index, column, item)
            if record.image_path:
                open_button = QPushButton("Fotoğrafı Aç")
                open_button.setObjectName("openCaptureButton")
                open_button.clicked.connect(
                    lambda _checked=False, path=record.image_path: self._open_capture(
                        path
                    )
                )
                self.table.setCellWidget(row_index, 5, open_button)
            else:
                missing_item = QTableWidgetItem("Yok")
                missing_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row_index, 5, missing_item)

    def _open_capture(self, image_path: str) -> None:
        resolved = self.plate_service.resolve_capture_path(image_path)
        if resolved is None or not resolved.is_file():
            QMessageBox.warning(
                self,
                "Fotoğraf bulunamadı",
                "Bu kayıtla ilişkili fotoğraf dosyası bulunamadı.",
            )
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(resolved))):
            QMessageBox.warning(
                self,
                "Fotoğraf açılamadı",
                "Fotoğraf varsayılan görüntüleyicide açılamadı.",
            )


class InsideVehiclesWidget(QWidget):
    def __init__(self, plate_service: PlateService, user: SessionUser) -> None:
        super().__init__()
        self.plate_service = plate_service
        self.user = user
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(14)
        header = QHBoxLayout()
        header.addWidget(QLabel("İçerideki Araçlar", objectName="pageTitle"))
        header.addStretch()
        refresh_button = QPushButton("Yenile")
        refresh_button.clicked.connect(self.refresh)
        header.addWidget(refresh_button)
        layout.addLayout(header)
        description = QLabel(
            "Son hareketi giriş olan araçlar gösterilir.", objectName="mutedLabel"
        )
        layout.addWidget(description)
        self.table = QTableWidget()
        prepare_table(self.table, ["Plaka", "Giriş zamanı", "Kamera"])
        layout.addWidget(self.table, 1)

    def refresh(self) -> None:
        try:
            vehicles = self.plate_service.get_vehicles_inside(self.user)
        except (ValueError, PermissionError) as exc:
            QMessageBox.warning(self, "Araçlar alınamadı", str(exc))
            return
        self._populate(vehicles)

    def _populate(self, vehicles: list[VehicleInside]) -> None:
        self.table.setRowCount(len(vehicles))
        for row_index, vehicle in enumerate(vehicles):
            values = (
                vehicle.plate,
                display_timestamp(vehicle.entry_time),
                vehicle.camera_name,
            )
            for column, value in enumerate(values):
                self.table.setItem(row_index, column, QTableWidgetItem(value))
