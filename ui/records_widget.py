from __future__ import annotations

from datetime import date
from functools import partial

from PySide6.QtCore import Qt
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
from app.plate_service import (
    PlateRecord,
    PlateRecordDaySummary,
    PlateService,
    VehicleInside,
)
from ui.display_helpers import display_timestamp
from ui.record_detail_dialog import RecordDetailDialog


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
        self._records: list[PlateRecord] = []
        self._records_by_id: dict[int, PlateRecord] = {}
        self._selected_date: date | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(14)
        header = QHBoxLayout()
        header.addWidget(QLabel("Plaka Kayıtları", objectName="pageTitle"))
        header.addStretch()
        self.back_button = QPushButton("Günlük Arşive Dön")
        self.back_button.setObjectName("recordsArchiveBackButton")
        self.back_button.clicked.connect(self._show_archive)
        self.back_button.hide()
        header.addWidget(self.back_button)
        layout.addLayout(header)

        self.view_label = QLabel("Günlük Arşiv", objectName="mutedLabel")
        layout.addWidget(self.view_label)

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
            ["Tarih", "Toplam Kayıt", "Giriş", "Çıkış", ""],
        )
        self.table.cellDoubleClicked.connect(self._handle_double_click)
        layout.addWidget(self.table, 1)

    def refresh(self) -> None:
        try:
            if self._selected_date is None:
                summaries = self.plate_service.get_record_day_summaries(
                    self.user,
                    self.search_input.text(),
                    self.direction_filter.currentData(),
                )
            else:
                records = self.plate_service.search_records_for_local_date(
                    self.user,
                    self._selected_date,
                    self.search_input.text(),
                    self.direction_filter.currentData(),
                )
        except (ValueError, PermissionError) as exc:
            QMessageBox.warning(self, "Kayıtlar alınamadı", str(exc))
            return
        if self._selected_date is None:
            self._populate_day_summaries(summaries)
        else:
            self._populate(records)

    def _populate_day_summaries(
        self, summaries: list[PlateRecordDaySummary]
    ) -> None:
        self._records = []
        self._records_by_id = {}
        sorting_enabled = self.table.isSortingEnabled()
        self.table.setSortingEnabled(False)
        self.table.clearContents()
        prepare_table(
            self.table,
            ["Tarih", "Toplam Kayıt", "Giriş", "Çıkış", ""],
        )
        self.table.setRowCount(len(summaries))
        for row_index, summary in enumerate(summaries):
            values = (
                summary.date.strftime("%d.%m.%Y"),
                str(summary.total_count),
                str(summary.entry_count),
                str(summary.exit_count),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, summary.date.isoformat())
                if column > 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row_index, column, item)
            open_button = QPushButton("Aç")
            open_button.setObjectName("openRecordDayButton")
            open_button.setProperty("localDate", summary.date.isoformat())
            open_button.clicked.connect(partial(self._open_day, summary.date))
            self.table.setCellWidget(row_index, 4, open_button)
        self.table.setSortingEnabled(sorting_enabled)

    def _populate(self, records: list[PlateRecord]) -> None:
        self._records = records
        self._records_by_id = {record.id: record for record in records}
        sorting_enabled = self.table.isSortingEnabled()
        self.table.setSortingEnabled(False)
        self.table.clearContents()
        prepare_table(
            self.table,
            ["Plaka", "Giriş / Çıkış", "Kamera", "Saat", "Fotoğraf"],
        )
        self.table.setRowCount(len(records))
        for row_index, record in enumerate(records):
            values = (
                record.plate,
                "Giriş" if record.direction is Direction.ENTRY else "Çıkış",
                record.camera_name,
                display_timestamp(record.timestamp),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, record.id)
                if column == 1:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row_index, column, item)
            if record.image_path:
                open_button = QPushButton("Aç")
                open_button.setObjectName("openRecordPhotoButton")
                open_button.setProperty("recordId", record.id)
                open_button.setToolTip("Kayıt detayını ve araç fotoğrafını aç")
                open_button.clicked.connect(
                    partial(self._open_record_detail, record.id)
                )
                self.table.setCellWidget(row_index, 4, open_button)
            else:
                photo_item = QTableWidgetItem("-")
                photo_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row_index, 4, photo_item)
        self.table.setSortingEnabled(sorting_enabled)

    def _handle_double_click(self, row: int, column: int) -> None:
        if self._selected_date is None:
            date_item = self.table.item(row, 0)
            if date_item is None:
                return
            value = date_item.data(Qt.ItemDataRole.UserRole)
            if not isinstance(value, str):
                return
            try:
                self._open_day(date.fromisoformat(value))
            except ValueError:
                return
            return
        self._show_record_detail(row, column)

    def _open_day(self, local_date: date) -> None:
        self._selected_date = local_date
        self.view_label.setText(local_date.strftime("%d.%m.%Y Kayıtları"))
        self.back_button.show()
        self.refresh()

    def _show_archive(self) -> None:
        self._selected_date = None
        self.view_label.setText("Günlük Arşiv")
        self.back_button.hide()
        self.refresh()

    def _show_record_detail(self, row: int, _column: int) -> None:
        record_item = self.table.item(row, 0)
        if record_item is None:
            return
        record_id = record_item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(record_id, int):
            return
        self._open_record_detail(record_id)

    def _open_record_detail(self, record_id: int) -> None:
        record = self._records_by_id.get(record_id)
        if record is None:
            return
        dialog = RecordDetailDialog(record, self.plate_service, self)
        dialog.exec()


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
