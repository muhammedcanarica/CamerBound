from __future__ import annotations

import platform
import subprocess
from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.camera import Direction
from app.plate_service import PlateRecord, PlateService
from ui.display_helpers import display_plate, display_timestamp


MISSING_PHOTO_TEXT = "Bu kayıt için araç fotoğrafı bulunamadı."


class AspectRatioPixmapLabel(QLabel):
    """Display one pixmap responsively without stretching or up-front caching."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._source_pixmap: QPixmap | None = None
        self.setObjectName("recordPhotoLabel")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(420, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_source_pixmap(self, pixmap: QPixmap) -> None:
        self._source_pixmap = pixmap
        self.setText("")
        self._scale_pixmap()

    def show_placeholder(self) -> None:
        self._source_pixmap = None
        self.clear()
        self.setText(MISSING_PHOTO_TEXT)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._scale_pixmap()

    def _scale_pixmap(self) -> None:
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return
        target_size = self.contentsRect().size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            return
        self.setPixmap(
            self._source_pixmap.scaled(
                target_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class RecordDetailDialog(QDialog):
    def __init__(
        self,
        record: PlateRecord,
        plate_service: PlateService,
        parent: QWidget | None = None,
        *,
        explorer_launcher: Callable[..., object] = subprocess.Popen,
        system_name_provider: Callable[[], str] = platform.system,
    ) -> None:
        super().__init__(parent)
        self.record = record
        self.plate_service = plate_service
        self._explorer_launcher = explorer_launcher
        self._system_name_provider = system_name_provider
        self.setObjectName("recordDetailDialog")
        self.setWindowTitle("Kayıt Detayı")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setMinimumSize(600, 520)
        self.resize(760, 640)
        self._build_ui()
        self._load_photo()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(16)
        layout.addWidget(QLabel("Kayıt Detayı", objectName="pageTitle"))

        photo_frame = QFrame(objectName="recordPhotoFrame")
        photo_layout = QVBoxLayout(photo_frame)
        photo_layout.setContentsMargins(10, 10, 10, 10)
        self.photo_label = AspectRatioPixmapLabel()
        photo_layout.addWidget(self.photo_label)
        layout.addWidget(photo_frame, 1)

        metadata_frame = QFrame(objectName="contentCard")
        metadata_layout = QFormLayout(metadata_frame)
        metadata_layout.setContentsMargins(18, 16, 18, 16)
        metadata_layout.setHorizontalSpacing(28)
        metadata_layout.setVerticalSpacing(10)
        self.plate_value = self._detail_value(
            display_plate(self.record.plate), "recordPlateValue"
        )
        self.timestamp_value = self._detail_value(
            display_timestamp(self.record.timestamp), "recordTimestampValue"
        )
        self.direction_value = self._detail_value(
            "Giriş" if self.record.direction is Direction.ENTRY else "Çıkış",
            "recordDirectionValue",
        )
        self.camera_value = self._detail_value(
            self.record.camera_name, "recordCameraValue"
        )
        self.confidence_value = self._detail_value(
            f"%{self.record.confidence * 100:.1f}", "recordConfidenceValue"
        )
        metadata_layout.addRow("Plaka", self.plate_value)
        metadata_layout.addRow("Tarih / Saat", self.timestamp_value)
        metadata_layout.addRow("Yön", self.direction_value)
        metadata_layout.addRow("Kamera", self.camera_value)
        metadata_layout.addRow("OCR Güveni", self.confidence_value)
        layout.addWidget(metadata_frame)

        footer = QHBoxLayout()
        self.open_file_button = QPushButton("Dosyada Aç")
        self.open_file_button.setObjectName("openCaptureFileButton")
        self.open_file_button.setEnabled(False)
        self.open_file_button.clicked.connect(self._open_file)
        close_button = QPushButton("Kapat")
        close_button.clicked.connect(self.accept)
        footer.addWidget(self.open_file_button)
        footer.addStretch()
        footer.addWidget(close_button)
        layout.addLayout(footer)

    def _load_photo(self) -> None:
        resolved = self.plate_service.resolve_capture_path(self.record.image_path)
        if resolved is None or not resolved.is_file():
            self.photo_label.show_placeholder()
            return

        pixmap = QPixmap(str(resolved))
        if pixmap.isNull():
            self.photo_label.show_placeholder()
            return
        self.photo_label.set_source_pixmap(pixmap)
        self.open_file_button.setEnabled(True)

    def _open_file(self) -> None:
        resolved = self.plate_service.resolve_capture_path(self.record.image_path)
        if resolved is None or not resolved.is_file():
            self.open_file_button.setEnabled(False)
            self.photo_label.show_placeholder()
            QMessageBox.warning(
                self,
                "Fotoğraf bulunamadı",
                "Fotoğraf dosyası bulunamadı.",
            )
            return
        if self._system_name_provider() != "Windows":
            QMessageBox.warning(
                self,
                "Dosya konumu açılamadı",
                "Bu özellik yalnızca Windows'ta kullanılabilir.",
            )
            return
        try:
            self._explorer_launcher(
                ["explorer.exe", f"/select,{resolved}"],
                shell=False,
            )
        except (OSError, subprocess.SubprocessError):
            QMessageBox.warning(
                self,
                "Dosya konumu açılamadı",
                "Fotoğrafın bulunduğu klasör Windows Explorer'da açılamadı.",
            )

    @staticmethod
    def _detail_value(text: str, object_name: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName(object_name)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return label
