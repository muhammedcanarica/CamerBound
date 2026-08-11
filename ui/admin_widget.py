from __future__ import annotations

from functools import partial
import threading

from PySide6.QtCore import Qt, Signal, Slot

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QFrame,
)

from app.auth import AuthService, Role, SessionUser, UserExistsError, ValidationError
from app.camera import Camera, CameraService, CameraStatus, Direction
from app.config import ConfigError, update_plate_roi
from app.ocr_models import OcrModelError, collect_model_diagnostics, select_ocr_backend
from app.plate_recognition import PlateRecognitionService
from ui.roi_calibration_dialog import RoiCalibrationDialog
from ui.records_widget import display_timestamp, prepare_table


class UsersAdminWidget(QWidget):
    def __init__(self, auth_service: AuthService, user: SessionUser) -> None:
        super().__init__()
        self.auth_service = auth_service
        self.user = user
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(14)
        layout.addWidget(QLabel("Kullanıcı Yönetimi", objectName="pageTitle"))

        create_group = QGroupBox("Yeni kullanıcı")
        form = QHBoxLayout(create_group)
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Kullanıcı adı")
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("En az 8 karakter şifre")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.role_input = QComboBox()
        self.role_input.addItem("USER", Role.USER)
        self.role_input.addItem("ADMIN", Role.ADMIN)
        create_button = QPushButton("Kullanıcı Oluştur")
        create_button.clicked.connect(self._create_user)
        form.addWidget(self.username_input, 1)
        form.addWidget(self.password_input, 1)
        form.addWidget(self.role_input)
        form.addWidget(create_button)
        layout.addWidget(create_group)

        self.table = QTableWidget()
        prepare_table(self.table, ["ID", "Kullanıcı adı", "Rol", "Oluşturulma"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table, 1)

    def _create_user(self) -> None:
        try:
            try:
                normalized_role = Role(self.role_input.currentData())
            except (TypeError, ValueError) as exc:
                raise ValidationError("Geçersiz kullanıcı rolü.") from exc
            self.auth_service.create_user(
                self.user,
                self.username_input.text(),
                self.password_input.text(),
                normalized_role,
            )
        except (UserExistsError, ValidationError, PermissionError) as exc:
            QMessageBox.warning(self, "Kullanıcı oluşturulamadı", str(exc))
            return

        self.username_input.clear()
        self.password_input.clear()
        QMessageBox.information(self, "Başarılı", "Kullanıcı oluşturuldu.")
        self.refresh()

    def refresh(self) -> None:
        try:
            users = self.auth_service.list_users(self.user)
        except PermissionError as exc:
            QMessageBox.warning(self, "Yetki hatası", str(exc))
            return
        self.table.setRowCount(len(users))
        for row_index, user in enumerate(users):
            values = (
                str(user["id"]),
                user["username"],
                user["role"],
                display_timestamp(user["created_at"]),
            )
            for column, value in enumerate(values):
                self.table.setItem(row_index, column, QTableWidgetItem(value))


class CameraSettingsWidget(QWidget):
    diagnostics_ready = Signal(str)

    def __init__(
        self,
        camera_service: CameraService,
        user: SessionUser,
        recognition_service: PlateRecognitionService | None = None,
    ) -> None:
        super().__init__()
        self.camera_service = camera_service
        self.user = user
        self.recognition_service = recognition_service
        self._editors: dict[int, tuple[QLineEdit, QLineEdit, QComboBox, QCheckBox]] = {}
        self._camera_buttons: dict[int, tuple[QPushButton, QPushButton]] = {}
        self._camera_layout: QVBoxLayout
        self._build_ui()
        self.diagnostics_ready.connect(self._show_diagnostics)

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("cameraSettingsScroll")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.content_widget = QWidget()
        self.content_widget.setObjectName("cameraSettingsContent")
        self.scroll_area.viewport().setObjectName("cameraSettingsViewport")
        for background_widget in (self.scroll_area.viewport(), self.content_widget):
            background_widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(self.content_widget)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(14)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        layout.addWidget(QLabel("Kamera Ayarları", objectName="pageTitle"))
        description = QLabel(
            "Webcam index, RTSP URL veya video dosyası kaynağı yerel veritabanında saklanır. "
            "Kamera credential koruması sonraki aşamadadır.",
            objectName="mutedLabel",
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        self._camera_layout = QVBoxLayout()
        self._camera_layout.setSpacing(14)
        layout.addLayout(self._camera_layout)
        diagnostics_group = QGroupBox("OCR Tanılama")
        diagnostics_layout = QVBoxLayout(diagnostics_group)
        self.diagnostics_label = QLabel("Model durumu kontrol edilmedi.")
        self.diagnostics_label.setWordWrap(True)
        diagnostics_button = QPushButton("Modelleri Kontrol Et")
        diagnostics_button.clicked.connect(self._run_diagnostics)
        diagnostics_layout.addWidget(self.diagnostics_label)
        diagnostics_layout.addWidget(diagnostics_button)
        layout.addWidget(diagnostics_group)
        layout.addStretch()
        self.scroll_area.setWidget(self.content_widget)
        root_layout.addWidget(self.scroll_area)

    def refresh(self) -> None:
        while self._camera_layout.count():
            item = self._camera_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._editors.clear()
        self._camera_buttons.clear()
        for camera in self.camera_service.list_cameras():
            self._add_camera_editor(camera)

    def _add_camera_editor(self, camera: Camera) -> None:
        group = QGroupBox(f"Kamera #{camera.id}")
        group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        form = QFormLayout(group)
        form.setContentsMargins(14, 18, 14, 14)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(9)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        name_input = QLineEdit(camera.name)
        url_input = QLineEdit(camera.stream_url)
        url_input.setPlaceholderText("RTSP URL, video dosyası veya webcam index (örn. 0)")
        direction_input = QComboBox()
        direction_input.addItem("Giriş", Direction.ENTRY)
        direction_input.addItem("Çıkış", Direction.EXIT)
        direction_input.setCurrentIndex(0 if camera.direction is Direction.ENTRY else 1)
        enabled_input = QCheckBox("Aktif")
        enabled_input.setChecked(camera.enabled)
        save_button = QPushButton("Kaydet")
        save_button.setMinimumHeight(36)
        save_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        save_button.clicked.connect(partial(self._save_camera, camera.id))
        roi_button = QPushButton("Plaka Alanını Kalibre Et")
        roi_button.setMinimumHeight(36)
        roi_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        roi_button.setEnabled(self.recognition_service is not None)
        roi_button.clicked.connect(partial(self._calibrate_roi, camera.id))
        form.addRow("Ad", name_input)
        form.addRow("Kamera Kaynağı", url_input)
        form.addRow("Yön", direction_input)
        form.addRow("Durum", enabled_input)
        form.addRow("", save_button)
        form.addRow("", roi_button)
        self._camera_layout.addWidget(group)
        self._editors[camera.id] = (name_input, url_input, direction_input, enabled_input)
        self._camera_buttons[camera.id] = (save_button, roi_button)

    def _save_camera(self, camera_id: int) -> None:
        name, url, direction, enabled = self._editors[camera_id]
        try:
            normalized_direction = Direction(direction.currentData())
            previous = self.camera_service.get_camera(camera_id)
            was_running = self.camera_service.is_camera_running(camera_id)
            updated = self.camera_service.update_camera(
                self.user,
                camera_id,
                name.text(),
                url.text(),
                normalized_direction,
                enabled.isChecked(),
            )
            restart_required = previous.direction is not updated.direction
            if was_running:
                stop_status = self.camera_service.stop_camera(camera_id)
                if stop_status is CameraStatus.ERROR:
                    raise RuntimeError("Kamera güvenli şekilde durdurulamadı.")
            if (
                not restart_required
                and updated.enabled
                and updated.stream_url
            ):
                self.camera_service.start_camera(camera_id)
        except (TypeError, ValueError, ValidationError, PermissionError, RuntimeError) as exc:
            QMessageBox.warning(self, "Kamera kaydedilemedi", str(exc))
            return
        if restart_required:
            message = (
                "Kamera ayarları kaydedildi. Yön değişikliğinin OCR akışına güvenli uygulanması "
                "için Dashboard oturumunu yeniden açın."
            )
        else:
            message = "Kamera ayarları kaydedildi ve canlı akışa uygulandı."
        QMessageBox.information(
            self,
            "Başarılı",
            message,
        )

    def _calibrate_roi(self, camera_id: int) -> None:
        if self.recognition_service is None:
            return
        frame = self.camera_service.get_latest_frame(camera_id)
        if frame is None:
            QMessageBox.information(
                self,
                "Görüntü yok",
                "Kalibrasyon için kamerayı Dashboard'da başlatıp bir görüntü bekleyin.",
            )
            return
        camera = self.camera_service.get_camera(camera_id)
        dialog = RoiCalibrationDialog(
            frame,
            self.recognition_service.config.roi_for(camera.direction),
            self,
        )
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        try:
            config = update_plate_roi(
                self.recognition_service.settings_path,
                camera.direction,
                dialog.selected_roi(),
            )
            self.recognition_service.apply_config(config)
        except ConfigError as exc:
            QMessageBox.warning(self, "ROI kaydedilemedi", str(exc))
            return
        QMessageBox.information(
            self,
            "ROI kaydedildi",
            "Yeni plaka alanı OCR'a uygulandı; kamera önizlemesi çalışmaya devam ediyor.",
        )

    def _run_diagnostics(self) -> None:
        if self.recognition_service is None:
            self.diagnostics_label.setText("OCR servisi bu ekrana bağlı değil.")
            return
        self.diagnostics_label.setText("OCR modelleri arka planda kontrol ediliyor...")
        config = self.recognition_service.config

        def check() -> None:
            try:
                selection = select_ocr_backend(
                    config.model_root,
                    config.ocr_backend,
                    load_onnx=True,
                )
            except OcrModelError as exc:
                lines = [f"OCR modelleri: Kullanılamıyor ({exc})", "Backend: Kullanılamıyor"]
            else:
                checks = collect_model_diagnostics(
                    config.model_root,
                    backend=selection.backend,
                    load_onnx=selection.backend.value == "onnx",
                )
                lines = [
                    f"{item.name}: {'Hazır' if item.ok else 'Kullanılamıyor'}"
                    for item in checks
                ]
                lines.append(f"Backend: {selection.label}")
            lines.append(f"Servis: {self.recognition_service.get_status().value}")
            self.diagnostics_ready.emit("\n".join(lines))

        threading.Thread(target=check, name="ocr-diagnostics", daemon=True).start()

    @Slot(str)
    def _show_diagnostics(self, message: str) -> None:
        self.diagnostics_label.setText(message)
