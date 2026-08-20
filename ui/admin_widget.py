from __future__ import annotations

from functools import partial
import sqlite3
import threading

from PySide6.QtCore import Qt, Signal, Slot

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
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

from app.audit import AuditService
from app.auth import AuthService, Role, SessionUser, UserExistsError, ValidationError
from app.camera import Camera, CameraService, CameraStatus, Direction
from app.camera_credentials import split_camera_source_credentials
from app.config import (
    ConfigError,
    application_root,
    update_plate_roi,
    update_record_retention,
)
from app.ocr_models import OcrModelError, collect_model_diagnostics, select_ocr_backend
from app.plate_recognition import PlateRecognitionService
from app.plate_service import PlateService
from app.time_diagnostics import (
    TimeDiagnosticsResult,
    TimeDiagnosticsService,
    TimeSyncStatus,
)
from ui.roi_calibration_dialog import RoiCalibrationDialog
from ui.display_helpers import display_timestamp
from ui.records_widget import prepare_table


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


class CameraCredentialsDialog(QDialog):
    def __init__(self, camera: Camera, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("cameraCredentialsDialog")
        self.setWindowTitle("Kamera Erişim Bilgileri")
        self.setModal(True)
        self.setMinimumWidth(480)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(14)

        description = QLabel(
            "Bu işlem kameranın kendi kullanıcı adı veya şifresini değiştirmez. "
            "Yalnızca CamerBound'un kameraya bağlanırken kullandığı bilgileri günceller."
        )
        description.setObjectName("cameraCredentialsDescription")
        description.setWordWrap(True)
        layout.addWidget(description)

        form = QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)
        self.username_input = QLineEdit(camera.username or "")
        self.username_input.setObjectName("cameraCredentialUsernameInput")
        self.password_input = QLineEdit()
        self.password_input.setObjectName("cameraCredentialPasswordInput")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Kamera Kullanıcı Adı", self.username_input)
        form.addRow("Bağlantı Şifresi", self.password_input)
        layout.addLayout(form)

        if camera.username and camera.has_password:
            password_help = QLabel("Mevcut şifreyi korumak için boş bırakın.")
            password_help.setObjectName("cameraCredentialPasswordHelp")
            password_help.setWordWrap(True)
            password_help.setStyleSheet("color: #6d7890;")
            layout.addWidget(password_help)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Save
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("İptal")
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setText("Kaydet")
        self.buttons.rejected.connect(self.reject)
        self.buttons.accepted.connect(self.accept)
        layout.addWidget(self.buttons)


class CameraSettingsWidget(QWidget):
    diagnostics_ready = Signal(str)
    time_diagnostics_ready = Signal(object)
    records_changed = Signal()

    def __init__(
        self,
        camera_service: CameraService,
        user: SessionUser,
        plate_service: PlateService,
        recognition_service: PlateRecognitionService | None = None,
        audit_service: AuditService | None = None,
        time_diagnostics_service: TimeDiagnosticsService | None = None,
    ) -> None:
        super().__init__()
        self.camera_service = camera_service
        self.user = user
        self.plate_service = plate_service
        self.recognition_service = recognition_service
        self.audit_service = audit_service
        self.time_diagnostics_service = (
            time_diagnostics_service or TimeDiagnosticsService()
        )
        self._time_diagnostics_running = False
        self.settings_path = (
            recognition_service.settings_path
            if recognition_service is not None
            else application_root() / "config" / "settings.json"
        )
        self._editors: dict[
            int,
            tuple[QLineEdit, QLineEdit, QComboBox, QCheckBox],
        ] = {}
        self._camera_buttons: dict[int, tuple[QPushButton, QPushButton]] = {}
        self._camera_layout: QVBoxLayout
        self._build_ui()
        self.diagnostics_ready.connect(self._show_diagnostics)
        self.time_diagnostics_ready.connect(self._show_time_diagnostics)

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
            "Kamera şifreleri Windows DPAPI ile mevcut Windows kullanıcısına bağlı "
            "olarak korunur ve loglarda gösterilmez.",
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
        capture_buttons = QHBoxLayout()
        self.capture_entry_recognition_frame_button = QPushButton(
            "İlk ENTRY Detector Karesini Kaydet"
        )
        self.capture_exit_recognition_frame_button = QPushButton(
            "İlk EXIT Detector Karesini Kaydet"
        )
        self.capture_entry_recognition_frame_button.clicked.connect(
            partial(self._arm_recognition_frame_capture, Direction.ENTRY)
        )
        self.capture_exit_recognition_frame_button.clicked.connect(
            partial(self._arm_recognition_frame_capture, Direction.EXIT)
        )
        capture_buttons.addWidget(self.capture_entry_recognition_frame_button)
        capture_buttons.addWidget(self.capture_exit_recognition_frame_button)
        diagnostics_layout.addWidget(self.diagnostics_label)
        diagnostics_layout.addWidget(diagnostics_button)
        diagnostics_layout.addLayout(capture_buttons)
        layout.addWidget(diagnostics_group)
        self.time_group = QGroupBox("Saat Durumu")
        self.time_group.setObjectName("timeDiagnosticsGroup")
        time_layout = QVBoxLayout(self.time_group)
        time_form = QFormLayout()
        self.local_time_label = QLabel("-")
        self.utc_time_label = QLabel("-")
        self.timezone_name_label = QLabel("-")
        self.utc_offset_label = QLabel("-")
        self.windows_time_label = QLabel("Kontrol edilmedi")
        self.time_source_label = QLabel("-")
        for label in (
            self.local_time_label,
            self.utc_time_label,
            self.timezone_name_label,
            self.utc_offset_label,
            self.windows_time_label,
            self.time_source_label,
        ):
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        time_form.addRow("Yerel Saat:", self.local_time_label)
        time_form.addRow("UTC:", self.utc_time_label)
        time_form.addRow("Saat Dilimi:", self.timezone_name_label)
        time_form.addRow("UTC Ofseti:", self.utc_offset_label)
        time_form.addRow("Windows Time:", self.windows_time_label)
        time_form.addRow("Saat Kaynağı:", self.time_source_label)
        time_layout.addLayout(time_form)
        self.time_status_label = QLabel("Saat durumu kontrol edilmedi.")
        self.time_status_label.setObjectName("timeDiagnosticsStatus")
        self.time_status_label.setWordWrap(True)
        time_layout.addWidget(self.time_status_label)
        self.refresh_time_button = QPushButton("Yenile")
        self.refresh_time_button.setObjectName("refreshTimeDiagnosticsButton")
        self.refresh_time_button.clicked.connect(self._refresh_time_diagnostics)
        time_layout.addWidget(self.refresh_time_button)
        layout.addWidget(self.time_group)
        self.retention_group = QGroupBox("Veri Saklama")
        self.retention_group.setObjectName("dataRetentionGroup")
        retention_layout = QVBoxLayout(self.retention_group)
        retention_form = QFormLayout()
        self.retention_combo = QComboBox()
        self.retention_combo.setObjectName("recordRetentionCombo")
        for label, days in (
            ("30 gün", 30),
            ("90 gün", 90),
            ("180 gün", 180),
            ("Süresiz", 0),
        ):
            self.retention_combo.addItem(label, days)
        retention_form.addRow(
            "Plaka kayıtlarını saklama süresi:", self.retention_combo
        )
        retention_layout.addLayout(retention_form)
        self.save_retention_button = QPushButton("Saklama Politikasını Kaydet")
        self.save_retention_button.clicked.connect(self._save_retention_policy)
        retention_layout.addWidget(self.save_retention_button)
        self.total_records_label = QLabel("Toplam kayıt: 0")
        self.oldest_record_label = QLabel("En eski kayıt: -")
        retention_layout.addWidget(self.total_records_label)
        retention_layout.addWidget(self.oldest_record_label)
        self.cleanup_old_button = QPushButton("Eski Kayıtları Şimdi Temizle")
        self.cleanup_old_button.setObjectName("cleanupOldRecordsButton")
        self.cleanup_old_button.clicked.connect(self._cleanup_old_records)
        retention_layout.addWidget(self.cleanup_old_button)
        danger_title = QLabel("Tehlikeli Bölge")
        danger_title.setStyleSheet("color:#b42318; font-weight:700;")
        retention_layout.addWidget(danger_title)
        self.delete_all_records_button = QPushButton(
            "Tüm Plaka Kayıtlarını Temizle"
        )
        self.delete_all_records_button.setObjectName("deleteAllPlateRecordsButton")
        self.delete_all_records_button.setStyleSheet(
            "background:#b42318; border-color:#b42318; color:white;"
        )
        self.delete_all_records_button.clicked.connect(self._delete_all_records)
        retention_layout.addWidget(self.delete_all_records_button)
        layout.addWidget(self.retention_group)
        if self.audit_service is not None:
            self.audit_group = QGroupBox("Güvenlik Günlüğü")
            self.audit_group.setObjectName("securityAuditGroup")
            audit_layout = QVBoxLayout(self.audit_group)
            self.audit_table = QTableWidget()
            self.audit_table.setObjectName("securityAuditTable")
            prepare_table(
                self.audit_table,
                ["Tarih / Saat", "Kullanıcı", "İşlem", "Detay"],
            )
            self.audit_table.setMinimumHeight(260)
            audit_layout.addWidget(self.audit_table)
            layout.addWidget(self.audit_group)
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
        self._refresh_retention()
        self._refresh_audit_logs()

    def _refresh_audit_logs(self) -> None:
        if self.audit_service is None:
            return
        try:
            logs = self.audit_service.get_recent_logs(self.user, 200)
        except (PermissionError, sqlite3.Error) as exc:
            self.audit_group.setToolTip(str(exc))
            self.audit_table.setRowCount(0)
            return
        self.audit_group.setToolTip("")
        self.audit_table.setRowCount(len(logs))
        for row_index, entry in enumerate(logs):
            values = (
                display_timestamp(entry.timestamp),
                entry.username,
                entry.action,
                entry.details or "",
            )
            for column, value in enumerate(values):
                self.audit_table.setItem(row_index, column, QTableWidgetItem(value))

    def _refresh_retention(self) -> None:
        index = self.retention_combo.findData(self.plate_service.record_retention_days)
        if index >= 0:
            self.retention_combo.setCurrentIndex(index)
        unlimited = self.plate_service.record_retention_days == 0
        self.cleanup_old_button.setEnabled(not unlimited)
        self.cleanup_old_button.setToolTip(
            "Saklama süresi Süresiz olduğu için otomatik temizlenecek kayıt yok."
            if unlimited
            else ""
        )
        try:
            stats = self.plate_service.get_record_stats(self.user)
        except (PermissionError, sqlite3.Error) as exc:
            self.total_records_label.setText("Toplam kayıt: -")
            self.oldest_record_label.setText("En eski kayıt: -")
            self.retention_group.setToolTip(str(exc))
            return
        self.retention_group.setToolTip("")
        self.total_records_label.setText(f"Toplam kayıt: {stats.total_records}")
        oldest = (
            display_timestamp(stats.oldest_timestamp)
            if stats.oldest_timestamp is not None
            else "-"
        )
        self.oldest_record_label.setText(f"En eski kayıt: {oldest}")

    def _save_retention_policy(self) -> None:
        try:
            AuthService.require_admin(self.user)
            retention_days = int(self.retention_combo.currentData())
            updated_config = update_record_retention(
                self.settings_path,
                retention_days,
            )
            self.plate_service.set_record_retention_days(
                updated_config.record_retention_days,
                actor=self.user,
            )
            if self.recognition_service is not None:
                self.recognition_service.config = updated_config
        except (ConfigError, PermissionError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Saklama politikası kaydedilemedi", str(exc))
            return

        self._refresh_retention()
        self._refresh_audit_logs()
        if retention_days == 0:
            message = "Plaka kayıtları süresiz saklanacak."
        else:
            message = f"Plaka kayıtları {retention_days} gün saklanacak."
        QMessageBox.information(self, "Başarılı", message)

    def _cleanup_old_records(self) -> None:
        retention_days = self.plate_service.record_retention_days
        if retention_days == 0:
            QMessageBox.information(
                self,
                "Temizlenecek kayıt yok",
                "Saklama süresi Süresiz olduğu için otomatik temizlenecek kayıt yok.",
            )
            return
        answer = QMessageBox.question(
            self,
            "Eski kayıtları temizle",
            f"{retention_days} günden eski plaka kayıtları kalıcı olarak silinecek.\n\n"
            "Devam etmek istiyor musunuz?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            deleted = self.plate_service.delete_records_older_than(self.user)
        except (PermissionError, ValueError, sqlite3.Error) as exc:
            QMessageBox.warning(self, "Kayıtlar temizlenemedi", str(exc))
            return
        self._refresh_retention()
        self._refresh_audit_logs()
        self.records_changed.emit()
        QMessageBox.information(
            self,
            "Temizleme tamamlandı",
            f"{deleted} eski kayıt silindi.",
        )

    def _delete_all_records(self) -> None:
        answer = QMessageBox.warning(
            self,
            "Tüm kayıtları temizle",
            "Tüm plaka hareket kayıtları kalıcı olarak silinecek.\n"
            "Bu işlem geri alınamaz.",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Ok:
            return
        confirmation, accepted = QInputDialog.getText(
            self,
            "Onay gerekli",
            "Devam etmek için TEMIZLE yazın:",
        )
        if not accepted:
            return
        if confirmation != "TEMIZLE":
            QMessageBox.warning(
                self,
                "Onay başarısız",
                "TEMIZLE metni tam olarak yazılmadığı için kayıtlar silinmedi.",
            )
            return
        try:
            self.plate_service.delete_all_records(self.user)
        except (PermissionError, sqlite3.Error) as exc:
            QMessageBox.warning(self, "Kayıtlar temizlenemedi", str(exc))
            return
        self._refresh_retention()
        self._refresh_audit_logs()
        self.records_changed.emit()
        QMessageBox.information(
            self,
            "Temizleme tamamlandı",
            "Tüm plaka kayıtları temizlendi.",
        )

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
        if self.user.role is Role.ADMIN:
            credential_group = QGroupBox("Kamera Erişim Bilgileri")
            credential_group.setObjectName(f"cameraCredentialsGroup-{camera.id}")
            credential_layout = QVBoxLayout(credential_group)
            credential_status = QLabel(
                "✓ Erişim bilgileri kayıtlı"
                if camera.username and camera.has_password
                else "⚠ Erişim bilgileri ayarlanmamış"
            )
            credential_status.setObjectName(f"cameraCredentialStatus-{camera.id}")
            credential_description = QLabel(
                "CamerBound, kameraya bağlanmak için bu kullanıcı adı ve şifreyi kullanır."
            )
            credential_description.setWordWrap(True)
            credential_description.setObjectName(
                f"cameraCredentialDescription-{camera.id}"
            )
            credential_button = QPushButton(
                "Erişim Bilgilerini Güncelle"
                if camera.username and camera.has_password
                else "Erişim Bilgilerini Ayarla"
            )
            credential_button.setObjectName(f"cameraCredentialButton-{camera.id}")
            credential_button.clicked.connect(
                partial(self._open_camera_credentials, camera.id)
            )
            credential_layout.addWidget(credential_status)
            credential_layout.addWidget(credential_description)
            credential_layout.addWidget(credential_button)
            form.addRow(credential_group)
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
            previous = self.camera_service.get_camera(camera_id)
            try:
                source_has_credentials = split_camera_source_credentials(
                    url.text()
                ).had_credentials
            except (UnicodeError, ValueError):
                url.setText(previous.stream_url)
                raise ValidationError("Kamera kaynağı geçersiz.") from None
            if source_has_credentials:
                url.setText(previous.stream_url)
                raise ValidationError(
                    "Kamera kaynağı alanına kullanıcı adı veya şifre eklemeyin. "
                    "Bunun için Kamera Erişim Bilgileri bölümünü kullanın."
                )
            normalized_direction = Direction(direction.currentData())
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
        self._refresh_audit_logs()
        QMessageBox.information(
            self,
            "Başarılı",
            message,
        )

    def _open_camera_credentials(self, camera_id: int) -> None:
        try:
            AuthService.require_admin(self.user)
            camera = self.camera_service.get_camera(camera_id)
        except (PermissionError, ValidationError) as exc:
            QMessageBox.warning(self, "Erişim bilgileri açılamadı", str(exc))
            return

        dialog = CameraCredentialsDialog(camera, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        username = dialog.username_input.text().strip()
        password = dialog.password_input.text()
        dialog.password_input.clear()
        if not username:
            QMessageBox.warning(
                self,
                "Erişim bilgileri kaydedilemedi",
                "Kamera kullanıcı adı boş bırakılamaz.",
            )
            return
        if not camera.has_password and not password:
            QMessageBox.warning(
                self,
                "Erişim bilgileri kaydedilemedi",
                "Bağlantı şifresi boş bırakılamaz.",
            )
            return

        try:
            self.camera_service.update_camera(
                self.user,
                camera.id,
                camera.name,
                camera.stream_url,
                camera.direction,
                camera.enabled,
                username=username,
                password=password,
            )
        except (TypeError, ValueError, ValidationError, PermissionError) as exc:
            QMessageBox.warning(
                self,
                "Erişim bilgileri kaydedilemedi",
                str(exc),
            )
            return
        finally:
            del password

        self.refresh()
        QMessageBox.information(
            self,
            "Başarılı",
            "Kamera erişim bilgileri kaydedildi.",
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
            health = self.recognition_service.get_runtime_health()
            last_frame = (
                "yok"
                if health.last_frame_age_seconds is None
                else f"{health.last_frame_age_seconds:.1f} sn önce"
            )
            lines.append(
                "Runtime: "
                f"frame={health.frames_ingested} (son={last_frame}), "
                f"detector={health.detector_frames_processed} "
                f"(live={health.live_detector_frames_processed}, "
                f"replay={health.replay_detector_frames_processed}), "
                f"(hit={health.detector_hits}, miss={health.detector_misses}), "
                f"OCR={health.ocr_jobs_processed}/{health.ocr_jobs_queued}, "
                f"aday={health.valid_candidates}, "
                f"hata={health.ocr_inference_errors}, kayıt={health.saved_records}"
            )
            lines.append(
                "Kararlar: "
                f"bekleyen={health.awaiting_confirmation}, "
                f"canlı-2frame={health.confirmed_live_multiframe}, "
                f"kayıt={health.saved_records}"
            )
            lines.append(
                "Discard / Ret: "
                f"tek-frame-red={health.single_frame_discarded}, "
                f"track-end-red={health.track_end_discarded}"
            )
            lines.append(
                "Son OCR işi: "
                + (
                    "yok"
                    if health.last_ocr_job_age_seconds is None
                    else f"{health.last_ocr_job_age_seconds:.1f} sn önce"
                )
                + ", inference="
                + (
                    "yok"
                    if health.last_inference_ok is None
                    else "OK" if health.last_inference_ok else "HATA"
                )
                + f", aday={health.last_candidate or 'yok'}, "
                + (
                    "durum=yok"
                    if health.last_state is None
                    else f"durum={health.last_state.value}"
                )
            )
            lines.append(
                f"Queue: {health.queue_depth}, drop={health.dropped_jobs}, "
                f"stale={health.stale_jobs} "
                f"(detector-crop={health.stale_detector_crop}, "
                f"zero-fallback={health.stale_zero_detection_fallback}, "
                f"static-rescue={health.stale_static_rescue})"
            )
            lines.append(
                "Detector pass: "
                f"raw={health.raw_detector_calls} (hit={health.raw_hits}), "
                f"enhanced={health.enhanced_detector_calls} "
                f"(hit={health.enhanced_hits}), "
                f"tiled-event={health.tiled_recovery_events}, "
                f"tile-call={health.tiled_detector_calls} "
                f"(hit={health.tiled_hits}); süre raw/enhanced/tiled="
                f"{health.raw_detector_ms:.0f}/{health.enhanced_detector_ms:.0f}/"
                f"{health.tiled_detector_ms:.0f} ms"
            )
            def latency_pair(mean: float | None, p95: float | None) -> str:
                if mean is None or p95 is None:
                    return "yok"
                return f"{mean:.1f}/{p95:.1f} ms"

            def latency_triplet(
                mean: float | None,
                p95: float | None,
                maximum: float | None,
            ) -> str:
                if mean is None or p95 is None or maximum is None:
                    return "yok"
                return f"{mean:.1f}/{p95:.1f}/{maximum:.1f} ms"

            if health.detector_mean_ms is not None:
                lines.append(
                    "Latency (son 128): "
                    f"detector={latency_pair(health.detector_mean_ms, health.detector_p95_ms)}, "
                    f"OCR={latency_pair(health.ocr_mean_ms, health.ocr_p95_ms)}, "
                    f"queue={latency_pair(health.queue_wait_mean_ms, health.queue_wait_p95_ms)}, "
                    f"e2e={latency_pair(health.end_to_end_mean_ms, health.end_to_end_p95_ms)} "
                    "(mean/p95)"
                )
            for buffer in health.buffers:
                direction = buffer.direction.value if buffer.direction is not None else "?"
                lines.append(
                    f"Buffer kamera={buffer.camera_id}/{direction}: "
                    f"depth={buffer.ring_depth}/{buffer.frame_cap}, "
                    f"history={buffer.effective_ring_duration_ms:.0f}/"
                    f"{buffer.configured_duration_ms} ms, "
                    f"ingest={buffer.recognition_ingest_fps:.2f} FPS, "
                    f"ring={buffer.estimated_ring_memory_mb:.1f} MiB, "
                    f"event={buffer.active_event_frames}"
                )
            for metric in health.direction_metrics:
                lines.append(
                    f"Yön {metric.direction.value}: "
                    f"live-ingest={metric.frames_ingested}, "
                    f"live-detector={metric.live_detector_frames_processed} "
                    f"(hit={metric.live_detector_hits}, miss={metric.live_detector_misses}), "
                    f"replay-detector={metric.replay_detector_frames_processed} "
                    f"(hit={metric.replay_detector_hits}, miss={metric.replay_detector_misses}), "
                    f"superseded={metric.live_frames_superseded_before_detector}, "
                    f"OCR={metric.ocr_jobs_queued}, aday={metric.valid_candidates}, "
                    f"kayıt={metric.saved_records}"
                )
                lines.append(
                    f"Yön {metric.direction.value} latency: "
                    f"detector mean/p95="
                    f"{latency_pair(metric.detector_mean_ms, metric.detector_p95_ms)}, "
                    f"live-age mean/p95/max="
                    f"{latency_triplet(metric.live_detector_frame_age_mean_ms, metric.live_detector_frame_age_p95_ms, metric.live_detector_frame_age_max_ms)}, "
                    f"replay-age mean/p95/max="
                    f"{latency_triplet(metric.replay_detector_frame_age_mean_ms, metric.replay_detector_frame_age_p95_ms, metric.replay_detector_frame_age_max_ms)}, "
                    f"raw/enhanced/tile-call={metric.raw_detector_calls}/"
                    f"{metric.enhanced_detector_calls}/{metric.tiled_detector_calls}"
                )
            for metric in health.ocr_job_metrics:
                candidate_yield = (
                    0.0
                    if metric.jobs_processed == 0
                    else metric.valid_candidates / metric.jobs_processed * 100.0
                )
                lines.append(
                    f"OCR iş {metric.job_type.value}: "
                    f"processed={metric.jobs_processed}, "
                    f"inference-call={metric.inference_calls}, "
                    f"inference mean/p95="
                    f"{latency_pair(metric.inference_mean_ms, metric.inference_p95_ms)}, "
                    f"valid={metric.valid_candidates} ({candidate_yield:.1f}%)"
                )
            self.diagnostics_ready.emit("\n".join(lines))

        threading.Thread(target=check, name="ocr-diagnostics", daemon=True).start()

    def _arm_recognition_frame_capture(
        self,
        direction: Direction,
        _checked: bool = False,
    ) -> None:
        if self.recognition_service is None:
            self.diagnostics_label.setText("OCR servisi bu ekrana bağlı değil.")
            return
        if not self.recognition_service.arm_raw_capture(direction):
            self.diagnostics_label.setText(
                "Recognition worker henüz hazır değil; servis ACTIVE olduktan sonra tekrar deneyin."
            )
            return
        self.diagnostics_label.setText(
            f"İlk canlı {direction.value} detector hit'i için tek-shot kayıt hazır. "
            "Araç gelene kadar bekleyecek; full frame ve configured ROI "
            "debug/recognition-frames altında oluşturulacak."
        )

    @Slot(str)
    def _show_diagnostics(self, message: str) -> None:
        self.diagnostics_label.setText(message)

    def _refresh_time_diagnostics(self) -> None:
        if self._time_diagnostics_running:
            return
        self._time_diagnostics_running = True
        self.refresh_time_button.setEnabled(False)
        self.time_status_label.setText("Saat durumu kontrol ediliyor...")

        def check() -> None:
            result = self.time_diagnostics_service.check()
            self.time_diagnostics_ready.emit(result)

        threading.Thread(target=check, name="time-diagnostics", daemon=True).start()

    @Slot(object)
    def _show_time_diagnostics(self, result: TimeDiagnosticsResult) -> None:
        self._time_diagnostics_running = False
        self.refresh_time_button.setEnabled(True)
        self.local_time_label.setText(result.local_time.strftime("%d.%m.%Y %H:%M:%S"))
        self.utc_time_label.setText(result.utc_time.strftime("%d.%m.%Y %H:%M:%S"))
        self.timezone_name_label.setText(result.timezone_name)
        self.utc_offset_label.setText(result.utc_offset)
        self.windows_time_label.setText(
            "Çalışıyor" if result.windows_time_available else "Kullanılamıyor"
        )
        self.time_source_label.setText(result.time_source or "-")
        prefix = {
            TimeSyncStatus.SYNCED: "✓ ",
            TimeSyncStatus.WARNING: "⚠ ",
            TimeSyncStatus.UNAVAILABLE: "⚠ ",
            TimeSyncStatus.UNKNOWN: "? ",
        }[result.sync_status]
        self.time_status_label.setText(prefix + result.status_message)
