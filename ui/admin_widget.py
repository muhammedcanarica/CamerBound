from __future__ import annotations

from functools import partial

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
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

from app.auth import AuthService, Role, SessionUser, UserExistsError, ValidationError
from app.camera import Camera, CameraService, Direction
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
            self.auth_service.create_user(
                self.user,
                self.username_input.text(),
                self.password_input.text(),
                self.role_input.currentData(),
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
    def __init__(self, camera_service: CameraService, user: SessionUser) -> None:
        super().__init__()
        self.camera_service = camera_service
        self.user = user
        self._editors: dict[int, tuple[QLineEdit, QLineEdit, QComboBox, QCheckBox]] = {}
        self._camera_layout: QVBoxLayout
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(14)
        layout.addWidget(QLabel("Kamera Ayarları", objectName="pageTitle"))
        layout.addWidget(
            QLabel(
                "RTSP bilgileri yerel veritabanında saklanır. Credential koruması sonraki aşamadadır.",
                objectName="mutedLabel",
            )
        )
        self._camera_layout = QVBoxLayout()
        layout.addLayout(self._camera_layout)
        layout.addStretch()

    def refresh(self) -> None:
        while self._camera_layout.count():
            item = self._camera_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._editors.clear()
        for camera in self.camera_service.list_cameras():
            self._add_camera_editor(camera)

    def _add_camera_editor(self, camera: Camera) -> None:
        group = QGroupBox(f"Kamera #{camera.id}")
        form = QFormLayout(group)
        name_input = QLineEdit(camera.name)
        url_input = QLineEdit(camera.stream_url)
        url_input.setPlaceholderText("rtsp://...")
        direction_input = QComboBox()
        direction_input.addItem("Giriş", Direction.ENTRY)
        direction_input.addItem("Çıkış", Direction.EXIT)
        direction_input.setCurrentIndex(0 if camera.direction is Direction.ENTRY else 1)
        enabled_input = QCheckBox("Aktif")
        enabled_input.setChecked(camera.enabled)
        save_button = QPushButton("Kaydet")
        save_button.clicked.connect(partial(self._save_camera, camera.id))
        form.addRow("Ad", name_input)
        form.addRow("RTSP URL", url_input)
        form.addRow("Yön", direction_input)
        form.addRow("Durum", enabled_input)
        form.addRow("", save_button)
        self._camera_layout.addWidget(group)
        self._editors[camera.id] = (name_input, url_input, direction_input, enabled_input)

    def _save_camera(self, camera_id: int) -> None:
        name, url, direction, enabled = self._editors[camera_id]
        try:
            self.camera_service.update_camera(
                self.user,
                camera_id,
                name.text(),
                url.text(),
                direction.currentData(),
                enabled.isChecked(),
            )
        except (ValidationError, PermissionError) as exc:
            QMessageBox.warning(self, "Kamera kaydedilemedi", str(exc))
            return
        QMessageBox.information(
            self,
            "Başarılı",
            "Kamera ayarları kaydedildi. Değişikliğin canlı akışa uygulanması için "
            "Dashboard oturumunu yeniden açın.",
        )
