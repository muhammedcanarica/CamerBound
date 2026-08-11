from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.auth import AuthService, AuthenticationError, SessionUser


class LoginWindow(QMainWindow):
    authenticated = Signal(object)

    def __init__(self, auth_service: AuthService) -> None:
        super().__init__()
        self.auth_service = auth_service
        self.setWindowTitle("Plaka Takip Sistemi - Giriş")
        self.setMinimumSize(480, 520)
        self.resize(560, 620)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget(objectName="loginRoot")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(70, 60, 70, 60)
        root_layout.addStretch()

        card = QFrame(objectName="loginCard")
        card.setMaximumWidth(420)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(36, 38, 36, 38)
        card_layout.setSpacing(14)

        title = QLabel("Plaka Takip Sistemi", objectName="appTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle = QLabel("Güvenli yerel oturum", objectName="mutedLabel")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)

        username_label = QLabel("Kullanıcı adı")
        self.username_input = QLineEdit()
        self.username_input.setObjectName("usernameInput")
        self.username_input.setPlaceholderText("Kullanıcı adınızı girin")
        self.username_input.setClearButtonEnabled(True)

        password_label = QLabel("Şifre")
        self.password_input = QLineEdit()
        self.password_input.setObjectName("passwordInput")
        self.password_input.setPlaceholderText("Şifrenizi girin")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_input.returnPressed.connect(self._attempt_login)

        self.login_button = QPushButton("Giriş Yap")
        self.login_button.setObjectName("loginButton")
        self.login_button.clicked.connect(self._attempt_login)

        card_layout.addWidget(title)
        card_layout.addWidget(subtitle)
        card_layout.addSpacing(15)
        card_layout.addWidget(username_label)
        card_layout.addWidget(self.username_input)
        card_layout.addWidget(password_label)
        card_layout.addWidget(self.password_input)
        card_layout.addSpacing(8)
        card_layout.addWidget(self.login_button)

        root_layout.addWidget(card, alignment=Qt.AlignmentFlag.AlignCenter)
        root_layout.addStretch()
        self.setCentralWidget(root)

    def _attempt_login(self) -> None:
        try:
            user = self.auth_service.authenticate(
                self.username_input.text(), self.password_input.text()
            )
        except AuthenticationError as exc:
            QMessageBox.warning(self, "Giriş başarısız", str(exc))
            self.password_input.clear()
            self.password_input.setFocus()
            return

        self.password_input.clear()
        self.authenticated.emit(user)

    def reset(self) -> None:
        self.username_input.clear()
        self.password_input.clear()
        self.username_input.setFocus()
