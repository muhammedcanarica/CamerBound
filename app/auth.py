from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from enum import StrEnum

import bcrypt

from app.audit import AuditAction, AuditService
from app.database import Database
from app.time_utils import to_utc_storage, utc_now


class Role(StrEnum):
    ADMIN = "ADMIN"
    USER = "USER"


@dataclass(frozen=True, slots=True)
class SessionUser:
    id: int
    username: str
    role: Role


class AuthenticationError(ValueError):
    pass


class AuthorizationError(PermissionError):
    pass


class ValidationError(ValueError):
    pass


class UserExistsError(ValidationError):
    pass


class AuthService:
    # Development bootstrap account. It must be changed before production use.
    DEFAULT_ADMIN_USERNAME = "admin"
    DEFAULT_ADMIN_PASSWORD = "admin123"

    def __init__(
        self,
        database: Database,
        audit_service: AuditService | None = None,
    ) -> None:
        self.database = database
        self.audit_service = audit_service or AuditService(database)

    def ensure_default_admin(self) -> bool:
        with self.database.connection() as connection:
            user_count = connection.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            if user_count > 0:
                return False
            connection.execute(
                """
                INSERT INTO users (username, password_hash, role, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    self.DEFAULT_ADMIN_USERNAME,
                    self._hash_password(self.DEFAULT_ADMIN_PASSWORD),
                    Role.ADMIN.value,
                    to_utc_storage(utc_now()),
                ),
            )
        return True

    def authenticate(self, username: str, password: str) -> SessionUser:
        normalized_username = username.strip()
        if not normalized_username or not password:
            self.audit_service.try_log(
                AuditAction.LOGIN_FAILURE,
                username=normalized_username,
            )
            raise AuthenticationError("Kullanıcı adı ve şifre zorunludur.")

        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT id, username, password_hash, role FROM users WHERE username = ?",
                (normalized_username,),
            ).fetchone()

        if row is None or not self._password_matches(password, row["password_hash"]):
            self.audit_service.try_log(
                AuditAction.LOGIN_FAILURE,
                username=normalized_username,
            )
            raise AuthenticationError("Kullanıcı adı veya şifre hatalı.")

        user = SessionUser(
            id=row["id"], username=row["username"], role=Role(row["role"])
        )
        self.audit_service.try_log(AuditAction.LOGIN_SUCCESS, actor=user)
        return user

    def create_user(
        self, actor: SessionUser, username: str, password: str, role: Role | str
    ) -> SessionUser:
        try:
            normalized_role = role if isinstance(role, Role) else Role(role)
        except (TypeError, ValueError) as exc:
            raise ValidationError("Geçersiz kullanıcı rolü.") from exc

        self.require_admin(actor)
        normalized_username = username.strip()
        self._validate_new_user(normalized_username, password)

        try:
            with self.database.connection() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO users (username, password_hash, role, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        normalized_username,
                        self._hash_password(password),
                        normalized_role.value,
                        to_utc_storage(utc_now()),
                    ),
                )
                user_id = int(cursor.lastrowid)
        except sqlite3.IntegrityError as exc:
            raise UserExistsError("Bu kullanıcı adı zaten kullanılıyor.") from exc

        created = SessionUser(
            id=user_id,
            username=normalized_username,
            role=normalized_role,
        )
        self.audit_service.try_log(
            AuditAction.USER_CREATED,
            actor=actor,
            details=(
                f"created_username={created.username}; role={created.role.value}"
            ),
        )
        return created

    def list_users(self, actor: SessionUser) -> list[sqlite3.Row]:
        self.require_admin(actor)
        with self.database.connection() as connection:
            return connection.execute(
                "SELECT id, username, role, created_at FROM users ORDER BY username"
            ).fetchall()

    @staticmethod
    def require_admin(actor: SessionUser) -> None:
        if actor.role is not Role.ADMIN:
            raise AuthorizationError("Bu işlem yalnızca ADMIN rolüyle yapılabilir.")

    @staticmethod
    def require_authenticated(actor: SessionUser | None) -> SessionUser:
        if actor is None:
            raise AuthorizationError("Bu işlem için giriş yapmalısınız.")
        return actor

    @staticmethod
    def _hash_password(password: str) -> str:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    @staticmethod
    def _password_matches(password: str, password_hash: str) -> bool:
        try:
            return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
        except ValueError:
            return False

    @staticmethod
    def _validate_new_user(username: str, password: str) -> None:
        if len(username) < 3 or len(username) > 50:
            raise ValidationError("Kullanıcı adı 3-50 karakter arasında olmalıdır.")
        if len(password) < 8:
            raise ValidationError("Şifre en az 8 karakter olmalıdır.")
