from __future__ import annotations

import base64
import ctypes
import sys
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import quote, unquote, urlsplit, urlunsplit


SUPPORTED_CAMERA_URL_SCHEMES = {"rtsp", "http", "https"}
CRYPTPROTECT_UI_FORBIDDEN = 0x01


class CredentialProtectionError(RuntimeError):
    """Raised without including credential material in the message."""


class PasswordProtector(Protocol):
    def protect(self, plaintext: str) -> str: ...

    def unprotect(self, protected_value: str) -> str: ...


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", ctypes.c_ulong),
        ("pbData", ctypes.POINTER(ctypes.c_ubyte)),
    ]


class DpapiPasswordProtector:
    """Protect camera passwords with Windows DPAPI current-user scope."""

    def protect(self, plaintext: str) -> str:
        if sys.platform != "win32":
            raise CredentialProtectionError(
                "Camera passwords require Windows DPAPI."
            )
        protected = self._crypt_protect(plaintext.encode("utf-8"))
        return base64.b64encode(protected).decode("ascii")

    def unprotect(self, protected_value: str) -> str:
        if sys.platform != "win32":
            raise CredentialProtectionError(
                "Camera passwords require Windows DPAPI."
            )
        try:
            encrypted = base64.b64decode(
                protected_value.encode("ascii"),
                validate=True,
            )
            plaintext = self._crypt_unprotect(encrypted)
            return plaintext.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise CredentialProtectionError(
                "The protected camera password could not be opened."
            ) from exc

    @staticmethod
    def _input_blob(value: bytes) -> tuple[_DataBlob, ctypes.Array[ctypes.c_char]]:
        buffer = ctypes.create_string_buffer(value, max(1, len(value)))
        blob = _DataBlob(
            len(value),
            ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)),
        )
        return blob, buffer

    @classmethod
    def _crypt_protect(cls, plaintext: bytes) -> bytes:
        input_blob, input_buffer = cls._input_blob(plaintext)
        output_blob = _DataBlob()
        crypt32 = ctypes.windll.crypt32
        kernel32 = ctypes.windll.kernel32
        crypt32.CryptProtectData.argtypes = [
            ctypes.POINTER(_DataBlob),
            ctypes.c_wchar_p,
            ctypes.POINTER(_DataBlob),
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.POINTER(_DataBlob),
        ]
        crypt32.CryptProtectData.restype = ctypes.c_int
        kernel32.LocalFree.argtypes = [ctypes.c_void_p]
        kernel32.LocalFree.restype = ctypes.c_void_p
        success = crypt32.CryptProtectData(
            ctypes.byref(input_blob),
            "CamerBound camera password",
            None,
            None,
            None,
            CRYPTPROTECT_UI_FORBIDDEN,
            ctypes.byref(output_blob),
        )
        del input_buffer
        if not success:
            raise CredentialProtectionError(
                "The camera password could not be protected with Windows DPAPI."
            )
        try:
            return ctypes.string_at(output_blob.pbData, output_blob.cbData)
        finally:
            kernel32.LocalFree(ctypes.cast(output_blob.pbData, ctypes.c_void_p))

    @classmethod
    def _crypt_unprotect(cls, protected_value: bytes) -> bytes:
        input_blob, input_buffer = cls._input_blob(protected_value)
        output_blob = _DataBlob()
        crypt32 = ctypes.windll.crypt32
        kernel32 = ctypes.windll.kernel32
        crypt32.CryptUnprotectData.argtypes = [
            ctypes.POINTER(_DataBlob),
            ctypes.POINTER(ctypes.c_wchar_p),
            ctypes.POINTER(_DataBlob),
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.POINTER(_DataBlob),
        ]
        crypt32.CryptUnprotectData.restype = ctypes.c_int
        kernel32.LocalFree.argtypes = [ctypes.c_void_p]
        kernel32.LocalFree.restype = ctypes.c_void_p
        success = crypt32.CryptUnprotectData(
            ctypes.byref(input_blob),
            None,
            None,
            None,
            None,
            CRYPTPROTECT_UI_FORBIDDEN,
            ctypes.byref(output_blob),
        )
        del input_buffer
        if not success:
            raise CredentialProtectionError(
                "The protected camera password could not be opened."
            )
        try:
            return ctypes.string_at(output_blob.pbData, output_blob.cbData)
        finally:
            kernel32.LocalFree(ctypes.cast(output_blob.pbData, ctypes.c_void_p))


@dataclass(frozen=True, slots=True)
class ParsedCameraSource:
    stream_url: str
    username: str | None = None
    password: str | None = None

    @property
    def had_credentials(self) -> bool:
        return self.username is not None or self.password is not None


def split_camera_source_credentials(source: str) -> ParsedCameraSource:
    """Remove URL userinfo while preserving the path, query, and fragment."""
    value = source.strip()
    try:
        parsed = urlsplit(value)
    except ValueError:
        return ParsedCameraSource(value)
    if "@" not in parsed.netloc:
        return ParsedCameraSource(value)
    if parsed.scheme.lower() not in SUPPORTED_CAMERA_URL_SCHEMES:
        raise ValueError("Camera URL credentials use an unsupported scheme.")

    username = (
        unquote(parsed.username, encoding="utf-8", errors="strict")
        if parsed.username is not None
        else None
    )
    password = (
        unquote(parsed.password, encoding="utf-8", errors="strict")
        if parsed.password is not None
        else None
    )
    credential_free_netloc = parsed.netloc.rsplit("@", 1)[1]
    clean_url = urlunsplit(
        (
            parsed.scheme,
            credential_free_netloc,
            parsed.path,
            parsed.query,
            parsed.fragment,
        )
    )
    return ParsedCameraSource(clean_url, username, password)


def build_authenticated_camera_source(
    stream_url: str,
    username: str,
    password: str,
) -> str:
    """Build an in-memory userinfo URL for OpenCV without changing storage."""
    parsed_source = split_camera_source_credentials(stream_url)
    parsed = urlsplit(parsed_source.stream_url)
    if (
        parsed.scheme.lower() not in SUPPORTED_CAMERA_URL_SCHEMES
        or not parsed.netloc
    ):
        raise CredentialProtectionError(
            "Camera credentials can only be used with HTTP, HTTPS, or RTSP URLs."
        )
    userinfo = f"{quote(username, safe='')}:{quote(password, safe='')}"
    return urlunsplit(
        (
            parsed.scheme,
            f"{userinfo}@{parsed.netloc}",
            parsed.path,
            parsed.query,
            parsed.fragment,
        )
    )


def is_authenticatable_camera_source(stream_url: str) -> bool:
    try:
        parsed = urlsplit(stream_url)
    except ValueError:
        return False
    return parsed.scheme.lower() in SUPPORTED_CAMERA_URL_SCHEMES and bool(
        parsed.netloc
    )
