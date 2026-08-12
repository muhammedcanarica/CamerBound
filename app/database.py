from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from app.time_utils import to_utc_storage


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL COLLATE NOCASE UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('ADMIN', 'USER')),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S+00:00', 'now'))
);

CREATE TABLE IF NOT EXISTS cameras (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    stream_url TEXT NOT NULL DEFAULT '',
    direction TEXT NOT NULL CHECK (direction IN ('ENTRY', 'EXIT')),
    enabled INTEGER NOT NULL DEFAULT 0 CHECK (enabled IN (0, 1))
);

CREATE TABLE IF NOT EXISTS plate_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('ENTRY', 'EXIT')),
    camera_id INTEGER NOT NULL,
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    timestamp TEXT NOT NULL,
    image_path TEXT NULL,
    FOREIGN KEY (camera_id) REFERENCES cameras(id) ON UPDATE CASCADE ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_plate_records_plate ON plate_records(plate);
CREATE INDEX IF NOT EXISTS idx_plate_records_timestamp ON plate_records(timestamp);
CREATE INDEX IF NOT EXISTS idx_plate_records_camera_plate_time
ON plate_records(camera_id, plate, timestamp DESC);
"""


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connection() as connection:
            connection.executescript(SCHEMA)
            self._migrate_plate_records(connection)
            self._seed_cameras(connection)
            self._normalize_legacy_timestamps(connection)

    @staticmethod
    def _migrate_plate_records(connection: sqlite3.Connection) -> None:
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(plate_records)").fetchall()
        }
        if "image_path" not in columns:
            connection.execute("ALTER TABLE plate_records ADD COLUMN image_path TEXT NULL")

    @staticmethod
    def _seed_cameras(connection: sqlite3.Connection) -> None:
        camera_count = connection.execute("SELECT COUNT(*) FROM cameras").fetchone()[0]
        if camera_count == 0:
            connection.executemany(
                """
                INSERT INTO cameras (name, stream_url, direction, enabled)
                VALUES (?, '', ?, 0)
                """,
                (("Giriş Kamerası", "ENTRY"), ("Çıkış Kamerası", "EXIT")),
            )

    @staticmethod
    def _normalize_legacy_timestamps(connection: sqlite3.Connection) -> None:
        for table, column in (
            ("users", "created_at"),
            ("plate_records", "timestamp"),
        ):
            rows = connection.execute(
                f"SELECT id, {column} FROM {table}"
            ).fetchall()
            updates: list[tuple[str, int]] = []
            for row in rows:
                original = row[column]
                try:
                    normalized = to_utc_storage(original)
                except (TypeError, ValueError):
                    continue
                if normalized != original:
                    updates.append((normalized, row["id"]))
            if updates:
                connection.executemany(
                    f"UPDATE {table} SET {column} = ? WHERE id = ?",
                    updates,
                )
