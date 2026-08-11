from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL COLLATE NOCASE UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('ADMIN', 'USER')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
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
    FOREIGN KEY (camera_id) REFERENCES cameras(id) ON UPDATE CASCADE ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_plate_records_plate ON plate_records(plate);
CREATE INDEX IF NOT EXISTS idx_plate_records_timestamp ON plate_records(timestamp);
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
            self._seed_cameras(connection)

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
