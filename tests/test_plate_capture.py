from __future__ import annotations

import re
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from app.database import Database
from app.plate_capture import PlateCaptureService


class PlateCaptureStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_directory.name)
        self.service = PlateCaptureService(
            self.root / "data" / "captures",
            self.root,
            max_width=960,
            jpeg_quality=60,
        )

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def test_capture_uses_safe_year_month_path_and_writes_resized_jpeg(self) -> None:
        frame = np.full((1200, 2000, 3), 180, dtype=np.uint8)

        image_path = self.service.save_capture(
            frame,
            "../../34 abc 123",
            "../ENTRY",
            datetime(2026, 8, 12, 8, 20, 31, tzinfo=timezone.utc),
            42,
        )

        self.assertIsNotNone(image_path)
        self.assertTrue(image_path.startswith("data/captures/2026/08/"))
        self.assertRegex(
            Path(image_path).name,
            re.compile(r"^[A-Z0-9]+_20260812_082031_[A-Z0-9]+_42\.jpg$"),
        )
        target = self.service.resolve_reference(image_path)
        self.assertIsNotNone(target)
        self.assertTrue(target.is_file())
        decoded = cv2.imread(str(target))
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.shape[1], 960)
        self.assertLess(target.stat().st_size, frame.nbytes)

    def test_disabled_capture_does_not_write_a_file(self) -> None:
        service = PlateCaptureService(
            self.root / "data" / "captures",
            self.root,
            enabled=False,
        )

        result = service.save_capture(
            np.zeros((100, 200, 3), dtype=np.uint8),
            "34ABC123",
            "ENTRY",
            datetime.now(timezone.utc),
            1,
        )

        self.assertIsNone(result)
        self.assertEqual(list(self.root.rglob("*.jpg")), [])

    def test_reference_resolution_rejects_path_traversal(self) -> None:
        self.assertIsNone(self.service.resolve_reference("../../outside.jpg"))
        self.assertIsNone(self.service.resolve_reference(str(self.root / "absolute.jpg")))


class PlateRecordMigrationTests(unittest.TestCase):
    def test_legacy_database_gains_nullable_image_path_without_losing_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            database_path = Path(temp_directory) / "legacy.db"
            connection = sqlite3.connect(database_path)
            connection.executescript(
                """
                CREATE TABLE plate_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    camera_id INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp TEXT NOT NULL
                );
                INSERT INTO plate_records
                    (plate, direction, camera_id, confidence, timestamp)
                VALUES
                    ('34LEGACY', 'ENTRY', 1, 0.9, '2026-08-12T08:00:00+00:00');
                """
            )
            connection.commit()
            connection.close()

            database = Database(database_path)
            database.initialize()

            with database.connection() as connection:
                columns = {
                    row["name"]
                    for row in connection.execute("PRAGMA table_info(plate_records)")
                }
                row = connection.execute(
                    "SELECT plate, image_path FROM plate_records"
                ).fetchone()
            self.assertIn("image_path", columns)
            self.assertEqual(row["plate"], "34LEGACY")
            self.assertIsNone(row["image_path"])


if __name__ == "__main__":
    unittest.main()
