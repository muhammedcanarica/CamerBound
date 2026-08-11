from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from app.time_utils import parse_utc_timestamp, to_utc_storage
from ui.records_widget import display_timestamp


class TimestampTests(unittest.TestCase):
    def test_storage_normalizes_aware_and_naive_values_to_utc(self) -> None:
        plus_three = timezone(timedelta(hours=3))

        self.assertEqual(
            to_utc_storage(datetime(2026, 8, 11, 13, 0, tzinfo=plus_three)),
            "2026-08-11T10:00:00+00:00",
        )
        self.assertEqual(
            to_utc_storage(datetime(2026, 8, 11, 10, 0)),
            "2026-08-11T10:00:00+00:00",
        )

    def test_sqlite_current_timestamp_is_treated_as_utc(self) -> None:
        parsed = parse_utc_timestamp("2026-08-11 10:00:00")

        self.assertEqual(parsed, datetime(2026, 8, 11, 10, 0, tzinfo=timezone.utc))

    def test_display_timestamp_uses_system_local_timezone(self) -> None:
        utc_value = datetime(2026, 8, 11, 10, 0, tzinfo=timezone.utc)
        expected = utc_value.astimezone().strftime("%d.%m.%Y %H:%M:%S")

        self.assertEqual(display_timestamp("2026-08-11T10:00:00+00:00"), expected)
        self.assertEqual(display_timestamp("2026-08-11 10:00:00"), expected)
        self.assertEqual(display_timestamp("2026-08-11T13:00:00+03:00"), expected)

    def test_display_timestamp_preserves_invalid_legacy_value(self) -> None:
        self.assertEqual(display_timestamp("legacy-value"), "legacy-value")


if __name__ == "__main__":
    unittest.main()
