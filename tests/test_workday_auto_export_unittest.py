import unittest
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from app.config import Config
from app.database import Database
from app.services.exporter import compute_workday_range
from app.services.scheduling import WorkHoursScheduler


class TestWorkdayRange(unittest.TestCase):
    def test_compute_workday_range_berlin_winter(self):
        # 2024-01-02 is winter time in Europe/Berlin (UTC+1).
        now = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
        start, end = compute_workday_range(now, "Europe/Berlin", "09:00", "18:00")
        self.assertEqual(start.isoformat(), "2024-01-02T08:00:00+00:00")
        self.assertEqual(end.isoformat(), "2024-01-02T17:00:00+00:00")


class _RecorderStub:
    def __init__(self):
        self.started = 0
        self.stopped = 0

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1


class _DbStub:
    def __init__(self):
        self.sessions_started = 0
        self.sessions_ended = 0

    def add_session(self, _start_ts: str) -> int:
        self.sessions_started += 1
        return self.sessions_started

    def end_session(self, _session_id: int, _end_ts: str) -> None:
        self.sessions_ended += 1


class TestSchedulerEdge(unittest.TestCase):
    def test_workday_end_hook_fires_once_on_true_to_false(self):
        cfg = Config()
        cfg.work_hours.enabled = True
        cfg.work_hours.timezone = "UTC"
        cfg.work_hours.work_days = ["TUE"]
        cfg.work_hours.work_start = "09:00"
        cfg.work_hours.work_end = "18:00"

        recorder = _RecorderStub()
        db = _DbStub()

        calls = []

        def on_end(now_local: datetime):
            calls.append(now_local.isoformat())

        seq = [
            datetime(2024, 1, 2, 10, 0, tzinfo=ZoneInfo("UTC")),  # in hours
            datetime(2024, 1, 2, 19, 0, tzinfo=ZoneInfo("UTC")),  # out of hours
            datetime(2024, 1, 2, 19, 30, tzinfo=ZoneInfo("UTC")),  # still out
        ]
        idx = {"i": 0}

        def now_provider(_tz):
            i = idx["i"]
            idx["i"] = min(i + 1, len(seq) - 1)
            return seq[i]

        from app.state import AppState

        scheduler = WorkHoursScheduler(
            cfg, AppState(), recorder, db, on_workday_end=on_end, now_provider=now_provider
        )
        scheduler.tick()  # establish last_in_hours=True
        scheduler.tick()  # transition to False => hook
        scheduler.tick()  # no additional hook

        self.assertEqual(len(calls), 1)


class TestDbDedupe(unittest.TestCase):
    def test_find_export(self):
        # In-memory DB is fine; schema init should succeed.
        db = Database(":memory:")
        export_id = db.add_export(
            "workday",
            "2024-01-02T08:00:00+00:00",
            "2024-01-02T17:00:00+00:00",
            "/tmp/export.txt",
        )
        found = db.find_export(
            "workday",
            "2024-01-02T08:00:00+00:00",
            "2024-01-02T17:00:00+00:00",
        )
        self.assertIsNotNone(found)
        self.assertEqual(found["id"], export_id)


if __name__ == "__main__":
    unittest.main()

