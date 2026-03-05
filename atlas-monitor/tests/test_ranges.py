from datetime import datetime, timezone

from app.services.exporter import compute_range
from app.config import Config


def test_compute_range_30m():
    cfg = Config()
    now = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
    start, end = compute_range("30m", now, None, tz=cfg.work_hours.timezone)
    assert end == now
    assert (end - start).total_seconds() == 1800


def test_compute_range_today():
    cfg = Config()
    now = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
    start, end = compute_range("today", now, None, tz=cfg.work_hours.timezone)
    assert end == now
    assert start.tzinfo is not None
