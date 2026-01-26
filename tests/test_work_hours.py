from datetime import datetime
from zoneinfo import ZoneInfo

from app.config import Config
from app.services.scheduling import is_within_work_hours


def test_work_hours_basic():
    cfg = Config().work_hours
    tz = ZoneInfo(cfg.timezone)
    dt = datetime(2024, 1, 2, 10, 0, tzinfo=tz)
    assert is_within_work_hours(cfg, dt) is True
    dt = datetime(2024, 1, 2, 20, 0, tzinfo=tz)
    assert is_within_work_hours(cfg, dt) is False
