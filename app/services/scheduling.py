import logging
import threading
import time
from datetime import datetime, time as dtime, timezone
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from app.state import AppState

logger = logging.getLogger(__name__)


class WorkHoursScheduler:
    def __init__(
        self,
        config,
        state: AppState,
        recorder,
        db,
        on_workday_end: Optional[Callable[[datetime], None]] = None,
        now_provider: Optional[Callable[[ZoneInfo], datetime]] = None,
    ):
        self._config = config
        self._state = state
        self._recorder = recorder
        self._db = db
        self._on_workday_end = on_workday_end
        self._last_in_hours: Optional[bool] = None
        self._now_provider = now_provider or (lambda tz: datetime.now(tz))
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._tick()
        self._thread.start()

    def shutdown(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception as exc:
                logger.exception("Scheduler error: %s", exc)
            time.sleep(30)

    def _tick(self) -> None:
        cfg = self._config.work_hours
        if not cfg.enabled:
            return
        tz = ZoneInfo(cfg.timezone)
        now = self._now_provider(tz)
        in_hours = is_within_work_hours(cfg, now)

        if self._last_in_hours is True and not in_hours and self._on_workday_end:
            try:
                self._on_workday_end(now)
            except Exception as exc:
                logger.exception("Workday-end hook failed: %s", exc)
        self._last_in_hours = in_hours

        with self._state.lock:
            if self._state.manual_override is None:
                desired = in_hours
            else:
                desired = self._state.manual_override
                if not in_hours and self._state.manual_override is False:
                    self._state.manual_override = None

        if desired and not self._state.recording:
            logger.info("Scheduler starting recorder.")
            self._recorder.start()
            self._state.recording = True
            self._state.recording_since = datetime.now(timezone.utc).isoformat()
            if self._state.current_session_id is None:
                self._state.current_session_id = self._db.add_session(datetime.now(timezone.utc).isoformat())
        if not desired and self._state.recording:
            logger.info("Scheduler stopping recorder.")
            self._recorder.stop()
            self._state.recording = False
            self._state.recording_since = None
            self._state.last_segment_start = None
            if self._state.current_session_id is not None:
                self._db.end_session(self._state.current_session_id, datetime.now(timezone.utc).isoformat())
                self._state.current_session_id = None

    def tick(self) -> None:
        self._tick()


def is_within_work_hours(cfg, now: datetime) -> bool:
    weekday = now.strftime("%a").upper()[:3]
    in_day = weekday in cfg.work_days
    start = dtime.fromisoformat(cfg.work_start)
    end = dtime.fromisoformat(cfg.work_end)
    return in_day and (start <= now.time() <= end)
