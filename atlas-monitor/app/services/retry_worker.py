import logging
import threading
import time
from datetime import datetime, timezone

from app.db.database import Database

logger = logging.getLogger(__name__)


class RetryWorker:
    def __init__(self, db: Database, queue, config):
        self._db = db
        self._queue = queue
        self._config = config.retry
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        if not self._config.enabled:
            return
        self._thread.start()

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)

    def _backoff_sec(self, attempts: int) -> int:
        base = self._config.base_backoff_sec
        delay = base * (2 ** max(0, attempts - 1))
        return min(delay, self._config.max_backoff_sec)

    def _eligible(self, row) -> bool:
        attempts = int(row["attempts"] or 0)
        if attempts >= self._config.max_attempts:
            return False
        last = row["last_attempt_at"]
        if not last:
            return True
        last_dt = datetime.fromisoformat(last)
        next_dt = last_dt + time_delta(self._backoff_sec(attempts))
        return datetime.now(timezone.utc) >= next_dt

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception as exc:
                logger.exception("Retry worker error: %s", exc)
            time.sleep(self._config.poll_interval_sec)

    def _tick(self) -> None:
        rows = self._db.list_failed_segments()
        for row in rows:
            if not self._eligible(row):
                continue
            self._db.update_segment_status(row["id"], "pending", None)
            self._queue.put(int(row["id"]))


def time_delta(seconds: int):
    from datetime import timedelta

    return timedelta(seconds=seconds)
