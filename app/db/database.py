import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Iterable, List, Optional


class Database:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS segments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    start_ts TEXT NOT NULL,
                    end_ts TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    attempts INTEGER DEFAULT 0,
                    last_attempt_at TEXT,
                    created_ts TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS transcripts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    segment_id INTEGER,
                    start_ts TEXT NOT NULL,
                    end_ts TEXT NOT NULL,
                    speaker TEXT NOT NULL,
                    text TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    speaker TEXT PRIMARY KEY,
                    vector TEXT NOT NULL,
                    updated_ts TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_ts TEXT NOT NULL,
                    end_ts TEXT
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS exports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_ts TEXT NOT NULL,
                    range_label TEXT NOT NULL,
                    start_ts TEXT NOT NULL,
                    end_ts TEXT NOT NULL,
                    file_path TEXT NOT NULL
                )
                """
            )
        self._migrate()

    def _migrate(self) -> None:
        with self._lock, self._conn:
            cur = self._conn.execute("PRAGMA table_info(segments)")
            cols = {row["name"] for row in cur.fetchall()}
            if "attempts" not in cols:
                self._conn.execute("ALTER TABLE segments ADD COLUMN attempts INTEGER DEFAULT 0")
            if "last_attempt_at" not in cols:
                self._conn.execute("ALTER TABLE segments ADD COLUMN last_attempt_at TEXT")

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def add_segment(self, file_path: str, start_ts: str, end_ts: str) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO segments (file_path, start_ts, end_ts, status, created_ts, attempts, last_attempt_at)
                VALUES (?, ?, ?, ?, ?, 0, NULL)
                """,
                (file_path, start_ts, end_ts, "pending", self._utc_now()),
            )
            return int(cur.lastrowid)

    def update_segment_status(self, segment_id: int, status: str, error: Optional[str] = None) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE segments SET status = ?, error = ? WHERE id = ?",
                (status, error, segment_id),
            )

    def mark_attempt(self, segment_id: int) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE segments
                SET attempts = COALESCE(attempts, 0) + 1,
                    last_attempt_at = ?
                WHERE id = ?
                """,
                (self._utc_now(), segment_id),
            )

    def get_segment(self, segment_id: int):
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM segments WHERE id = ?", (segment_id,)
            )
            return cur.fetchone()

    def list_failed_segments(self) -> List[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT * FROM segments
                WHERE status = 'failed'
                ORDER BY end_ts DESC
                """
            )
            return list(cur.fetchall())

    def delete_segment(self, segment_id: int) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE segments SET status = 'deleted', error = 'deleted' WHERE id = ?",
                (segment_id,),
            )

    def delete_segment_row(self, segment_id: int) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM segments WHERE id = ?", (segment_id,))

    def add_transcripts(self, segment_id: int, rows: Iterable[dict]) -> None:
        with self._lock, self._conn:
            self._conn.executemany(
                """
                INSERT INTO transcripts (segment_id, start_ts, end_ts, speaker, text)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        segment_id,
                        row["start_ts"],
                        row["end_ts"],
                        row["speaker"],
                        row["text"],
                    )
                    for row in rows
                ],
            )

    def list_transcripts_between(self, start_ts: str, end_ts: str) -> List[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT * FROM transcripts
                WHERE start_ts < ? AND end_ts > ?
                ORDER BY start_ts ASC
                """,
                (end_ts, start_ts),
            )
            return list(cur.fetchall())

    def set_embedding(self, speaker: str, vector: List[float]) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO embeddings (speaker, vector, updated_ts)
                VALUES (?, ?, ?)
                ON CONFLICT(speaker) DO UPDATE SET vector = excluded.vector, updated_ts = excluded.updated_ts
                """,
                (speaker, json.dumps(vector), self._utc_now()),
            )

    def get_embedding(self, speaker: str) -> Optional[List[float]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT vector FROM embeddings WHERE speaker = ?", (speaker,)
            )
            row = cur.fetchone()
        if not row:
            return None
        return json.loads(row["vector"])

    def add_session(self, start_ts: str) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO sessions (start_ts) VALUES (?)", (start_ts,)
            )
            return int(cur.lastrowid)

    def end_session(self, session_id: int, end_ts: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE sessions SET end_ts = ? WHERE id = ?",
                (end_ts, session_id),
            )

    def get_latest_session(self):
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM sessions ORDER BY id DESC LIMIT 1"
            )
            return cur.fetchone()

    def add_export(self, range_label: str, start_ts: str, end_ts: str, file_path: str) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO exports (created_ts, range_label, start_ts, end_ts, file_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                (self._utc_now(), range_label, start_ts, end_ts, file_path),
            )
            return int(cur.lastrowid)

    def get_export(self, export_id: int):
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM exports WHERE id = ?", (export_id,)
            )
            return cur.fetchone()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
