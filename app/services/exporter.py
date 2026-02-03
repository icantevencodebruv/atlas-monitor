import os
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from app.db.database import Database


def _split_sentences(text: str):
    parts = re.split(r"(?<=[.!?])\\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def format_dialogue(rows):
    lines = []
    for row in rows:
        sentences = _split_sentences(row["text"])
        if not sentences:
            continue
        for sentence in sentences:
            lines.append(f"{row['speaker']}: {sentence}")
    return "\n".join(lines) + ("\n" if lines else "")


def _parse_with_tz(value: str, tz: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(tz))
    return dt.astimezone(ZoneInfo("UTC"))


def compute_range(range_label: str, now: datetime, session_row, start: str = None, end: str = None, tz: str = "UTC"):
    if range_label == "30m":
        return now - timedelta(minutes=30), now
    if range_label == "60m":
        return now - timedelta(minutes=60), now
    if range_label == "today":
        local_now = now.astimezone(ZoneInfo(tz))
        start_local = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        return start_local.astimezone(ZoneInfo("UTC")), now
    if range_label == "session":
        if session_row is None:
            return now, now
        start_ts = datetime.fromisoformat(session_row["start_ts"])
        end_ts = datetime.fromisoformat(session_row["end_ts"]) if session_row["end_ts"] else now
        return start_ts, end_ts
    if range_label == "custom":
        if not start or not end:
            return now, now
        return _parse_with_tz(start, tz), _parse_with_tz(end, tz)
    return now, now


def build_export(db: Database, range_label: str, start_ts: datetime, end_ts: datetime, exports_dir: str) -> int:
    os.makedirs(exports_dir, exist_ok=True)
    rows = db.list_transcripts_between(start_ts.isoformat(), end_ts.isoformat())
    content = format_dialogue(rows)
    filename = f"export_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.txt"
    file_path = os.path.join(exports_dir, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return db.add_export(range_label, start_ts.isoformat(), end_ts.isoformat(), file_path)
