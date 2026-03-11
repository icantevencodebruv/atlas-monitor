from datetime import datetime, timedelta, timezone

from app.database import Database


def test_transcript_ordering(tmp_path):
    db_path = tmp_path / "app.db"
    db = Database(str(db_path))
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = [
        {"start_ts": (base + timedelta(seconds=30)).isoformat(), "end_ts": (base + timedelta(seconds=40)).isoformat(), "speaker": "Hugo", "text": "later"},
        {"start_ts": (base + timedelta(seconds=10)).isoformat(), "end_ts": (base + timedelta(seconds=20)).isoformat(), "speaker": "Leon", "text": "earlier"},
    ]
    db.add_transcripts(1, rows)
    result = db.list_transcripts_between(base.isoformat(), (base + timedelta(seconds=50)).isoformat())
    assert result[0]["text"] == "earlier"
    assert result[1]["text"] == "later"
