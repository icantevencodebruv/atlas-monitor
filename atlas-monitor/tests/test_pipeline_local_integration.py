import os
from datetime import datetime, timedelta, timezone

import numpy as np

from app.asr.manager import select_backend
from app.asr.pipeline_local import PipelineLocalBackend
from app.config import Config
from app.db.database import Database
from app.services.audio_utils import write_wav_int16
from app.services.diarization import SpeakerIdentifier
from app.services.transcription import TranscriptionWorker
from app.state import AppState


class _DummyPipelineBackendSuccess:
    name = "pipeline_local"
    languages = ["en", "de"]

    def transcribe_segment(self, _wav_path, segment_start_ts, *_args, **_kwargs):
        start = datetime.fromisoformat(segment_start_ts)
        end = start + timedelta(seconds=1)
        return [
            {
                "start_ts": start.isoformat(),
                "end_ts": end.isoformat(),
                "speaker": "Hugo",
                "text": "pipeline text",
                "low_confidence": False,
            }
        ]


class _DummyPipelineBackendFail:
    name = "pipeline_local"
    languages = ["en", "de"]

    def transcribe_segment(self, _wav_path, segment_start_ts, *_args, **_kwargs):
        start = datetime.fromisoformat(segment_start_ts)
        end = start + timedelta(seconds=1)
        return [
            {
                "start_ts": start.isoformat(),
                "end_ts": end.isoformat(),
                "speaker": "Unknown",
                "text": "",
                "low_confidence": True,
            }
        ]


def test_select_backend_pipeline_local_requested(monkeypatch):
    cfg = Config()
    cfg.asr.backend = "pipeline_local"
    monkeypatch.setattr(PipelineLocalBackend, "is_available", classmethod(lambda cls: True))
    backend = select_backend(cfg)
    assert backend.name == "pipeline_local"


def test_pipeline_backend_worker_path_saves_transcript_and_deletes_audio(tmp_path):
    db = Database(str(tmp_path / "app.db"))
    audio_path = tmp_path / "segment.wav"
    start = datetime.now(timezone.utc)
    write_wav_int16(str(audio_path), np.zeros(16000, dtype=np.int16), 16000)
    segment_id = db.add_segment(str(audio_path), start.isoformat(), start.isoformat())

    worker = TranscriptionWorker(
        db=db,
        queue=None,
        asr_backend=_DummyPipelineBackendSuccess(),
        diarizer=SpeakerIdentifier(),
        config=Config(),
        state=AppState(),
    )
    worker._process_segment(segment_id)

    rows = db.list_transcripts_between(
        (start - timedelta(minutes=1)).isoformat(),
        (start + timedelta(minutes=1)).isoformat(),
    )
    assert len(rows) == 1
    assert rows[0]["speaker"] == "Hugo"
    assert rows[0]["text"] == "pipeline text"
    assert not os.path.exists(audio_path)


def test_pipeline_backend_worker_path_marks_failed_on_empty_text(tmp_path):
    db = Database(str(tmp_path / "app.db"))
    audio_path = tmp_path / "segment.wav"
    start = datetime.now(timezone.utc)
    write_wav_int16(str(audio_path), np.zeros(16000, dtype=np.int16), 16000)
    segment_id = db.add_segment(str(audio_path), start.isoformat(), start.isoformat())

    worker = TranscriptionWorker(
        db=db,
        queue=None,
        asr_backend=_DummyPipelineBackendFail(),
        diarizer=SpeakerIdentifier(),
        config=Config(),
        state=AppState(),
    )
    worker._process_segment(segment_id)

    row = db.get_segment(segment_id)
    assert row["status"] == "failed"
    assert os.path.exists(audio_path)
