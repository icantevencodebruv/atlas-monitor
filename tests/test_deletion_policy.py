import os
from datetime import datetime, timezone

import numpy as np

from app.config import Config
from app.db.database import Database
from app.services.audio_utils import write_wav_int16
from app.services.diarization import SpeakerIdentifier
from app.services.transcription import TranscriptionWorker
from app.services.vad import VadSegment
import app.services.transcription as transcription_mod


class DummyBackendSuccess:
    name = "dummy"
    languages = ["en", "de"]

    def transcribe(self, wav_path: str) -> str:
        return "hello"


class DummyBackendFail:
    name = "dummy"
    languages = ["en", "de"]

    def transcribe(self, wav_path: str) -> str:
        return ""


def test_audio_deleted_after_processing(tmp_path):
    db_path = tmp_path / "app.db"
    db = Database(str(db_path))
    audio_path = tmp_path / "segment.wav"
    write_wav_int16(str(audio_path), np.zeros(16000, dtype=np.int16), 16000)
    segment_id = db.add_segment(
        str(audio_path),
        datetime.now(timezone.utc).isoformat(),
        datetime.now(timezone.utc).isoformat(),
    )
    diarizer = SpeakerIdentifier()
    original = transcription_mod.vad_split
    transcription_mod.vad_split = lambda *args, **kwargs: [VadSegment(0.0, 1.0)]
    try:
        worker = TranscriptionWorker(db, None, DummyBackendSuccess(), diarizer, Config())
        worker._process_segment(segment_id)
    finally:
        transcription_mod.vad_split = original
    assert not os.path.exists(audio_path)


def test_audio_kept_on_failure(tmp_path):
    db_path = tmp_path / "app.db"
    db = Database(str(db_path))
    audio_path = tmp_path / "segment.wav"
    write_wav_int16(str(audio_path), np.zeros(16000, dtype=np.int16), 16000)
    segment_id = db.add_segment(
        str(audio_path),
        datetime.now(timezone.utc).isoformat(),
        datetime.now(timezone.utc).isoformat(),
    )
    diarizer = SpeakerIdentifier()
    original = transcription_mod.vad_split
    transcription_mod.vad_split = lambda *args, **kwargs: [VadSegment(0.0, 1.0)]
    try:
        worker = TranscriptionWorker(db, None, DummyBackendFail(), diarizer, Config())
        worker._process_segment(segment_id)
    finally:
        transcription_mod.vad_split = original
    assert os.path.exists(audio_path)
