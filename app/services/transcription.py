import logging
import os
import re
import threading
from datetime import datetime, timedelta
from queue import Queue, Empty
from typing import List, Tuple

import numpy as np

from app.database import Database
from app.audio.utils import read_wav_int16, write_wav_int16
from app.services.diarization import SpeakerIdentifier, compute_embedding
from app.services.transcript_qa import TranscriptQA
from app.services.vad import vad_split

logger = logging.getLogger(__name__)


NON_SPEECH_PATTERNS = [
    r"^\s*\[.*\]\s*$",
    r"^\s*\(.*\)\s*$",
    r"^\s*\*.*\*\s*$",
    r"\b(blank audio|music|musik|musique|typing|keyboard|clacking|scissors|cough)\b",
    # YouTube / ASR boilerplate hallucinations
    r"\bthank\s+you\s+for\s+(watching|viewing)\b",
    r"\bthanks?\s+for\s+watching\b",
    r"\bsubtitles?\s+by\b",
    r"\blike\s+and\s+subscribe\b",
    r"\bsubscribe\s+(for\s+more|to\s+my)\b",
    r"\bdon.t\s+forget\s+to\s+(like|subscribe|comment)\b",
]


def _letter_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = sum(1 for ch in text if ch.isalpha())
    return letters / max(len(text), 1)


def is_non_speech(text: str) -> bool:
    if not text:
        return True
    t = text.strip().lower()
    for pattern in NON_SPEECH_PATTERNS:
        if re.search(pattern, t):
            return True
    if _letter_ratio(t) < 0.2:
        return True
    return False


class TranscriptionWorker:
    def __init__(
        self,
        db: Database,
        queue: Queue,
        asr_backend,
        diarizer: SpeakerIdentifier,
        config,
        state,
    ):
        self._db = db
        self._queue = queue
        self._backend = asr_backend
        self._diarizer = diarizer
        self._config = config
        self._state = state
        self._qa = TranscriptQA(config.llm_qa.model_dump())
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def shutdown(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                segment_id = self._queue.get(timeout=0.5)
            except Empty:
                continue
            try:
                self._process_segment(segment_id)
            except Exception as exc:
                logger.exception("Segment %s failed: %s", segment_id, exc)
                self._db.update_segment_status(segment_id, "failed", str(exc))

    def _process_segment(self, segment_id: int) -> None:
        segment = self._db.get_segment(segment_id)
        if not segment:
            return
        file_path = segment["file_path"]
        self._db.mark_attempt(segment_id)
        self._db.update_segment_status(segment_id, "processing")
        audio, sample_rate = read_wav_int16(file_path)

        segments = vad_split(
            audio,
            sample_rate,
            self._config.transcription.vad_aggressiveness,
            self._config.transcription.min_utterance_sec,
            self._config.transcription.max_utterance_sec,
            self._config.transcription.max_silence_sec,
        )

        fragments: List[dict] = []
        had_text = False
        for seg in segments:
            start = int(seg.start_sec * sample_rate)
            end = int(seg.end_sec * sample_rate)
            chunk = audio[start:end]
            if chunk.size == 0:
                continue
            emb = compute_embedding(chunk, sample_rate)
            speaker, low_confidence = self._select_speaker(emb)

            chunk_path = file_path + f".{start}_{end}.wav"
            write_wav_int16(chunk_path, chunk, sample_rate)
            try:
                text = self._backend.transcribe(chunk_path)
            finally:
                try:
                    os.remove(chunk_path)
                except OSError:
                    pass
            if not text:
                continue
            had_text = True
            if is_non_speech(text):
                continue
            qa_result = self._qa.evaluate(speaker, text.strip())
            if qa_result.action == "filtered":
                continue
            frag_start = datetime.fromisoformat(segment["start_ts"]) + timedelta(seconds=seg.start_sec)
            frag_end = datetime.fromisoformat(segment["start_ts"]) + timedelta(seconds=seg.end_sec)
            fragments.append(
                {
                    "start_ts": frag_start.isoformat(),
                    "end_ts": frag_end.isoformat(),
                    "speaker": speaker,
                    "text": qa_result.corrected_text,
                    "original_text": qa_result.original_text,
                    "low_confidence": low_confidence,
                }
            )

        if fragments:
            self._db.add_transcripts(segment_id, fragments)
            self._db.update_segment_status(segment_id, "done")
            os.remove(file_path)
            return

        if segments and not had_text:
            self._db.update_segment_status(segment_id, "failed", "no transcription output")
            return
        if segments and had_text:
            self._db.update_segment_status(segment_id, "done")
            os.remove(file_path)
            return

        self._db.update_segment_status(segment_id, "done")
        os.remove(file_path)

    def _select_speaker(self, embedding: np.ndarray) -> Tuple[str, bool]:
        lock_mode = getattr(self._state, "speaker_lock", "auto")
        if lock_mode == "hugo":
            return "Hugo", False
        if lock_mode == "leon":
            return "Leon", False
        if self._config.diarization.require_both_enrolled:
            if not (self._diarizer.has_embedding("Hugo") and self._diarizer.has_embedding("Leon")):
                return "Unknown", True
        best_speaker, best_score, second_score = self._diarizer.best_match(embedding)
        if not best_speaker:
            return "Unknown", True
        margin = second_score - best_score
        if best_score > self._config.diarization.max_distance:
            return "Unknown", True
        if margin < self._config.diarization.min_margin:
            return "Unknown", True
        soft_max = self._config.diarization.max_distance * 0.85
        soft_margin = self._config.diarization.min_margin * 1.5
        low_confidence = best_score > soft_max or margin < soft_margin
        return best_speaker, low_confidence
