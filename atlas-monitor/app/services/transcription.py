import logging
import os
import re
import threading
import unicodedata
from datetime import datetime, timedelta
from queue import Queue, Empty
from typing import List, Tuple

import numpy as np

from app.db.database import Database
from app.services.audio_utils import read_wav_int16, write_wav_int16
from app.services.diarization import SpeakerIdentifier, compute_embedding
from app.services.vad import vad_split

logger = logging.getLogger(__name__)


NON_SPEECH_PATTERNS = [
    r"^\s*\[.*\]\s*$",
    r"^\s*\(.*\)\s*$",
    r"^\s*\*.*\*\s*$",
    r"\b(blank audio|music|musik|musique|typing|keyboard|clacking|scissors|cough)\b",
]
SUPPORTED_SPEAKERS = {"Hugo", "Leon"}
MIN_LATIN_LETTER_RATIO = 0.85
MIN_PIPELINE_LANG_CONFIDENCE = 0.60
MIN_TEXT_QUALITY = 0.40
LOW_VALUE_SINGLE_WORDS = {
    "a",
    "an",
    "and",
    "ah",
    "eh",
    "hm",
    "i",
    "ja",
    "ne",
    "oh",
    "the",
    "uh",
    "um",
    "und",
}


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


def _is_latin_letter(ch: str) -> bool:
    if not ch.isalpha():
        return False
    try:
        return "LATIN" in unicodedata.name(ch)
    except ValueError:
        return False


def _latin_letter_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    latin_letters = sum(1 for ch in letters if _is_latin_letter(ch))
    return latin_letters / float(len(letters))


def is_supported_alphabet(text: str) -> bool:
    return _latin_letter_ratio(text) >= MIN_LATIN_LETTER_RATIO


def _language_allowed(row: dict) -> bool:
    lang = row.get("language")
    if lang is None:
        return True
    lang_norm = str(lang).strip().lower()
    if lang_norm not in {"en", "de"}:
        return False
    conf = row.get("language_confidence")
    if conf is None:
        return True
    try:
        conf_value = float(conf)
    except (TypeError, ValueError):
        return False
    return conf_value >= MIN_PIPELINE_LANG_CONFIDENCE


def _quality_allowed(row: dict) -> bool:
    quality = row.get("text_quality")
    if quality is None:
        return True
    try:
        return float(quality) >= MIN_TEXT_QUALITY
    except (TypeError, ValueError):
        return False


def _has_substantive_content(text: str) -> bool:
    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿß']+", text.lower())
    if not words:
        return False
    if len(words) == 1:
        token = words[0]
        if len(token) < 4:
            return False
        if token in LOW_VALUE_SINGLE_WORDS:
            return False
    if len(words) == 2:
        low_value_count = sum(1 for token in words if token in LOW_VALUE_SINGLE_WORDS or len(token) <= 2)
        if low_value_count == 2:
            return False
    return True


def should_store_fragment(row: dict, text: str) -> bool:
    speaker = str(row.get("speaker", "Unknown")).strip()
    if speaker not in SUPPORTED_SPEAKERS:
        return False
    if bool(row.get("low_confidence")):
        return False
    if is_non_speech(text):
        return False
    if not _has_substantive_content(text):
        return False
    if not is_supported_alphabet(text):
        return False
    if not _quality_allowed(row):
        return False
    if not _language_allowed(row):
        return False
    return True


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
        if hasattr(self._backend, "transcribe_segment"):
            raw_fragments = self._backend.transcribe_segment(
                file_path,
                segment["start_ts"],
                self._diarizer,
                self._config.diarization,
                getattr(self._state, "speaker_lock", "auto"),
            )
            fragments: List[dict] = []
            had_text = False
            for row in raw_fragments:
                text = str(row.get("text", "")).strip()
                if not text:
                    continue
                had_text = True
                if not should_store_fragment(row, text):
                    continue
                fragments.append(
                    {
                        "start_ts": row["start_ts"],
                        "end_ts": row["end_ts"],
                        "speaker": row.get("speaker", "Unknown"),
                        "text": text,
                        "low_confidence": bool(row.get("low_confidence")),
                    }
                )
            if fragments:
                self._db.add_transcripts(segment_id, fragments)
                self._db.update_segment_status(segment_id, "done")
                os.remove(file_path)
                return
            if raw_fragments and not had_text:
                self._db.update_segment_status(segment_id, "failed", "no transcription output")
                return
            if raw_fragments and had_text:
                self._db.update_segment_status(segment_id, "done")
                os.remove(file_path)
                return
            self._db.update_segment_status(segment_id, "done")
            os.remove(file_path)
            return

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
            row = {
                "speaker": speaker,
                "low_confidence": low_confidence,
            }
            if not should_store_fragment(row, text):
                continue
            frag_start = datetime.fromisoformat(segment["start_ts"]) + timedelta(seconds=seg.start_sec)
            frag_end = datetime.fromisoformat(segment["start_ts"]) + timedelta(seconds=seg.end_sec)
            fragments.append(
                {
                    "start_ts": frag_start.isoformat(),
                    "end_ts": frag_end.isoformat(),
                    "speaker": speaker,
                    "text": text.strip(),
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
