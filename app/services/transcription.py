import logging
import os
import threading
from datetime import datetime, timedelta
from queue import Queue, Empty
from typing import List

from app.db.database import Database
from app.services.audio_utils import read_wav_int16, write_wav_int16
from app.services.diarization import SpeakerIdentifier, compute_embedding
from app.services.vad import vad_split

logger = logging.getLogger(__name__)


class TranscriptionWorker:
    def __init__(
        self,
        db: Database,
        queue: Queue,
        asr_backend,
        diarizer: SpeakerIdentifier,
        config,
    ):
        self._db = db
        self._queue = queue
        self._backend = asr_backend
        self._diarizer = diarizer
        self._config = config
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
        for seg in segments:
            start = int(seg.start_sec * sample_rate)
            end = int(seg.end_sec * sample_rate)
            chunk = audio[start:end]
            if chunk.size == 0:
                continue
            emb = compute_embedding(chunk, sample_rate)
            speaker = self._diarizer.assign(emb)

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
            frag_start = datetime.fromisoformat(segment["start_ts"]) + timedelta(seconds=seg.start_sec)
            frag_end = datetime.fromisoformat(segment["start_ts"]) + timedelta(seconds=seg.end_sec)
            fragments.append(
                {
                    "start_ts": frag_start.isoformat(),
                    "end_ts": frag_end.isoformat(),
                    "speaker": speaker,
                    "text": text.strip(),
                }
            )

        if fragments:
            self._db.add_transcripts(segment_id, fragments)
            self._db.update_segment_status(segment_id, "done")
            os.remove(file_path)
            return

        if segments:
            self._db.update_segment_status(segment_id, "failed", "no transcription output")
            return

        self._db.update_segment_status(segment_id, "done")
        os.remove(file_path)
