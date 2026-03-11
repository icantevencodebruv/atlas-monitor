import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import sounddevice as sd

from app.services.diarization import compute_embedding
from app.audio.utils import write_wav_int16

logger = logging.getLogger(__name__)


class EnrollmentService:
    def __init__(
        self,
        db,
        diarizer,
        audio_config,
        diarization_config,
        device_lock: threading.Lock,
        audio_dir: str,
    ):
        self._db = db
        self._diarizer = diarizer
        self._audio_config = audio_config
        self._diarization_config = diarization_config
        self._device_lock = device_lock
        self._audio_dir = audio_dir
        self._active_lock = threading.Lock()
        self._active_speaker: Optional[str] = None
        self._active_started_at: Optional[float] = None
        self._stop_event = threading.Event()

    def _snr_db(self, audio: np.ndarray) -> float:
        if audio.size == 0:
            return -float("inf")
        audio_f = audio.astype(np.float32)
        signal_rms = float(np.sqrt(np.mean(audio_f ** 2)) + 1e-6)
        abs_audio = np.abs(audio_f)
        noise_threshold = float(np.percentile(abs_audio, 20))
        noise_samples = audio_f[abs_audio <= noise_threshold]
        if noise_samples.size == 0:
            noise_rms = 1e-6
        else:
            noise_rms = float(np.sqrt(np.mean(noise_samples ** 2)) + 1e-6)
        return 20.0 * float(np.log10(signal_rms / noise_rms))

    def stop_active(self) -> dict:
        with self._active_lock:
            speaker = self._active_speaker
            if speaker is None:
                return {"stopping": False, "speaker": None}
            self._stop_event.set()
            return {"stopping": True, "speaker": speaker}

    def _record_with_stop(self, duration_sec: int) -> tuple[np.ndarray, bool]:
        frame_count = int(duration_sec * self._audio_config.sample_rate)
        chunk_frames = max(int(self._audio_config.sample_rate * 0.1), 1)
        chunks = []
        captured = 0
        cancelled = False

        with self._device_lock:
            with sd.InputStream(
                samplerate=self._audio_config.sample_rate,
                channels=self._audio_config.channels,
                dtype="int16",
            ) as stream:
                while captured < frame_count:
                    if self._stop_event.is_set():
                        cancelled = True
                        break
                    to_read = min(chunk_frames, frame_count - captured)
                    block, _ = stream.read(to_read)
                    chunks.append(block.copy())
                    captured += block.shape[0]

        if chunks:
            audio = np.concatenate(chunks, axis=0)
        else:
            audio = np.empty((0, self._audio_config.channels), dtype=np.int16)
        return audio, cancelled

    def enroll(self, speaker: str, duration_sec: int) -> dict:
        with self._active_lock:
            if self._active_speaker is not None:
                raise RuntimeError("enrollment_in_progress")
            self._active_speaker = speaker
            self._active_started_at = time.monotonic()
            self._stop_event.clear()

        try:
            logger.info("Recording enrollment for %s (%s seconds).", speaker, duration_sec)
            audio, cancelled = self._record_with_stop(duration_sec)
            audio = np.squeeze(audio)
            if audio.ndim > 1:
                audio = audio.mean(axis=1).astype(np.int16)
            actual_sec = float(audio.shape[0]) / float(self._audio_config.sample_rate)
            if cancelled:
                logger.info("Enrollment cancelled for %s after %.1f seconds.", speaker, actual_sec)
                return {"speaker": speaker, "duration_sec": round(actual_sec, 1), "cancelled": True}
            if audio.size == 0:
                raise RuntimeError("no_audio_captured")

            emb = compute_embedding(audio, self._audio_config.sample_rate)
            self._db.set_embedding(speaker, emb.tolist())
            self._diarizer.set_embedding(speaker, emb)
            snr_db = self._snr_db(audio)
            min_snr = float(self._diarization_config.enrollment_min_snr_db)
            quality_ok = True
            if min_snr is not None and snr_db < min_snr:
                quality_ok = False
                logger.warning(
                    "Enrollment quality low for %s (snr=%.1f dB, min=%.1f dB).",
                    speaker,
                    snr_db,
                    min_snr,
                )
            filename = f"enroll_{speaker.lower()}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.wav"
            write_wav_int16(os.path.join(self._audio_dir, filename), audio, self._audio_config.sample_rate)
            return {
                "speaker": speaker,
                "snr_db": round(snr_db, 1),
                "duration_sec": round(actual_sec, 1),
                "quality_ok": quality_ok,
                "cancelled": False,
            }
        finally:
            with self._active_lock:
                self._active_speaker = None
                self._active_started_at = None
                self._stop_event.clear()
