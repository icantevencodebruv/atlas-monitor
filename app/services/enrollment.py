import logging
import os
import threading
from datetime import datetime, timezone

import numpy as np
import sounddevice as sd

from app.services.diarization import compute_embedding
from app.services.audio_utils import write_wav_int16

logger = logging.getLogger(__name__)


class EnrollmentService:
    def __init__(self, db, diarizer, audio_config, device_lock: threading.Lock, audio_dir: str):
        self._db = db
        self._diarizer = diarizer
        self._audio_config = audio_config
        self._device_lock = device_lock
        self._audio_dir = audio_dir

    def enroll(self, speaker: str, duration_sec: int) -> None:
        frame_count = int(duration_sec * self._audio_config.sample_rate)
        logger.info("Recording enrollment for %s (%s seconds).", speaker, duration_sec)
        with self._device_lock:
            audio = sd.rec(
                frames=frame_count,
                samplerate=self._audio_config.sample_rate,
                channels=self._audio_config.channels,
                dtype="int16",
                blocking=True,
            )
        audio = np.squeeze(audio)
        if audio.ndim > 1:
            audio = audio.mean(axis=1).astype(np.int16)
        emb = compute_embedding(audio, self._audio_config.sample_rate)
        self._db.set_embedding(speaker, emb.tolist())
        self._diarizer.set_embedding(speaker, emb)
        filename = f"enroll_{speaker.lower()}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.wav"
        write_wav_int16(os.path.join(self._audio_dir, filename), audio, self._audio_config.sample_rate)
