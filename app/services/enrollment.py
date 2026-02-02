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

    def enroll(self, speaker: str, duration_sec: int) -> dict:
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
        snr_db = self._snr_db(audio)
        actual_sec = float(audio.shape[0]) / float(self._audio_config.sample_rate)
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
        }
