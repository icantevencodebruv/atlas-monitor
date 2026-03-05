import logging
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

from app.services.recorder import select_input_device

logger = logging.getLogger(__name__)


def _noise_rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 1e-6
    audio_f = audio.astype(np.float32)
    abs_audio = np.abs(audio_f)
    noise_threshold = float(np.percentile(abs_audio, 20))
    noise_samples = audio_f[abs_audio <= noise_threshold]
    if noise_samples.size == 0:
        return 1e-6
    return float(np.sqrt(np.mean(noise_samples ** 2)) + 1e-6)


def _snr_db(audio: np.ndarray) -> float:
    if audio.size == 0:
        return -float("inf")
    audio_f = audio.astype(np.float32)
    signal_rms = float(np.sqrt(np.mean(audio_f ** 2)) + 1e-6)
    noise_rms = _noise_rms(audio)
    return 20.0 * float(np.log10(signal_rms / noise_rms))


def _normalize(audio: np.ndarray) -> np.ndarray:
    audio = np.squeeze(audio)
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.int16)
    return audio


def _record_block(
    duration_sec: float,
    sample_rate: int,
    channels: int,
    device_lock: threading.Lock,
    preferred_hostapi: Optional[str] = None,
    preferred_name: Optional[str] = None,
) -> Optional[np.ndarray]:
    frames = int(duration_sec * sample_rate)
    if frames <= 0:
        return None
    acquired = device_lock.acquire(timeout=0.1)
    if not acquired:
        return None
    try:
        last_exc = None
        for attempt in range(2):
            try:
                select_input_device(preferred_hostapi, preferred_name)
                audio = sd.rec(
                    frames=frames,
                    samplerate=sample_rate,
                    channels=channels,
                    dtype="int16",
                    blocking=True,
                )
                return _normalize(audio)
            except Exception as exc:
                last_exc = exc
                logger.warning("Audio probe failed on attempt %s: %s", attempt + 1, exc)
        if last_exc is not None:
            logger.warning("Audio probe exhausted retries: %s", last_exc)
        return None
    finally:
        device_lock.release()


def measure_level(
    sample_rate: int,
    channels: int,
    device_lock: threading.Lock,
    preferred_hostapi: Optional[str] = None,
    preferred_name: Optional[str] = None,
) -> dict:
    audio = _record_block(0.15, sample_rate, channels, device_lock, preferred_hostapi, preferred_name)
    if audio is None:
        return {"busy": True}
    audio_f = audio.astype(np.float32)
    rms = float(np.sqrt(np.mean(audio_f ** 2)) + 1e-6)
    peak = float(np.max(np.abs(audio_f)) + 1e-6)
    snr_db = _snr_db(audio)
    noise_rms = _noise_rms(audio)
    noise_dbfs = 20.0 * float(np.log10(noise_rms / 32768.0))
    return {
        "busy": False,
        "rms": rms / 32768.0,
        "peak": peak / 32768.0,
        "snr_db": round(snr_db, 1),
        "noise_floor": round(noise_dbfs, 1),
    }


def test_recording(
    duration_sec: float,
    sample_rate: int,
    channels: int,
    device_lock: threading.Lock,
    preferred_hostapi: Optional[str] = None,
    preferred_name: Optional[str] = None,
) -> dict:
    audio = _record_block(
        duration_sec,
        sample_rate,
        channels,
        device_lock,
        preferred_hostapi,
        preferred_name,
    )
    if audio is None:
        return {"ok": False, "error": "device_busy"}
    snr_db = _snr_db(audio)
    audio_f = audio.astype(np.float32)
    rms = float(np.sqrt(np.mean(audio_f ** 2)) + 1e-6)
    peak = float(np.max(np.abs(audio_f)) + 1e-6)
    return {
        "ok": True,
        "duration_sec": duration_sec,
        "snr_db": round(snr_db, 1),
        "rms": rms / 32768.0,
        "peak": peak / 32768.0,
    }
