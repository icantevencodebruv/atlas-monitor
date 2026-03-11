import wave
from typing import Tuple

import numpy as np


def write_wav_int16(path: str, audio: np.ndarray, sample_rate: int) -> None:
    audio = np.asarray(audio, dtype=np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())


def read_wav_int16(path: str) -> Tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        frames = wf.getnframes()
        data = wf.readframes(frames)
    audio = np.frombuffer(data, dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return audio, sample_rate
