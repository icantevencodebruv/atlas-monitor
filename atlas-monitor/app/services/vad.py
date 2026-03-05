from dataclasses import dataclass
from typing import List

import numpy as np
import webrtcvad


@dataclass
class VadSegment:
    start_sec: float
    end_sec: float


def _frame_bytes(pcm: bytes, offset: int, size: int) -> bytes:
    return pcm[offset : offset + size]


def vad_split(
    audio: np.ndarray,
    sample_rate: int,
    aggressiveness: int,
    min_utterance_sec: float,
    max_utterance_sec: float,
    max_silence_sec: float,
    frame_ms: int = 20,
) -> List[VadSegment]:
    vad = webrtcvad.Vad(int(aggressiveness))
    pcm = np.asarray(audio, dtype=np.int16).tobytes()
    frame_size = int(sample_rate * frame_ms / 1000)
    frame_bytes = frame_size * 2
    total_frames = len(pcm) // frame_bytes

    segments: List[VadSegment] = []
    speech_active = False
    speech_start = 0.0
    last_speech = 0.0

    for i in range(total_frames):
        start = i * frame_bytes
        frame = _frame_bytes(pcm, start, frame_bytes)
        is_speech = vad.is_speech(frame, sample_rate)
        t = (i * frame_size) / sample_rate

        if is_speech:
            if not speech_active:
                speech_active = True
                speech_start = t
            last_speech = t
        elif speech_active:
            silence = t - last_speech
            if silence >= max_silence_sec:
                end_t = t
                segments.append(VadSegment(speech_start, end_t))
                speech_active = False

    if speech_active:
        end_t = total_frames * frame_size / sample_rate
        segments.append(VadSegment(speech_start, end_t))

    filtered: List[VadSegment] = []
    for seg in segments:
        duration = seg.end_sec - seg.start_sec
        if duration < min_utterance_sec:
            continue
        if duration > max_utterance_sec:
            chunks = int(duration // max_utterance_sec) + 1
            step = duration / chunks
            for idx in range(chunks):
                s = seg.start_sec + idx * step
                e = min(seg.start_sec + (idx + 1) * step, seg.end_sec)
                if e - s >= min_utterance_sec:
                    filtered.append(VadSegment(s, e))
        else:
            filtered.append(seg)
    return filtered
