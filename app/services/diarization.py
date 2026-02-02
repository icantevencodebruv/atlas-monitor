from typing import Dict, Optional

import numpy as np
from python_speech_features import mfcc


def compute_embedding(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.int16).astype(np.float32)
    feats = mfcc(audio, samplerate=sample_rate, numcep=13, nfft=512)
    if feats.size == 0:
        return np.zeros(13, dtype=np.float32)
    emb = feats.mean(axis=0)
    norm = np.linalg.norm(emb) + 1e-6
    return emb / norm


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))


class SpeakerIdentifier:
    def __init__(self):
        self._embeddings: Dict[str, np.ndarray] = {}

    def load_embedding(self, speaker: str, vector: Optional[list]) -> None:
        if vector is None:
            return
        self._embeddings[speaker] = np.asarray(vector, dtype=np.float32)

    def set_embedding(self, speaker: str, vector: np.ndarray) -> None:
        self._embeddings[speaker] = vector.astype(np.float32)

    def has_embedding(self, speaker: str) -> bool:
        emb = self._embeddings.get(speaker)
        return emb is not None and emb.size > 0

    def best_match(self, embedding: np.ndarray) -> tuple[Optional[str], float, float]:
        scores = []
        for speaker, ref in self._embeddings.items():
            if ref is None or ref.size == 0:
                continue
            scores.append((speaker, cosine_distance(embedding, ref)))
        if not scores:
            return None, float("inf"), float("inf")
        scores.sort(key=lambda item: item[1])
        best_speaker, best_score = scores[0]
        second_score = scores[1][1] if len(scores) > 1 else float("inf")
        return best_speaker, best_score, second_score

    def assign(self, embedding: np.ndarray) -> str:
        best_speaker, _, _ = self.best_match(embedding)
        return best_speaker or "Unknown"
