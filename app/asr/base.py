from abc import ABC, abstractmethod
from typing import List


class ASRBackend(ABC):
    name: str = "base"
    languages: List[str] = []

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        raise NotImplementedError

    @abstractmethod
    def transcribe(self, wav_path: str) -> str:
        raise NotImplementedError

    def supports_languages(self, required: List[str]) -> bool:
        if not self.languages:
            return False
        return all(lang in self.languages for lang in required)
