import logging
from typing import List

from app.asr.base import ASRBackend

logger = logging.getLogger(__name__)


class NemoParakeetBackend(ASRBackend):
    name = "windows_nemo_parakeet_cuda"

    def __init__(self, model_name: str, languages: List[str]):
        self.model_name = model_name
        self.languages = languages
        self._model = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            import torch  # noqa: F401
            import nemo  # noqa: F401

            return torch.cuda.is_available()
        except Exception:
            return False

    def _load(self) -> None:
        if self._model is not None:
            return
        from nemo.collections.asr.models import EncDecCTCModelBPE
        import torch

        if self.model_name and self.model_name.endswith(".nemo"):
            self._model = EncDecCTCModelBPE.restore_from(self.model_name)
        else:
            if not self.model_name:
                raise RuntimeError("Nemo Parakeet model_name not set.")
            self._model = EncDecCTCModelBPE.from_pretrained(self.model_name)
        self._model = self._model.to(torch.device("cuda"))
        logger.info("Loaded NeMo Parakeet model on CUDA.")

    def transcribe(self, wav_path: str) -> str:
        self._load()
        texts = self._model.transcribe([wav_path])
        if not texts:
            return ""
        return texts[0].strip()
