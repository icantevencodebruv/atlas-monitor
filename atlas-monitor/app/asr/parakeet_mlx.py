import logging
import os
from typing import List

from app.asr.base import ASRBackend

logger = logging.getLogger(__name__)


class ParakeetMLXBackend(ASRBackend):
    name = "mac_parakeet_mlx"

    def __init__(self, model_id: str, hf_cache_dir: str, hf_cache_dir_parent: str, hf_offline: bool, languages: List[str]):
        self.model_id = model_id
        self.hf_cache_dir = hf_cache_dir
        self.hf_cache_dir_parent = hf_cache_dir_parent
        self.hf_offline = hf_offline
        self.languages = languages
        self._model = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            import mlx  # noqa: F401
            return True
        except Exception:
            return False

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.model_id:
            raise RuntimeError("Parakeet MLX model_id not set.")
        if self.hf_offline:
            os.environ["HF_HOME"] = self.hf_cache_dir_parent
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from parakeet_mlx import from_pretrained

        self._model = from_pretrained(self.model_id, cache_dir=self.hf_cache_dir)
        logger.info("Loaded Parakeet MLX model.")

    def transcribe(self, wav_path: str) -> str:
        self._load()
        result = self._model.transcribe(wav_path, chunk_duration=120.0, overlap_duration=15.0)
        if hasattr(result, "text"):
            return str(result.text).strip()
        return str(result).strip()

    def precheck_offline_cache(self) -> None:
        if not self.hf_offline:
            return
        os.environ["HF_HOME"] = self.hf_cache_dir_parent
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if not self.hf_cache_dir or not os.path.isdir(self.hf_cache_dir):
            raise RuntimeError(
                f"Parakeet MLX offline cache missing at {self.hf_cache_dir}."
            )
        has_file = False
        for _, _, files in os.walk(self.hf_cache_dir):
            if files:
                has_file = True
                break
        if not has_file:
            raise RuntimeError(
                f"Parakeet MLX offline cache is empty at {self.hf_cache_dir}."
            )
