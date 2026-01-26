import json
import logging
import os
import subprocess
import tempfile
from typing import Optional

from app.asr.base import ASRBackend

logger = logging.getLogger(__name__)


class WhisperCppBackend(ASRBackend):
    name = "whisper_cpp"

    def __init__(self, model_path: str, language: str, binary_path: str):
        self.model_path = model_path
        self.language = language
        self.binary_path = binary_path
        self.languages = ["en", "de"]

    @classmethod
    def is_available(cls) -> bool:
        try:
            import whispercpp  # noqa: F401

            return True
        except Exception:
            return True

    def _resolve_binary(self) -> Optional[str]:
        if self.binary_path and os.path.exists(self.binary_path):
            return self.binary_path
        for candidate in ["whisper-cli", "main", "whisper.cpp"]:
            path = os.path.join(".", "bin", "whisper.cpp", candidate)
            if os.path.exists(path):
                return path
        return None

    def _transcribe_with_python(self, wav_path: str) -> Optional[str]:
        try:
            import whispercpp

            model = whispercpp.Whisper(self.model_path)
            result = model.transcribe(wav_path, language=None if self.language == "auto" else self.language)
            if isinstance(result, dict) and "text" in result:
                return str(result["text"]).strip()
            return str(result).strip()
        except Exception:
            return None

    def _transcribe_with_cli(self, wav_path: str) -> str:
        binary = self._resolve_binary()
        if not binary:
            raise RuntimeError("whisper.cpp binary not found.")
        if not os.path.exists(self.model_path):
            raise RuntimeError("whisper.cpp model not found.")
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "out.json")
            cmd = [
                binary,
                "-m",
                self.model_path,
                "-f",
                wav_path,
                "-oj",
                "-of",
                json_path,
            ]
            if self.language:
                cmd.extend(["-l", self.language])
            logger.info("Running whisper.cpp CLI.")
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return str(data.get("text", "")).strip()
        return ""

    def transcribe(self, wav_path: str) -> str:
        text = self._transcribe_with_python(wav_path)
        if text is not None:
            return text
        return self._transcribe_with_cli(wav_path)
