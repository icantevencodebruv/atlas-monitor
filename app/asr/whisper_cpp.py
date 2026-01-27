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
            model_name = None
            if os.path.exists(self.model_path):
                # Map ggml model filenames to whispercpp model names when possible.
                name = os.path.basename(self.model_path)
                if name.startswith("ggml-") and name.endswith(".bin"):
                    model_name = name.replace("ggml-", "").replace(".bin", "")
                    # whispercpp supports these canonical names.
                    if model_name not in whispercpp.utils.MODELS_URL:
                        model_name = None
            else:
                model_name = self.model_path

            if model_name:
                model = whispercpp.Whisper.from_pretrained(model_name)
                result = model.transcribe_from_file(wav_path)
            else:
                # Unsupported local model for python backend.
                return None
            if isinstance(result, dict) and "text" in result:
                return str(result["text"]).strip()
            return str(result).strip()
        except Exception as exc:
            logger.warning("whispercpp python backend failed: %s", exc)
            return None

    def _transcribe_with_cli(self, wav_path: str) -> str:
        binary = self._resolve_binary()
        if not binary:
            raise RuntimeError("whisper.cpp binary not found.")
        if not os.path.exists(self.model_path):
            raise RuntimeError("whisper.cpp model not found.")
        with tempfile.TemporaryDirectory() as tmpdir:
            json_prefix = os.path.join(tmpdir, "out")
            cmd = [
                binary,
                "-m",
                self.model_path,
                "-f",
                wav_path,
                "-oj",
                "-of",
                json_prefix,
            ]
            if self.language:
                cmd.extend(["-l", self.language])
            logger.info("Running whisper.cpp CLI.")
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            json_path = json_prefix + ".json"
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                text = data.get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()
                # whisper.cpp JSON schema (newer) stores segments under "transcription".
                if isinstance(data.get("transcription"), list):
                    parts = []
                    for item in data["transcription"]:
                        if isinstance(item, dict):
                            seg_text = item.get("text")
                            if isinstance(seg_text, str) and seg_text.strip():
                                parts.append(seg_text.strip())
                    if parts:
                        return " ".join(parts).strip()
        return ""

    def transcribe(self, wav_path: str) -> str:
        text = self._transcribe_with_python(wav_path)
        if text is not None:
            return text
        return self._transcribe_with_cli(wav_path)
