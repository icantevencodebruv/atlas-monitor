"""
TranscriptQA — post-ASR quality filter and correction layer.

Architecture
-----------
Sits between the ASR backend and the database write in TranscriptionWorker.
The full pipeline per chunk:

    raw WAV  →  ASR  →  is_non_speech (regex gate)  →  TranscriptQA.evaluate()  →  DB

When llm_qa.enabled is False the stub is used: every fragment passes through
unchanged, with original_text == text and action == "pass".

When enabled and the model loads successfully, _evaluate_with_llm() calls the
GGUF model via llama-cpp-python. On any model load or inference failure the
class falls back to stub behaviour (fail open — transcription is never blocked).
"""

import logging
import os
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Filter speech transcripts. Output the text unchanged, or say EMPTY for non-speech noise."
)

# Few-shot examples injected as prior conversation turns.
_FEW_SHOT: list[dict] = [
    {"role": "user", "content": "(upbeat music)"},
    {"role": "assistant", "content": "EMPTY"},
    {"role": "user", "content": "[MUSIC]"},
    {"role": "assistant", "content": "EMPTY"},
    {"role": "user", "content": "[BLANK_AUDIO]"},
    {"role": "assistant", "content": "EMPTY"},
    {"role": "user", "content": "[MUSIC PLAYING]"},
    {"role": "assistant", "content": "EMPTY"},
    {"role": "user", "content": "Thank you for watching!"},
    {"role": "assistant", "content": "EMPTY"},
    {"role": "user", "content": "Subtitles by the Amara community"},
    {"role": "assistant", "content": "EMPTY"},
    {"role": "user", "content": "Thanks for watching, subscribe for more"},
    {"role": "assistant", "content": "EMPTY"},
    {"role": "user", "content": "[Subscribe]"},
    {"role": "assistant", "content": "EMPTY"},
    {"role": "user", "content": "ლლლლ"},
    {"role": "assistant", "content": "EMPTY"},
    {"role": "user", "content": "ʕ ʔ ʔ ʔ"},
    {"role": "assistant", "content": "EMPTY"},
    # Keep: real speech in any language
    {"role": "user", "content": "Ich glaube wir sollten das morgen besprechen"},
    {"role": "assistant", "content": "Ich glaube wir sollten das morgen besprechen"},
    {"role": "user", "content": "ja genau also ich hab das gemacht"},
    {"role": "assistant", "content": "ja genau also ich hab das gemacht"},
    {"role": "user", "content": "and then I was like äh basically we need to refactor this"},
    {"role": "assistant", "content": "and then I was like äh basically we need to refactor this"},
    {"role": "user", "content": "ja"},
    {"role": "assistant", "content": "ja"},
    {"role": "user", "content": "uh huh"},
    {"role": "assistant", "content": "uh huh"},
    {"role": "user", "content": "the the the project is is ready"},
    {"role": "assistant", "content": "the the the project is is ready"},
    {"role": "user", "content": "Tizian hat gesagt er kommt um drei"},
    {"role": "assistant", "content": "Tizian hat gesagt er kommt um drei"},
]


@dataclass
class QAResult:
    original_text: str
    corrected_text: str
    was_modified: bool
    action: Literal["pass", "corrected", "filtered"]


class TranscriptQA:
    def __init__(self, config: dict) -> None:
        self.enabled: bool = bool(config.get("enabled", False))
        self.model_path: str = config.get("model_path", "")
        self.max_tokens: int = int(config.get("max_tokens", 256))
        self._model = None

        if self.enabled and self.model_path:
            abs_path = (
                self.model_path
                if os.path.isabs(self.model_path)
                else os.path.join(os.getcwd(), self.model_path)
            )
            try:
                import llama_cpp  # type: ignore

                self._model = llama_cpp.Llama(
                    model_path=abs_path,
                    n_ctx=2048,
                    n_gpu_layers=-1,
                    verbose=False,
                )
                logger.info("TranscriptQA: loaded model from %s", abs_path)
            except Exception as exc:
                logger.warning(
                    "TranscriptQA: failed to load model from %s (%s) — falling back to stub",
                    abs_path,
                    exc,
                )
                self._model = None

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def evaluate(self, speaker: str, text: str) -> QAResult:
        """
        Evaluate a single transcript fragment.

        Returns a QAResult with:
          - original_text: what came out of ASR (unmodified)
          - corrected_text: what goes into the DB (may differ when LLM active)
          - was_modified: True only when text actually changed
          - action: "pass" | "corrected" | "filtered"

        When disabled or model not loaded (stub), every call returns action="pass"
        with the text unchanged. "filtered" means the fragment should be discarded
        entirely (corrected_text will be empty).
        """
        if not self.enabled or self._model is None:
            return QAResult(
                original_text=text,
                corrected_text=text,
                was_modified=False,
                action="pass",
            )
        return self._evaluate_with_llm(speaker, text)

    def _evaluate_with_llm(self, speaker: str, text: str) -> QAResult:
        """Call the GGUF model and parse its response. Never raises."""
        try:
            messages = (
                [{"role": "system", "content": SYSTEM_PROMPT}]
                + _FEW_SHOT
                + [{"role": "user", "content": text}]
            )
            response = self._model.create_chat_completion(
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=0.0,
            )
            raw: str = response["choices"][0]["message"]["content"].strip()

            # Hallucination guard: if model echoed the system prompt or a fragment of it,
            # or produced output much longer than input, fall back to pass-through.
            _SYSTEM_FRAGMENTS = ("filter speech", "copy it back", "real speech", "original text")
            if (
                any(frag in raw.lower() for frag in _SYSTEM_FRAGMENTS)
                or (text and len(raw) > len(text) * 1.3)
            ):
                logger.debug("TranscriptQA: hallucination guard triggered, passing through")
                return QAResult(
                    original_text=text,
                    corrected_text=text,
                    was_modified=False,
                    action="pass",
                )

            # Classify
            if not raw or raw.upper() == "EMPTY":
                return QAResult(
                    original_text=text,
                    corrected_text="",
                    was_modified=True,
                    action="filtered",
                )
            if raw == text.strip() or raw.lower() == text.strip().lower():
                return QAResult(
                    original_text=text,
                    corrected_text=text,
                    was_modified=False,
                    action="pass",
                )
            return QAResult(
                original_text=text,
                corrected_text=raw,
                was_modified=True,
                action="corrected",
            )
        except Exception as exc:
            logger.warning("TranscriptQA: inference error (%s) — passing through original", exc)
            return QAResult(
                original_text=text,
                corrected_text=text,
                was_modified=False,
                action="pass",
            )
