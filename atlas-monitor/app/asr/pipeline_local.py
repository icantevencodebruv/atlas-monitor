import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from app.asr.base import ASRBackend
from app.services.audio_utils import read_wav_int16
from app.services.diarization import compute_embedding

logger = logging.getLogger(__name__)

_EN_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "have",
    "he", "her", "his", "i", "if", "in", "is", "it", "its", "me", "my", "not", "of",
    "on", "or", "our", "she", "so", "that", "the", "their", "them", "there", "they",
    "this", "to", "us", "was", "we", "were", "what", "when", "who", "with", "you", "your",
}

_DE_STOPWORDS = {
    "aber", "als", "am", "an", "auch", "auf", "aus", "bei", "bin", "bis", "da", "dann",
    "das", "dass", "dein", "dem", "den", "der", "des", "die", "doch", "du", "ein", "eine",
    "einer", "einem", "einen", "er", "es", "für", "hat", "ich", "im", "in", "ist", "ja",
    "kein", "mit", "nach", "nicht", "oder", "sein", "sie", "so", "und", "uns", "von",
    "war", "was", "wenn", "wie", "wir", "zu",
}

_WORD_RE = re.compile(r"[a-zA-ZäöüßÄÖÜ']+")
_COMMON_STOPWORDS = _EN_STOPWORDS | _DE_STOPWORDS


@dataclass
class _ChunkCandidate:
    text: str
    model_lang: str
    detected_lang: str
    detected_conf: float
    asr_conf: float
    text_quality: float
    score: float


class PipelineLocalBackend(ASRBackend):
    name = "pipeline_local"

    def __init__(self, cfg):
        self.cfg = cfg
        self.languages = ["en", "de"]
        self.asr_engine = str(getattr(cfg, "asr_engine", "vosk")).lower().strip() or "vosk"
        self.silero_model_path = str(getattr(cfg, "silero_model_path", "") or "")
        self.fasttext_model_path = str(cfg.fasttext_model_path)
        self.vosk_model_en_path = str(cfg.vosk_model_en_path)
        self.vosk_model_de_path = str(cfg.vosk_model_de_path)
        self.wav2vec2_model_en_path = str(getattr(cfg, "wav2vec2_model_en_path", "") or "")
        self.wav2vec2_model_de_path = str(getattr(cfg, "wav2vec2_model_de_path", "") or "")
        self.pyannote_model_path = str(cfg.pyannote_model_path)
        self.pyannote_auth_token = str(
            (getattr(cfg, "pyannote_auth_token", "") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or "")
        )
        raw_hf_cache_dir = str(getattr(cfg, "hf_cache_dir", "") or "")
        self.hf_cache_dir = str(Path(raw_hf_cache_dir).resolve()) if raw_hf_cache_dir else ""
        self.hf_offline = bool(cfg.hf_offline)
        self.silero_threshold = float(cfg.silero_threshold)
        self.silero_min_speech_ms = int(cfg.silero_min_speech_ms)
        self.silero_min_silence_ms = int(cfg.silero_min_silence_ms)
        self.silero_speech_pad_ms = int(cfg.silero_speech_pad_ms)
        self.merge_gap_sec = float(cfg.merge_gap_sec)

        self._silero_model = None
        self._silero_get_speech_timestamps = None
        self._fasttext_model = None
        self._vosk_models: Dict[str, Any] = {}
        self._wav2vec2_models: Dict[str, Tuple[Any, Any]] = {}
        self._pyannote_pipeline = None
        self._pyannote_runtime_config: Optional[str] = None
        self._fasttext_disabled = False
        self._fasttext_warned = False
        self._set_offline_env()

    @classmethod
    def is_available(cls) -> bool:
        try:
            import fasttext  # noqa: F401
            import torch  # noqa: F401
            import torchaudio  # noqa: F401
            import transformers  # noqa: F401
            import vosk  # noqa: F401
            from pyannote.audio import Pipeline  # noqa: F401
            from silero_vad import get_speech_timestamps, load_silero_vad  # noqa: F401

            return True
        except Exception:
            return False

    def _set_offline_env(self) -> None:
        os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        if not self.hf_offline:
            return
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if self.hf_cache_dir:
            os.environ["HF_HUB_CACHE"] = self.hf_cache_dir
            os.environ["HUGGINGFACE_HUB_CACHE"] = self.hf_cache_dir
            os.environ["HF_HOME"] = self.hf_cache_dir

    def precheck_offline_cache(self) -> None:
        self._set_offline_env()
        missing = []
        pyannote_cfg_path: Optional[Path] = None
        if not os.path.exists(self.fasttext_model_path):
            missing.append(self.fasttext_model_path)
        if self.silero_model_path and not os.path.exists(self.silero_model_path):
            missing.append(self.silero_model_path)
        if self.asr_engine == "vosk":
            if not os.path.exists(self.vosk_model_en_path):
                missing.append(self.vosk_model_en_path)
            if not os.path.exists(self.vosk_model_de_path):
                missing.append(self.vosk_model_de_path)
        if self.asr_engine == "wav2vec2":
            if not os.path.exists(self.wav2vec2_model_en_path):
                missing.append(self.wav2vec2_model_en_path)
            if not os.path.exists(self.wav2vec2_model_de_path):
                missing.append(self.wav2vec2_model_de_path)
        pyannote_ref = str(self.pyannote_model_path or "").strip()
        pyannote_exists = bool(pyannote_ref) and os.path.exists(pyannote_ref)
        pyannote_is_repo_id = ("/" in pyannote_ref) and (not pyannote_exists)

        if pyannote_exists and os.path.isdir(pyannote_ref):
            cfg_file = os.path.join(self.pyannote_model_path, "config.yaml")
            if not os.path.exists(cfg_file):
                missing.append(f"{self.pyannote_model_path}/config.yaml")
            else:
                pyannote_cfg_path = Path(cfg_file)
        elif pyannote_exists and os.path.isfile(pyannote_ref):
            pyannote_cfg_path = Path(pyannote_ref)
        elif self.hf_offline or not pyannote_is_repo_id:
            missing.append(self.pyannote_model_path)

        if self.hf_offline and pyannote_cfg_path is not None:
            missing.extend(self._missing_pyannote_dependencies(pyannote_cfg_path))

        if missing:
            raise RuntimeError(
                "pipeline_local offline assets missing: " + ", ".join(missing)
            )

    def transcribe(self, wav_path: str) -> str:
        audio, sample_rate = read_wav_int16(wav_path)
        text, _, _ = self._transcribe_chunk(audio, sample_rate)
        return text.strip()

    def transcribe_segment(
        self,
        wav_path: str,
        segment_start_ts: str,
        diarizer,
        diarization_config,
        speaker_lock: str = "auto",
    ) -> List[dict]:
        audio, sample_rate = read_wav_int16(wav_path)
        if audio.size == 0:
            return []

        speech_ranges = self._speech_ranges(audio, sample_rate)
        if not speech_ranges:
            return []

        turns = self._pyannote_turns(wav_path)
        label_speaker_map = self._map_turn_labels(
            turns, audio, sample_rate, diarizer, diarization_config, speaker_lock
        )

        base_ts = datetime.fromisoformat(segment_start_ts)
        fragments: List[dict] = []
        had_text = False

        for start_sec, end_sec in speech_ranges:
            start_i = max(0, int(start_sec * sample_rate))
            end_i = min(audio.shape[0], int(end_sec * sample_rate))
            chunk = audio[start_i:end_i]
            if chunk.size == 0:
                continue

            text, language, language_conf = self._transcribe_chunk(chunk, sample_rate)
            text = text.strip()
            if not text:
                continue
            had_text = True

            label = self._dominant_label(start_sec, end_sec, turns)
            if label and label in label_speaker_map:
                speaker, low_confidence = label_speaker_map[label]
            else:
                emb = compute_embedding(chunk, sample_rate)
                speaker, low_confidence = self._select_speaker(
                    emb, diarizer, diarization_config, speaker_lock
                )

            frag_start = base_ts + timedelta(seconds=start_sec)
            frag_end = base_ts + timedelta(seconds=end_sec)
            fragments.append(
                {
                    "start_ts": frag_start.isoformat(),
                    "end_ts": frag_end.isoformat(),
                    "speaker": speaker,
                    "text": text,
                    "low_confidence": low_confidence,
                    "language": language,
                    "language_confidence": round(float(language_conf), 4),
                    "text_quality": round(float(self._text_quality_score(text, language)), 4),
                }
            )

        if not had_text:
            raise RuntimeError("no transcription output")

        return self._merge_adjacent_fragments(fragments)

    def _load_silero(self) -> None:
        if self._silero_model is not None and self._silero_get_speech_timestamps is not None:
            return
        from silero_vad import get_speech_timestamps, load_silero_vad

        if self.silero_model_path:
            try:
                self._silero_model = load_silero_vad(model_path=self.silero_model_path)
            except TypeError:
                self._silero_model = load_silero_vad()
        else:
            self._silero_model = load_silero_vad()
        self._silero_get_speech_timestamps = get_speech_timestamps

    def _load_fasttext(self):
        if self._fasttext_model is not None:
            return self._fasttext_model
        import fasttext

        self._fasttext_model = fasttext.load_model(self.fasttext_model_path)
        return self._fasttext_model

    def _load_vosk_model(self, lang: str):
        key = "de" if lang == "de" else "en"
        if key in self._vosk_models:
            return self._vosk_models[key]
        from vosk import Model

        model_path = self.vosk_model_de_path if key == "de" else self.vosk_model_en_path
        self._vosk_models[key] = Model(model_path)
        return self._vosk_models[key]

    def _load_pyannote_pipeline(self):
        if self._pyannote_pipeline is not None:
            return self._pyannote_pipeline
        from pyannote.audio import Pipeline

        model_ref = self.pyannote_model_path
        if model_ref and os.path.isdir(model_ref):
            cfg_file = Path(model_ref) / "config.yaml"
            if cfg_file.exists():
                model_ref = str(cfg_file.resolve())
            else:
                model_ref = str(Path(model_ref).resolve())
        elif model_ref and os.path.exists(model_ref):
            model_ref = str(Path(model_ref).resolve())
        if self.hf_offline and model_ref and os.path.isfile(model_ref):
            model_ref = self._build_pyannote_runtime_config(Path(model_ref))

        kwargs: Dict[str, Any] = {}
        if self.pyannote_auth_token:
            kwargs["use_auth_token"] = self.pyannote_auth_token
        if self.hf_cache_dir:
            kwargs["cache_dir"] = self.hf_cache_dir
        try:
            self._pyannote_pipeline = Pipeline.from_pretrained(model_ref, **kwargs)
        except TypeError:
            kwargs.pop("cache_dir", None)
            self._pyannote_pipeline = Pipeline.from_pretrained(model_ref, **kwargs)
        return self._pyannote_pipeline

    def _missing_pyannote_dependencies(self, pyannote_cfg_path: Path) -> List[str]:
        try:
            with pyannote_cfg_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as exc:
            return [f"{pyannote_cfg_path}: unreadable ({exc})"]

        params = (
            cfg.get("pipeline", {})
            .get("params", {})
        )
        refs = [params.get("segmentation"), params.get("embedding")]
        missing = []
        for ref in refs:
            if not isinstance(ref, str) or not ref:
                continue
            if os.path.exists(ref):
                continue
            if "/" not in ref:
                continue
            if not self._is_hf_model_cached(ref):
                missing.append(f"{self.hf_cache_dir}/models--{ref.replace('/', '--')}")
        return missing

    def _is_hf_model_cached(self, repo_id: str) -> bool:
        if not self.hf_cache_dir:
            return False
        repo_cache_dir = Path(self.hf_cache_dir) / f"models--{repo_id.replace('/', '--')}"
        snapshots_dir = repo_cache_dir / "snapshots"
        if not snapshots_dir.exists() or not snapshots_dir.is_dir():
            return False
        for snapshot in snapshots_dir.iterdir():
            if not snapshot.is_dir():
                continue
            files = [p.name for p in snapshot.iterdir() if p.is_file() or p.is_symlink()]
            has_config = "config.yaml" in files or "config.json" in files
            has_weights = any(
                name.endswith((".bin", ".safetensors", ".ckpt", ".onnx")) for name in files
            )
            if has_config and has_weights:
                return True
        return False

    def _build_pyannote_runtime_config(self, config_path: Path) -> str:
        try:
            with config_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            return str(config_path)

        params = cfg.get("pipeline", {}).get("params", {})
        changed = False
        for key in ("segmentation", "embedding"):
            ref = params.get(key)
            if not isinstance(ref, str) or not ref or os.path.exists(ref) or "/" not in ref:
                continue
            model_file = self._resolve_cached_model_file(ref)
            if model_file:
                params[key] = model_file
                changed = True

        if not changed:
            return str(config_path)

        fd, temp_path = tempfile.mkstemp(prefix="pyannote_runtime_", suffix=".yaml")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        self._pyannote_runtime_config = temp_path
        return temp_path

    def _resolve_cached_model_file(self, repo_id: str) -> Optional[str]:
        if not self.hf_cache_dir:
            return None
        repo_cache_dir = Path(self.hf_cache_dir) / f"models--{repo_id.replace('/', '--')}"
        snapshots_dir = repo_cache_dir / "snapshots"
        if not snapshots_dir.exists() or not snapshots_dir.is_dir():
            return None

        candidates: List[Path] = []
        ref_main = repo_cache_dir / "refs" / "main"
        if ref_main.exists():
            sha = ref_main.read_text(encoding="utf-8").strip()
            candidate = snapshots_dir / sha
            if candidate.exists() and candidate.is_dir():
                candidates.append(candidate)

        for snap in snapshots_dir.iterdir():
            if snap.is_dir():
                candidates.append(snap)

        model_names = (
            "pytorch_model.bin",
            "model.safetensors",
            "model.bin",
            "weights.ckpt",
        )
        for snap in candidates:
            for name in model_names:
                path = snap / name
                if path.exists():
                    return str(path.resolve())
        return None

    def _load_wav2vec2_model(self, lang: str) -> Tuple[Any, Any]:
        key = "de" if lang == "de" else "en"
        if key in self._wav2vec2_models:
            return self._wav2vec2_models[key]
        from transformers import AutoModelForCTC, AutoProcessor

        model_path = self.wav2vec2_model_de_path if key == "de" else self.wav2vec2_model_en_path
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=self.hf_offline)
        model = AutoModelForCTC.from_pretrained(model_path, local_files_only=self.hf_offline)
        model.eval()
        self._wav2vec2_models[key] = (processor, model)
        return self._wav2vec2_models[key]

    def _speech_ranges(self, audio: np.ndarray, sample_rate: int) -> List[Tuple[float, float]]:
        self._load_silero()
        import torch
        import torchaudio

        audio_f = np.asarray(audio, dtype=np.int16).astype(np.float32) / 32768.0
        tensor = torch.from_numpy(audio_f)
        target_sr = int(sample_rate)
        if target_sr != 16000:
            tensor = torchaudio.functional.resample(tensor, target_sr, 16000)
            target_sr = 16000

        raw = self._silero_get_speech_timestamps(
            tensor,
            self._silero_model,
            sampling_rate=target_sr,
            threshold=self.silero_threshold,
            min_speech_duration_ms=self.silero_min_speech_ms,
            min_silence_duration_ms=self.silero_min_silence_ms,
            speech_pad_ms=self.silero_speech_pad_ms,
        )

        ranges: List[Tuple[float, float]] = []
        for item in raw:
            start = float(item.get("start", 0)) / float(target_sr)
            end = float(item.get("end", 0)) / float(target_sr)
            if end > start:
                ranges.append((start, end))
        return ranges

    def _detect_language(self, text: str) -> Tuple[str, float]:
        clean = text.strip()
        if not clean:
            return "unknown", 0.0
        heuristic_lang, heuristic_conf = self._heuristic_detect_language(clean)
        if self._fasttext_disabled:
            return heuristic_lang, heuristic_conf
        try:
            model = self._load_fasttext()
            labels, scores = model.predict(clean.replace("\n", " "), k=1)
        except Exception as exc:
            self._fasttext_disabled = True
            if not self._fasttext_warned:
                logger.warning(
                    "fastText language-id failed; disabling language-id and using model-language fallback: %s",
                    exc,
                )
                self._fasttext_warned = True
            else:
                logger.debug("fastText language-id still unavailable: %s", exc)
            return heuristic_lang, heuristic_conf
        if not labels:
            return heuristic_lang, heuristic_conf
        label = str(labels[0]).replace("__label__", "").lower()
        lang = "unknown"
        if label.startswith("de"):
            lang = "de"
        elif label.startswith("en"):
            lang = "en"
        score = float(scores[0]) if scores else 0.0
        if lang not in {"en", "de"}:
            return heuristic_lang, heuristic_conf
        if heuristic_lang in {"en", "de"} and heuristic_lang != lang and heuristic_conf > (score + 0.12):
            return heuristic_lang, heuristic_conf
        return lang, max(score, heuristic_conf * 0.8)

    def _tokenize_words(self, text: str) -> List[str]:
        return [w.lower() for w in _WORD_RE.findall(text)]

    def _heuristic_detect_language(self, text: str) -> Tuple[str, float]:
        words = self._tokenize_words(text)
        if not words:
            return "unknown", 0.0
        total = float(len(words))
        en_hits = sum(1 for w in words if w in _EN_STOPWORDS)
        de_hits = sum(1 for w in words if w in _DE_STOPWORDS)
        umlaut_hits = sum(1 for w in words if any(ch in w for ch in "äöüß"))

        en_score = en_hits / total
        de_score = (de_hits / total) + (0.12 * min(umlaut_hits, 3))

        if en_score < 0.06 and de_score < 0.06:
            return "unknown", 0.25 if len(words) >= 3 else 0.15

        if de_score >= en_score:
            gap = de_score - en_score
            conf = min(0.96, 0.45 + de_score * 1.8 + gap * 0.6)
            return "de", conf
        gap = en_score - de_score
        conf = min(0.96, 0.45 + en_score * 1.8 + gap * 0.6)
        return "en", conf

    def _text_quality_score(self, text: str, lang_hint: str) -> float:
        stripped = text.strip()
        if not stripped:
            return 0.0
        words = self._tokenize_words(stripped)
        if not words:
            return 0.0
        alpha_chars = sum(1 for ch in stripped if ch.isalpha())
        alpha_ratio = alpha_chars / max(len(stripped), 1)
        total = float(len(words))
        short_ratio = sum(1 for w in words if len(w) <= 2) / total
        long_ratio = sum(1 for w in words if len(w) >= 16) / total
        unique_ratio = len(set(words)) / total

        if lang_hint == "de":
            stop_hits = sum(1 for w in words if w in _DE_STOPWORDS)
        elif lang_hint == "en":
            stop_hits = sum(1 for w in words if w in _EN_STOPWORDS)
        else:
            stop_hits = 0
        stop_cov = stop_hits / total

        score = 0.0
        score += 0.35 * alpha_ratio
        score += 0.25 * max(0.0, 1.0 - short_ratio)
        score += 0.15 * max(0.0, 1.0 - min(1.0, long_ratio * 2.0))
        score += 0.15 * min(1.0, unique_ratio * 1.15)
        score += 0.10 * min(1.0, stop_cov * 3.0)
        if len(words) <= 1:
            score -= 0.45
            if words[0] in _COMMON_STOPWORDS:
                score -= 0.25
        elif len(words) == 2 and all(w in _COMMON_STOPWORDS for w in words):
            score -= 0.20
        return max(0.0, min(1.0, score))

    def _decode_vosk(self, audio: np.ndarray, sample_rate: int, lang: str) -> Tuple[str, float]:
        from vosk import KaldiRecognizer

        model = self._load_vosk_model(lang)
        rec = KaldiRecognizer(model, float(sample_rate))
        rec.SetWords(True)
        pcm = np.asarray(audio, dtype=np.int16).tobytes()
        step = max(1, int(sample_rate * 2 * 0.25))
        parts: List[str] = []
        conf_values: List[float] = []
        for i in range(0, len(pcm), step):
            if rec.AcceptWaveform(pcm[i : i + step]):
                data = json.loads(rec.Result())
                txt = str(data.get("text", "")).strip()
                if txt:
                    parts.append(txt)
                for item in data.get("result", []) or []:
                    try:
                        conf_values.append(float(item.get("conf")))
                    except (TypeError, ValueError):
                        pass
        final = json.loads(rec.FinalResult())
        final_text = str(final.get("text", "")).strip()
        if final_text:
            parts.append(final_text)
        for item in final.get("result", []) or []:
            try:
                conf_values.append(float(item.get("conf")))
            except (TypeError, ValueError):
                pass
        avg_conf = float(sum(conf_values) / len(conf_values)) if conf_values else 0.0
        return " ".join(parts).strip(), avg_conf

    def _candidate_score(self, text: str, model_lang: str, asr_conf: float) -> _ChunkCandidate:
        detected_lang, detected_conf = self._detect_language(text)
        text_quality = self._text_quality_score(text, model_lang)
        score = 0.0
        score += 0.55 * text_quality
        score += 0.35 * max(0.0, min(1.0, asr_conf))
        if detected_lang == model_lang:
            score += 0.10 * max(0.25, min(1.0, detected_conf))
        elif detected_lang in {"en", "de"} and detected_lang != model_lang:
            score -= 0.08
        else:
            score += 0.02
        if len(text.strip()) < 4:
            score -= 0.18
        return _ChunkCandidate(
            text=text.strip(),
            model_lang=model_lang,
            detected_lang=detected_lang,
            detected_conf=detected_conf,
            asr_conf=asr_conf,
            text_quality=text_quality,
            score=score,
        )

    def _transcribe_chunk_vosk(self, audio: np.ndarray, sample_rate: int) -> Tuple[str, str, float]:
        text_en, conf_en = self._decode_vosk(audio, sample_rate, "en")
        text_de, conf_de = self._decode_vosk(audio, sample_rate, "de")
        candidates = []
        if text_en:
            candidates.append(self._candidate_score(text_en, "en", conf_en))
        if text_de:
            candidates.append(self._candidate_score(text_de, "de", conf_de))
        if not candidates:
            return "", "unknown", 0.0
        best = max(candidates, key=lambda c: c.score)
        lang = best.detected_lang if best.detected_lang in {"en", "de"} else best.model_lang
        lang_conf = best.detected_conf if best.detected_lang in {"en", "de"} else max(0.45, best.text_quality * 0.9)
        return best.text, lang, lang_conf

    def _decode_wav2vec2(self, audio: np.ndarray, sample_rate: int, lang: str) -> str:
        import torch
        import torchaudio

        processor, model = self._load_wav2vec2_model(lang)
        audio_f = np.asarray(audio, dtype=np.int16).astype(np.float32) / 32768.0
        tensor = torch.from_numpy(audio_f)
        if sample_rate != 16000:
            tensor = torchaudio.functional.resample(tensor, sample_rate, 16000)
            sample_rate = 16000
        inputs = processor(tensor.numpy(), sampling_rate=sample_rate, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted = torch.argmax(logits, dim=-1)
        text = processor.batch_decode(predicted, skip_special_tokens=True)[0]
        return str(text).strip()

    def _transcribe_chunk_wav2vec2(self, audio: np.ndarray, sample_rate: int) -> Tuple[str, str, float]:
        text_en = self._decode_wav2vec2(audio, sample_rate, "en")
        text_de = self._decode_wav2vec2(audio, sample_rate, "de")
        candidates = []
        if text_en:
            candidates.append(self._candidate_score(text_en, "en", 0.5))
        if text_de:
            candidates.append(self._candidate_score(text_de, "de", 0.5))
        if not candidates:
            return "", "unknown", 0.0
        best = max(candidates, key=lambda c: c.score)
        lang = best.detected_lang if best.detected_lang in {"en", "de"} else best.model_lang
        lang_conf = best.detected_conf if best.detected_lang in {"en", "de"} else max(0.45, best.text_quality * 0.9)
        return best.text, lang, lang_conf

    def _transcribe_chunk(self, audio: np.ndarray, sample_rate: int) -> Tuple[str, str, float]:
        if self.asr_engine == "vosk":
            return self._transcribe_chunk_vosk(audio, sample_rate)
        if self.asr_engine == "wav2vec2":
            return self._transcribe_chunk_wav2vec2(audio, sample_rate)
        raise RuntimeError(f"Unsupported pipeline_local asr_engine: {self.asr_engine}")

    def _pyannote_turns(self, wav_path: str) -> List[Tuple[float, float, str]]:
        pipeline = self._load_pyannote_pipeline()
        diarization = pipeline({"audio": wav_path})
        turns: List[Tuple[float, float, str]] = []
        for segment, _, label in diarization.itertracks(yield_label=True):
            start = float(segment.start)
            end = float(segment.end)
            if end > start:
                turns.append((start, end, str(label)))
        turns.sort(key=lambda x: x[0])
        return turns

    def _map_turn_labels(
        self,
        turns: List[Tuple[float, float, str]],
        audio: np.ndarray,
        sample_rate: int,
        diarizer,
        diarization_config,
        speaker_lock: str,
    ) -> Dict[str, Tuple[str, bool]]:
        label_buffers: Dict[str, List[np.ndarray]] = {}
        for start, end, label in turns:
            a = max(0, int(start * sample_rate))
            b = min(audio.shape[0], int(end * sample_rate))
            if b <= a:
                continue
            label_buffers.setdefault(label, []).append(audio[a:b])

        mapping: Dict[str, Tuple[str, bool]] = {}
        for label, chunks in label_buffers.items():
            if not chunks:
                continue
            merged = np.concatenate(chunks)
            emb = compute_embedding(merged, sample_rate)
            mapping[label] = self._select_speaker(
                emb, diarizer, diarization_config, speaker_lock
            )
        return mapping

    def _dominant_label(
        self, start_sec: float, end_sec: float, turns: List[Tuple[float, float, str]]
    ) -> Optional[str]:
        best_label = None
        best_overlap = 0.0
        for turn_start, turn_end, label in turns:
            overlap = max(0.0, min(end_sec, turn_end) - max(start_sec, turn_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = label
        return best_label

    def _select_speaker(
        self,
        embedding: np.ndarray,
        diarizer,
        diarization_config,
        speaker_lock: str,
    ) -> Tuple[str, bool]:
        mode = (speaker_lock or "auto").lower()
        if mode == "hugo":
            return "Hugo", False
        if mode == "leon":
            return "Leon", False
        if diarizer is None or diarization_config is None:
            return "Unknown", True
        if diarization_config.require_both_enrolled:
            if not (diarizer.has_embedding("Hugo") and diarizer.has_embedding("Leon")):
                return "Unknown", True
        best_speaker, best_score, second_score = diarizer.best_match(embedding)
        if not best_speaker:
            return "Unknown", True
        margin = second_score - best_score
        if best_score > diarization_config.max_distance:
            return "Unknown", True
        if margin < diarization_config.min_margin:
            return "Unknown", True
        soft_max = diarization_config.max_distance * 0.85
        soft_margin = diarization_config.min_margin * 1.5
        low_confidence = best_score > soft_max or margin < soft_margin
        return best_speaker, bool(low_confidence)

    def _merge_adjacent_fragments(self, rows: List[dict]) -> List[dict]:
        if not rows:
            return rows
        sorted_rows = sorted(rows, key=lambda r: r["start_ts"])
        merged: List[dict] = []
        for row in sorted_rows:
            if not merged:
                merged.append(dict(row))
                continue
            prev = merged[-1]
            same_speaker = prev.get("speaker") == row.get("speaker")
            same_conf = bool(prev.get("low_confidence")) == bool(row.get("low_confidence"))
            prev_end = datetime.fromisoformat(prev["end_ts"])
            row_start = datetime.fromisoformat(row["start_ts"])
            close = (row_start - prev_end).total_seconds() <= self.merge_gap_sec
            if same_speaker and same_conf and close:
                prev["end_ts"] = row["end_ts"]
                prev_text = str(prev.get("text", "")).strip()
                row_text = str(row.get("text", "")).strip()
                if row_text:
                    prev["text"] = f"{prev_text} {row_text}".strip()
                continue
            merged.append(dict(row))
        return merged
