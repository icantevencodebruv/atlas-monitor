import os
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 7070
    open_browser: bool = True


class StorageConfig(BaseModel):
    db_path: str = "./data/app.db"
    audio_dir: str = "./data/audio"
    exports_dir: str = "./data/exports"
    logs_dir: str = "./data/logs"


class AudioConfig(BaseModel):
    sample_rate: int = 16000
    channels: int = 1
    segment_seconds: int = 30
    device_hostapi_preference: dict = Field(default_factory=dict)
    input_device_name: Optional[str] = None


class WorkHoursConfig(BaseModel):
    timezone: str = "Europe/Berlin"
    enabled: bool = True
    work_days: List[str] = ["MON", "TUE", "WED", "THU", "FRI"]
    work_start: str = "09:00"
    work_end: str = "18:00"


class NemoConfig(BaseModel):
    model_name: str = ""
    languages: List[str] = ["en"]


class MlxConfig(BaseModel):
    model_id: str = "mlx-community/parakeet-tdt-0.6b-v3"
    hf_cache_dir: str = "./models/parakeet_mlx"
    hf_cache_dir_parent: str = "./models"
    hf_offline: bool = False
    languages: List[str] = ["en", "de"]


class WhisperCppConfig(BaseModel):
    model_path: str = "./models/ggml-large-v3.bin"
    language: str = "auto"
    binary_path: str = ""


class PipelineLocalConfig(BaseModel):
    asr_engine: str = "vosk"
    silero_model_path: str = "./models/silero/silero_vad.jit"
    silero_threshold: float = 0.5
    silero_min_speech_ms: int = 250
    silero_min_silence_ms: int = 120
    silero_speech_pad_ms: int = 80
    merge_gap_sec: float = 0.35
    fasttext_model_path: str = "./models/lid.176.bin"
    vosk_model_en_path: str = "./models/vosk/vosk-model-en"
    vosk_model_de_path: str = "./models/vosk/vosk-model-de"
    wav2vec2_model_en_path: str = "./models/wav2vec2/wav2vec2-en"
    wav2vec2_model_de_path: str = "./models/wav2vec2/wav2vec2-de"
    pyannote_model_path: str = "./models/pyannote/speaker-diarization-community-1"
    pyannote_auth_token: str = ""
    hf_cache_dir: str = "./models/hf_cache"
    hf_offline: bool = True


class ASRConfig(BaseModel):
    backend: str = "auto"
    required_languages: List[str] = ["de", "en"]
    windows_nemo_parakeet_cuda: NemoConfig = Field(default_factory=NemoConfig)
    mac_parakeet_mlx: MlxConfig = Field(default_factory=MlxConfig)
    whisper_cpp: WhisperCppConfig = Field(default_factory=WhisperCppConfig)
    pipeline_local: PipelineLocalConfig = Field(default_factory=PipelineLocalConfig)


class TranscriptionConfig(BaseModel):
    vad_aggressiveness: int = 2
    min_utterance_sec: float = 0.6
    max_utterance_sec: float = 30.0
    max_silence_sec: float = 0.8
    enrollment_seconds: int = 30


class DiarizationConfig(BaseModel):
    max_distance: float = 0.25
    min_margin: float = 0.05
    require_both_enrolled: bool = True
    enrollment_min_snr_db: float = 12.0
    reference_locked: bool = False


class RetryConfig(BaseModel):
    enabled: bool = True
    max_attempts: int = 5
    base_backoff_sec: int = 30
    max_backoff_sec: int = 900
    poll_interval_sec: int = 15


class Config(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    work_hours: WorkHoursConfig = Field(default_factory=WorkHoursConfig)
    asr: ASRConfig = Field(default_factory=ASRConfig)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)


def load_config(path: str) -> Config:
    if not os.path.exists(path):
        cfg = Config()
        cfg.app.host = "127.0.0.1"
        return cfg
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = Config.model_validate(data)
    cfg.app.host = "127.0.0.1"
    return cfg


def save_config(path: str, cfg: Config) -> None:
    data = cfg.model_dump()
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
