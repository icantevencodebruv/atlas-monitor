import logging
import platform

from app.asr.nemo_parakeet import NemoParakeetBackend
from app.asr.parakeet_mlx import ParakeetMLXBackend
from app.asr.pipeline_local import PipelineLocalBackend
from app.asr.whisper_cpp import WhisperCppBackend

logger = logging.getLogger(__name__)


def _build_backend(name: str, config):
    if name == "windows_nemo_parakeet_cuda":
        cfg = config.asr.windows_nemo_parakeet_cuda
        return NemoParakeetBackend(cfg.model_name, cfg.languages)
    if name == "mac_parakeet_mlx":
        cfg = config.asr.mac_parakeet_mlx
        return ParakeetMLXBackend(
            cfg.model_id,
            cfg.hf_cache_dir,
            cfg.hf_cache_dir_parent,
            cfg.hf_offline,
            cfg.languages,
        )
    if name == "whisper_cpp":
        cfg = config.asr.whisper_cpp
        return WhisperCppBackend(cfg.model_path, cfg.language, cfg.binary_path)
    if name == "pipeline_local":
        cfg = config.asr.pipeline_local
        return PipelineLocalBackend(cfg)
    raise ValueError("Unknown backend")


def _backend_available(name: str) -> bool:
    if name == "windows_nemo_parakeet_cuda":
        return NemoParakeetBackend.is_available()
    if name == "mac_parakeet_mlx":
        return ParakeetMLXBackend.is_available()
    if name == "whisper_cpp":
        return WhisperCppBackend.is_available()
    if name == "pipeline_local":
        return PipelineLocalBackend.is_available()
    return False


def select_backend(config):
    required = config.asr.required_languages
    requested = config.asr.backend
    if requested and requested != "auto":
        if _backend_available(requested):
            backend = _build_backend(requested, config)
            if backend.supports_languages(required):
                logger.info("Using configured ASR backend: %s", requested)
                return backend
            logger.warning("Configured backend missing required languages, falling back.")
        else:
            logger.warning("Configured backend not available, falling back.")

    system = platform.system().lower()
    if system.startswith("win") and _backend_available("windows_nemo_parakeet_cuda"):
        backend = _build_backend("windows_nemo_parakeet_cuda", config)
        if backend.supports_languages(required):
            logger.info("Auto-selected NeMo Parakeet backend.")
            return backend
        logger.warning("NeMo Parakeet lacks required languages, falling back to whisper.cpp.")
    if system.startswith("darwin") and _backend_available("mac_parakeet_mlx"):
        backend = _build_backend("mac_parakeet_mlx", config)
        if backend.supports_languages(required):
            logger.info("Auto-selected Parakeet MLX backend.")
            return backend
        logger.warning("Parakeet MLX lacks required languages, falling back to whisper.cpp.")

    logger.info("Using whisper.cpp backend.")
    return _build_backend("whisper_cpp", config)
