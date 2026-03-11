import logging
import os
import threading
from datetime import datetime
from queue import Queue

from app.asr.manager import select_backend
from app.config import load_config, save_config
from app.database import Database
from app.logging_setup import setup_logging
from app.services.diarization import SpeakerIdentifier
from app.services.enrollment import EnrollmentService
from app.services.exporter import build_export, compute_workday_range
from app.services.recorder import SegmentRecorder, select_input_device
from app.services.retry_worker import RetryWorker
from app.services.scheduling import WorkHoursScheduler
from app.services.transcription import TranscriptionWorker
from app.state import AppState

CONFIG_PATH = os.environ.get("APP_CONFIG", "config.yaml")

config = load_config(CONFIG_PATH)
setup_logging(config.storage.logs_dir)
logger = logging.getLogger(__name__)
config_lock = threading.Lock()

if isinstance(config.audio.input_device_name, str) and config.audio.input_device_name.strip() == "":
    config.audio.input_device_name = None

os.makedirs(config.storage.audio_dir, exist_ok=True)
os.makedirs(config.storage.exports_dir, exist_ok=True)
os.makedirs(config.storage.logs_dir, exist_ok=True)

db = Database(config.storage.db_path)
state = AppState()
device_lock = threading.Lock()
state.reference_locked = config.diarization.reference_locked

diarizer = SpeakerIdentifier()
for _speaker in ["Hugo", "Leon"]:
    diarizer.load_embedding(_speaker, db.get_embedding(_speaker))

backend = select_backend(config)
backend_error: str | None = None
segment_queue: Queue = Queue()


def _mark_segment_start(ts: datetime) -> None:
    with state.lock:
        state.last_segment_start = ts.isoformat()


recorder = SegmentRecorder(
    db=db,
    audio_dir=config.storage.audio_dir,
    segment_seconds=config.audio.segment_seconds,
    sample_rate=config.audio.sample_rate,
    channels=config.audio.channels,
    hostapi_preference=config.audio.device_hostapi_preference.get(
        "windows" if os.name == "nt" else "macos"
    ),
    input_device_name=config.audio.input_device_name,
    queue=segment_queue,
    device_lock=device_lock,
    on_segment_start=_mark_segment_start,
)
worker = TranscriptionWorker(db, segment_queue, backend, diarizer, config, state)


def _auto_export_workday(now_local: datetime) -> None:
    cfg = config.work_hours
    if not cfg.enabled:
        return
    try:
        start_ts, end_ts = compute_workday_range(now_local, cfg.timezone, cfg.work_start, cfg.work_end)
    except Exception as exc:
        logger.warning("Auto export skipped (invalid workday range): %s", exc)
        return
    start_iso = start_ts.isoformat()
    end_iso = end_ts.isoformat()
    if start_ts >= end_ts:
        logger.warning("Auto export skipped (start >= end): %s .. %s", start_iso, end_iso)
        return
    existing = db.find_export("workday", start_iso, end_iso)
    if existing:
        logger.info("Auto export skipped (already exists): workday %s .. %s", start_iso, end_iso)
        return
    export_id = build_export(db, "workday", start_ts, end_ts, config.storage.exports_dir)
    logger.info("Auto export created: id=%s workday %s .. %s", export_id, start_iso, end_iso)


scheduler = WorkHoursScheduler(config, state, recorder, db, on_workday_end=_auto_export_workday)
retry_worker = RetryWorker(db, segment_queue, config)
enroller = EnrollmentService(
    db,
    diarizer,
    config.audio,
    config.diarization,
    device_lock,
    config.storage.audio_dir,
)


def _persist_config() -> None:
    with config_lock:
        save_config(CONFIG_PATH, config)
