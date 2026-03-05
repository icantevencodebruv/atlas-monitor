import logging
import os
import threading
from queue import Queue
import webbrowser
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel

from app.asr.manager import select_backend
from app.config import load_config, save_config
from app.db.database import Database
from app.logging_setup import setup_logging
from app.services.diarization import SpeakerIdentifier
from app.services.enrollment import EnrollmentService
from app.services.exporter import build_export, compute_range, compute_workday_range
from app.services.audio_probe import measure_level, test_recording
from app.services.recorder import SegmentRecorder, list_input_devices, set_input_device, select_input_device
from app.services.retry_worker import RetryWorker
from app.services.scheduling import WorkHoursScheduler, is_within_work_hours
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
for speaker in ["Hugo", "Leon"]:
    diarizer.load_embedding(speaker, db.get_embedding(speaker))

backend = select_backend(config)
backend_error = None
segment_queue = Queue()


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

app = FastAPI()


def _persist_config() -> None:
    with config_lock:
        save_config(CONFIG_PATH, config)


def _auto_mode_ready() -> bool:
    if not config.diarization.require_both_enrolled:
        return True
    return diarizer.has_embedding("Hugo") and diarizer.has_embedding("Leon")


def _is_within_dir(path: str, base_dir: str) -> bool:
    try:
        real_path = os.path.realpath(path)
        real_base = os.path.realpath(base_dir)
        return os.path.commonpath([real_path, real_base]) == real_base
    except Exception:
        return False


def _ensure_setup_mutable() -> None:
    with state.lock:
        recording = state.recording
        reference_locked = state.reference_locked
    if recording:
        raise HTTPException(status_code=400, detail="Stop recording before changing speaker references.")
    if reference_locked:
        raise HTTPException(status_code=400, detail="Reference locked. Click Reroll to update.")


def _preferred_hostapi() -> str | None:
    return config.audio.device_hostapi_preference.get("windows" if os.name == "nt" else "macos")


class ExportRequest(BaseModel):
    range: str
    start: str | None = None
    end: str | None = None


class RecordRequest(BaseModel):
    lock_mode: str | None = None
    set_only: bool = False


class DeviceSelectRequest(BaseModel):
    index: int | None = None
    name: str | None = None


@app.on_event("startup")
def _startup():
    worker.start()
    scheduler.start()
    retry_worker.start()
    global backend_error
    try:
        if hasattr(backend, "precheck_offline_cache"):
            backend.precheck_offline_cache()
    except Exception as exc:
        backend_error = str(exc)
        logger.error("ASR backend precheck failed: %s", backend_error)
    if config.app.open_browser:
        url = f"http://{config.app.host}:{config.app.port}"
        webbrowser.open(url)


@app.on_event("shutdown")
def _shutdown():
    scheduler.shutdown()
    worker.shutdown()
    retry_worker.shutdown()
    recorder.shutdown()
    db.close()


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(_HOME_HTML)


@app.get("/setup", response_class=HTMLResponse)
def setup_page():
    return HTMLResponse(_SETUP_HTML)


@app.get("/mic", response_class=HTMLResponse)
def mic_page():
    return HTMLResponse(_MIC_HTML)


@app.get("/admin", response_class=HTMLResponse)
def admin_page():
    return HTMLResponse(_ADMIN_HTML)


@app.get("/status")
def status():
    enrolled = {
        "Hugo": diarizer.has_embedding("Hugo"),
        "Leon": diarizer.has_embedding("Leon"),
    }
    in_hours = is_within_work_hours(config.work_hours, datetime.now(ZoneInfo(config.work_hours.timezone)))
    return {
        "status": "recording" if state.recording else "idle",
        "manual_override": state.manual_override,
        "backend": backend.name,
        "backend_error": backend_error,
        "speaker_lock": state.speaker_lock,
        "reference_locked": state.reference_locked,
        "enrolled": enrolled,
        "auto_ready": (enrolled["Hugo"] and enrolled["Leon"]) if config.diarization.require_both_enrolled else True,
        "recording_since": state.recording_since,
        "last_segment_start": state.last_segment_start,
        "segment_seconds": config.audio.segment_seconds,
        "enrollment_seconds": config.transcription.enrollment_seconds,
        "work_hours_enabled": config.work_hours.enabled,
        "in_work_hours": in_hours,
        "timezone": config.work_hours.timezone,
        "input_device_name": config.audio.input_device_name,
    }


@app.get("/admin/failed")
def admin_failed():
    rows = db.list_failed_segments()
    payload = []
    for row in rows:
        start = datetime.fromisoformat(row["start_ts"])
        end = datetime.fromisoformat(row["end_ts"])
        duration = (end - start).total_seconds()
        payload.append(
            {
                "id": row["id"],
                "start_ts": row["start_ts"],
                "end_ts": row["end_ts"],
                "duration_sec": duration,
                "attempts": row["attempts"] or 0,
                "error": row["error"] or "",
            }
        )
    return {"rows": payload}


@app.post("/record/start")
def record_start(req: RecordRequest | None = None):
    if req and req.lock_mode is not None:
        lock_mode = req.lock_mode.strip().lower()
        if lock_mode not in {"auto", "hugo", "leon"}:
            raise HTTPException(status_code=400, detail="Invalid speaker mode.")
        if lock_mode == "auto" and not _auto_mode_ready():
            raise HTTPException(status_code=400, detail="Auto mode requires both speakers enrolled.")
        with state.lock:
            state.speaker_lock = lock_mode
    with state.lock:
        active_lock_mode = state.speaker_lock
    if active_lock_mode == "auto" and not _auto_mode_ready():
        raise HTTPException(status_code=400, detail="Auto mode requires both speakers enrolled.")
    if req and req.set_only:
        return {"status": "recording" if state.recording else "idle", "speaker_lock": state.speaker_lock}
    with state.lock:
        state.manual_override = True
        state.recording = True
        state.recording_since = datetime.now(timezone.utc).isoformat()
    recorder.start()
    if state.current_session_id is None:
        state.current_session_id = db.add_session(datetime.now(timezone.utc).isoformat())
    return {"status": "recording"}


@app.post("/record/stop")
def record_stop():
    with state.lock:
        state.manual_override = False
        state.recording = False
        state.recording_since = None
        state.last_segment_start = None
    recorder.stop()
    if state.current_session_id is not None:
        db.end_session(state.current_session_id, datetime.now(timezone.utc).isoformat())
        state.current_session_id = None
    return {"status": "idle"}


@app.post("/record/resume")
def record_resume():
    with state.lock:
        state.manual_override = None
    scheduler.tick()
    return {"status": "recording" if state.recording else "idle", "manual_override": state.manual_override}


@app.post("/export")
def export_range(req: ExportRequest):
    now = datetime.now(timezone.utc)
    session_row = db.get_latest_session()
    if req.range == "custom" and (not req.start or not req.end):
        raise HTTPException(status_code=400, detail="Custom range requires start and end timestamps.")
    try:
        start_ts, end_ts = compute_range(
            req.range, now, session_row, req.start, req.end, config.work_hours.timezone
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid date/time: {exc}") from exc
    if start_ts >= end_ts:
        raise HTTPException(status_code=400, detail="Start must be before end.")
    export_id = build_export(db, req.range, start_ts, end_ts, config.storage.exports_dir)
    return {"id": export_id}


@app.get("/exports/recent")
def exports_recent(limit: int = 8):
    rows = db.list_exports(limit)
    payload = []
    for row in rows:
        file_path = row["file_path"]
        size_bytes = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        payload.append(
            {
                "id": row["id"],
                "created_ts": row["created_ts"],
                "range_label": row["range_label"],
                "start_ts": row["start_ts"],
                "end_ts": row["end_ts"],
                "file_path": row["file_path"],
                "size_bytes": size_bytes,
            }
        )
    return {"rows": payload}


@app.get("/transcripts/recent")
def transcripts_recent(
    minutes: int = 30,
    search: str = "",
    include_unknown: bool = False,
    include_low_confidence: bool = False,
):
    now = datetime.now(timezone.utc)
    start = now - timedelta(minutes=max(1, min(minutes, 240)))
    rows = db.list_transcripts_between(start.isoformat(), now.isoformat())
    query = search.strip().lower()
    payload = []
    for row in rows:
        speaker = row["speaker"]
        text = row["text"]
        low_conf = bool(row["low_confidence"])
        if not include_unknown and speaker == "Unknown":
            continue
        if not include_low_confidence and low_conf:
            continue
        if query and query not in text.lower():
            continue
        payload.append(
            {
                "start_ts": row["start_ts"],
                "end_ts": row["end_ts"],
                "speaker": speaker,
                "text": text,
                "low_confidence": low_conf,
            }
        )
    return {"rows": payload}


@app.get("/audio/devices")
def audio_devices():
    devices = list_input_devices()
    return {"devices": devices, "selected": config.audio.input_device_name}


@app.post("/audio/device")
def audio_device(req: DeviceSelectRequest):
    if state.recording:
        raise HTTPException(status_code=400, detail="Stop recording before changing the device.")
    if req.index is None and not req.name:
        raise HTTPException(status_code=400, detail="Device index or name required.")
    if req.name and req.name.lower() == "auto":
        config.audio.input_device_name = None
        select_input_device(_preferred_hostapi())
        recorder.update_preferred_device(None)
        _persist_config()
        return {"status": "ok", "device_name": None}
    if req.index is not None:
        set_input_device(req.index)
        devices = list_input_devices()
        match = next((d for d in devices if d["index"] == req.index), None)
        if not match:
            raise HTTPException(status_code=404, detail="Device not found.")
        config.audio.input_device_name = match["name"]
        recorder.update_preferred_device(match["name"])
        _persist_config()
        return {"status": "ok", "device_name": match["name"]}
    devices = list_input_devices()
    match = next((d for d in devices if d["name"].lower() == req.name.lower()), None)
    if not match:
        raise HTTPException(status_code=404, detail="Device not found.")
    set_input_device(match["index"])
    config.audio.input_device_name = match["name"]
    recorder.update_preferred_device(match["name"])
    _persist_config()
    return {"status": "ok", "device_name": match["name"]}


@app.get("/audio/level")
def audio_level():
    return measure_level(
        config.audio.sample_rate,
        config.audio.channels,
        device_lock,
        _preferred_hostapi(),
        config.audio.input_device_name,
    )


@app.post("/audio/test")
def audio_test():
    return test_recording(
        3.0,
        config.audio.sample_rate,
        config.audio.channels,
        device_lock,
        _preferred_hostapi(),
        config.audio.input_device_name,
    )


@app.get("/download/{export_id}")
def download(export_id: int):
    export = db.get_export(export_id)
    if not export:
        raise HTTPException(status_code=404, detail="Export not found")
    file_path = export["file_path"]
    if not _is_within_dir(file_path, config.storage.exports_dir):
        raise HTTPException(status_code=400, detail="Invalid export path")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Export file not found")
    return FileResponse(file_path, filename=os.path.basename(file_path))


@app.post("/admin/segment/{segment_id}/retry")
def admin_retry(segment_id: int):
    segment = db.get_segment(segment_id)
    if not segment:
        raise HTTPException(status_code=404, detail="Segment not found")
    db.update_segment_status(segment_id, "pending", None)
    segment_queue.put(segment_id)
    return {"status": "queued"}


@app.post("/admin/segment/{segment_id}/delete")
def admin_delete(segment_id: int):
    segment = db.get_segment(segment_id)
    if not segment:
        raise HTTPException(status_code=404, detail="Segment not found")
    file_path = segment["file_path"]
    if not _is_within_dir(file_path, config.storage.audio_dir):
        raise HTTPException(status_code=400, detail="Invalid segment path")
    if os.path.exists(file_path):
        os.remove(file_path)
    db.delete_segment(segment_id)
    return {"status": "deleted"}


@app.get("/admin/segment/{segment_id}/export")
def admin_export(segment_id: int):
    segment = db.get_segment(segment_id)
    if not segment:
        raise HTTPException(status_code=404, detail="Segment not found")
    if segment["status"] != "failed":
        raise HTTPException(status_code=400, detail="Export only allowed for failed segments")
    file_path = segment["file_path"]
    if not _is_within_dir(file_path, config.storage.audio_dir):
        raise HTTPException(status_code=400, detail="Invalid segment path")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    zip_name = f"segment_{segment_id}.zip"
    zip_path = os.path.join(config.storage.exports_dir, zip_name)
    os.makedirs(config.storage.exports_dir, exist_ok=True)
    import zipfile

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(file_path, arcname=os.path.basename(file_path))
    return FileResponse(zip_path, filename=zip_name)


@app.post("/setup/enroll/hugo")
def enroll_hugo():
    if state.reference_locked:
        raise HTTPException(status_code=400, detail="Reference locked. Click Reroll to update.")
    if state.recording:
        raise HTTPException(status_code=400, detail="Stop recording before enrollment.")
    try:
        result = enroller.enroll("Hugo", config.transcription.enrollment_seconds)
    except Exception as exc:
        msg = str(exc).lower()
        if "enrollment_in_progress" in msg:
            raise HTTPException(status_code=409, detail="Another enrollment is in progress.") from exc
        if "no_audio_captured" in msg:
            raise HTTPException(status_code=400, detail="No audio captured. Check Mic settings and retry.") from exc
        if "portaudio" in msg or "inputstream" in msg:
            logger.warning("Enrollment failed for Hugo: %s", exc)
            raise HTTPException(
                status_code=503,
                detail="Microphone is unavailable. Check Mic settings and close other apps using the mic, then retry.",
            ) from exc
        logger.exception("Unexpected enrollment failure for Hugo.")
        raise
    if result.get("cancelled"):
        return JSONResponse({"status": "cancelled", **result})
    return JSONResponse({"status": "ok", **result})


@app.post("/setup/enroll/leon")
def enroll_leon():
    if state.reference_locked:
        raise HTTPException(status_code=400, detail="Reference locked. Click Reroll to update.")
    if state.recording:
        raise HTTPException(status_code=400, detail="Stop recording before enrollment.")
    try:
        result = enroller.enroll("Leon", config.transcription.enrollment_seconds)
    except Exception as exc:
        msg = str(exc).lower()
        if "enrollment_in_progress" in msg:
            raise HTTPException(status_code=409, detail="Another enrollment is in progress.") from exc
        if "no_audio_captured" in msg:
            raise HTTPException(status_code=400, detail="No audio captured. Check Mic settings and retry.") from exc
        if "portaudio" in msg or "inputstream" in msg:
            logger.warning("Enrollment failed for Leon: %s", exc)
            raise HTTPException(
                status_code=503,
                detail="Microphone is unavailable. Check Mic settings and close other apps using the mic, then retry.",
            ) from exc
        logger.exception("Unexpected enrollment failure for Leon.")
        raise
    if result.get("cancelled"):
        return JSONResponse({"status": "cancelled", **result})
    return JSONResponse({"status": "ok", **result})


@app.post("/setup/enroll/stop")
def stop_enrollment():
    stop = enroller.stop_active()
    if stop["stopping"]:
        return {"status": "stopping", "speaker": stop["speaker"]}
    return {"status": "idle", "speaker": None}


@app.post("/setup/clear/hugo")
def clear_hugo():
    _ensure_setup_mutable()
    db.clear_embedding("Hugo")
    diarizer.remove_embedding("Hugo")
    return {"status": "cleared"}


@app.post("/setup/clear/leon")
def clear_leon():
    _ensure_setup_mutable()
    db.clear_embedding("Leon")
    diarizer.remove_embedding("Leon")
    return {"status": "cleared"}


@app.post("/setup/reroll")
def reroll_references():
    with state.lock:
        recording = state.recording
    if recording:
        raise HTTPException(status_code=400, detail="Stop recording before rerolling references.")
    with state.lock:
        state.reference_locked = False
    config.diarization.reference_locked = False
    db.clear_embedding("Hugo")
    db.clear_embedding("Leon")
    diarizer.remove_embedding("Hugo")
    diarizer.remove_embedding("Leon")
    _persist_config()
    return {"status": "unlocked"}


@app.post("/setup/lock_reference")
def lock_reference():
    with state.lock:
        recording = state.recording
    if recording:
        raise HTTPException(status_code=400, detail="Stop recording before locking references.")
    if not (diarizer.has_embedding("Hugo") and diarizer.has_embedding("Leon")):
        raise HTTPException(status_code=400, detail="Enroll both Hugo and Leon before locking references.")
    with state.lock:
        state.reference_locked = True
    config.diarization.reference_locked = True
    _persist_config()
    return {"status": "locked"}


_HOME_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <title>Overview • Atlas Monitor</title>
    <style>
      :root {
        --bg-primary: radial-gradient(circle at 20% 20%, #0f172a, #030712);
        --bg-surface: rgba(15, 23, 42, 0.6);
        --bg-surface-strong: rgba(15, 23, 42, 0.8);
        --border-glass: rgba(56, 107, 255, 0.35);
        --text-primary: #e2e8f0;
        --text-secondary: rgba(226, 232, 240, 0.7);
        --accent: #5b8dff;
        --accent-glow: rgba(91, 141, 255, 0.45);
        --shadow-glass: 0 30px 80px rgba(8, 12, 24, 0.7);
        --success: #4bde97;
        --warn: #ffb94d;
        --error: #ff6b6b;
        --space-xs: 0.28rem;
        --space-sm: 0.55rem;
        --space-md: 1.1rem;
        --space-lg: 1.65rem;
        --space-xl: 2.2rem;
        --radius-sm: 8px;
        --radius-md: 16px;
        --transition: 180ms ease-in-out;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        min-height: 100vh;
        font-family: "Inter", "Segoe UI", sans-serif;
        color: var(--text-primary);
        background: var(--bg-primary);
        overflow-x: hidden;
      }
      a { color: inherit; text-decoration: none; }
      .page { max-width: 1520px; margin: 0 auto; padding: 28px 24px 56px; }
      .top-nav {
        display: grid;
        grid-template-columns: 1fr auto auto;
        align-items: center;
        gap: 20px;
        margin-bottom: 24px;
      }
      .brand-title {
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }
      .brand-subtitle { color: var(--text-secondary); font-size: 0.85rem; margin-top: 6px; }
      .nav-links { display: flex; gap: 16px; }
      .nav-link {
        padding: 8px 14px;
        border-radius: 999px;
        color: var(--text-secondary);
        border: 1px solid transparent;
        transition: var(--transition);
      }
      .nav-link.active, .nav-link:hover {
        color: var(--text-primary);
        border-color: var(--border-glass);
        background: rgba(91, 141, 255, 0.12);
      }
      .status-pill {
        justify-self: end;
        display: inline-flex;
        align-items: center;
        gap: 10px;
        padding: 10px 18px;
        border-radius: 999px;
        border: 1px solid var(--border-glass);
        background: var(--bg-surface-strong);
        font-size: 0.75rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
      }
      .status-pill::before {
        content: "";
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: var(--warn);
        box-shadow: 0 0 12px rgba(255, 185, 77, 0.6);
      }
      .status-pill.recording {
        border-color: rgba(75, 222, 151, 0.5);
        color: #eafff5;
        animation: pulse 2.4s ease-in-out infinite;
      }
      .status-pill.recording::before {
        background: var(--success);
        box-shadow: 0 0 16px rgba(75, 222, 151, 0.6);
      }
      @keyframes pulse {
        0% { box-shadow: 0 0 0 rgba(91, 141, 255, 0.0); }
        50% { box-shadow: 0 0 18px rgba(91, 141, 255, 0.35); }
        100% { box-shadow: 0 0 0 rgba(91, 141, 255, 0.0); }
      }
      .main-grid {
        display: grid;
        grid-template-columns: minmax(300px, 360px) minmax(0, 1fr);
        gap: 20px;
        align-items: start;
      }
      .sidebar { display: flex; flex-direction: column; gap: 18px; }
      .content { display: flex; flex-direction: column; gap: 18px; min-width: 0; }
      .panel {
        position: relative;
        padding: 22px;
        border-radius: var(--radius-md);
        background: var(--bg-surface);
        border: 1px solid var(--border-glass);
        box-shadow: var(--shadow-glass);
        backdrop-filter: blur(18px);
        overflow: hidden;
      }
      .panel::after {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: inherit;
        background: linear-gradient(140deg, rgba(255, 255, 255, 0.12), transparent 35%);
        pointer-events: none;
        opacity: 0.6;
      }
      .panel > * { position: relative; z-index: 1; }
      .panel-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 16px;
      }
      h2 { margin: 0; font-size: 1.4rem; font-weight: 600; }
      .muted { color: var(--text-secondary); font-size: 0.85rem; margin: 0; }
      .label {
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-secondary);
      }
      label.label { display: block; }
      .value { font-size: 0.95rem; font-weight: 600; }
      .stats { display: grid; gap: 14px; margin-bottom: 16px; }
      .button-row { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 14px; }
      .btn {
        border: 1px solid var(--border-glass);
        background: rgba(255, 255, 255, 0.04);
        color: var(--text-primary);
        padding: 10px 16px;
        border-radius: var(--radius-sm);
        font-weight: 600;
        cursor: pointer;
        transition: var(--transition);
      }
      .btn:hover { transform: translateY(-2px); box-shadow: 0 12px 24px rgba(8, 12, 24, 0.35); }
      .btn.primary {
        border: none;
        background: linear-gradient(135deg, #5b8dff, #7a6bff);
        color: #f8fbff;
        box-shadow: 0 12px 30px rgba(91, 141, 255, 0.35);
      }
      .btn.secondary { background: rgba(91, 141, 255, 0.12); }
      .btn.ghost { background: transparent; }
      .segmented {
        display: inline-flex;
        gap: 6px;
        padding: 6px;
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid var(--border-glass);
        margin-top: 8px;
      }
      .segmented input { display: none; }
      .segmented label {
        padding: 8px 14px;
        border-radius: 999px;
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text-secondary);
        cursor: pointer;
      }
      .segmented input:checked + label {
        background: rgba(91, 141, 255, 0.2);
        color: var(--text-primary);
        box-shadow: inset 0 0 0 1px rgba(91, 141, 255, 0.6);
      }
      .badge-row { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; }
      .badge, .chip {
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        border: 1px solid var(--border-glass);
        color: var(--text-secondary);
        background: rgba(91, 141, 255, 0.08);
      }
      .badge.good { color: #e9fff4; background: rgba(75, 222, 151, 0.25); border-color: rgba(75, 222, 151, 0.45); }
      .badge.warn { color: #fff2cf; background: rgba(255, 185, 77, 0.25); border-color: rgba(255, 185, 77, 0.45); }
      .badge.error { color: #ffe1e1; background: rgba(255, 107, 107, 0.2); border-color: rgba(255, 107, 107, 0.5); }
      .notice {
        margin-top: 12px;
        padding: 10px 12px;
        border-radius: var(--radius-sm);
        border: 1px solid rgba(255, 185, 77, 0.4);
        background: rgba(255, 185, 77, 0.16);
        font-size: 0.78rem;
        color: #fff0c8;
      }
      .alert {
        margin-top: 12px;
        padding: 12px 14px;
        border-radius: var(--radius-sm);
        border: 1px solid rgba(255, 107, 107, 0.4);
        background: rgba(255, 107, 107, 0.18);
        color: #ffe3e3;
        display: none;
      }
      .alert.show { display: block; }
      .divider { height: 1px; background: rgba(91, 141, 255, 0.2); margin: 14px 0; }
      select, input {
        width: 100%;
        margin-top: 6px;
        padding: 10px 12px;
        border-radius: var(--radius-sm);
        border: 1px solid rgba(255,255,255,0.1);
        background: rgba(255,255,255,0.08);
        color: var(--text-primary);
      }
      .input-row { display: grid; grid-template-columns: 1fr auto; gap: 10px; align-items: end; }
      .input-row.wide { grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); margin-top: 12px; }
      .field { display: flex; flex-direction: column; gap: 6px; }
      .meter {
        margin-top: 14px;
        height: 10px;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        overflow: hidden;
      }
      .meter-bar {
        height: 100%;
        width: 8%;
        background: linear-gradient(90deg, #5b8dff, #4bde97);
        transition: width 120ms ease-in-out;
      }
      .meter-meta { display: flex; align-items: center; justify-content: space-between; margin-top: 8px; }
      .inline { font-size: 0.8rem; color: var(--text-secondary); }
      .controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
      .controls input { width: 180px; }
      .toggle { display: inline-flex; align-items: center; gap: 6px; font-size: 0.78rem; color: var(--text-secondary); }
      .transcript-list {
        margin-top: 12px;
        max-height: min(560px, calc(100vh - 360px));
        overflow: auto;
        display: grid;
        gap: 10px;
        scrollbar-gutter: stable;
        overscroll-behavior: contain;
      }
      .jump-latest { display: none; }
      .jump-latest.show { display: inline-flex; }
      .chip.inline { display: inline-flex; align-items: center; gap: 8px; }
      .transcript-item {
        padding: 10px 12px;
        border-radius: var(--radius-sm);
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(7, 12, 24, 0.45);
      }
      .transcript-meta {
        display: flex;
        justify-content: space-between;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-secondary);
      }
      .transcript-text { margin-top: 6px; font-size: 0.92rem; line-height: 1.4; }
      .transcript-text { word-break: break-word; }
      .transcript-item.low { border-color: rgba(255, 185, 77, 0.45); }
      .export-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 10px;
        margin-top: 10px;
      }
      .link { color: var(--accent); font-weight: 600; }
      .progress {
        height: 6px;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        margin-top: 10px;
        overflow: hidden;
        display: none;
      }
      .progress.active { display: block; }
      .progress::after {
        content: "";
        display: block;
        height: 100%;
        width: 40%;
        background: linear-gradient(90deg, rgba(91, 141, 255, 0.2), rgba(91, 141, 255, 0.9));
        animation: slide 1.1s ease-in-out infinite;
      }
      @keyframes slide {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(250%); }
      }
      .error-inline { color: #ffc5c5; font-size: 0.8rem; margin-top: 8px; }
      .export-list { margin-top: 14px; display: grid; gap: 8px; font-size: 0.8rem; color: var(--text-secondary); }
      .export-row { display: flex; justify-content: space-between; }
      @media (max-width: 900px) {
        .top-nav { grid-template-columns: 1fr; justify-items: start; }
        .main-grid { grid-template-columns: 1fr; }
        .status-pill { justify-self: start; }
      }
    </style>
  </head>
  <body data-theme="dark">
    <div class="page">
      <header class="top-nav">
        <div>
          <div class="brand-title">Atlas Monitor</div>
          <div class="brand-subtitle">Overview • Offline recorder • diarized transcription • localhost only</div>
        </div>
        <nav class="nav-links">
          <a class="nav-link active" href="/">Overview</a>
          <a class="nav-link" href="/setup">Setup</a>
          <a class="nav-link" href="/mic">Mic</a>
          <a class="nav-link" href="/admin">Admin</a>
        </nav>
        <div class="status-pill idle" id="statusPill">Idle</div>
      </header>

      <div class="main-grid">
        <aside class="sidebar">
          <section class="panel">
            <div class="panel-head">
              <h2>Capture</h2>
              <div class="chip" id="scheduleState">--</div>
            </div>
            <div class="stats">
              <div>
                <div class="label">Recording since</div>
                <div class="value" id="recordingSince">--</div>
              </div>
              <div>
                <div class="label">Segment ends in</div>
                <div class="value" id="segmentCountdown">--</div>
              </div>
            </div>
            <div class="button-row">
              <button class="btn primary" onclick="startRecording()">Start</button>
              <button class="btn ghost" onclick="post('/record/stop')">Stop</button>
              <button class="btn secondary" id="resumeScheduleBtn" onclick="post('/record/resume')">Resume schedule</button>
            </div>
            <div class="divider"></div>
            <label class="label">Speaker mode</label>
            <div class="segmented" role="radiogroup" aria-label="Speaker lock">
              <input id="lock-auto" type="radio" name="speakerLock" value="auto" checked>
              <label for="lock-auto">Auto</label>
              <input id="lock-hugo" type="radio" name="speakerLock" value="hugo">
              <label for="lock-hugo">Hugo only</label>
              <input id="lock-leon" type="radio" name="speakerLock" value="leon">
              <label for="lock-leon">Leon only</label>
            </div>
            <div class="badge-row">
              <div class="badge" id="badgeHugo">Hugo: --</div>
              <div class="badge" id="badgeLeon">Leon: --</div>
            </div>
            <p class="muted" style="margin-top: 12px;">Manage enrollment references in <a class="link" href="/setup">Setup</a>.</p>
            <div class="notice" id="enrollHint" style="display:none;"></div>
            <div class="alert" id="error"></div>
          </section>

          <section class="panel">
            <div class="panel-head">
              <h2>Scheduler</h2>
              <div class="chip" id="workHoursBadge">--</div>
            </div>
            <p class="muted">Work hours auto-start based on your configured schedule.</p>
            <div class="badge-row">
              <div class="badge" id="manualBadge">Manual: --</div>
              <div class="badge" id="timezoneBadge">TZ: --</div>
            </div>
          </section>
        </aside>

        <section class="content">
          <section class="panel">
            <div class="panel-head">
              <h2>Transcript preview</h2>
              <div class="controls">
                <label class="label">Window</label>
                <select id="previewMinutes">
                  <option value="15">15m</option>
                  <option value="30" selected>30m</option>
                  <option value="60">60m</option>
                </select>
                <input id="searchInput" placeholder="Search transcripts"/>
                <label class="toggle">
                  <input type="checkbox" id="hideUnknown" checked/>
                  <span>Hide Unknown</span>
                </label>
                <label class="toggle">
                  <input type="checkbox" id="hideLow" checked/>
                  <span>Hide low confidence</span>
                </label>
                <button class="btn ghost jump-latest" id="jumpLatest" onclick="jumpToLatest()">Jump to latest</button>
                <div class="chip inline" id="transcriptStatus" title="Live transcript refresh">Live</div>
              </div>
            </div>
            <div class="transcript-list" id="transcriptList"></div>
          </section>

          <section class="panel">
            <div class="panel-head">
              <h2>Exports</h2>
              <div class="chip">Timezone: <span id="tzLabel">--</span></div>
            </div>
            <div class="export-grid">
              <button class="btn" onclick="exportRange('30m')">Last 30m</button>
              <button class="btn" onclick="exportRange('60m')">Last 60m</button>
              <button class="btn" onclick="exportRange('today')">Today</button>
              <button class="btn" onclick="exportRange('session')">Session</button>
            </div>
            <div class="input-row wide">
              <div class="field">
                <label class="label" for="customStart">Custom start</label>
                <input id="customStart" type="datetime-local"/>
              </div>
              <div class="field">
                <label class="label" for="customEnd">Custom end</label>
                <input id="customEnd" type="datetime-local"/>
              </div>
            </div>
            <div class="button-row">
              <button class="btn primary" onclick="exportCustom()">Export custom</button>
              <a class="link" id="downloadLink" href="#">Download latest export</a>
            </div>
            <div class="progress" id="exportProgress"></div>
            <div class="error-inline" id="exportError"></div>
            <div class="export-list" id="exportList"></div>
          </section>
        </section>
      </div>
    </div>
    <script>
      const escapeHtml = (value) => String(value ?? "").replace(/[&<>\"]/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}[c]));
      const resumeBtn = document.getElementById('resumeScheduleBtn');
      const errorEl = document.getElementById('error');

      function showPageError(msg) {
        if (!msg) {
          errorEl.textContent = "";
          errorEl.classList.remove("show");
          return;
        }
        errorEl.textContent = msg;
        errorEl.classList.add("show");
      }

      async function fetchJson(url, options) {
        const response = await fetch(url, options);
        let data = {};
        try {
          data = await response.json();
        } catch (_) {
          data = {};
        }
        if (!response.ok) {
          throw new Error(data.detail || `Request failed (${response.status})`);
        }
        return data;
      }

      function selectedLock() {
        const selected = document.querySelector('input[name="speakerLock"]:checked');
        return selected ? selected.value : 'auto';
      }

      function formatDuration(seconds) {
        const s = Math.max(0, Math.floor(seconds));
        const h = Math.floor(s / 3600);
        const m = Math.floor((s % 3600) / 60);
        const sec = s % 60;
        if (h > 0) return `${h}h ${m}m`;
        return `${m}m ${sec}s`;
      }

      function formatSince(iso) {
        if (!iso) return '--';
        const dt = new Date(iso);
        if (Number.isNaN(dt.getTime())) return '--';
        const diff = (Date.now() - dt.getTime()) / 1000;
        return `${dt.toLocaleTimeString()} • ${formatDuration(diff)}`;
      }

      function formatCountdown(iso, segmentSeconds) {
        if (!iso || !segmentSeconds) return '--';
        const dt = new Date(iso);
        if (Number.isNaN(dt.getTime())) return '--';
        const elapsed = (Date.now() - dt.getTime()) / 1000;
        const remaining = Math.max(0, segmentSeconds - elapsed);
        return `${Math.ceil(remaining)}s`;
      }

      async function refresh() {
        try {
          const data = await fetchJson('/status');
          const pill = document.getElementById('statusPill');
          pill.textContent = data.status === 'recording' ? 'Recording' : 'Idle';
          pill.className = 'status-pill ' + (data.status === 'recording' ? 'recording' : 'idle');
          if (data.backend_error) {
            showPageError('ASR error: ' + data.backend_error);
          } else {
            showPageError('');
          }
          const hugoOk = data.enrolled && data.enrolled.Hugo;
          const leonOk = data.enrolled && data.enrolled.Leon;
          updateBadge('badgeHugo', 'Hugo', hugoOk);
          updateBadge('badgeLeon', 'Leon', leonOk);
          const hint = document.getElementById('enrollHint');
          if (data.speaker_lock === 'auto' && !data.auto_ready) {
            hint.style.display = 'block';
            hint.textContent = 'Enroll required for Auto mode. Please enroll both speakers.';
          } else {
            hint.style.display = 'none';
            hint.textContent = '';
          }
          if (data.speaker_lock) {
            const target = document.querySelector(`input[name="speakerLock"][value="${data.speaker_lock}"]`);
            if (target) {
              target.checked = true;
            }
          }
          document.getElementById('recordingSince').textContent = formatSince(data.recording_since);
          document.getElementById('segmentCountdown').textContent = formatCountdown(data.last_segment_start, data.segment_seconds);
          const scheduleState = document.getElementById('scheduleState');
          if (!data.work_hours_enabled) {
            scheduleState.textContent = 'Schedule off';
          } else if (data.manual_override === null) {
            scheduleState.textContent = data.in_work_hours ? 'Scheduled: On' : 'Scheduled: Off';
          } else {
            scheduleState.textContent = 'Manual override';
          }
          resumeBtn.style.display = data.manual_override === null ? 'none' : 'inline-flex';
          document.getElementById('timezoneBadge').textContent = `TZ: ${data.timezone}`;
          document.getElementById('tzLabel').textContent = data.timezone;
          document.getElementById('workHoursBadge').textContent = data.in_work_hours ? 'In hours' : 'Out of hours';
          document.getElementById('manualBadge').textContent = data.manual_override === null ? 'Manual: Auto' : 'Manual: Override';
        } catch (err) {
          showPageError(err.message || 'Failed to load status.');
        }
      }

      function updateBadge(id, name, ok) {
        const el = document.getElementById(id);
        if (!el) return;
        el.textContent = name + ': ' + (ok ? 'Enrolled' : 'Not enrolled');
        el.className = 'badge ' + (ok ? 'good' : 'warn');
      }

      async function post(url, body) {
        await fetchJson(url, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(body || {}),
        });
        await refresh();
      }

      async function startRecording() {
        try {
          await post('/record/start', {lock_mode: selectedLock()});
        } catch (err) {
          showPageError(err.message || 'Failed to start recording.');
        }
      }

      async function setLockMode(mode) {
        try {
          await post('/record/start', {lock_mode: mode, set_only: true});
        } catch (err) {
          showPageError(err.message || 'Failed to set speaker mode.');
        }
      }

      function showProgress(show) {
        const bar = document.getElementById('exportProgress');
        bar.classList.toggle('active', show);
      }

      async function exportRange(range) {
        const exportError = document.getElementById('exportError');
        exportError.textContent = '';
        showProgress(true);
        try {
          const data = await fetchJson('/export', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({range}),
          });
          document.getElementById('downloadLink').href = '/download/' + data.id;
          await loadRecentExports();
        } catch (err) {
          exportError.textContent = err.message || 'Export failed.';
        } finally {
          showProgress(false);
        }
      }

      async function exportCustom() {
        const exportError = document.getElementById('exportError');
        exportError.textContent = '';
        const start = document.getElementById('customStart').value;
        const end = document.getElementById('customEnd').value;
        showProgress(true);
        try {
          const data = await fetchJson('/export', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({range: 'custom', start, end}),
          });
          document.getElementById('downloadLink').href = '/download/' + data.id;
          await loadRecentExports();
        } catch (err) {
          exportError.textContent = err.message || 'Export failed.';
        } finally {
          showProgress(false);
        }
      }

      function formatBytes(bytes) {
        if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
      }

      async function loadRecentExports() {
        const list = document.getElementById('exportList');
        list.innerHTML = '<div class="inline">Loading exports...</div>';
        try {
          const data = await fetchJson('/exports/recent');
          list.innerHTML = '';
          for (const row of data.rows || []) {
            const time = new Date(row.created_ts).toLocaleString();
            const size = formatBytes(row.size_bytes || 0);
            const div = document.createElement('div');
            div.className = 'export-row';
            div.innerHTML = `<span>${time} • ${row.range_label}</span><span>${size}</span>`;
            list.appendChild(div);
          }
          if (!(data.rows || []).length) {
            list.innerHTML = '<div class="inline">No exports yet.</div>';
          }
        } catch (err) {
          list.innerHTML = `<div class="inline">Failed to load exports: ${escapeHtml(err.message || 'Unknown error')}</div>`;
        }
      }

      async function loadTranscripts() {
        const list = document.getElementById('transcriptList');
        const jumpBtn = document.getElementById('jumpLatest');
        const statusChip = document.getElementById('transcriptStatus');
        if (!window.__atlasTranscript) {
          window.__atlasTranscript = {
            inFlight: false,
            lastQueryKey: '',
            lastSig: '',
            hasNewWhileAway: false,
            lastOkAt: 0,
          };
          // Clear "new updates" indicator when user returns to the bottom.
          list.addEventListener('scroll', () => {
            const nearBottom = (list.scrollTop + list.clientHeight) >= (list.scrollHeight - 24);
            if (nearBottom && window.__atlasTranscript.hasNewWhileAway) {
              window.__atlasTranscript.hasNewWhileAway = false;
              jumpBtn.classList.remove('show');
              jumpBtn.textContent = 'Jump to latest';
            }
          }, {passive: true});
        }
        const tState = window.__atlasTranscript;
        if (tState.inFlight) return;
        tState.inFlight = true;
        const wasNearBottom = (list.scrollTop + list.clientHeight) >= (list.scrollHeight - 24);
        const prevScrollTop = list.scrollTop;
        const minutes = document.getElementById('previewMinutes').value;
        const search = document.getElementById('searchInput').value;
        const includeUnknown = !document.getElementById('hideUnknown').checked;
        const includeLow = !document.getElementById('hideLow').checked;
        const params = new URLSearchParams({
          minutes,
          search,
          include_unknown: includeUnknown,
          include_low_confidence: includeLow,
        });
        const queryKey = params.toString();
        const isNewQuery = queryKey !== tState.lastQueryKey;
        tState.lastQueryKey = queryKey;
        if (!list.dataset.loaded) {
          list.innerHTML = '<div class="inline">Loading transcripts...</div>';
        }
        statusChip.textContent = 'Live';
        try {
          const data = await fetchJson('/transcripts/recent?' + params.toString());
          const rows = data.rows || [];
          const last = rows.length ? rows[rows.length - 1] : null;
          const lastSig = last ? `${last.start_ts}|${last.speaker}|${(last.text || '').slice(0, 64)}` : '';
          const gotNew = Boolean(tState.lastSig && lastSig && lastSig !== tState.lastSig);
          list.innerHTML = '';
          for (const row of rows) {
            const item = document.createElement('div');
            item.className = 'transcript-item' + (row.low_confidence ? ' low' : '');
            const time = new Date(row.start_ts).toLocaleTimeString();
            const badge = row.low_confidence ? ' • low' : '';
            item.innerHTML = `
              <div class="transcript-meta">
                <span>${escapeHtml(row.speaker)}${badge}</span>
                <span>${time}</span>
              </div>
              <div class="transcript-text">${escapeHtml(row.text)}</div>
            `;
            list.appendChild(item);
          }
          if (!rows.length) {
            list.innerHTML = '<div class="inline">No transcripts in this window.</div>';
          }
          list.dataset.loaded = '1';
          tState.lastSig = lastSig;

          if (isNewQuery) {
            tState.hasNewWhileAway = false;
            jumpBtn.classList.remove('show');
            jumpBtn.textContent = 'Jump to latest';
            // Default behavior: show the most recent content for the standard live view.
            if ((search || '').trim()) {
              list.scrollTop = 0;
            } else {
              list.scrollTop = list.scrollHeight;
            }
          } else if (wasNearBottom) {
            list.scrollTop = list.scrollHeight;
          } else {
            // Keep the user's reading position steady when they're scrolled up.
            list.scrollTop = prevScrollTop;
            if (gotNew) {
              tState.hasNewWhileAway = true;
              jumpBtn.classList.add('show');
              jumpBtn.textContent = 'New updates • Jump to latest';
            }
          }

          tState.lastOkAt = Date.now();
          statusChip.textContent = 'Live';
        } catch (err) {
          if (!list.dataset.loaded) {
            list.innerHTML = `<div class="inline">Failed to load transcripts: ${escapeHtml(err.message || 'Unknown error')}</div>`;
          }
          statusChip.textContent = 'Live (error)';
        }
        finally {
          tState.inFlight = false;
        }
      }

      function jumpToLatest() {
        const list = document.getElementById('transcriptList');
        const jumpBtn = document.getElementById('jumpLatest');
        const tState = window.__atlasTranscript;
        list.scrollTop = list.scrollHeight;
        if (tState) tState.hasNewWhileAway = false;
        jumpBtn.classList.remove('show');
        jumpBtn.textContent = 'Jump to latest';
      }

      document.querySelectorAll('input[name="speakerLock"]').forEach((el) => {
        el.addEventListener('change', () => setLockMode(el.value));
      });
      document.getElementById('previewMinutes').addEventListener('change', loadTranscripts);
      document.getElementById('searchInput').addEventListener('input', loadTranscripts);
      document.getElementById('hideUnknown').addEventListener('change', loadTranscripts);
      document.getElementById('hideLow').addEventListener('change', loadTranscripts);

      refresh();
      loadRecentExports();
      loadTranscripts();
      const refreshTimer = setInterval(refresh, 3000);
      const transcriptTimer = setInterval(loadTranscripts, 5000);
      const exportTimer = setInterval(loadRecentExports, 12000);
      window.addEventListener('beforeunload', () => {
        clearInterval(refreshTimer);
        clearInterval(transcriptTimer);
        clearInterval(exportTimer);
      });
    </script>
  </body>
</html>
"""

_MIC_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <title>Mic Settings • Atlas Monitor</title>
    <style>
      :root {
        --bg-primary: radial-gradient(circle at 20% 20%, #0f172a, #030712);
        --bg-surface: rgba(15, 23, 42, 0.6);
        --border-glass: rgba(56, 107, 255, 0.35);
        --text-primary: #e2e8f0;
        --text-secondary: rgba(226, 232, 240, 0.7);
        --radius-sm: 8px;
        --radius-md: 16px;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        min-height: 100vh;
        font-family: "Inter", "Segoe UI", sans-serif;
        color: var(--text-primary);
        background: var(--bg-primary);
        overflow-x: hidden;
      }
      .page { max-width: 1200px; margin: 0 auto; padding: 28px 24px 56px; }
      .top-nav {
        display: grid;
        grid-template-columns: 1fr auto;
        align-items: center;
        gap: 20px;
        margin-bottom: 24px;
      }
      .brand-title {
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }
      .brand-subtitle { color: var(--text-secondary); font-size: 0.85rem; margin-top: 6px; }
      .nav-links { display: flex; gap: 16px; }
      .nav-link {
        padding: 8px 14px;
        border-radius: 999px;
        color: var(--text-secondary);
        border: 1px solid transparent;
        transition: 180ms ease-in-out;
        text-decoration: none;
      }
      .nav-link.active, .nav-link:hover {
        color: var(--text-primary);
        border-color: var(--border-glass);
        background: rgba(91, 141, 255, 0.12);
      }
      .panel {
        position: relative;
        padding: 22px;
        border-radius: var(--radius-md);
        background: var(--bg-surface);
        border: 1px solid var(--border-glass);
        box-shadow: 0 30px 80px rgba(8, 12, 24, 0.7);
        backdrop-filter: blur(18px);
        overflow: hidden;
      }
      .panel::after {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: inherit;
        background: linear-gradient(140deg, rgba(255, 255, 255, 0.12), transparent 35%);
        pointer-events: none;
        opacity: 0.6;
      }
      .panel > * { position: relative; z-index: 1; }
      .panel-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
      h1, h2 { margin: 0; }
      h1 { font-size: 1.4rem; }
      .muted { color: var(--text-secondary); font-size: 0.85rem; margin: 0; }
      .input-row { display: grid; grid-template-columns: minmax(260px, 1fr) auto; gap: 10px; align-items: end; margin-top: 14px; }
      select {
        width: 100%;
        padding: 10px 12px;
        border-radius: var(--radius-sm);
        border: 1px solid rgba(255,255,255,0.1);
        background: rgba(255,255,255,0.08);
        color: var(--text-primary);
      }
      .btn {
        border: 1px solid var(--border-glass);
        background: rgba(255,255,255,0.04);
        color: var(--text-primary);
        padding: 10px 16px;
        border-radius: var(--radius-sm);
        font-weight: 600;
        cursor: pointer;
      }
      .btn.ghost { background: transparent; }
      .btn.primary { border: none; background: linear-gradient(135deg, #5b8dff, #7a6bff); }
      .button-row { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 14px; align-items: center; }
      .meter { margin-top: 14px; height: 10px; border-radius: 999px; background: rgba(255,255,255,0.08); overflow: hidden; }
      .meter-bar { height: 100%; width: 8%; background: linear-gradient(90deg, #5b8dff, #4bde97); transition: width 120ms ease-in-out; }
      .alert {
        margin-top: 12px;
        padding: 12px;
        border-radius: var(--radius-sm);
        border: 1px solid rgba(255, 107, 107, 0.4);
        background: rgba(255, 107, 107, 0.18);
        color: #ffe3e3;
        display: none;
      }
      .alert.show { display: block; }
      .chip {
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        border: 1px solid var(--border-glass);
        color: var(--text-secondary);
        background: rgba(91, 141, 255, 0.1);
      }
      @media (max-width: 900px) {
        .top-nav { grid-template-columns: 1fr; }
        .input-row { grid-template-columns: 1fr; }
      }
    </style>
  </head>
  <body data-theme="dark">
    <div class="page">
      <header class="top-nav">
        <div>
          <div class="brand-title">Atlas Monitor</div>
          <div class="brand-subtitle">Mic settings</div>
        </div>
        <nav class="nav-links">
          <a class="nav-link" href="/">Overview</a>
          <a class="nav-link" href="/setup">Setup</a>
          <a class="nav-link active" href="/mic">Mic</a>
          <a class="nav-link" href="/admin">Admin</a>
        </nav>
      </header>

      <section class="panel">
        <div class="panel-head">
          <h1>Microphone Settings</h1>
          <div class="chip" id="deviceStatus">Loading</div>
        </div>
        <p class="muted">Use this page for device selection and one-off signal checks. Live meter is opt-in.</p>
        <div class="input-row">
          <select id="deviceSelect"></select>
          <button class="btn primary" onclick="saveDevice()">Save device</button>
        </div>
        <div class="meter">
          <div class="meter-bar" id="levelBar"></div>
        </div>
        <div class="button-row">
          <button class="btn ghost" id="toggleMeter" onclick="toggleMeter()">Enable live meter</button>
          <button class="btn" onclick="testMic()">Test 3s</button>
          <div class="muted" id="levelText">Level idle</div>
          <div class="muted" id="testResult"></div>
        </div>
        <div class="alert" id="error"></div>
      </section>
    </div>
    <script>
      const errorEl = document.getElementById("error");
      let meterTimer = null;

      function showError(msg) {
        if (!msg) {
          errorEl.textContent = "";
          errorEl.classList.remove("show");
          return;
        }
        errorEl.textContent = msg;
        errorEl.classList.add("show");
      }

      async function fetchJson(url, options) {
        const response = await fetch(url, options);
        let data = {};
        try {
          data = await response.json();
        } catch (_) {
          data = {};
        }
        if (!response.ok) {
          throw new Error(data.detail || `Request failed (${response.status})`);
        }
        return data;
      }

      async function loadDevices() {
        const select = document.getElementById("deviceSelect");
        select.innerHTML = '<option>Loading devices...</option>';
        try {
          const data = await fetchJson('/audio/devices');
          select.innerHTML = '';
          const autoOption = document.createElement('option');
          autoOption.value = 'auto';
          autoOption.textContent = 'Auto (first available)';
          select.appendChild(autoOption);
          for (const dev of data.devices || []) {
            const option = document.createElement('option');
            option.value = dev.name;
            option.textContent = `${dev.name} • ${dev.hostapi}`;
            select.appendChild(option);
          }
          select.value = data.selected || 'auto';
          document.getElementById('deviceStatus').textContent = data.selected || 'Auto';
          showError("");
        } catch (err) {
          document.getElementById('deviceStatus').textContent = 'Unavailable';
          select.innerHTML = '<option>No devices available</option>';
          showError(err.message || 'Failed to load devices.');
        }
      }

      async function saveDevice() {
        const value = document.getElementById('deviceSelect').value;
        try {
          await fetchJson('/audio/device', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({name: value}),
          });
          document.getElementById('deviceStatus').textContent = value === 'auto' ? 'Auto' : value;
          showError("");
        } catch (err) {
          showError(err.message || 'Failed to save device.');
        }
      }

      async function pollLevel() {
        try {
          const data = await fetchJson('/audio/level');
          const bar = document.getElementById('levelBar');
          const text = document.getElementById('levelText');
          if (data.busy) {
            bar.style.width = '5%';
            text.textContent = 'Paused while recorder is active';
            return;
          }
          const rms = Math.min(1, data.rms || 0);
          bar.style.width = Math.min(100, Math.round(rms * 140)) + '%';
          text.textContent = `RMS ${(rms * 100).toFixed(1)}% • SNR ${data.snr_db} dB • Noise ${data.noise_floor} dBFS`;
        } catch (err) {
          showError(err.message || 'Failed to read audio level.');
        }
      }

      function toggleMeter() {
        const btn = document.getElementById('toggleMeter');
        if (meterTimer) {
          clearInterval(meterTimer);
          meterTimer = null;
          btn.textContent = 'Enable live meter';
          document.getElementById('levelText').textContent = 'Level idle';
          document.getElementById('levelBar').style.width = '8%';
          return;
        }
        pollLevel();
        meterTimer = setInterval(pollLevel, 1400);
        btn.textContent = 'Stop meter';
      }

      async function testMic() {
        const output = document.getElementById('testResult');
        output.textContent = 'Testing...';
        try {
          const data = await fetchJson('/audio/test', {method: 'POST'});
          output.textContent = data.ok ? `SNR ${data.snr_db} dB` : 'Test failed (device busy)';
          showError("");
        } catch (err) {
          output.textContent = 'Test failed';
          showError(err.message || 'Failed to run mic test.');
        }
      }

      loadDevices();
      window.addEventListener('beforeunload', () => {
        if (meterTimer) {
          clearInterval(meterTimer);
        }
      });
    </script>
  </body>
</html>
"""

_SETUP_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <title>Setup • Atlas Monitor</title>
    <style>
      :root {
        --bg-primary: radial-gradient(circle at 20% 20%, #0f172a, #030712);
        --bg-surface: rgba(15, 23, 42, 0.6);
        --bg-surface-strong: rgba(15, 23, 42, 0.8);
        --border-glass: rgba(56, 107, 255, 0.35);
        --text-primary: #e2e8f0;
        --text-secondary: rgba(226, 232, 240, 0.7);
        --accent: #5b8dff;
        --success: #4bde97;
        --warn: #ffb94d;
        --error: #ff6b6b;
        --radius-sm: 8px;
        --radius-md: 16px;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        min-height: 100vh;
        font-family: "Inter", "Segoe UI", sans-serif;
        color: var(--text-primary);
        background: var(--bg-primary);
        overflow-x: hidden;
      }
      a { color: inherit; text-decoration: none; }
      .page { max-width: 1400px; margin: 0 auto; padding: 28px 24px 56px; }
      .top-nav {
        display: grid;
        grid-template-columns: 1fr auto;
        align-items: center;
        gap: 20px;
        margin-bottom: 24px;
      }
      .brand-title {
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }
      .brand-subtitle { color: var(--text-secondary); font-size: 0.85rem; margin-top: 6px; }
      .nav-links { display: flex; gap: 16px; }
      .nav-link {
        padding: 8px 14px;
        border-radius: 999px;
        color: var(--text-secondary);
        border: 1px solid transparent;
        transition: 180ms ease-in-out;
      }
      .nav-link.active, .nav-link:hover {
        color: var(--text-primary);
        border-color: var(--border-glass);
        background: rgba(91, 141, 255, 0.12);
      }
      .panel {
        position: relative;
        padding: 22px;
        border-radius: var(--radius-md);
        background: var(--bg-surface);
        border: 1px solid var(--border-glass);
        box-shadow: 0 30px 80px rgba(8, 12, 24, 0.7);
        backdrop-filter: blur(18px);
        margin-bottom: 18px;
      }
      .panel::after {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: inherit;
        background: linear-gradient(140deg, rgba(255, 255, 255, 0.12), transparent 35%);
        pointer-events: none;
        opacity: 0.6;
      }
      .panel > * { position: relative; z-index: 1; }
      .panel-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
      h1 { margin: 0; font-size: 1.6rem; }
      h2 { margin: 0; font-size: 1.2rem; }
      .muted { color: var(--text-secondary); font-size: 0.85rem; }
      .badge {
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        border: 1px solid var(--border-glass);
        color: var(--text-secondary);
        background: rgba(91, 141, 255, 0.08);
      }
      .badge.good { color: #eafff5; background: rgba(75, 222, 151, 0.25); border-color: rgba(75, 222, 151, 0.45); }
      .badge.warn { color: #fff2cf; background: rgba(255, 185, 77, 0.25); border-color: rgba(255, 185, 77, 0.45); }
      .btn {
        border: 1px solid var(--border-glass);
        background: rgba(255,255,255,0.04);
        color: var(--text-primary);
        padding: 10px 16px;
        border-radius: var(--radius-sm);
        font-weight: 600;
        cursor: pointer;
        transition: 180ms ease-in-out;
      }
      .btn.primary {
        border: none;
        background: linear-gradient(135deg, #5b8dff, #7a6bff);
      }
      .btn.ghost { background: transparent; }
      .btn:disabled {
        opacity: 0.45;
        cursor: not-allowed;
      }
      .button-row { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 14px; }
      .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }
      .meter { margin-top: 14px; height: 10px; border-radius: 999px; background: rgba(255,255,255,0.08); overflow: hidden; }
      .meter-bar { height: 100%; width: 10%; background: linear-gradient(90deg, #5b8dff, #4bde97); transition: width 120ms ease-in-out; }
      .result { margin-top: 10px; font-size: 0.82rem; color: var(--text-secondary); }
      .result.warn { color: #fff2cf; }
      @media (max-width: 900px) {
        .top-nav { grid-template-columns: 1fr; }
      }
    </style>
  </head>
  <body data-theme="dark">
    <div class="page">
      <header class="top-nav">
        <div>
          <div class="brand-title">Atlas Monitor</div>
          <div class="brand-subtitle">Speaker enrollment & reference lock</div>
        </div>
        <nav class="nav-links">
          <a class="nav-link" href="/">Overview</a>
          <a class="nav-link active" href="/setup">Setup</a>
          <a class="nav-link" href="/mic">Mic</a>
          <a class="nav-link" href="/admin">Admin</a>
        </nav>
      </header>

      <section class="panel">
        <div class="panel-head">
          <h1>Reference lock</h1>
          <div class="badge" id="refState">--</div>
        </div>
        <p class="muted">Reroll clears both embeddings and unlocks re-enrollment. Lock reference to prevent changes.</p>
        <div class="button-row">
          <button class="btn" onclick="rerollReferences()">Reroll references</button>
          <button class="btn ghost" id="lockBtn" onclick="lockReferences()">Lock reference</button>
          <button class="btn ghost" id="stopEnrollBtn" style="display:none;" onclick="stopEnrollment()">STOP RECORDING</button>
        </div>
        <div class="meter">
          <div class="meter-bar" id="levelBar"></div>
        </div>
        <div class="button-row">
          <button class="btn ghost" id="toggleMeter" onclick="toggleMeter()">Enable live meter</button>
          <div class="muted" id="levelText">Live level --</div>
          <div class="muted" id="enrollTimer">Enrollment timer --</div>
        </div>
      </section>

      <section class="grid">
        <div class="panel">
          <div class="panel-head">
            <h2>Hugo</h2>
            <div class="badge" id="statusHugo">--</div>
          </div>
          <p class="muted">Speak naturally for 20–40 seconds. Keep a steady distance to the mic.</p>
          <div class="button-row">
            <button class="btn primary" data-enroll onclick="enroll('/setup/enroll/hugo', 'Hugo')">Re-enroll Hugo</button>
            <button class="btn ghost" onclick="clearEnroll('hugo', 'Hugo')">Clear</button>
          </div>
          <div class="result" id="resultHugo"></div>
        </div>
        <div class="panel">
          <div class="panel-head">
            <h2>Leon</h2>
            <div class="badge" id="statusLeon">--</div>
          </div>
          <p class="muted">Speak in your normal cadence for the full duration.</p>
          <div class="button-row">
            <button class="btn primary" data-enroll onclick="enroll('/setup/enroll/leon', 'Leon')">Re-enroll Leon</button>
            <button class="btn ghost" onclick="clearEnroll('leon', 'Leon')">Clear</button>
          </div>
          <div class="result" id="resultLeon"></div>
        </div>
      </section>
    </div>
    <script>
      const setupError = document.createElement('div');
      setupError.className = 'result warn';
      setupError.style.marginTop = '10px';
      document.querySelector('.panel').appendChild(setupError);
      const lockBtn = document.getElementById('lockBtn');
      const stopEnrollBtn = document.getElementById('stopEnrollBtn');
      const enrollTimerText = document.getElementById('enrollTimer');
      const enrollButtons = Array.from(document.querySelectorAll('[data-enroll]'));
      let meterTimer = null;
      let enrollmentTimer = null;
      let activeEnrollment = null;
      let enrollmentSeconds = 30;

      function showSetupError(msg) {
        setupError.textContent = msg || '';
      }

      async function fetchJson(url, options) {
        const response = await fetch(url, options);
        let data = {};
        try {
          data = await response.json();
        } catch (_) {
          data = {};
        }
        if (!response.ok) {
          throw new Error(data.detail || `Request failed (${response.status})`);
        }
        return data;
      }

      function formatClock(totalSeconds) {
        const whole = Math.max(0, Math.floor(totalSeconds));
        const mins = Math.floor(whole / 60).toString().padStart(2, '0');
        const secs = (whole % 60).toString().padStart(2, '0');
        return `${mins}:${secs}`;
      }

      function setBadge(id, ok) {
        const el = document.getElementById(id);
        if (!el) return;
        el.textContent = ok ? 'Enrolled' : 'Not enrolled';
        el.className = 'badge ' + (ok ? 'good' : 'warn');
      }

      function setEnrollUiActive(active) {
        for (const btn of enrollButtons) {
          btn.disabled = active;
        }
        stopEnrollBtn.style.display = active ? 'inline-block' : 'none';
      }

      function updateEnrollTimer() {
        if (!activeEnrollment) {
          enrollTimerText.textContent = `Enrollment timer ${formatClock(enrollmentSeconds)} target`;
          return;
        }
        const elapsed = (Date.now() - activeEnrollment.startedAt) / 1000;
        const remaining = Math.max(0, enrollmentSeconds - elapsed);
        enrollTimerText.textContent = `Recording ${activeEnrollment.name} ${formatClock(elapsed)} / ${formatClock(enrollmentSeconds)} • ${formatClock(remaining)} left`;
      }

      function startEnrollmentTimer(name) {
        activeEnrollment = {name, startedAt: Date.now()};
        updateEnrollTimer();
        if (enrollmentTimer) {
          clearInterval(enrollmentTimer);
        }
        enrollmentTimer = setInterval(updateEnrollTimer, 250);
      }

      function stopEnrollmentTimer() {
        activeEnrollment = null;
        if (enrollmentTimer) {
          clearInterval(enrollmentTimer);
          enrollmentTimer = null;
        }
        updateEnrollTimer();
      }

      async function refreshStatus() {
        try {
          const data = await fetchJson('/status');
          enrollmentSeconds = Math.max(1, Number(data.enrollment_seconds) || enrollmentSeconds);
          setBadge('statusHugo', data.enrolled && data.enrolled.Hugo);
          setBadge('statusLeon', data.enrolled && data.enrolled.Leon);
          const bothEnrolled = Boolean(data.enrolled && data.enrolled.Hugo && data.enrolled.Leon);
          lockBtn.disabled = !bothEnrolled;
          lockBtn.title = bothEnrolled ? '' : 'Enroll both Hugo and Leon before locking.';
          const ref = document.getElementById('refState');
          ref.textContent = data.reference_locked ? 'Locked' : 'Unlocked';
          ref.className = 'badge ' + (data.reference_locked ? 'good' : 'warn');
          showSetupError('');
        } catch (err) {
          showSetupError(err.message || 'Failed to load setup status.');
        }
      }
      async function enroll(url, name) {
        if (activeEnrollment) {
          showSetupError('Another enrollment is already recording.');
          return;
        }
        const resultEl = document.getElementById('result' + name);
        resultEl.textContent = 'Recording...';
        resultEl.className = 'result';
        setEnrollUiActive(true);
        startEnrollmentTimer(name);
        try {
          const data = await fetchJson(url, {method: 'POST'});
          if (data.status === 'ok') {
            const snr = Number.isFinite(data.snr_db) ? data.snr_db.toFixed(1) : '--';
            const duration = Number.isFinite(data.duration_sec) ? data.duration_sec.toFixed(1) : '--';
            resultEl.textContent = 'Done. ' + duration + 's, SNR ' + snr + ' dB.';
            if (data.quality_ok === false) {
              resultEl.textContent += ' Low quality — please re-enroll.';
              resultEl.className = 'result warn';
            }
          } else if (data.status === 'cancelled') {
            const duration = Number.isFinite(data.duration_sec) ? data.duration_sec.toFixed(1) : '--';
            resultEl.textContent = `Stopped at ${duration}s. Enrollment not saved.`;
            resultEl.className = 'result warn';
          } else {
            resultEl.textContent = data.detail || 'Error.';
          }
          showSetupError('');
        } catch (err) {
          resultEl.textContent = err.message || 'Enrollment failed.';
          resultEl.className = 'result warn';
          showSetupError(err.message || 'Enrollment failed.');
        } finally {
          stopEnrollmentTimer();
          setEnrollUiActive(false);
        }
        await refreshStatus();
      }

      async function stopEnrollment() {
        if (!activeEnrollment) {
          return;
        }
        try {
          const data = await fetchJson('/setup/enroll/stop', {method: 'POST'});
          if (data.status === 'stopping') {
            showSetupError(`Stopping ${data.speaker || 'enrollment'}...`);
          }
        } catch (err) {
          showSetupError(err.message || 'Failed to stop enrollment.');
        }
      }

      async function clearEnroll(speaker, name) {
        try {
          await fetchJson(`/setup/clear/${speaker}`, {method: 'POST'});
          const resultEl = document.getElementById('result' + name);
          resultEl.textContent = 'Cleared.';
          showSetupError('');
        } catch (err) {
          showSetupError(err.message || 'Failed to clear enrollment.');
        }
        await refreshStatus();
      }
      async function rerollReferences() {
        try {
          await fetchJson('/setup/reroll', {method: 'POST'});
          showSetupError('');
        } catch (err) {
          showSetupError(err.message || 'Failed to reroll references.');
        }
        await refreshStatus();
      }
      async function lockReferences() {
        if (lockBtn.disabled) {
          showSetupError('Enroll both Hugo and Leon before locking.');
          return;
        }
        try {
          await fetchJson('/setup/lock_reference', {method: 'POST'});
          showSetupError('');
        } catch (err) {
          showSetupError(err.message || 'Failed to lock references.');
        }
        await refreshStatus();
      }

      function toggleMeter() {
        const btn = document.getElementById('toggleMeter');
        if (meterTimer) {
          clearInterval(meterTimer);
          meterTimer = null;
          btn.textContent = 'Enable live meter';
          document.getElementById('levelText').textContent = 'Live level --';
          document.getElementById('levelBar').style.width = '10%';
          return;
        }
        pollLevel();
        meterTimer = setInterval(pollLevel, 1400);
        btn.textContent = 'Stop meter';
      }

      async function pollLevel() {
        try {
          const data = await fetchJson('/audio/level');
          const bar = document.getElementById('levelBar');
          const text = document.getElementById('levelText');
          if (data.busy) {
            bar.style.width = '5%';
            text.textContent = 'Live level paused (recording)';
            return;
          }
          const rms = Math.min(1, data.rms || 0);
          bar.style.width = Math.min(100, Math.round(rms * 140)) + '%';
          text.textContent = `Live level ${(rms * 100).toFixed(1)}% • SNR ${data.snr_db} dB • Noise ${data.noise_floor} dBFS`;
        } catch (err) {
          showSetupError(err.message || 'Failed to read mic level.');
        }
      }
      refreshStatus();
      updateEnrollTimer();
      const setupTimer = setInterval(refreshStatus, 3000);
      // Live meter is opt-in to avoid macOS mic indicator blinking.
      window.addEventListener('beforeunload', () => {
        clearInterval(setupTimer);
        if (meterTimer) {
          clearInterval(meterTimer);
        }
        if (enrollmentTimer) {
          clearInterval(enrollmentTimer);
        }
      });
    </script>
  </body>
</html>
"""

_ADMIN_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <title>Admin • Atlas Monitor</title>
    <style>
      :root {
        --bg-primary: radial-gradient(circle at 20% 20%, #0f172a, #030712);
        --bg-surface: rgba(15, 23, 42, 0.6);
        --border-glass: rgba(56, 107, 255, 0.35);
        --text-primary: #e2e8f0;
        --text-secondary: rgba(226, 232, 240, 0.7);
        --accent: #5b8dff;
        --error: #ff6b6b;
        --radius-md: 16px;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        min-height: 100vh;
        font-family: "Inter", "Segoe UI", sans-serif;
        color: var(--text-primary);
        background: var(--bg-primary);
        overflow-x: hidden;
      }
      .page { max-width: 1520px; margin: 0 auto; padding: 28px 24px 56px; }
      .top-nav {
        display: grid;
        grid-template-columns: 1fr auto;
        align-items: center;
        gap: 20px;
        margin-bottom: 24px;
      }
      .brand-title {
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }
      .brand-subtitle { color: var(--text-secondary); font-size: 0.85rem; margin-top: 6px; }
      .nav-links { display: flex; gap: 16px; }
      .nav-link {
        padding: 8px 14px;
        border-radius: 999px;
        color: var(--text-secondary);
        border: 1px solid transparent;
        transition: 180ms ease-in-out;
      }
      .nav-link.active, .nav-link:hover {
        color: var(--text-primary);
        border-color: var(--border-glass);
        background: rgba(91, 141, 255, 0.12);
      }
      .panel {
        position: relative;
        padding: 22px;
        border-radius: var(--radius-md);
        background: var(--bg-surface);
        border: 1px solid var(--border-glass);
        box-shadow: 0 30px 80px rgba(8, 12, 24, 0.7);
        backdrop-filter: blur(18px);
      }
      .panel::after {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: inherit;
        background: linear-gradient(140deg, rgba(255, 255, 255, 0.12), transparent 35%);
        pointer-events: none;
        opacity: 0.6;
      }
      .panel > * { position: relative; z-index: 1; }
      h2 { margin: 0 0 16px; font-size: 1.2rem; text-transform: uppercase; letter-spacing: 0.08em; }
      .table-wrap { overflow: auto; border-radius: 14px; border: 1px solid rgba(255,255,255,0.08); }
      table { border-collapse: collapse; width: 100%; min-width: 760px; }
      th, td {
        padding: 10px 12px;
        font-size: 0.78rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        text-align: left;
        vertical-align: top;
      }
      th {
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.7rem;
        background: rgba(15, 19, 26, 0.9);
        position: sticky;
        top: 0;
      }
      tbody tr:nth-child(odd) { background: rgba(7, 12, 24, 0.45); }
      tbody tr:hover { background: rgba(91, 141, 255, 0.1); }
      .btn {
        border: 1px solid var(--border-glass);
        background: rgba(255,255,255,0.04);
        color: var(--text-primary);
        padding: 6px 10px;
        border-radius: 8px;
        font-size: 0.7rem;
        margin-right: 6px;
        cursor: pointer;
      }
      .btn.danger { border-color: rgba(255, 107, 107, 0.5); color: #ffdede; }
    </style>
  </head>
  <body data-theme="dark">
    <div class="page">
      <header class="top-nav">
        <div>
          <div class="brand-title">Atlas Monitor</div>
          <div class="brand-subtitle">Failed segments</div>
        </div>
        <nav class="nav-links">
          <a class="nav-link" href="/">Overview</a>
          <a class="nav-link" href="/setup">Setup</a>
          <a class="nav-link" href="/mic">Mic</a>
          <a class="nav-link active" href="/admin">Admin</a>
        </nav>
      </header>

      <section class="panel">
        <h2>Failed segments</h2>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Start</th>
                <th>End</th>
                <th>Duration</th>
                <th>Attempts</th>
                <th>Error</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody id="rows"></tbody>
          </table>
        </div>
      </section>
    </div>
    <script>
      const escapeHtml = (value) => String(value ?? "").replace(/[&<>\"]/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}[c]));

      async function fetchJson(url, options) {
        const response = await fetch(url, options);
        let data = {};
        try {
          data = await response.json();
        } catch (_) {
          data = {};
        }
        if (!response.ok) {
          throw new Error(data.detail || `Request failed (${response.status})`);
        }
        return data;
      }

      async function load() {
        const body = document.getElementById('rows');
        body.innerHTML = '<tr><td colspan="7">Loading failed segments...</td></tr>';
        try {
          const data = await fetchJson('/admin/failed');
          body.innerHTML = '';
          for (const row of data.rows || []) {
            const id = Number(row.id);
            if (!Number.isFinite(id)) {
              continue;
            }
            const durationSec = Number(row.duration_sec || 0);
            const tr = document.createElement('tr');
            tr.innerHTML = `
              <td>${id}</td>
              <td>${escapeHtml(row.start_ts)}</td>
              <td>${escapeHtml(row.end_ts)}</td>
              <td>${durationSec.toFixed(1)}s</td>
              <td>${escapeHtml(row.attempts)}</td>
              <td>${escapeHtml(row.error || '')}</td>
              <td>
                <button class="btn" onclick="retryNow(${id})">Retry</button>
                <button class="btn" onclick="exportZip(${id})">Export ZIP</button>
                <button class="btn danger" onclick="deleteSeg(${id})">Delete</button>
              </td>`;
            body.appendChild(tr);
          }
          if (!(data.rows || []).length) {
            body.innerHTML = '<tr><td colspan="7">No failed segments.</td></tr>';
          }
        } catch (err) {
          body.innerHTML = `<tr><td colspan="7">Failed to load: ${escapeHtml(err.message || 'Unknown error')}</td></tr>`;
        }
      }
      async function retryNow(id) {
        try {
          await fetchJson(`/admin/segment/${id}/retry`, {method: 'POST'});
        } catch (_) {
          // Keep the UI responsive even on transient retry errors.
        }
        await load();
      }
      async function deleteSeg(id) {
        try {
          await fetchJson(`/admin/segment/${id}/delete`, {method: 'POST'});
        } catch (_) {
          // Keep the UI responsive even on transient delete errors.
        }
        await load();
      }
      function exportZip(id) {
        window.location.href = `/admin/segment/${id}/export`;
      }
      load();
      const adminTimer = setInterval(load, 5000);
      window.addEventListener('beforeunload', () => clearInterval(adminTimer));
    </script>
  </body>
</html>
"""
