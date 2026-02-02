import logging
import os
import threading
from queue import Queue
import webbrowser
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel

from app.asr.manager import select_backend
from app.config import load_config
from app.db.database import Database
from app.logging_setup import setup_logging
from app.services.diarization import SpeakerIdentifier
from app.services.enrollment import EnrollmentService
from app.services.exporter import build_export, compute_range
from app.services.recorder import SegmentRecorder
from app.services.retry_worker import RetryWorker
from app.services.scheduling import WorkHoursScheduler
from app.services.transcription import TranscriptionWorker
from app.state import AppState

CONFIG_PATH = os.environ.get("APP_CONFIG", "config.yaml")

config = load_config(CONFIG_PATH)
setup_logging(config.storage.logs_dir)
logger = logging.getLogger(__name__)

os.makedirs(config.storage.audio_dir, exist_ok=True)
os.makedirs(config.storage.exports_dir, exist_ok=True)
os.makedirs(config.storage.logs_dir, exist_ok=True)

db = Database(config.storage.db_path)
state = AppState()
device_lock = threading.Lock()

diarizer = SpeakerIdentifier()
for speaker in ["Hugo", "Leon"]:
    diarizer.load_embedding(speaker, db.get_embedding(speaker))

backend = select_backend(config)
backend_error = None
segment_queue = Queue()
recorder = SegmentRecorder(
    db=db,
    audio_dir=config.storage.audio_dir,
    segment_seconds=config.audio.segment_seconds,
    sample_rate=config.audio.sample_rate,
    channels=config.audio.channels,
    hostapi_preference=config.audio.device_hostapi_preference.get(
        "windows" if os.name == "nt" else "macos"
    ),
    queue=segment_queue,
    device_lock=device_lock,
)
worker = TranscriptionWorker(db, segment_queue, backend, diarizer, config, state)
scheduler = WorkHoursScheduler(config, state, recorder, db)
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


class ExportRequest(BaseModel):
    range: str
    start: str | None = None
    end: str | None = None


class RecordRequest(BaseModel):
    lock_mode: str | None = None
    set_only: bool = False


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


@app.get("/admin", response_class=HTMLResponse)
def admin_page():
    return HTMLResponse(_ADMIN_HTML)


@app.get("/status")
def status():
    enrolled = {
        "Hugo": diarizer.has_embedding("Hugo"),
        "Leon": diarizer.has_embedding("Leon"),
    }
    return {
        "status": "recording" if state.recording else "idle",
        "manual_override": state.manual_override,
        "backend": backend.name,
        "backend_error": backend_error,
        "speaker_lock": state.speaker_lock,
        "enrolled": enrolled,
        "auto_ready": (enrolled["Hugo"] and enrolled["Leon"]) if config.diarization.require_both_enrolled else True,
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
    if req and req.lock_mode:
        lock_mode = req.lock_mode.lower()
        if lock_mode in {"auto", "hugo", "leon"}:
            with state.lock:
                state.speaker_lock = lock_mode
    if req and req.set_only:
        return {"status": "recording" if state.recording else "idle", "speaker_lock": state.speaker_lock}
    with state.lock:
        state.manual_override = True
        state.recording = True
    recorder.start()
    if state.current_session_id is None:
        state.current_session_id = db.add_session(datetime.now(timezone.utc).isoformat())
    return {"status": "recording"}


@app.post("/record/stop")
def record_stop():
    with state.lock:
        state.manual_override = False
        state.recording = False
    recorder.stop()
    if state.current_session_id is not None:
        db.end_session(state.current_session_id, datetime.now(timezone.utc).isoformat())
        state.current_session_id = None
    return {"status": "idle"}


@app.post("/export")
def export_range(req: ExportRequest):
    now = datetime.now(timezone.utc)
    session_row = db.get_latest_session()
    start_ts, end_ts = compute_range(
        req.range, now, session_row, req.start, req.end, config.work_hours.timezone
    )
    export_id = build_export(db, req.range, start_ts, end_ts, config.storage.exports_dir)
    return {"id": export_id}


@app.get("/download/{export_id}")
def download(export_id: int):
    export = db.get_export(export_id)
    if not export:
        raise HTTPException(status_code=404, detail="Export not found")
    return FileResponse(export["file_path"], filename=os.path.basename(export["file_path"]))


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
    if state.recording:
        raise HTTPException(status_code=400, detail="Stop recording before enrollment.")
    result = enroller.enroll("Hugo", config.transcription.enrollment_seconds)
    return JSONResponse({"status": "ok", **result})


@app.post("/setup/enroll/leon")
def enroll_leon():
    if state.recording:
        raise HTTPException(status_code=400, detail="Stop recording before enrollment.")
    result = enroller.enroll("Leon", config.transcription.enrollment_seconds)
    return JSONResponse({"status": "ok", **result})


_HOME_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <title>Atlas Monitor</title>
    <style>
      :root {
        --bg: #0b0e12;
        --bg-alt: #0f131a;
        --surface: #151a22;
        --surface-strong: #1b2230;
        --border: #263042;
        --text: #e9edf2;
        --muted: #a3adbb;
        --accent: #4f8cff;
        --accent-2: #28d17c;
        --danger: #ff5c7a;
        --warning: #f4b244;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        min-height: 100vh;
        font-family: "Avenir Next", "Futura", "Gill Sans", "Trebuchet MS", sans-serif;
        color: var(--text);
        background:
          radial-gradient(circle at 10% 20%, rgba(79, 140, 255, 0.15), transparent 45%),
          radial-gradient(circle at 80% 10%, rgba(40, 209, 124, 0.12), transparent 35%),
          linear-gradient(160deg, #0b0e12 0%, #101520 50%, #0b0e12 100%);
      }
      a { color: inherit; text-decoration: none; }
      .app { max-width: 1120px; margin: 0 auto; padding: 28px 24px 56px; }
      .top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 16px;
      }
      .brand-title {
        font-size: 24px;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        font-weight: 700;
      }
      .brand-subtitle { color: var(--muted); font-size: 14px; margin-top: 6px; }
      .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        padding: 10px 18px;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: rgba(21, 26, 34, 0.9);
        font-size: 12px;
        letter-spacing: 0.12em;
        text-transform: uppercase;
      }
      .status-pill::before {
        content: "";
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: var(--warning);
        box-shadow: 0 0 10px rgba(244, 178, 68, 0.4);
      }
      .status-pill.recording {
        border-color: rgba(40, 209, 124, 0.4);
        color: #d7ffe9;
      }
      .status-pill.recording::before {
        background: var(--accent-2);
        box-shadow: 0 0 12px rgba(40, 209, 124, 0.7);
      }
      .grid {
        margin-top: 26px;
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 20px;
      }
      .card {
        background: rgba(21, 26, 34, 0.95);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 14px 30px rgba(0, 0, 0, 0.35);
      }
      .card h2 { margin: 0 0 12px; font-size: 20px; }
      .card p { margin: 0 0 14px; color: var(--muted); }
      .button-row { display: flex; flex-wrap: wrap; gap: 10px; }
      .btn {
        border: 1px solid var(--border);
        background: var(--surface-strong);
        color: var(--text);
        padding: 12px 18px;
        border-radius: 12px;
        font-weight: 600;
        letter-spacing: 0.02em;
        cursor: pointer;
        transition: transform 0.15s ease, box-shadow 0.2s ease, border 0.2s ease;
      }
      .btn:hover { transform: translateY(-1px); box-shadow: 0 10px 18px rgba(0, 0, 0, 0.35); }
      .btn.primary {
        border: none;
        background: linear-gradient(130deg, #4f8cff, #6b6bff);
        color: #f6f8ff;
      }
      .btn.ghost { background: transparent; }
      .segmented {
        display: inline-flex;
        gap: 6px;
        padding: 6px;
        border-radius: 999px;
        background: var(--bg-alt);
        border: 1px solid var(--border);
      }
      .segmented input { display: none; }
      .segmented label {
        padding: 8px 14px;
        border-radius: 999px;
        font-size: 12px;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
        cursor: pointer;
      }
      .segmented input:checked + label {
        background: rgba(79, 140, 255, 0.2);
        color: var(--text);
        box-shadow: inset 0 0 0 1px rgba(79, 140, 255, 0.55);
      }
      .badge-row { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; }
      .badge {
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 11px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        border: 1px solid var(--border);
        color: var(--muted);
      }
      .badge.good { color: #d5ffeb; border-color: rgba(40, 209, 124, 0.5); }
      .badge.warn { color: #ffe1b8; border-color: rgba(244, 178, 68, 0.5); }
      .notice {
        margin-top: 12px;
        padding: 10px 12px;
        border-radius: 12px;
        border: 1px solid rgba(244, 178, 68, 0.4);
        background: rgba(244, 178, 68, 0.12);
        font-size: 12px;
        color: #ffe2b3;
      }
      .alert {
        margin-top: 12px;
        padding: 12px 14px;
        border-radius: 12px;
        border: 1px solid rgba(255, 92, 122, 0.4);
        background: rgba(255, 92, 122, 0.12);
        color: #ffdbe2;
        display: none;
      }
      .alert.show { display: block; }
      .export-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
      }
      .input-row {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 12px;
        margin-top: 12px;
      }
      label { font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); }
      input {
        width: 100%;
        margin-top: 6px;
        padding: 10px 12px;
        border-radius: 10px;
        border: 1px solid var(--border);
        background: var(--bg-alt);
        color: var(--text);
      }
      .link-row { margin-top: 14px; }
      .link-row a { color: var(--accent); font-weight: 600; }
      .footer {
        margin-top: 24px;
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        color: var(--muted);
        font-size: 13px;
      }
      .footer a { color: var(--muted); text-decoration: underline; }
      @media (min-width: 900px) {
        .export-grid { grid-template-columns: repeat(4, minmax(0, 1fr)); }
      }
    </style>
  </head>
  <body>
    <div class="app">
      <div class="top">
        <div>
          <div class="brand-title">Atlas Monitor</div>
          <div class="brand-subtitle">Offline recorder • diarized transcription • localhost only</div>
        </div>
        <div class="status-pill idle" id="statusPill">Idle</div>
      </div>

      <div class="grid">
        <section class="card">
          <h2>Capture</h2>
          <p>Manual control for recording and speaker lock.</p>
          <div class="button-row">
            <button class="btn primary" onclick="startRecording()">Start</button>
            <button class="btn ghost" onclick="post('/record/stop')">Stop</button>
          </div>
          <div style="margin-top: 16px;">
            <label>Speaker lock</label>
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
            <div class="notice" id="enrollHint" style="display: none;"></div>
          </div>
          <div class="alert" id="error"></div>
        </section>

        <section class="card">
          <h2>Export</h2>
          <p>Instant exports with custom range support.</p>
          <div class="export-grid">
            <button class="btn" onclick="exportRange('30m')">Last 30m</button>
            <button class="btn" onclick="exportRange('60m')">Last 60m</button>
            <button class="btn" onclick="exportRange('today')">Today</button>
            <button class="btn" onclick="exportRange('session')">Session</button>
          </div>
          <div class="input-row">
            <label>Custom start (ISO)
              <input id="customStart" placeholder="YYYY-MM-DDTHH:MM:SS+00:00"/>
            </label>
            <label>Custom end (ISO)
              <input id="customEnd" placeholder="YYYY-MM-DDTHH:MM:SS+00:00"/>
            </label>
          </div>
          <div class="button-row" style="margin-top: 12px;">
            <button class="btn" onclick="exportCustom()">Export custom</button>
          </div>
          <div class="link-row">
            <a id="downloadLink" href="#">Download latest export</a>
          </div>
        </section>
      </div>

      <div class="footer">
        <a href="/setup">Setup speaker enrollment</a>
        <a href="/admin">Admin</a>
      </div>
    </div>
    <script>
      function selectedLock() {
        const selected = document.querySelector('input[name="speakerLock"]:checked');
        return selected ? selected.value : 'auto';
      }
      async function refresh() {
        const res = await fetch('/status');
        const data = await res.json();
        const pill = document.getElementById('statusPill');
        pill.textContent = data.status === 'recording' ? 'Recording' : 'Idle';
        pill.className = 'status-pill ' + (data.status === 'recording' ? 'recording' : 'idle');
        const errorEl = document.getElementById('error');
        if (data.backend_error) {
          errorEl.textContent = 'ASR error: ' + data.backend_error;
          errorEl.classList.add('show');
        } else {
          errorEl.textContent = '';
          errorEl.classList.remove('show');
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
          if (target) { target.checked = true; }
        }
      }
      function updateBadge(id, name, ok) {
        const el = document.getElementById(id);
        if (!el) return;
        el.textContent = name + ': ' + (ok ? 'Enrolled' : 'Not enrolled');
        el.className = 'badge ' + (ok ? 'good' : 'warn');
      }
      async function post(url, body) {
        await fetch(url, {method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body || {})});
        refresh();
      }
      async function startRecording() {
        await post('/record/start', {lock_mode: selectedLock()});
      }
      async function setLockMode(mode) {
        await post('/record/start', {lock_mode: mode, set_only: true});
      }
      async function exportRange(range) {
        const res = await fetch('/export', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({range})});
        const data = await res.json();
        document.getElementById('downloadLink').href = '/download/' + data.id;
      }
      async function exportCustom() {
        const start = document.getElementById('customStart').value;
        const end = document.getElementById('customEnd').value;
        const res = await fetch('/export', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({range: 'custom', start, end})});
        const data = await res.json();
        document.getElementById('downloadLink').href = '/download/' + data.id;
      }
      document.querySelectorAll('input[name="speakerLock"]').forEach((el) => {
        el.addEventListener('change', () => setLockMode(el.value));
      });
      refresh();
      setInterval(refresh, 3000);
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
        --bg: #0b0e12;
        --bg-alt: #0f131a;
        --surface: #151a22;
        --surface-strong: #1b2230;
        --border: #263042;
        --text: #e9edf2;
        --muted: #a3adbb;
        --accent: #4f8cff;
        --accent-2: #28d17c;
        --warning: #f4b244;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        min-height: 100vh;
        font-family: "Avenir Next", "Futura", "Gill Sans", "Trebuchet MS", sans-serif;
        color: var(--text);
        background:
          radial-gradient(circle at 15% 15%, rgba(79, 140, 255, 0.15), transparent 45%),
          radial-gradient(circle at 85% 0%, rgba(40, 209, 124, 0.1), transparent 40%),
          linear-gradient(160deg, #0b0e12 0%, #101520 50%, #0b0e12 100%);
      }
      .app { max-width: 980px; margin: 0 auto; padding: 28px 24px 56px; }
      h1 { margin: 0 0 8px; font-size: 26px; letter-spacing: 0.12em; text-transform: uppercase; }
      .subtitle { color: var(--muted); margin-bottom: 24px; }
      .grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 18px;
      }
      .card {
        background: rgba(21, 26, 34, 0.95);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 14px 30px rgba(0, 0, 0, 0.35);
      }
      .card-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 10px;
      }
      .card h3 { margin: 0; font-size: 18px; }
      .badge {
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 11px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        border: 1px solid var(--border);
        color: var(--muted);
      }
      .badge.good { color: #d5ffeb; border-color: rgba(40, 209, 124, 0.5); }
      .badge.warn { color: #ffe1b8; border-color: rgba(244, 178, 68, 0.5); }
      .tip { color: var(--muted); font-size: 13px; margin: 0 0 14px; }
      .btn {
        border: none;
        background: linear-gradient(130deg, #4f8cff, #6b6bff);
        color: #f6f8ff;
        padding: 12px 18px;
        border-radius: 12px;
        font-weight: 600;
        letter-spacing: 0.02em;
        cursor: pointer;
      }
      .result { margin-top: 12px; font-size: 13px; color: var(--muted); }
      .result.warn { color: #ffe2b3; }
      .links { margin-top: 24px; color: var(--muted); font-size: 13px; }
      .links a { color: var(--muted); text-decoration: underline; margin-right: 12px; }
    </style>
  </head>
  <body>
    <div class="app">
      <h1>Speaker enrollment</h1>
      <div class="subtitle">Speak naturally for 20–40 seconds. Keep a steady distance to the mic.</div>

      <div class="grid">
        <div class="card">
          <div class="card-head">
            <h3>Hugo</h3>
            <div class="badge" id="statusHugo">--</div>
          </div>
          <p class="tip">Aim for calm, consistent volume. Avoid keyboard noise.</p>
          <button class="btn" onclick="enroll('/setup/enroll/hugo', 'Hugo')">Record Hugo</button>
          <div class="result" id="resultHugo"></div>
        </div>
        <div class="card">
          <div class="card-head">
            <h3>Leon</h3>
            <div class="badge" id="statusLeon">--</div>
          </div>
          <p class="tip">Speak in your normal cadence for the full duration.</p>
          <button class="btn" onclick="enroll('/setup/enroll/leon', 'Leon')">Record Leon</button>
          <div class="result" id="resultLeon"></div>
        </div>
      </div>

      <div class="links">
        <a href="/">Back to recorder</a>
        <a href="/admin">Admin</a>
      </div>
    </div>
    <script>
      function setBadge(id, ok) {
        const el = document.getElementById(id);
        if (!el) return;
        el.textContent = ok ? 'Enrolled' : 'Not enrolled';
        el.className = 'badge ' + (ok ? 'good' : 'warn');
      }
      async function refreshStatus() {
        const res = await fetch('/status');
        const data = await res.json();
        setBadge('statusHugo', data.enrolled && data.enrolled.Hugo);
        setBadge('statusLeon', data.enrolled && data.enrolled.Leon);
      }
      async function enroll(url, name) {
        const resultEl = document.getElementById('result' + name);
        resultEl.textContent = 'Recording...';
        resultEl.className = 'result';
        const res = await fetch(url, {method:'POST'});
        const data = await res.json();
        if (data.status === 'ok') {
          const snr = Number.isFinite(data.snr_db) ? data.snr_db.toFixed(1) : '--';
          const duration = Number.isFinite(data.duration_sec) ? data.duration_sec.toFixed(1) : '--';
          resultEl.textContent = 'Done. ' + duration + 's, SNR ' + snr + ' dB.';
          if (data.quality_ok === false) {
            resultEl.textContent += ' Low quality — please re-enroll.';
            resultEl.className = 'result warn';
          }
        } else {
          resultEl.textContent = 'Error.';
        }
        refreshStatus();
      }
      refreshStatus();
      setInterval(refreshStatus, 3000);
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
        --bg: #0b0e12;
        --surface: #151a22;
        --surface-strong: #1b2230;
        --border: #263042;
        --text: #e9edf2;
        --muted: #a3adbb;
        --accent: #4f8cff;
        --danger: #ff5c7a;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        min-height: 100vh;
        font-family: "Avenir Next", "Futura", "Gill Sans", "Trebuchet MS", sans-serif;
        color: var(--text);
        background: linear-gradient(160deg, #0b0e12 0%, #101520 50%, #0b0e12 100%);
      }
      .app { max-width: 1200px; margin: 0 auto; padding: 28px 24px 56px; }
      h2 { margin: 0 0 16px; font-size: 22px; letter-spacing: 0.08em; text-transform: uppercase; }
      .table-wrap {
        background: rgba(21, 26, 34, 0.95);
        border: 1px solid var(--border);
        border-radius: 16px;
        overflow: auto;
        box-shadow: 0 12px 26px rgba(0, 0, 0, 0.35);
      }
      table {
        border-collapse: collapse;
        width: 100%;
        min-width: 760px;
      }
      th, td {
        padding: 10px 12px;
        font-size: 12px;
        border-bottom: 1px solid var(--border);
        text-align: left;
        vertical-align: top;
      }
      th {
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 11px;
        background: rgba(15, 19, 26, 0.9);
        position: sticky;
        top: 0;
      }
      tbody tr:nth-child(odd) { background: rgba(15, 19, 26, 0.6); }
      tbody tr:hover { background: rgba(79, 140, 255, 0.08); }
      .btn {
        border: 1px solid var(--border);
        background: var(--surface-strong);
        color: var(--text);
        padding: 6px 10px;
        border-radius: 10px;
        font-size: 11px;
        margin-right: 6px;
        cursor: pointer;
      }
      .btn.danger {
        border-color: rgba(255, 92, 122, 0.5);
        color: #ffdbe2;
      }
      .links { margin-top: 18px; font-size: 13px; color: var(--muted); }
      .links a { color: var(--muted); text-decoration: underline; margin-right: 12px; }
    </style>
  </head>
  <body>
    <div class="app">
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
      <div class="links">
        <a href="/">Back to recorder</a>
        <a href="/setup">Setup enrollment</a>
      </div>
    </div>
    <script>
      async function load() {
        const res = await fetch('/admin/failed');
        const data = await res.json();
        const body = document.getElementById('rows');
        body.innerHTML = '';
        for (const row of data.rows) {
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td>${row.id}</td>
            <td>${row.start_ts}</td>
            <td>${row.end_ts}</td>
            <td>${row.duration_sec.toFixed(1)}s</td>
            <td>${row.attempts}</td>
            <td>${row.error || ''}</td>
            <td>
              <button class="btn" onclick="retryNow(${row.id})">Retry</button>
              <button class="btn" onclick="exportZip(${row.id})">Export ZIP</button>
              <button class="btn danger" onclick="deleteSeg(${row.id})">Delete</button>
            </td>`;
          body.appendChild(tr);
        }
      }
      async function retryNow(id) {
        await fetch(`/admin/segment/${id}/retry`, {method:'POST'});
        load();
      }
      async function deleteSeg(id) {
        await fetch(`/admin/segment/${id}/delete`, {method:'POST'});
        load();
      }
      function exportZip(id) {
        window.location.href = `/admin/segment/${id}/export`;
      }
      load();
      setInterval(load, 5000);
    </script>
  </body>
</html>
"""
