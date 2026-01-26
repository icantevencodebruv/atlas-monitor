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
worker = TranscriptionWorker(db, segment_queue, backend, diarizer, config)
scheduler = WorkHoursScheduler(config, state, recorder, db)
retry_worker = RetryWorker(db, segment_queue, config)
enroller = EnrollmentService(db, diarizer, config.audio, device_lock, config.storage.audio_dir)

app = FastAPI()


class ExportRequest(BaseModel):
    range: str
    start: str | None = None
    end: str | None = None


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
    return {
        "status": "recording" if state.recording else "idle",
        "manual_override": state.manual_override,
        "backend": backend.name,
        "backend_error": backend_error,
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
def record_start():
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
    enroller.enroll("Hugo", config.transcription.enrollment_seconds)
    return JSONResponse({"status": "ok"})


@app.post("/setup/enroll/leon")
def enroll_leon():
    if state.recording:
        raise HTTPException(status_code=400, detail="Stop recording before enrollment.")
    enroller.enroll("Leon", config.transcription.enrollment_seconds)
    return JSONResponse({"status": "ok"})


_HOME_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <title>Recorder</title>
    <style>
      body { font-family: Helvetica, Arial, sans-serif; margin: 24px; }
      .status { font-size: 24px; margin-bottom: 16px; }
      button { margin: 4px; padding: 8px 12px; }
      .row { margin-top: 12px; }
    </style>
  </head>
  <body>
    <div class="status" id="status">status: ...</div>
    <div id="error" style="color: #b00020;"></div>
    <div>
      <button onclick="post('/record/start')">Start</button>
      <button onclick="post('/record/stop')">Stop</button>
    </div>
    <div class="row">
      <button onclick="exportRange('30m')">Export 30m</button>
      <button onclick="exportRange('60m')">Export 60m</button>
      <button onclick="exportRange('today')">Export Today</button>
      <button onclick="exportRange('session')">Export Session</button>
    </div>
    <div class="row">
      <label>Custom start (ISO): <input id="customStart" size="24"/></label>
      <label>Custom end (ISO): <input id="customEnd" size="24"/></label>
      <button onclick="exportCustom()">Export Custom</button>
    </div>
    <div class="row">
      <a id="downloadLink" href="#">Download latest export</a>
    </div>
    <div class="row">
      <a href="/setup">Setup speaker enrollment</a>
    </div>
    <script>
      async function refresh() {
        const res = await fetch('/status');
        const data = await res.json();
        document.getElementById('status').textContent = 'status: ' + data.status;
        document.getElementById('error').textContent = data.backend_error ? ('ASR error: ' + data.backend_error) : '';
      }
      async function post(url, body) {
        await fetch(url, {method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body || {})});
        refresh();
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
    <title>Setup</title>
    <style>
      body { font-family: Helvetica, Arial, sans-serif; margin: 24px; }
      button { margin: 8px; padding: 8px 12px; }
    </style>
  </head>
  <body>
    <h2>Speaker enrollment</h2>
    <p>Click a button and speak naturally for 20-40 seconds.</p>
    <button onclick="enroll('/setup/enroll/hugo')">Enroll Hugo</button>
    <button onclick="enroll('/setup/enroll/leon')">Enroll Leon</button>
    <p id="result"></p>
    <a href="/">Back</a>
    <script>
      async function enroll(url) {
        document.getElementById('result').textContent = 'Recording...';
        const res = await fetch(url, {method:'POST'});
        const data = await res.json();
        document.getElementById('result').textContent = data.status === 'ok' ? 'Done.' : 'Error.';
      }
    </script>
  </body>
</html>
"""

_ADMIN_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <title>Admin</title>
    <style>
      body { font-family: Helvetica, Arial, sans-serif; margin: 24px; }
      table { border-collapse: collapse; width: 100%; }
      th, td { border: 1px solid #ccc; padding: 6px; font-size: 12px; }
      button { margin-right: 6px; }
    </style>
  </head>
  <body>
    <h2>Failed segments</h2>
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
              <button onclick="retryNow(${row.id})">Retry</button>
              <button onclick="exportZip(${row.id})">Export ZIP</button>
              <button onclick="deleteSeg(${row.id})">Delete</button>
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
