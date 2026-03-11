from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel

from app.api import context

router = APIRouter()


class RecordRequest(BaseModel):
    lock_mode: str | None = None
    set_only: bool = False


@router.post("/record/start")
def record_start(req: RecordRequest | None = None):
    if req and req.lock_mode:
        lock_mode = req.lock_mode.lower()
        if lock_mode in {"auto", "hugo", "leon"}:
            with context.state.lock:
                context.state.speaker_lock = lock_mode
    if req and req.set_only:
        return {
            "status": "recording" if context.state.recording else "idle",
            "speaker_lock": context.state.speaker_lock,
        }
    with context.state.lock:
        context.state.manual_override = True
        context.state.recording = True
        context.state.recording_since = datetime.now(timezone.utc).isoformat()
    context.recorder.start()
    if context.state.current_session_id is None:
        context.state.current_session_id = context.db.add_session(
            datetime.now(timezone.utc).isoformat()
        )
    return {"status": "recording"}


@router.post("/record/stop")
def record_stop():
    with context.state.lock:
        context.state.manual_override = False
        context.state.recording = False
        context.state.recording_since = None
        context.state.last_segment_start = None
    context.recorder.stop()
    if context.state.current_session_id is not None:
        context.db.end_session(
            context.state.current_session_id,
            datetime.now(timezone.utc).isoformat(),
        )
        context.state.current_session_id = None
    return {"status": "idle"}


@router.post("/record/resume")
def record_resume():
    with context.state.lock:
        context.state.manual_override = None
    context.scheduler.tick()
    return {
        "status": "recording" if context.state.recording else "idle",
        "manual_override": context.state.manual_override,
    }
