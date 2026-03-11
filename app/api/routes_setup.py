import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.api import context

router = APIRouter()
logger = logging.getLogger(__name__)


def _enroll_speaker(speaker: str) -> JSONResponse:
    if context.state.reference_locked:
        raise HTTPException(status_code=400, detail="Reference locked. Click Reroll to update.")
    if context.state.recording:
        raise HTTPException(status_code=400, detail="Stop recording before enrollment.")
    try:
        result = context.enroller.enroll(speaker, context.config.transcription.enrollment_seconds)
    except Exception as exc:
        msg = str(exc).lower()
        if "enrollment_in_progress" in msg:
            raise HTTPException(status_code=409, detail="Another enrollment is in progress.") from exc
        if "no_audio_captured" in msg:
            raise HTTPException(
                status_code=400, detail="No audio captured. Check Mic settings and retry."
            ) from exc
        if "portaudio" in msg or "inputstream" in msg:
            logger.warning("Enrollment failed for %s: %s", speaker, exc)
            raise HTTPException(
                status_code=503,
                detail="Microphone is unavailable. Check Mic settings and close other apps using the mic, then retry.",
            ) from exc
        logger.exception("Unexpected enrollment failure for %s.", speaker)
        raise
    if result.get("cancelled"):
        return JSONResponse({"status": "cancelled", **result})
    return JSONResponse({"status": "ok", **result})


@router.post("/setup/enroll/hugo")
def enroll_hugo():
    return _enroll_speaker("Hugo")


@router.post("/setup/enroll/leon")
def enroll_leon():
    return _enroll_speaker("Leon")


@router.post("/setup/enroll/stop")
def stop_enrollment():
    stop = context.enroller.stop_active()
    if stop["stopping"]:
        return {"status": "stopping", "speaker": stop["speaker"]}
    return {"status": "idle", "speaker": None}


@router.post("/setup/clear/hugo")
def clear_hugo():
    context.db.clear_embedding("Hugo")
    context.diarizer.remove_embedding("Hugo")
    return {"status": "cleared"}


@router.post("/setup/clear/leon")
def clear_leon():
    context.db.clear_embedding("Leon")
    context.diarizer.remove_embedding("Leon")
    return {"status": "cleared"}


@router.post("/setup/reroll")
def reroll_references():
    with context.state.lock:
        context.state.reference_locked = False
    context.config.diarization.reference_locked = False
    context.db.clear_embedding("Hugo")
    context.db.clear_embedding("Leon")
    context.diarizer.remove_embedding("Hugo")
    context.diarizer.remove_embedding("Leon")
    context._persist_config()
    return {"status": "unlocked"}


@router.post("/setup/lock_reference")
def lock_reference():
    with context.state.lock:
        context.state.reference_locked = True
    context.config.diarization.reference_locked = True
    context._persist_config()
    return {"status": "locked"}
