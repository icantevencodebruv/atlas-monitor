import os
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

from fastapi import APIRouter

from app.api import context
from app.services.scheduling import is_within_work_hours

router = APIRouter()


@router.get("/status")
def status():
    enrolled = {
        "Hugo": context.diarizer.has_embedding("Hugo"),
        "Leon": context.diarizer.has_embedding("Leon"),
    }
    in_hours = is_within_work_hours(
        context.config.work_hours,
        datetime.now(ZoneInfo(context.config.work_hours.timezone)),
    )
    return {
        "status": "recording" if context.state.recording else "idle",
        "manual_override": context.state.manual_override,
        "backend": context.backend.name,
        "backend_error": context.backend_error,
        "speaker_lock": context.state.speaker_lock,
        "reference_locked": context.state.reference_locked,
        "enrolled": enrolled,
        "auto_ready": (
            (enrolled["Hugo"] and enrolled["Leon"])
            if context.config.diarization.require_both_enrolled
            else True
        ),
        "recording_since": context.state.recording_since,
        "last_segment_start": context.state.last_segment_start,
        "segment_seconds": context.config.audio.segment_seconds,
        "enrollment_seconds": context.config.transcription.enrollment_seconds,
        "work_hours_enabled": context.config.work_hours.enabled,
        "in_work_hours": in_hours,
        "timezone": context.config.work_hours.timezone,
        "input_device_name": context.config.audio.input_device_name,
        "failed_count": len(context.db.list_failed_segments()),
    }


@router.get("/transcripts/recent")
def transcripts_recent(
    minutes: int = 30,
    search: str = "",
    include_unknown: bool = True,
    include_low_confidence: bool = True,
):
    now = datetime.now(timezone.utc)
    start = now - timedelta(minutes=max(1, min(minutes, 240)))
    rows = context.db.list_transcripts_between(start.isoformat(), now.isoformat())
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


@router.get("/health")
def health():
    from app.services.recorder import list_input_devices

    checks: dict = {}

    # backend
    checks["backend"] = "error" if context.backend_error else "ok"

    # mic: configured device present in device list
    try:
        device_name = context.config.audio.input_device_name
        if device_name is None:
            checks["mic"] = "ok"  # auto mode
        else:
            devices = list_input_devices()
            checks["mic"] = "ok" if any(d["name"] == device_name for d in devices) else "error"
    except Exception:
        checks["mic"] = "error"

    # asr_model: model file present on disk
    model_path = context.config.asr.whisper_cpp.model_path
    checks["asr_model"] = "ok" if os.path.exists(model_path) else "error"

    # llm_qa: skipped / ok / error
    llm_cfg = context.config.llm_qa
    if not llm_cfg.enabled:
        checks["llm_qa"] = "skipped"
    elif context.worker._qa.model_loaded:
        checks["llm_qa"] = "ok"
    else:
        checks["llm_qa"] = "error"

    checks["overall"] = "ok" if all(v in ("ok", "skipped") for v in checks.values()) else "error"
    return checks
