import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api import context
from app.audio.probe import measure_level, test_recording
from app.services.recorder import list_input_devices, set_input_device, select_input_device

router = APIRouter()


class DeviceSelectRequest(BaseModel):
    index: int | None = None
    name: str | None = None


@router.get("/audio/devices")
def audio_devices():
    devices = list_input_devices()
    return {"devices": devices, "selected": context.config.audio.input_device_name}


@router.post("/audio/device")
def audio_device(req: DeviceSelectRequest):
    if context.state.recording:
        raise HTTPException(status_code=400, detail="Stop recording before changing the device.")
    if req.index is None and not req.name:
        raise HTTPException(status_code=400, detail="Device index or name required.")
    if req.name and req.name.lower() == "auto":
        context.config.audio.input_device_name = None
        select_input_device(
            context.config.audio.device_hostapi_preference.get(
                "windows" if os.name == "nt" else "macos"
            )
        )
        context._persist_config()
        return {"status": "ok", "device_name": None}
    if req.index is not None:
        set_input_device(req.index)
        devices = list_input_devices()
        match = next((d for d in devices if d["index"] == req.index), None)
        if not match:
            raise HTTPException(status_code=404, detail="Device not found.")
        context.config.audio.input_device_name = match["name"]
        context._persist_config()
        return {"status": "ok", "device_name": match["name"]}
    devices = list_input_devices()
    match = next((d for d in devices if d["name"].lower() == req.name.lower()), None)
    if not match:
        raise HTTPException(status_code=404, detail="Device not found.")
    set_input_device(match["index"])
    context.config.audio.input_device_name = match["name"]
    context._persist_config()
    return {"status": "ok", "device_name": match["name"]}


@router.get("/audio/level")
def audio_level():
    return measure_level(context.config.audio.sample_rate, context.config.audio.channels, context.device_lock)


@router.post("/audio/test")
def audio_test():
    return test_recording(3.0, context.config.audio.sample_rate, context.config.audio.channels, context.device_lock)
