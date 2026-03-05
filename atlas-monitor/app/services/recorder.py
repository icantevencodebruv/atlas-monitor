import logging
import os
import threading
import time
from datetime import datetime, timezone
from queue import Queue
from typing import Optional, Callable

import numpy as np
import sounddevice as sd

from app.db.database import Database
from app.services.audio_utils import write_wav_int16

logger = logging.getLogger(__name__)


def list_input_devices() -> list[dict]:
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    hostapi_names = {idx: api["name"] for idx, api in enumerate(hostapis)}
    payload = []
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] <= 0:
            continue
        payload.append(
            {
                "index": idx,
                "name": dev["name"],
                "hostapi": hostapi_names.get(dev["hostapi"], ""),
                "max_input_channels": dev["max_input_channels"],
                "default_samplerate": dev.get("default_samplerate"),
            }
        )
    return payload


def _resolve_device_by_name(name: Optional[str]) -> Optional[int]:
    if not name:
        return None
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] <= 0:
            continue
        if dev["name"].lower() == name.lower():
            return idx
    return None


def select_input_device(preferred_hostapi: Optional[str], preferred_name: Optional[str] = None) -> Optional[int]:
    try:
        if preferred_name:
            idx = _resolve_device_by_name(preferred_name)
            if idx is not None:
                sd.default.device = (idx, None)
                logger.info("Selected input device %s.", preferred_name)
                return idx
        if not preferred_hostapi:
            return None
        hostapis = sd.query_hostapis()
        for api in hostapis:
            if api["name"] == preferred_hostapi:
                for dev_idx in api["devices"]:
                    dev = sd.query_devices(dev_idx)
                    if dev["max_input_channels"] > 0:
                        sd.default.device = (dev_idx, None)
                        logger.info("Selected input device %s on host API %s", dev["name"], preferred_hostapi)
                        return dev_idx
    except Exception as exc:
        logger.warning("Failed to select input device: %s", exc)
    return None


def set_input_device(index: int) -> None:
    sd.default.device = (index, None)
    dev = sd.query_devices(index)
    logger.info("Switched input device to %s.", dev["name"])


class SegmentRecorder:
    def __init__(
        self,
        db: Database,
        audio_dir: str,
        segment_seconds: int,
        sample_rate: int,
        channels: int,
        hostapi_preference: Optional[str],
        input_device_name: Optional[str],
        queue: Queue,
        device_lock: threading.Lock,
        on_segment_start: Optional[Callable[[datetime], None]] = None,
    ):
        self._db = db
        self._audio_dir = audio_dir
        self._segment_seconds = segment_seconds
        self._sample_rate = sample_rate
        self._channels = channels
        self._hostapi_preference = hostapi_preference
        self._input_device_name = input_device_name
        self._queue = queue
        self._device_lock = device_lock
        self._on_segment_start = on_segment_start
        self._recording = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        select_input_device(hostapi_preference, input_device_name)

    def start(self) -> None:
        self._recording = True
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._recording = False

    def shutdown(self) -> None:
        self._stop_event.set()
        self._recording = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def update_preferred_device(self, input_device_name: Optional[str]) -> None:
        self._input_device_name = input_device_name

    def _run(self) -> None:
        os.makedirs(self._audio_dir, exist_ok=True)
        while not self._stop_event.is_set():
            if not self._recording:
                time.sleep(0.5)
                continue
            start_time = datetime.now(timezone.utc)
            if self._on_segment_start:
                try:
                    self._on_segment_start(start_time)
                except Exception:
                    logger.exception("Segment start callback failed.")
            frame_count = int(self._segment_seconds * self._sample_rate)
            logger.info("Recording segment for %s seconds.", self._segment_seconds)
            with self._device_lock:
                select_input_device(self._hostapi_preference, self._input_device_name)
                audio = sd.rec(
                    frames=frame_count,
                    samplerate=self._sample_rate,
                    channels=self._channels,
                    dtype="int16",
                    blocking=True,
                )
            audio = np.squeeze(audio)
            if audio.ndim > 1:
                audio = audio.mean(axis=1).astype(np.int16)
            end_time = datetime.now(timezone.utc)
            filename = f"segment_{start_time.strftime('%Y%m%dT%H%M%S')}.wav"
            file_path = os.path.join(self._audio_dir, filename)
            write_wav_int16(file_path, audio, self._sample_rate)
            segment_id = self._db.add_segment(
                file_path,
                start_time.isoformat(),
                end_time.isoformat(),
            )
            self._queue.put(segment_id)
