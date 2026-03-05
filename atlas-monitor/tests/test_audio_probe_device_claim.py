import threading

import numpy as np

import app.services.audio_probe as audio_probe


def test_measure_level_claims_preferred_device(monkeypatch):
    calls = []

    def fake_select(hostapi, name):
        calls.append((hostapi, name))
        return 1

    def fake_rec(*, frames, samplerate, channels, dtype, blocking):
        return np.zeros((frames, channels), dtype=np.int16)

    monkeypatch.setattr(audio_probe, "select_input_device", fake_select)
    monkeypatch.setattr(audio_probe.sd, "rec", fake_rec)

    result = audio_probe.measure_level(
        16000,
        1,
        threading.Lock(),
        preferred_hostapi="Core Audio",
        preferred_name="Elgato Wave:3",
    )

    assert result["busy"] is False
    assert calls == [("Core Audio", "Elgato Wave:3")]


def test_test_recording_retries_and_reclaims_device(monkeypatch):
    calls = []
    attempts = {"n": 0}

    def fake_select(hostapi, name):
        calls.append((hostapi, name))
        return 1

    def fake_rec(*, frames, samplerate, channels, dtype, blocking):
        if attempts["n"] == 0:
            attempts["n"] += 1
            raise RuntimeError("device busy")
        return np.zeros((frames, channels), dtype=np.int16)

    monkeypatch.setattr(audio_probe, "select_input_device", fake_select)
    monkeypatch.setattr(audio_probe.sd, "rec", fake_rec)

    result = audio_probe.test_recording(
        0.5,
        16000,
        1,
        threading.Lock(),
        preferred_hostapi="Core Audio",
        preferred_name="Elgato Wave:3",
    )

    assert result["ok"] is True
    assert calls == [
        ("Core Audio", "Elgato Wave:3"),
        ("Core Audio", "Elgato Wave:3"),
    ]
