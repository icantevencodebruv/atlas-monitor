![Project Screenshot](<img width="1200" height="245" alt="azizrobotics-github" src="https://github.com/user-attachments/assets/83af5759-7bf6-41a3-92bb-2e2e8c45f8f4" />
)

# Atlas Monitor

Offline-only recorder with incremental transcription + diarization, built for two fixed speakers (Hugo, Leon) and a minimal local UI. No telemetry, no network calls at runtime.

<details>
<summary>ASCII Banner</summary>

<pre>
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃   █████╗ ████████╗██╗      █████╗ ███████╗     ███╗   ███╗ ██████╗      ┃
    ┃  ██╔══██╗╚══██╔══╝██║     ██╔══██╗██╔════╝     ████╗ ████║██╔═══██╗     ┃
    ┃  ███████║   ██║   ██║     ███████║███████╗     ██╔████╔██║██║   ██║     ┃
    ┃  ██╔══██║   ██║   ██║     ██╔══██║╚════██║     ██║╚██╔╝██║██║   ██║     ┃
    ┃  ██║  ██║   ██║   ███████╗██║  ██║███████║     ██║ ╚═╝ ██║╚██████╔╝     ┃
    ┃  ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚══════╝     ╚═╝     ╚═╝ ╚═════╝      ┃
    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

      REAL-TIME OFFLINE RECORDER + DIARIZED TRANSCRIPTION  •  127.0.0.1 ONLY
                        de/en diarization locked to Hugo + Leon

                         .--------------------.
                        /   .--------------.   \
                       |   /   .------.     \   |
                       |  |   /  /\    \     |  |
                       |  |   | |  |   |     |  |
                       |  |   \  \/   /      |  |
                       |   \   '------'     /   |
                        \   '--------------'   /
                         '--------.  .--------'
                                  |  |
                                  |  |
                                  |  |
                               ___|__|___
                              /__________\
</pre>

</details>

**Last update:** 05.02.2026

---

## About

**Atlas Monitor** is a localhost-only “control center” for:
- segmented microphone recording
- incremental transcription
- two-speaker diarization (Hugo / Leon) with a confidence gate to **Unknown**
- work-hours scheduling + manual override
- exports (session, today, custom ranges) and end-of-workday **workday** auto export

No accounts. No cloud. No telemetry.

---

## Features (Non‑Negotiables)

- Offline only at runtime (no network calls, no telemetry).
- Localhost only (`127.0.0.1`).
- Minimal UI with status + Start/Stop + scheduler controls.
- Mixed language transcription (German + English).
- Diarization labels **Hugo** or **Leon**; Auto mode can emit **Unknown** when below confidence.
- Work-hours scheduling with manual override.
- Raw audio deleted after successful transcription.
- Range exports: last 30m, last 60m, today (Europe/Berlin), session, custom.
- End-of-workday **workday** auto export (deduped).
- Output format: classic dialogue lines.

---

## Quick Start (Dev)

Recommended Python runtime for `pipeline_local`: `3.11`.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run.py
```

Open: `http://127.0.0.1:7070`

### Windows Quick Start (Dev)

```bat
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python run.py
```

---

## Configuration

Edit `config.yaml`.

Key sections:
- `asr.backend`: `auto|windows_nemo_parakeet_cuda|mac_parakeet_mlx|whisper_cpp|pipeline_local`
- `asr.required_languages`: defaults to `["de","en"]`
- `work_hours`: timezone and schedule
- `audio.segment_seconds`: 30 or 60 recommended

### Backend Selection Behavior

1. If `asr.backend` is explicit, it is attempted first.
2. Auto-selection:
   - Windows: NeMo Parakeet on CUDA if available.
   - macOS Apple Silicon: Parakeet MLX if available.
   - Otherwise: `whisper.cpp`.
3. If the selected backend does **not** support German+English, it falls back to `whisper.cpp`.

---

## Speaker Enrollment

Go to `http://127.0.0.1:7070/setup` and enroll Hugo + Leon (20–40 seconds each).

Embeddings are stored in SQLite and used for diarization.

### Enrollment UX Notes

- While enrolling, Setup shows a clear **STOP RECORDING** button plus an enrollment timer + countdown.
- References can be locked only after both speakers are enrolled (prevents accidental swaps).

### Speaker Lock + Confidence Gate (New)

- UI toggle on `/`: **Auto / Hugo only / Leon only**
- Auto mode uses a confidence gate and will label **Unknown** if thresholds are not met.
- Auto mode can require both speakers to be enrolled (configurable).

Relevant config (see `config.yaml`):

- `diarization.max_distance`
- `diarization.min_margin`
- `diarization.require_both_enrolled`
- `diarization.enrollment_min_snr_db`

---

## Offline Bundle (Airgapped Deployment)

On a connected machine:

```bash
python tools/offline_bundle/build_bundle.py --output offline_bundle
```

To include dev/test wheels:

```bash
python tools/offline_bundle/build_bundle.py --output offline_bundle --include-dev
```

This produces:

```
offline_bundle/
  app_source/    # full source tree
  wheelhouse/    # vendored Python wheels
  models/        # pinned model weights
```

Copy `offline_bundle` to the airgapped Mac Studio.

On the Mac Studio:

```bash
cd offline_bundle/app_source
cp -R ../wheelhouse .
cp -R ../models .
```

Then run the macOS installer (see below).

For airgapped Parakeet MLX:
- Pre-populate the Hugging Face cache into `models/` and set:
  - `asr.mac_parakeet_mlx.hf_cache_dir`
  - `asr.mac_parakeet_mlx.hf_cache_dir_parent`
  - `asr.mac_parakeet_mlx.hf_offline: true`

---

## Installers

### Windows (double click)

From File Explorer, double click:

```
tools/windows_installer/install.bat
```

This:
- Creates a local venv
- Installs dependencies (uses `wheelhouse/` if present)
- Sets a Scheduled Task for auto-start on login
- Launches the app immediately

### macOS (LaunchAgent)

```bash
chmod +x tools/macos_installer/install.sh
chmod +x tools/macos_installer/run_app.sh
tools/macos_installer/install.sh
```

This installs a LaunchAgent at login and starts the server.

### macOS .app Launcher

Double click `tools/macos_installer/HugoLeon Launcher.app` to start the LaunchAgent (if needed) and open the UI.
Keep the app inside the project folder so it can find `run.py`.

---

## Admin (Failed Segments)

Open `http://127.0.0.1:7070/admin` to review failed segments, retry, export raw audio ZIPs, or delete.

---

## UI Pages

- `/` Overview: recorder controls, scheduler state, transcript preview, exports.
- `/setup` Setup: enrollment + reference lock.
- `/mic` Mic: input device chooser, optional live meter, short test recording.
- `/admin` Admin: failed segment triage + exports.

---

## Data Layout

```
data/
  app.db
  logs/app.log
  exports/
  audio/        # segments are deleted after transcription
```

---

## Tests

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt -r requirements-dev.txt
python -m pytest
```

---

## Notes on ASR Backends

- `windows_nemo_parakeet_cuda`: requires NVIDIA GPU + NeMo ASR.
- `mac_parakeet_mlx`: requires MLX + `parakeet_mlx` Python module (configure in `config.yaml`).
- `whisper_cpp`: requires a local `ggml` model file + `whisper.cpp` binary or `whispercpp` Python package.
- `pipeline_local`: VAD + language-id + ASR + diarization pipeline (silero + fastText + Vosk/wav2vec2 + pyannote), fully local when models are pre-bundled.

You can override any backend in `config.yaml`.
