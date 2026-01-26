# atlas-monitor Offline Recorder + Diarized Transcription

Fully offline, localhost-only recorder with incremental transcription and speaker diarization into **Hugo** and **Leon**.

## Features (non-negotiables)
- Offline only at runtime (no network calls, no telemetry).
- Localhost only (`127.0.0.1`).
- Minimal UI with status + Start/Stop.
- Mixed language transcription (German + English).
- Diarization always labels **Hugo** or **Leon**.
- Work-hours scheduling with manual override.
- Raw audio deleted after successful transcription.
- Range exports: last 30m, last 60m, today (Europe/Berlin), session, custom.
- Output format: classic dialogue lines.

## Quick start (dev)
```bash
python -m venv .venv
.venv\\Scripts\\pip install -r requirements.txt
.venv\\Scripts\\python run.py
```

Open: `http://127.0.0.1:7070`

## Configuration
Edit `config.yaml`.

Key sections:
- `asr.backend`: `auto|windows_nemo_parakeet_cuda|mac_parakeet_mlx|whisper_cpp`
- `asr.required_languages`: defaults to `["de","en"]`
- `work_hours`: timezone and schedule
- `audio.segment_seconds`: 30 or 60 recommended

### Backend selection behavior
1. If `asr.backend` is explicit, it is attempted first.
2. Auto-selection:
   - Windows: NeMo Parakeet on CUDA if available.
   - macOS Apple Silicon: Parakeet MLX if available.
   - Otherwise: `whisper.cpp`.
3. If the selected backend does **not** support German+English, it falls back to `whisper.cpp`.

## Speaker enrollment
Go to `http://127.0.0.1:7070/setup` and enroll Hugo + Leon (20-40 seconds each).

Embeddings are stored in SQLite and used for diarization.

## Offline bundle (airgapped deployment)
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

## Windows installer (double click)
From File Explorer, double click:
```
tools/windows_installer/install.bat
```

This:
- Creates a local venv
- Installs dependencies (uses `wheelhouse/` if present)
- Sets a Scheduled Task for auto-start on login
- Launches the app immediately

## macOS installer (LaunchAgent)
```bash
chmod +x tools/macos_installer/install.sh
chmod +x tools/macos_installer/run_app.sh
tools/macos_installer/install.sh
```

This installs a LaunchAgent at login and starts the server.

## macOS .app launcher
Double click `tools/macos_installer/HugoLeon Launcher.app` to start the LaunchAgent (if needed) and open the UI.
Keep the app inside the project folder so it can find `run.py`.

## Admin (failed segments)
Open `http://127.0.0.1:7070/admin` to review failed segments, retry, export raw audio ZIPs, or delete.

## Data layout
```
data/
  app.db
  logs/app.log
  exports/
  audio/        # segments are deleted after transcription
```

## Tests
```bash
python -m venv .venv
.venv\\Scripts\\pip install -r requirements.txt -r requirements-dev.txt
python -m pytest
```

## Notes on ASR backends
- `windows_nemo_parakeet_cuda`: requires NVIDIA GPU + NeMo ASR.
- `mac_parakeet_mlx`: requires MLX + `parakeet_mlx` Python module (configure in `config.yaml`).
- `whisper_cpp`: requires a local `ggml` model file + `whisper.cpp` binary or `whispercpp` Python package.

You can override any backend in `config.yaml`.
