#!/bin/bash
# Atlas Monitor startup orchestrator.
# Run directly or via the .app bundle / launchd plist.
# Uses exec at the end so the menu bar app inherits the launchd job lifecycle.
set -euo pipefail

APP_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$APP_ROOT/.venv"
PYTHON="$VENV/bin/python"
ATLAS_DIR="$HOME/.atlas-monitor"
PID_FILE="$ATLAS_DIR/atlas.pid"
LOG_DIR="$ATLAS_DIR/logs"
LOG_FILE="$LOG_DIR/atlas-$(date +%Y%m%d).log"

# Read port from config.yaml (avoid Python startup cost in the health poll loop)
PORT=$(grep "^  port:" "$APP_ROOT/config.yaml" 2>/dev/null | awk '{print $2}' || echo 7070)

# ── 1. Already-running guard ──────────────────────────────────────────────────
if [ -f "$PID_FILE" ]; then
  EXISTING_PID=$(cat "$PID_FILE")
  if kill -0 "$EXISTING_PID" 2>/dev/null; then
    open "http://127.0.0.1:$PORT"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

if lsof -i ":$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
  open "http://127.0.0.1:$PORT"
  exit 0
fi

# ── 2. Venv check ────────────────────────────────────────────────────────────
if [ ! -x "$PYTHON" ]; then
  osascript -e 'display notification "Python venv missing. Run install.sh first." with title "Atlas Monitor" subtitle "Startup failed"'
  exit 1
fi

# ── 3. QA model pre-flight (auto-download if enabled and missing) ─────────────
QA_MODEL="$APP_ROOT/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
# Check if llm_qa.enabled is true in config
QA_ENABLED=$(python3 -c "
import yaml, sys
try:
    cfg = yaml.safe_load(open('$APP_ROOT/config.yaml'))
    print('yes' if cfg.get('llm_qa', {}).get('enabled') else '')
except Exception:
    print('')
" 2>/dev/null || true)

if [ -n "$QA_ENABLED" ] && [ ! -f "$QA_MODEL" ]; then
  bash "$APP_ROOT/tools/download_qa_model.sh"
fi

# ── 4. Log dir + rotation (keep last 7 days) ──────────────────────────────────
mkdir -p "$LOG_DIR"
find "$LOG_DIR" -name "atlas-*.log" -mtime +7 -delete 2>/dev/null || true

# ── 5. Start backend ──────────────────────────────────────────────────────────
cd "$APP_ROOT"
"$PYTHON" run.py >> "$LOG_FILE" 2>&1 &
BACKEND_PID=$!
mkdir -p "$ATLAS_DIR"
echo "$BACKEND_PID" > "$PID_FILE"

# ── 6. Health poll (500 ms interval, 15 s timeout) ────────────────────────────
TIMEOUT=15
START=$(date +%s)
until curl -sf "http://127.0.0.1:$PORT/status" >/dev/null 2>&1; do
  if (( $(date +%s) - START >= TIMEOUT )); then
    osascript -e 'display notification "Backend failed to start. Check logs in ~/.atlas-monitor/logs/" with title "Atlas Monitor" subtitle "Startup failed"'
    exit 1
  fi
  sleep 0.5
done

open "http://127.0.0.1:$PORT"

# ── 7. Start menu bar app (exec replaces this shell as launchd child) ─────────
exec "$PYTHON" "$APP_ROOT/app/menubar.py"
