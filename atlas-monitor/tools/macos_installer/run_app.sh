#!/bin/bash
set -euo pipefail

APP_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$APP_ROOT/.venv"
PYTHON="$VENV/bin/python"

if [ ! -x "$PYTHON" ]; then
  echo "Python venv missing. Run install.sh first."
  exit 1
fi

exec "$PYTHON" "$APP_ROOT/run.py"
