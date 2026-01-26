#!/bin/bash
set -euo pipefail

APP_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="$APP_ROOT/tools/macos_installer"
APP_NAME="HugoLeon Launcher.app"
APP_PATH="$OUT_DIR/$APP_NAME"
LAUNCHER="$APP_ROOT/tools/macos_installer/launcher.sh"

osacompile -o "$APP_PATH" -e "do shell script \"bash -lc '\\\"$LAUNCHER\\\"'\""
echo "Created $APP_PATH"
