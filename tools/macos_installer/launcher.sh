#!/bin/bash
set -euo pipefail

APP_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PLIST="$HOME/Library/LaunchAgents/com.hugoleon.recorder.plist"

if ! launchctl list | grep -q "com.hugoleon.recorder"; then
  if [ -f "$PLIST" ]; then
    launchctl load "$PLIST" >/dev/null 2>&1 || true
  fi
fi

open "http://127.0.0.1:7070"
