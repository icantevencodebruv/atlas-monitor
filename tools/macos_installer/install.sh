#!/bin/bash
set -euo pipefail

APP_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$APP_ROOT/.venv"
WHEELHOUSE="$APP_ROOT/wheelhouse"

if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi

PIP="$VENV/bin/pip"
if [ -d "$WHEELHOUSE" ]; then
  "$PIP" install --no-index --find-links "$WHEELHOUSE" -r "$APP_ROOT/requirements.txt"
else
"$PIP" install -r "$APP_ROOT/requirements.txt"
fi

chmod +x "$APP_ROOT/tools/macos_installer/run_app.sh"
chmod +x "$APP_ROOT/tools/macos_installer/launcher.sh"
chmod +x "$APP_ROOT/tools/macos_installer/HugoLeon Launcher.app/Contents/MacOS/launcher"

PLIST="$HOME/Library/LaunchAgents/com.hugoleon.recorder.plist"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.hugoleon.recorder</string>
  <key>ProgramArguments</key>
  <array>
    <string>$APP_ROOT/tools/macos_installer/run_app.sh</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
</dict>
</plist>
EOF

launchctl unload "$PLIST" >/dev/null 2>&1 || true
launchctl load "$PLIST"

echo "Installed LaunchAgent. Starting now..."
"$APP_ROOT/tools/macos_installer/run_app.sh"

echo "Launcher app is at $APP_ROOT/tools/macos_installer/HugoLeon Launcher.app"
