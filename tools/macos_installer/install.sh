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
chmod +x "$APP_ROOT/tools/macos_installer/launch.sh"
chmod +x "$APP_ROOT/tools/macos_installer/HugoLeon Launcher.app/Contents/MacOS/launcher"

# Migrate away from old label if present
OLD_PLIST="$HOME/Library/LaunchAgents/com.hugoleon.recorder.plist"
if [ -f "$OLD_PLIST" ]; then
  launchctl unload "$OLD_PLIST" >/dev/null 2>&1 || true
  rm -f "$OLD_PLIST"
fi

PLIST="$HOME/Library/LaunchAgents/com.atlas-monitor.app.plist"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.atlas-monitor.app</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>-l</string>
    <string>$APP_ROOT/tools/macos_installer/launch.sh</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <false/>
</dict>
</plist>
EOF

launchctl unload "$PLIST" >/dev/null 2>&1 || true
launchctl load "$PLIST"

echo "Installed LaunchAgent (com.atlas-monitor.app). Starting now..."
bash "$APP_ROOT/tools/macos_installer/launch.sh"

echo "Launcher app is at $APP_ROOT/tools/macos_installer/HugoLeon Launcher.app"
