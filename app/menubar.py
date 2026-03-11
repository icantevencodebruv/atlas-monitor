#!/usr/bin/env python3
"""
Atlas Monitor menu bar app (macOS only).

Launched by tools/macos_installer/launch.sh via exec, making this process
the direct child of launchd. Manages the backend lifecycle from the menu bar.

Title icons:
  ▲  backend responding
  ▽  backend down
  ↺  restarting
"""

import os
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime

import rumps

APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = sys.executable
PID_FILE = os.path.expanduser("~/.atlas-monitor/atlas.pid")
LOG_DIR = os.path.expanduser("~/.atlas-monitor/logs")
PORT = 7070  # fallback; overridden by config.yaml below

try:
    import yaml
    with open(os.path.join(APP_ROOT, "config.yaml")) as _f:
        _cfg = yaml.safe_load(_f)
    PORT = _cfg.get("app", {}).get("port", 7070)
except Exception:
    pass

_STATUS_URL = f"http://127.0.0.1:{PORT}/status"


def _backend_alive() -> bool:
    try:
        urllib.request.urlopen(_STATUS_URL, timeout=1)
        return True
    except Exception:
        return False


class AtlasApp(rumps.App):
    def __init__(self):
        super().__init__("Atlas", title="\u25b2")  # ▲
        self.menu = ["Open Atlas", "Restart", "View Logs", None, "Quit"]
        # Periodic health check every 5 seconds
        self._timer = rumps.Timer(self._health_tick, 5)
        self._timer.start()

    # ── health check ─────────────────────────────────────────────────────────

    def _health_tick(self, _):
        if self.title == "\u21ba":  # ↺ — restart in progress, don't interfere
            return
        self.title = "\u25b2" if _backend_alive() else "\u25bd"

    # ── helpers ───────────────────────────────────────────────────────────────

    def _read_pid(self):
        try:
            return int(open(PID_FILE).read().strip())
        except Exception:
            return None

    def _kill_backend(self):
        pid = self._read_pid()
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            os.remove(PID_FILE)
        except FileNotFoundError:
            pass

    def _start_backend(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        log_file = os.path.join(LOG_DIR, f"atlas-{datetime.now().strftime('%Y%m%d')}.log")
        proc = subprocess.Popen(
            [PYTHON, "run.py"],
            cwd=APP_ROOT,
            stdout=open(log_file, "a"),
            stderr=subprocess.STDOUT,
        )
        with open(PID_FILE, "w") as f:
            f.write(str(proc.pid))

    def _wait_backend(self, timeout=15) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            if _backend_alive():
                return True
            time.sleep(0.5)
        return False

    # ── menu actions ──────────────────────────────────────────────────────────

    @rumps.clicked("Open Atlas")
    def open_atlas(self, _):
        if not _backend_alive():
            # Backend is down — restart it first, then open
            self.title = "\u21ba"  # ↺
            self._kill_backend()
            self._start_backend()
            ok = self._wait_backend()
            self.title = "\u25b2" if ok else "\u25bd"
            if not ok:
                return
        os.system(f"open http://127.0.0.1:{PORT}")

    @rumps.clicked("Restart")
    def restart(self, _):
        self.title = "\u21ba"  # ↺
        self._kill_backend()
        time.sleep(0.5)
        self._start_backend()
        ok = self._wait_backend()
        self.title = "\u25b2" if ok else "\u25bd"

    @rumps.clicked("View Logs")
    def view_logs(self, _):
        os.makedirs(LOG_DIR, exist_ok=True)
        os.system(f"open '{LOG_DIR}'")

    @rumps.clicked("Quit")
    def quit_app(self, _):
        self._kill_backend()
        rumps.quit_application()


if __name__ == "__main__":
    AtlasApp().run()
