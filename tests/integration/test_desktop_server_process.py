"""The Electron app's headless server, run as a real process: it serves, and it exits when stdin closes."""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / "packaging" / "desktop" / "alchemist_server.py"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(url: str, accept: str = "application/json"):
    req = urllib.request.Request(url, headers={"Accept": accept})
    with urllib.request.urlopen(req, timeout=2) as resp:
        return resp.read().decode()


def test_server_serves_then_exits_when_stdin_closes(tmp_path):
    port = _free_port()
    env = {**os.environ, "HOME": str(tmp_path), "LOCALAPPDATA": str(tmp_path), "XDG_DATA_HOME": str(tmp_path)}
    proc = subprocess.Popen(
        [sys.executable, str(SERVER), "--port", str(port), "--watch-stdin"],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
    )
    try:
        health = None
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and proc.poll() is None:
            try:
                health = json.loads(_get(f"http://127.0.0.1:{port}/health"))
                break
            except OSError:
                time.sleep(0.5)
        assert health is not None, "server never answered /health"
        assert health["service"] == "alchemist-api"

        if (ROOT / "alchemist-web" / "dist" / "index.html").is_file():
            assert '<div id="root">' in _get(f"http://127.0.0.1:{port}/", accept="text/html")

        # DNS rebinding: a page on another domain pointed at 127.0.0.1 must be refused.
        rebound = urllib.request.Request(f"http://127.0.0.1:{port}/health", headers={"Host": f"attacker.example:{port}"})
        try:
            urllib.request.urlopen(rebound, timeout=2)
            raise AssertionError("a request with a foreign Host header was served")
        except urllib.error.HTTPError as err:
            assert err.code == 400
        assert json.loads(_get(f"http://localhost:{port}/health"))["service"] == "alchemist-api"

        # Sessions go to the per-user data dir, not the current directory.
        assert any(p.parent.name == "ALchemist" for p in tmp_path.rglob("cache"))

        proc.stdin.close()
        assert proc.wait(timeout=15) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
