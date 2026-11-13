"""Unit tests for the Electron app's headless server entry point (packaging/desktop/alchemist_server.py)."""

import importlib.util
import os
import threading
from pathlib import Path

import pytest

SERVER_PATH = Path(__file__).resolve().parents[2] / "packaging" / "desktop" / "alchemist_server.py"


@pytest.fixture(scope="module")
def server():
    # packaging/ is not a package, so load the module by path.
    spec = importlib.util.spec_from_file_location("alchemist_server", SERVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_port_is_required(server):
    with pytest.raises(SystemExit):
        server.parse_args([])


def test_parses_port_and_watch_flag(server):
    ns = server.parse_args(["--port", "38765", "--watch-stdin"])
    assert ns.port == 38765 and ns.watch_stdin is True


def test_watch_stdin_defaults_off(server):
    assert server.parse_args(["--port", "38765"]).watch_stdin is False


def test_watch_stdin_calls_on_eof_once(server):
    r, w = os.pipe()
    called = threading.Event()
    count = []

    def on_eof():
        count.append(1)
        called.set()

    t = server.watch_stdin(os.fdopen(r, "rb"), on_eof)
    os.close(w)
    assert called.wait(5)
    t.join(5)
    assert not t.is_alive()
    assert count == [1]
