"""A session's first load after a restart retrains its model; that must not
run on the server's event loop.

With ``retrain_on_load=True`` the first request for an unloaded session ID can
take seconds. An ``async def`` route that calls the store directly runs that
load on the loop thread, and a second request for the same ID then blocks the
loop on the per-ID lock, so every request stalls, ``/health`` included.

The test makes the first load of an ID block on an event and times ``/health``
while it does. ``with TestClient(app)`` shares one event loop between
requests, which is what makes a blocked loop visible.
"""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

import alchemist_core.session as core_session
from api.main import app
from api.services.session_store import session_store

# Safety net: if the loop is blocked, release the load after this long so the
# test fails instead of hanging.
_RELEASE_AFTER = 4.0
_HEALTH_DEADLINE = 1.5


@pytest.fixture
def api():
    with TestClient(app) as client:
        yield client


def _new_unloaded_session(api):
    sid = api.post("/api/v1/sessions", json={}).json()["session_id"]
    assert api.post(f"/api/v1/sessions/{sid}/variables",
                    json={"name": "x1", "type": "real", "min": 0, "max": 1}).status_code == 200
    session_store._sessions.pop(sid)  # as after a restart: the file remains
    return sid


@pytest.mark.parametrize("method, path", [
    ("get", "/api/v1/sessions/{sid}"),
    ("get", "/api/v1/sessions/{sid}/export"),
    ("post", "/api/v1/sessions/{sid}/backup"),
    ("post", "/api/v1/sessions/{sid}/lock"),
    ("get", "/api/v1/sessions/{sid}/lock"),
    ("delete", "/api/v1/sessions/{sid}/lock"),
])
def test_first_load_of_a_session_does_not_block_the_event_loop(api, monkeypatch, method, path):
    sid = _new_unloaded_session(api)
    entered, release = threading.Event(), threading.Event()
    real = core_session.OptimizationSession.load_session

    def slow_load(filepath, *args, **kwargs):
        entered.set()
        release.wait(_RELEASE_AFTER + 10)
        return real(filepath, *args, **kwargs)

    monkeypatch.setattr(core_session.OptimizationSession, "load_session", staticmethod(slow_load))
    timer = threading.Timer(_RELEASE_AFTER, release.set)
    timer.start()
    body = {"json": {"locked_by": "test"}} if method == "post" and path.endswith("/lock") else {}
    result = {}
    worker = threading.Thread(
        target=lambda: result.update(r=getattr(api, method)(path.format(sid=sid), **body)))
    worker.start()
    try:
        assert entered.wait(10), "the request never started loading the session"
        start = time.monotonic()
        health = api.get("/health")
        elapsed = time.monotonic() - start
        assert health.status_code == 200
        assert elapsed < _HEALTH_DEADLINE, (
            f"/health took {elapsed:.1f}s while {method.upper()} {path} was loading a session")
    finally:
        release.set()
        timer.cancel()
        worker.join(30)
        session_store.delete(sid)
    assert result["r"].status_code in (200, 201, 403, 404), result["r"].text
