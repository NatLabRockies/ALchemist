"""A suggestion or an auto-retrain must not run on the server's event loop.

``POST /acquisition/suggest`` called ``session.suggest_next`` inline in an
``async def`` route, and the auto-train after a recorded result called
``train_model`` the same way. While a qEI or a retrain ran, every other
request stalled: ``/health``, the web app's 30-s backup, a controller's
polling. The test blocks the slow call on an event and times ``/health``
meanwhile; see ``test_event_loop_not_blocked.py`` for the pattern.
"""

from __future__ import annotations

import threading
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import alchemist_core.session as core_session
from api.main import app
from api.services.session_store import session_store

_RELEASE_AFTER = 4.0
_HEALTH_DEADLINE = 1.5


@pytest.fixture
def api():
    with TestClient(app) as client:
        yield client


@pytest.fixture
def sid(api):
    sid = api.post("/api/v1/sessions", json={}).json()["session_id"]
    assert api.post(f"/api/v1/sessions/{sid}/variables",
                    json={"name": "x", "type": "real", "min": 0, "max": 1}).status_code == 200
    rows = [{"inputs": {"x": i / 10}, "output": float(i)} for i in range(6)]
    assert api.post(f"/api/v1/sessions/{sid}/experiments/batch",
                    json={"experiments": rows}).status_code == 200
    yield sid
    session_store.delete(sid)


def _health_while_blocked(api, monkeypatch, attr, slow, method, path, body):
    entered, release = threading.Event(), threading.Event()

    def blocking(self, *args, **kwargs):
        entered.set()
        release.wait(_RELEASE_AFTER + 10)
        return slow(self, *args, **kwargs)

    monkeypatch.setattr(core_session.OptimizationSession, attr, blocking)
    timer = threading.Timer(_RELEASE_AFTER, release.set)
    timer.start()
    result = {}
    worker = threading.Thread(target=lambda: result.update(r=getattr(api, method)(path, json=body)))
    worker.start()
    try:
        assert entered.wait(10), f"{method.upper()} {path} never reached {attr}"
        start = time.monotonic()
        health = api.get("/health")
        elapsed = time.monotonic() - start
        assert health.status_code == 200
        assert elapsed < _HEALTH_DEADLINE, (
            f"/health took {elapsed:.1f}s while {method.upper()} {path} was in {attr}")
    finally:
        release.set()
        timer.cancel()
        worker.join(30)
    return result["r"]


def test_a_slow_suggest_does_not_block_other_requests(api, sid, monkeypatch):
    session = session_store.get(sid)
    session.model = object()  # the route only checks that one exists

    def fake_suggest(self, *args, **kwargs):
        return pd.DataFrame([{"x": 0.5}])

    r = _health_while_blocked(api, monkeypatch, "suggest_next", fake_suggest, "post",
                              f"/api/v1/sessions/{sid}/acquisition/suggest",
                              {"strategy": "EI", "goal": "maximize", "n_suggestions": 1})
    assert r.status_code == 200, r.text
    assert r.json()["suggestions"] == [{"x": 0.5}]


def test_a_slow_auto_train_does_not_block_other_requests(api, sid, monkeypatch):
    def fake_train(self, *args, **kwargs):
        return {"metrics": {"rmse": 0.1, "r2": 0.9}, "hyperparameters": {}}

    r = _health_while_blocked(api, monkeypatch, "train_model", fake_train, "post",
                              f"/api/v1/sessions/{sid}/experiments?auto_train=true",
                              {"inputs": {"x": 0.7}, "output": 7.0})
    assert r.status_code == 200, r.text
    assert r.json()["model_trained"] is True
