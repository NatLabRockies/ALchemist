"""Every change a request makes to a session reaches disk.

Until 0.4.0 the server wrote a session file only on create, import,
lock/unlock and ``POST /save``. Experiments, queue transitions, trained models
and suggestions made in between lived in memory only, and a restart or crash
lost them; a lab controller had to call ``/save`` after every write to be
safe. The server now saves the session after every successful
POST/PUT/PATCH/DELETE under ``/api/v1/sessions/{id}/...``, except the routes
that only read it.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.services.session_store import SessionStore, session_store

client = TestClient(app)


def _on_disk(sid):
    return json.loads((session_store.persist_dir / f"{sid}.json").read_text())


def _restarted(sid):
    """The session as a freshly started server would load it."""
    return SessionStore(persist_dir=str(session_store.persist_dir)).get(sid)


@pytest.fixture
def sid():
    r = client.post("/api/v1/sessions", json={})
    assert r.status_code == 201
    sid = r.json()["session_id"]
    for name in ("x1", "x2"):
        r = client.post(f"/api/v1/sessions/{sid}/variables",
                        json={"name": name, "type": "real", "min": 0, "max": 1})
        assert r.status_code == 200, r.text
    yield sid
    client.delete(f"/api/v1/sessions/{sid}")


def _add_points(sid, n=6):
    for i in range(n):
        x1, x2 = (i + 0.5) / n, ((i * 3) % n + 0.5) / n
        r = client.post(f"/api/v1/sessions/{sid}/experiments",
                        json={"inputs": {"x1": x1, "x2": x2},
                              "output": -(x1 - 0.3) ** 2 - (x2 - 0.6) ** 2})
        assert r.status_code == 200, r.text


def test_a_variable_reaches_disk(sid):
    names = [v["name"] for v in _on_disk(sid)["search_space"]["variables"]]
    assert names == ["x1", "x2"]


def test_an_experiment_survives_a_restart(sid):
    _add_points(sid, 1)
    assert len(_restarted(sid).experiment_manager.get_data()) == 1


def test_queue_transitions_survive_a_restart(sid):
    r = client.post(f"/api/v1/sessions/{sid}/experiments/queue",
                    json={"items": [{"inputs": {"x1": 0.2, "x2": 0.4}}]})
    assert r.status_code in (200, 201), r.text
    item_id = _restarted(sid).queue.list()[0].id

    r = client.post(f"/api/v1/sessions/{sid}/experiments/queue/{item_id}/start")
    assert r.status_code == 200, r.text
    assert _restarted(sid).queue.list()[0].status == "running"

    r = client.post(f"/api/v1/sessions/{sid}/experiments/queue/{item_id}/complete",
                    json={"outputs": [1.5]})
    assert r.status_code == 200, r.text
    restarted = _restarted(sid)
    assert restarted.queue.list()[0].status == "done"
    assert len(restarted.experiment_manager.get_data()) == 1
    assert [p["id"] for p in restarted.get_provenance()] == [item_id]


def test_a_training_config_survives_a_restart(sid):
    _add_points(sid)
    r = client.post(f"/api/v1/sessions/{sid}/model/train",
                    json={"backend": "sklearn", "kernel": "Matern"})
    assert r.status_code == 200, r.text
    training = _on_disk(sid)["model_config"]["training_config"]
    assert training["backend"] == "sklearn"


def test_the_response_says_it_was_saved(sid):
    r = client.post(f"/api/v1/sessions/{sid}/variables",
                    json={"name": "x3", "type": "real", "min": 0, "max": 1})
    assert r.headers["X-ALchemist-Saved"] == "true"


def test_a_failed_save_keeps_the_change_and_says_so(sid, monkeypatch):
    """The change happened in memory; a 500 would invite a retry that applies
    it twice. The response stays a success and carries the failure."""
    monkeypatch.setattr(session_store, "persist_session_to_disk", lambda _sid: False)
    r = client.post(f"/api/v1/sessions/{sid}/variables",
                    json={"name": "x3", "type": "real", "min": 0, "max": 1})
    assert r.status_code == 200
    assert r.headers["X-ALchemist-Saved"] == "false"


def test_a_rejected_request_does_not_save(sid, monkeypatch):
    calls = []
    monkeypatch.setattr(session_store, "persist_session_to_disk", lambda s: calls.append(s) or True)
    r = client.post(f"/api/v1/sessions/{sid}/variables",
                    json={"name": "x1", "type": "real", "min": 0, "max": 1})
    assert r.status_code >= 400
    assert calls == []
    assert "X-ALchemist-Saved" not in r.headers


@pytest.mark.parametrize("path, body", [
    ("model/predict", {"inputs": [{"x1": 0.5, "x2": 0.5}]}),
    ("visualizations/contour", {"x_var": "x1", "y_var": "x2", "grid_resolution": 10}),
    ("acquisition/find-optimum", {"goal": "maximize"}),
])
def test_read_only_routes_do_not_save(sid, monkeypatch, path, body):
    _add_points(sid)
    assert client.post(f"/api/v1/sessions/{sid}/model/train",
                       json={"backend": "sklearn", "kernel": "Matern"}).status_code == 200
    calls = []
    monkeypatch.setattr(session_store, "persist_session_to_disk", lambda s: calls.append(s) or True)
    r = client.post(f"/api/v1/sessions/{sid}/{path}", json=body)
    assert r.status_code == 200, r.text
    assert calls == []


def test_deleting_a_session_does_not_write_it_back(sid):
    assert client.delete(f"/api/v1/sessions/{sid}").status_code == 204
    assert not (session_store.persist_dir / f"{sid}.json").exists()


def test_explicit_save_reports_a_failed_write(sid, monkeypatch):
    monkeypatch.setattr(session_store, "_save_to_disk",
                        lambda _sid: (_ for _ in ()).throw(OSError("disk full")))
    r = client.post(f"/api/v1/sessions/{sid}/save")
    assert r.status_code == 500


def test_explicit_save_of_an_unknown_session_is_404():
    assert client.post("/api/v1/sessions/no-such-session/save").status_code == 404
