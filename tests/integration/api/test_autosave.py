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


@pytest.mark.parametrize("path, kwargs", [
    ("model/predict", {"json": {"inputs": [{"x1": 0.5, "x2": 0.5}]}}),
    ("visualizations/contour", {"json": {"x_var": "x1", "y_var": "x2", "grid_resolution": 10}}),
    ("acquisition/find-optimum", {"json": {"goal": "maximize"}}),
    ("experiments/preview", {"files": {"file": ("pts.csv", b"x1,x2,Output\n0.1,0.2,1.0\n", "text/csv")}}),
    ("optimal-design/info", {"json": {"model_type": "linear"}}),
])
def test_read_only_routes_do_not_save(sid, monkeypatch, path, kwargs):
    _add_points(sid)
    assert client.post(f"/api/v1/sessions/{sid}/model/train",
                       json={"backend": "sklearn", "kernel": "Matern"}).status_code == 200
    calls = []
    monkeypatch.setattr(session_store, "persist_session_to_disk", lambda s: calls.append(s) or True)
    r = client.post(f"/api/v1/sessions/{sid}/{path}", **kwargs)
    assert r.status_code == 200, r.text
    assert calls == []


def test_browsers_may_read_the_saved_header():
    """X-ALchemist-Saved is not CORS-safelisted, so a browser client cannot
    read it unless the server exposes it."""
    r = client.get("/health", headers={"Origin": "http://localhost:5173"})
    exposed = [h.strip().lower() for h in r.headers.get("access-control-expose-headers", "").split(",")]
    assert "x-alchemist-saved" in exposed


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


def test_upload_logs_a_failed_write_and_still_creates_the_session(sid, monkeypatch, caplog):
    """The uploaded session exists in memory, so the 201 stands; the failed
    write is logged so it is not silent."""
    exported = client.get(f"/api/v1/sessions/{sid}/download")
    assert exported.status_code == 200, exported.text
    monkeypatch.setattr(session_store, "persist_session_to_disk", lambda _sid: False)
    with caplog.at_level("ERROR", logger="api.routers.sessions"):
        r = client.post("/api/v1/sessions/upload",
                        files={"file": ("s.json", exported.content, "application/json")})
    assert r.status_code == 201, r.text
    new_sid = r.json()["session_id"]
    try:
        assert any(new_sid in rec.getMessage() and rec.levelname == "ERROR"
                   for rec in caplog.records), caplog.text
    finally:
        client.delete(f"/api/v1/sessions/{new_sid}")


# --- a reloaded or imported session keeps its model ---------------------------
#
# The server's loaders rebuild the model from the saved training call. Until
# 0.4.0 the A16 bug made them retrain by accident; with it fixed they must ask.

def _train_botorch(sid):
    _add_points(sid, 8)
    r = client.post(f"/api/v1/sessions/{sid}/model/train",
                    json={"backend": "botorch", "kernel": "Matern",
                          "input_transform": "normalize", "output_transform": "standardize"})
    assert r.status_code == 200, r.text


def test_a_restarted_server_has_the_sessions_model_with_its_transforms(sid):
    _train_botorch(sid)
    restarted = _restarted(sid)
    assert restarted.model is not None
    config = restarted.get_last_training_config()
    assert config["backend"] == "botorch"
    assert config["input_transform_type"] == "normalize"
    assert config["output_transform_type"] == "standardize"
    assert restarted.model.input_transform_type == "normalize"
    assert restarted.model.output_transform_type == "standardize"


def test_a_saved_training_call_that_cannot_be_replayed_still_loads(sid, monkeypatch):
    """Regression guard for the retrain failure path (it passed before the
    loaders retrained, too; it pins that retraining on load cannot lose a
    session). A failed retrain is logged by the loader; the session comes back
    without a model rather than not at all."""
    from alchemist_core.session import OptimizationSession

    _train_botorch(sid)

    def boom(self, *args, **kwargs):
        raise RuntimeError("cannot fit")

    monkeypatch.setattr(OptimizationSession, "train_model", boom)
    restarted = _restarted(sid)
    assert restarted is not None
    assert restarted.model is None
    assert len(restarted.experiment_manager.get_data()) == 8


def test_a_session_with_no_data_to_retrain_on_still_loads(sid):
    """Regression guard for the retrain failure path, not a test of the change
    (it passed before the loaders retrained). The saved training call cannot be
    replayed without data; the session loads, without a model."""
    _train_botorch(sid)
    path = session_store.persist_dir / f"{sid}.json"
    data = json.loads(path.read_text())
    data["experiments"]["data"] = []
    data["experiments"]["n_total"] = 0
    path.write_text(json.dumps(data))
    restarted = _restarted(sid)
    assert restarted is not None
    assert restarted.model is None
    assert len(restarted.experiment_manager.get_data()) == 0


def test_store_import_gives_a_fitted_model(sid):
    _train_botorch(sid)
    exported = session_store.export_session(sid)
    new_sid = session_store.import_session(exported)
    try:
        session = session_store.get(new_sid)
        assert session.model is not None
        assert session.get_last_training_config()["input_transform_type"] == "normalize"
    finally:
        session_store.delete(new_sid)


def test_upload_gives_a_fitted_model(sid):
    _train_botorch(sid)
    exported = client.get(f"/api/v1/sessions/{sid}/download")
    r = client.post("/api/v1/sessions/upload",
                    files={"file": ("s.json", exported.content, "application/json")})
    assert r.status_code == 201, r.text
    new_sid = r.json()["session_id"]
    try:
        assert session_store.get(new_sid).model is not None
        assert _restarted(new_sid).model is not None
    finally:
        client.delete(f"/api/v1/sessions/{new_sid}")


def test_deleting_an_unloaded_session_whose_file_cannot_be_removed_is_a_500(sid, monkeypatch):
    """A file the OS will not unlink (a Windows scanner holding it) is a
    server error, not a 404 that tells the client the session is gone."""
    session_store._sessions.pop(sid)
    from pathlib import Path
    real = Path.unlink

    def unlink(self, *args, **kwargs):
        if self.name == f"{sid}.json":
            raise PermissionError("held open")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda _s: None)
    with TestClient(app, raise_server_exceptions=False) as c:
        assert c.delete(f"/api/v1/sessions/{sid}").status_code == 500


def test_a_failed_upload_leaves_no_temp_file(tmp_path, monkeypatch):
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    r = client.post("/api/v1/sessions/upload",
                    files={"file": ("s.json", b"{not json", "application/json")})
    assert r.status_code == 400
    assert list(tmp_path.iterdir()) == []
