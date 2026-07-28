import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


@pytest.fixture
def sid():
    r = client.post("/api/v1/sessions", json={"ttl_hours": 1})
    r.raise_for_status()
    s = r.json()["session_id"]
    client.post(f"/api/v1/sessions/{s}/variables",
                json={"name": "x", "type": "real", "min": 0.0, "max": 10.0})
    # seed a couple experiments so training is possible
    client.post(f"/api/v1/sessions/{s}/experiments",
                json={"inputs": {"x": 1.0}, "output": 2.0})
    client.post(f"/api/v1/sessions/{s}/experiments",
                json={"inputs": {"x": 3.0}, "output": 4.0})
    yield s
    client.delete(f"/api/v1/sessions/{s}")


def test_train_logs_config_changed(sid):
    r = client.post(f"/api/v1/sessions/{sid}/model/train",
                    json={"backend": "sklearn", "kernel": "Matern"})
    r.raise_for_status()
    g = client.get(f"/api/v1/sessions/{sid}/audit/config-changes")
    g.raise_for_status()
    changes = g.json()["changes"]
    model_changes = [c for c in changes if c["component"] == "model"]
    assert len(model_changes) >= 1
    assert model_changes[-1]["new"]["kernel"] == "Matern"
    assert model_changes[-1]["new"]["backend"] == "sklearn"
    assert model_changes[-1]["timestamp"]


def test_suggest_logs_config_changed_and_endpoint_shape(sid):
    client.post(f"/api/v1/sessions/{sid}/model/train",
                json={"backend": "sklearn", "kernel": "Matern"})
    r = client.post(f"/api/v1/sessions/{sid}/acquisition/suggest",
                    json={"strategy": "EI", "goal": "maximize", "n_suggestions": 1})
    r.raise_for_status()
    g = client.get(f"/api/v1/sessions/{sid}/audit/config-changes")
    g.raise_for_status()
    body = g.json()
    assert "changes" in body
    acq = [c for c in body["changes"] if c["component"] == "acquisition"]
    assert len(acq) >= 1
    assert acq[-1]["new"]["strategy"] == "EI"
    # every entry has the required shape
    for c in body["changes"]:
        assert set(["timestamp", "component", "old", "new"]).issubset(c.keys())


def test_config_changes_empty_for_new_session(sid):
    g = client.get(f"/api/v1/sessions/{sid}/audit/config-changes")
    g.raise_for_status()
    assert g.json() == {"changes": []}


def test_objective_label_change_still_logs_separately(sid):
    client.put(f"/api/v1/sessions/{sid}/objective-metadata",
               json={"metadata": {"Output": {"label": "area_x", "unit": "a.u."}}})
    # objective_label_changed is a distinct type; must NOT appear in config-changes
    g = client.get(f"/api/v1/sessions/{sid}/audit/config-changes")
    g.raise_for_status()
    assert all(c["component"] != "objective_label" for c in g.json()["changes"])
