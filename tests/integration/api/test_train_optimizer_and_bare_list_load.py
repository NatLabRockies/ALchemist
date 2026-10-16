"""`TrainModelRequest.optimizer` reaches the scikit-learn model (W42) and the
bare-list `/variables/load` is validated whole before the session changes (W36).
"""
import io
import json

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.services.session_store import session_store

client = TestClient(app)


@pytest.fixture
def sid():
    s = client.post("/api/v1/sessions", json={}).json()["session_id"]
    assert client.post(f"/api/v1/sessions/{s}/variables",
                       json={"name": "temperature", "type": "real", "min": 0, "max": 1}).status_code == 200
    yield s
    client.delete(f"/api/v1/sessions/{s}")


def _load(sid, payload):
    return client.post(f"/api/v1/sessions/{sid}/variables/load",
                       files={"file": ("variables.json", io.BytesIO(json.dumps(payload).encode()),
                                       "application/json")})


def _names(sid):
    return [v["name"] for v in client.get(f"/api/v1/sessions/{sid}/variables").json()["variables"]]


def test_sklearn_optimizer_is_passed_through(sid):
    rows = [{"inputs": {"temperature": i / 10}, "output": float(i)} for i in range(6)]
    assert client.post(f"/api/v1/sessions/{sid}/experiments/batch",
                       json={"experiments": rows}).status_code == 200
    r = client.post(f"/api/v1/sessions/{sid}/model/train",
                    json={"backend": "sklearn", "kernel": "RBF", "optimizer": "CG"})
    assert r.status_code == 200, r.text
    assert session_store.get(sid).model.optimizer == "CG"
    r = client.post(f"/api/v1/sessions/{sid}/model/train", json={"backend": "sklearn", "kernel": "RBF"})
    assert r.status_code == 200, r.text
    assert session_store.get(sid).model.optimizer == "L-BFGS-B"


def test_bare_list_load_with_a_duplicate_changes_nothing(sid):
    r = _load(sid, [
        {"name": "pressure", "type": "real", "min": 1, "max": 10},
        {"name": "temperature", "type": "real", "min": 0, "max": 1},  # already defined
        {"name": "catalyst", "type": "categorical", "categories": ["A", "B"]},
    ])
    assert r.status_code == 400, r.text
    assert "temperature" in r.json()["detail"]
    assert _names(sid) == ["temperature"]


def test_bare_list_load_with_a_bad_entry_changes_nothing(sid):
    r = _load(sid, [
        {"name": "pressure", "type": "real", "min": 1, "max": 10},
        {"name": "flow", "type": "real", "min": 1},  # no max
    ])
    assert r.status_code == 400, r.text
    assert _names(sid) == ["temperature"]


def test_a_good_bare_list_still_appends(sid):
    r = _load(sid, [{"name": "pressure", "type": "real", "min": 1, "max": 10},
                    {"name": "catalyst", "type": "categorical", "categories": ["A", "B"]}])
    assert r.status_code == 200, r.text
    assert _names(sid) == ["temperature", "pressure", "catalyst"]
