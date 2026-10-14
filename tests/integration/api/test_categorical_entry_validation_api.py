"""REST entry points refuse undeclared categorical values with a 400."""

import io

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


@pytest.fixture
def sid():
    s = client.post("/api/v1/sessions", json={"ttl_hours": 1}).json()["session_id"]
    for v in (
        {"name": "x1", "type": "real", "min": 0, "max": 10},
        {"name": "c", "type": "categorical", "categories": ["a", "b"]},
    ):
        client.post(f"/api/v1/sessions/{s}/variables", json=v).raise_for_status()
    yield s
    client.delete(f"/api/v1/sessions/{s}")


def test_record_batch_rejects_undeclared_value(sid):
    entries = [
        {"inputs": {"x1": 1.0, "c": "a"}, "output": 1.0},
        {"inputs": {"x1": 2.0, "c": "B"}, "output": 2.0},
    ]
    r = client.post(f"/api/v1/sessions/{sid}/experiments/record-batch", json={"entries": entries})
    assert r.status_code == 400, r.text
    assert "'c'" in r.json()["detail"] and "B" in r.json()["detail"]
    assert client.get(f"/api/v1/sessions/{sid}/experiments").json()["n_experiments"] == 0


def test_add_experiment_rejects_undeclared_value(sid):
    r = client.post(f"/api/v1/sessions/{sid}/experiments",
                    json={"inputs": {"x1": 1.0, "c": "B"}, "output": 1.0})
    assert r.status_code == 400, r.text


def test_upload_rejects_undeclared_value(sid):
    csv = io.BytesIO(b"x1,c,Output\n1,a,0.5\n2,B,0.7\n")
    r = client.post(f"/api/v1/sessions/{sid}/experiments/upload",
                    files={"file": ("data.csv", csv, "text/csv")})
    assert r.status_code == 400, r.text
    assert "'c'" in r.json()["detail"]


def test_upload_accepts_declared_values(sid):
    csv = io.BytesIO(b"x1,c,Output\n1,a,0.5\n2,b,0.7\n")
    r = client.post(f"/api/v1/sessions/{sid}/experiments/upload",
                    files={"file": ("data.csv", csv, "text/csv")})
    assert r.status_code == 200, r.text
