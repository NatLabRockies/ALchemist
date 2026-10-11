"""
The first Add Point on an empty session must not break GET /sessions/{id}.

Regression: with one row, get_data_summary returned std=NaN, the session GET
failed with 400 "Out of range float values are not JSON compliant", and the
web UI treated that as "Session not found" and cleared the session.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


@pytest.fixture
def sid():
    response = client.post("/api/v1/sessions", json={"ttl_hours": 1})
    response.raise_for_status()
    session_id = response.json()["session_id"]
    for payload in (
        {"name": "temperature", "type": "real", "min": 100.0, "max": 500.0},
        {"name": "pressure", "type": "real", "min": 1.0, "max": 10.0},
    ):
        client.post(f"/api/v1/sessions/{session_id}/variables", json=payload).raise_for_status()
    yield session_id
    client.delete(f"/api/v1/sessions/{session_id}")


def _add(sid, temperature, pressure, output, **params):
    return client.post(
        f"/api/v1/sessions/{sid}/experiments",
        params=params,
        json={"inputs": {"temperature": temperature, "pressure": pressure}, "output": output},
    )


def test_get_session_after_first_add_point(sid):
    # The web dialog sends auto_train=true by default ("Retrain model" ticked).
    resp = _add(sid, 200, 3, 0.5, auto_train="true")
    assert resp.status_code == 200, resp.text
    assert resp.json()["n_experiments"] == 1

    info = client.get(f"/api/v1/sessions/{sid}")
    assert info.status_code == 200, info.text
    data = info.json()["data"]
    assert data["n_experiments"] == 1
    assert data["target_stats"]["mean"] == pytest.approx(0.5)
    assert data["target_stats"]["std"] is None


def test_experiments_summary_after_first_add_point(sid):
    assert _add(sid, 200, 3, 0.5).status_code == 200
    summary = client.get(f"/api/v1/sessions/{sid}/experiments/summary")
    assert summary.status_code == 200, summary.text
    assert summary.json()["target_stats"]["std"] is None


def test_first_queue_completion_keeps_session_readable(sid):
    staged = client.post(
        f"/api/v1/sessions/{sid}/experiments/queue",
        json={"items": [{"inputs": {"temperature": 250.0, "pressure": 5.0}, "reason": "Initial Design"}]},
    )
    assert staged.status_code == 200, staged.text
    item_id = staged.json()["items"][-1]["id"]
    done = client.post(
        f"/api/v1/sessions/{sid}/experiments/queue/{item_id}/complete?auto_train=true",
        json={"outputs": [0.42], "actual_inputs": {"temperature": 250.0, "pressure": 5.0}},
    )
    assert done.status_code == 200, done.text

    info = client.get(f"/api/v1/sessions/{sid}")
    assert info.status_code == 200, info.text
    assert info.json()["data"]["n_experiments"] == 1


def test_second_add_point_reports_std(sid):
    assert _add(sid, 200, 3, 0.5).status_code == 200
    assert _add(sid, 300, 4, 0.7).status_code == 200
    info = client.get(f"/api/v1/sessions/{sid}")
    assert info.status_code == 200, info.text
    assert info.json()["data"]["target_stats"]["std"] == pytest.approx(0.1414, abs=1e-3)
