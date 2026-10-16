"""`GET /sessions/{id}` never fails because the data summary cannot be
computed (16), reports `target_columns` (17), and `record-batch` records a
whole request under one iteration when the entries give none (18).

A session whose data frame lacks a variable column (a variable added after
results were recorded) made `get_data_summary` raise, the route answered
400, and the web app lost the session. A 10-point design recorded through
`record-batch` without `iteration` got Iteration 0..9, one per row, while
the changelog promises one iteration per batch.
"""
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.services.session_store import session_store

client = TestClient(app)


@pytest.fixture
def sid():
    s = client.post("/api/v1/sessions", json={}).json()["session_id"]
    assert client.post(f"/api/v1/sessions/{s}/variables",
                       json={"name": "x", "type": "real", "min": 0, "max": 1}).status_code == 200
    yield s
    client.delete(f"/api/v1/sessions/{s}")


def _rows(sid):
    return client.get(f"/api/v1/sessions/{sid}/experiments").json()["experiments"]


# 16 -----------------------------------------------------------------------

def test_info_survives_a_data_frame_missing_a_variable_column(sid):
    import importlib
    from unittest.mock import patch
    store_module = importlib.import_module("api.services.session_store")
    rows = [{"inputs": {"x": i / 10}, "output": float(i)} for i in range(3)]
    assert client.post(f"/api/v1/sessions/{sid}/experiments/batch",
                       json={"experiments": rows}).status_code == 200
    # The inconsistent state, forced on the session object: a variable the
    # data frame has no column for.
    session_store.get(sid).search_space.add_variable("late", "real", min=0.0, max=1.0)
    # Asserted on the logger itself: other tests leave it not propagating,
    # so caplog would see nothing.
    with patch.object(store_module.logger, "error") as log_error:
        r = client.get(f"/api/v1/sessions/{sid}")
    assert r.status_code == 200, r.text
    data = r.json()["data"]
    assert data["n_experiments"] == 3
    assert "late" in data["error"]
    assert any("late" in str(call) for call in log_error.call_args_list)


def test_post_variables_reports_the_cores_message_as_400(sid, monkeypatch):
    from alchemist_core.session import OptimizationSession

    def refuse(self, *args, **kwargs):
        raise ValueError("Cannot add variable 'late' after results are recorded")

    monkeypatch.setattr(OptimizationSession, "add_variable", refuse)
    r = client.post(f"/api/v1/sessions/{sid}/variables",
                    json={"name": "late", "type": "real", "min": 0, "max": 1})
    assert r.status_code == 400, r.text
    assert "after results are recorded" in r.json()["detail"]


# 17 -----------------------------------------------------------------------

def test_info_reports_the_target_columns(sid):
    assert client.get(f"/api/v1/sessions/{sid}").json()["target_columns"] == ["Output"]


def test_info_reports_renamed_target_columns():
    import io
    s = client.post("/api/v1/sessions", json={}).json()["session_id"]
    try:
        for name in ("x1", "x2"):
            client.post(f"/api/v1/sessions/{s}/variables",
                        json={"name": name, "type": "real", "min": 0, "max": 10})
        csv = "x1,x2,conversion,selectivity\n1,2,0.5,0.9\n3,4,0.7,0.8\n"
        r = client.post(f"/api/v1/sessions/{s}/experiments/upload",
                        files={"file": ("d.csv", io.BytesIO(csv.encode()), "text/csv")},
                        params={"target_columns": "conversion,selectivity"})
        assert r.status_code == 200, r.text
        assert client.get(f"/api/v1/sessions/{s}").json()["target_columns"] == ["conversion", "selectivity"]
    finally:
        client.delete(f"/api/v1/sessions/{s}")


# 18 -----------------------------------------------------------------------

def test_record_batch_without_iteration_records_one_iteration_per_call(sid):
    entries = [{"inputs": {"x": i / 10}, "output": float(i)} for i in range(4)]
    r = client.post(f"/api/v1/sessions/{sid}/experiments/record-batch", json={"entries": entries})
    assert r.status_code == 200, r.text
    assert [row["Iteration"] for row in _rows(sid)] == [0, 0, 0, 0]
    # the next call is the next round
    r = client.post(f"/api/v1/sessions/{sid}/experiments/record-batch",
                    json={"entries": entries[:2]})
    assert r.status_code == 200, r.text
    assert [row["Iteration"] for row in _rows(sid)] == [0, 0, 0, 0, 1, 1]


def test_record_batch_mixes_queue_items_and_direct_rows_under_one_iteration(sid):
    r = client.post(f"/api/v1/sessions/{sid}/experiments/queue",
                    json={"items": [{"inputs": {"x": 0.1}}, {"inputs": {"x": 0.2}}]})
    ids = [item["id"] for item in r.json()["items"]]
    entries = [{"item_id": ids[0], "inputs": {"x": 0.1}, "output": 1.0},
               {"item_id": ids[1], "inputs": {"x": 0.2}, "output": 2.0},
               {"inputs": {"x": 0.3}, "output": 3.0}]
    r = client.post(f"/api/v1/sessions/{sid}/experiments/record-batch", json={"entries": entries})
    assert r.status_code == 200, r.text
    assert [row["Iteration"] for row in _rows(sid)] == [0, 0, 0]


def test_record_batch_keeps_an_explicit_iteration(sid):
    entries = [{"inputs": {"x": 0.1}, "output": 1.0, "iteration": 7},
               {"inputs": {"x": 0.2}, "output": 2.0, "iteration": 7}]
    assert client.post(f"/api/v1/sessions/{sid}/experiments/record-batch",
                       json={"entries": entries}).status_code == 200
    assert [row["Iteration"] for row in _rows(sid)] == [7, 7]
