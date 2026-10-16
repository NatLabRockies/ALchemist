"""`POST /experiments/queue` validates each row like `staged/batch` (A23) and
`POST /experiments/queue/{id}/complete` honours `iteration` (A22).

A queue row missing a variable used to be accepted as pending, and then
`staged/complete` answered 400 "Missing search space variables" for every
staged point until that row was deleted by id. And the completion route
accepted `iteration` in its body and ignored it (the row got the
auto-assigned iteration).
"""
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


@pytest.fixture
def sid():
    s = client.post("/api/v1/sessions", json={}).json()["session_id"]
    for v in ({"name": "x", "type": "real", "min": 0.0, "max": 1.0},
              {"name": "cat", "type": "categorical", "categories": ["a", "b"]}):
        assert client.post(f"/api/v1/sessions/{s}/variables", json=v).status_code == 200
    yield s
    client.delete(f"/api/v1/sessions/{s}")


def _queue(sid):
    return client.get(f"/api/v1/sessions/{sid}/experiments/queue").json()["items"]


def test_a_queue_row_missing_a_variable_is_refused_and_nothing_is_staged(sid):
    r = client.post(f"/api/v1/sessions/{sid}/experiments/queue",
                    json={"items": [{"inputs": {"x": 0.5, "cat": "a"}},
                                    {"inputs": {"x": 0.2}}]})
    assert r.status_code == 400, r.text
    assert "cat" in r.json()["detail"]
    assert _queue(sid) == []


def test_a_queue_row_with_an_unknown_category_is_refused(sid):
    r = client.post(f"/api/v1/sessions/{sid}/experiments/queue",
                    json={"items": [{"inputs": {"x": 0.5, "cat": "zzz"}}]})
    assert r.status_code == 400, r.text
    assert _queue(sid) == []


def test_a_bad_queue_row_cannot_wedge_staged_complete(sid):
    assert client.post(f"/api/v1/sessions/{sid}/experiments/queue",
                       json={"items": [{"inputs": {"x": 0.2}}]}).status_code == 400
    assert client.post(f"/api/v1/sessions/{sid}/experiments/staged/batch",
                       json={"experiments": [{"x": 0.5, "cat": "a"}]}).status_code == 200
    r = client.post(f"/api/v1/sessions/{sid}/experiments/staged/complete",
                    json={"outputs": [1.0]})
    assert r.status_code == 200, r.text
    assert r.json()["n_added"] == 1


def test_complete_queue_item_honours_iteration(sid):
    r = client.post(f"/api/v1/sessions/{sid}/experiments/queue",
                    json={"items": [{"inputs": {"x": 0.5, "cat": "a"}}]})
    item_id = r.json()["items"][0]["id"]
    r = client.post(f"/api/v1/sessions/{sid}/experiments/queue/{item_id}/complete",
                    json={"outputs": [2.0], "iteration": 7})
    assert r.status_code == 200, r.text
    rows = client.get(f"/api/v1/sessions/{sid}/experiments").json()["experiments"]
    assert [row["Iteration"] for row in rows] == [7]


def test_complete_queue_item_without_iteration_still_auto_assigns(sid):
    r = client.post(f"/api/v1/sessions/{sid}/experiments/queue",
                    json={"items": [{"inputs": {"x": 0.5, "cat": "a"}}]})
    item_id = r.json()["items"][0]["id"]
    assert client.post(f"/api/v1/sessions/{sid}/experiments/queue/{item_id}/complete",
                       json={"outputs": [2.0]}).status_code == 200
    rows = client.get(f"/api/v1/sessions/{sid}/experiments").json()["experiments"]
    assert rows[0]["Iteration"] is not None
