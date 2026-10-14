import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


@pytest.fixture
def sid():
    s = client.post("/api/v1/sessions", json={"ttl_hours": 1}).json()["session_id"]
    for v in (
        {"name": "x1", "type": "real", "min": 0, "max": 10},
        {"name": "x2", "type": "real", "min": 0, "max": 10},
        {"name": "c", "type": "categorical", "categories": ["a", "b"], "changes": "hard"},
        {"name": "z", "type": "real", "role": "observed"},
    ):
        client.post(f"/api/v1/sessions/{s}/variables", json=v).raise_for_status()
    yield s
    client.delete(f"/api/v1/sessions/{s}")


BODY = {
    "designs": {"hard": {"method": "full_factorial"}, "easy": {"method": "lhs", "n_points": 3}},
    "replicates": {"hard": 2},
    "reference": {"position": "start"},
    "random_seed": 5,
}


def _design(sid, body=BODY):
    return client.post(f"/api/v1/sessions/{sid}/split-plot-design", json=body)


def test_generates_points_and_info(sid):
    r = _design(sid)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_points"] == 2 * 2 * (3 + 1) == len(body["points"])
    assert {"WholePlot", "RunOrder", "Reason"} <= set(body["points"][0])
    assert body["info"]["n_plots"] == {"hard": 4}


def test_validation_errors_are_400(sid):
    r = _design(sid, {"designs": {"easy": {"method": "lhs", "n_points": 3}}})
    assert r.status_code == 400


def test_unknown_design_key_is_422(sid):
    r = _design(sid, {"designs": {"hard": {"method": "full_factorial", "bogus": 1},
                                  "easy": {"method": "lhs", "n_points": 3}}})
    assert r.status_code == 422


def test_metadata_survives_staging_and_record_batch(sid):
    points = _design(sid).json()["points"]
    staged = [
        {**{k: p[k] for k in ("x1", "x2", "c", "WholePlot", "RunOrder")}, "_reason": p["Reason"]}
        for p in points[:4]
    ]
    r = client.post(f"/api/v1/sessions/{sid}/experiments/staged/batch", json={"experiments": staged})
    assert r.status_code == 200, r.text
    ids = r.json()["ids"]
    entries = [
        {"item_id": i, "inputs": {"x1": p["x1"], "x2": p["x2"], "c": p["c"], "z": 25.0}, "output": 1.0}
        for i, p in zip(ids, points[:4])
    ]
    r = client.post(f"/api/v1/sessions/{sid}/experiments/record-batch", json={"entries": entries})
    assert r.status_code == 200, r.text
    rows = client.get(f"/api/v1/sessions/{sid}/experiments").json()["experiments"]
    assert [int(row["WholePlot"]) for row in rows] == [p["WholePlot"] for p in points[:4]]
    assert [int(row["RunOrder"]) for row in rows] == [1, 2, 3, 4]
    assert rows[0]["Reason"] == "Reference (split-plot)"


def test_record_batch_requires_observed_values(sid):
    entries = [{"inputs": {"x1": 1.0, "x2": 1.0, "c": "a"}, "output": 1.0}]
    r = client.post(f"/api/v1/sessions/{sid}/experiments/record-batch", json={"entries": entries})
    assert r.status_code == 400 and "z" in r.json()["detail"]
