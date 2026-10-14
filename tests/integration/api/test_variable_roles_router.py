import io
import json

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


@pytest.fixture
def sid():
    r = client.post("/api/v1/sessions", json={"ttl_hours": 1})
    assert r.status_code == 201
    s = r.json()["session_id"]
    yield s
    client.delete(f"/api/v1/sessions/{s}")


def _add(sid, body):
    return client.post(f"/api/v1/sessions/{sid}/variables", json=body)


def test_role_and_changes_round_trip(sid):
    assert _add(sid, {"name": "x1", "type": "real", "min": 0, "max": 1}).status_code == 200
    assert _add(sid, {"name": "c", "type": "categorical", "categories": ["a", "b"], "changes": "hard"}).status_code == 200
    assert _add(sid, {"name": "z", "type": "real", "role": "observed"}).status_code == 200
    assert _add(sid, {"name": "k", "type": "categorical", "categories": ["p", "q"], "role": "observed"}).status_code == 200
    rows = {v["name"]: v for v in client.get(f"/api/v1/sessions/{sid}/variables").json()["variables"]}
    assert (rows["x1"]["role"], rows["x1"]["changes"]) == ("controllable", "easy")
    assert rows["c"]["changes"] == "hard"
    assert rows["z"]["role"] == "observed" and rows["z"]["bounds"] is None
    assert rows["k"]["categories"] == ["p", "q"]


def test_controllable_real_still_requires_bounds(sid):
    assert _add(sid, {"name": "x1", "type": "real", "min": 0}).status_code == 422


def test_observed_with_hard_changes_is_rejected(sid):
    r = _add(sid, {"name": "z", "type": "real", "role": "observed", "changes": "hard"})
    assert r.status_code == 400 and "observed" in r.json()["detail"]


def test_update_can_change_changes(sid):
    _add(sid, {"name": "x1", "type": "real", "min": 0, "max": 1})
    r = client.put(f"/api/v1/sessions/{sid}/variables/x1",
                   json={"name": "x1", "type": "real", "min": 0, "max": 1, "changes": "very_hard"})
    assert r.status_code == 200 and r.json()["variable"]["changes"] == "very_hard"


def test_export_then_load_preserves_role_and_changes(sid):
    _add(sid, {"name": "c", "type": "categorical", "categories": ["a", "b"], "changes": "hard"})
    _add(sid, {"name": "z", "type": "real", "role": "observed"})
    # The default export is a bare JSON array; /variables/load takes it as a file upload.
    exported = client.get(f"/api/v1/sessions/{sid}/variables/export").json()
    r = client.post("/api/v1/sessions", json={"ttl_hours": 1})
    other = r.json()["session_id"]
    try:
        buf = io.BytesIO(json.dumps(exported).encode())
        load = client.post(
            f"/api/v1/sessions/{other}/variables/load",
            files={"file": ("space.json", buf, "application/json")},
        )
        assert load.status_code == 200, load.text
        rows = {v["name"]: v for v in client.get(f"/api/v1/sessions/{other}/variables").json()["variables"]}
        assert rows["c"]["changes"] == "hard" and rows["z"]["role"] == "observed"
    finally:
        client.delete(f"/api/v1/sessions/{other}")
