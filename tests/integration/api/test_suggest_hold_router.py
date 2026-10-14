import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app

# Fixture artifact: a third-party library (pyro) emits a SyntaxWarning when first imported.
pytestmark = pytest.mark.filterwarnings("ignore::SyntaxWarning")

client = TestClient(app)
OFFSET = {"a": 0.0, "b": 8.0}


@pytest.fixture
def sid():
    s = client.post("/api/v1/sessions", json={"ttl_hours": 1}).json()["session_id"]
    for v in (
        {"name": "x1", "type": "real", "min": 0, "max": 10},
        {"name": "c", "type": "categorical", "categories": ["a", "b"], "changes": "hard"},
        {"name": "z", "type": "real", "role": "observed"},
    ):
        client.post(f"/api/v1/sessions/{s}/variables", json=v).raise_for_status()
    rng = np.random.default_rng(0)
    for i in range(14):
        c = "ab"[i % 2]
        x1, z = float(rng.uniform(0, 10)), float(rng.uniform(20, 30))
        client.post(f"/api/v1/sessions/{s}/experiments",
                    json={"inputs": {"x1": x1, "c": c, "z": z},
                          "output": -(x1 - 6) ** 2 + OFFSET[c] + 0.1 * z}).raise_for_status()
    r = client.post(f"/api/v1/sessions/{s}/model/train", json={"backend": "botorch", "kernel": "Matern"})
    assert r.status_code == 200, r.text
    yield s
    client.delete(f"/api/v1/sessions/{s}")


def _suggest(sid, **body):
    return client.post(f"/api/v1/sessions/{sid}/acquisition/suggest", json={"strategy": "EI", **body})


def test_hold_and_observed_round_trip_with_diagnostics(sid):
    r = _suggest(sid, hold={"c": "a"}, observed={"z": 25.0})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["suggestions"][0]["c"] == "a"
    assert set(body["suggestions"][0]) == {"x1", "c"}
    d = body["diagnostics"][0]
    assert d["held"] == ["c"] and 0.0 <= d["knowledge_ratio"] <= 1.0


def test_missing_observed_value_is_a_400(sid):
    r = _suggest(sid)
    assert r.status_code == 400 and "z" in r.json()["detail"]


def test_bad_hold_value_is_a_400(sid):
    r = _suggest(sid, hold={"c": "nope"}, observed={"z": 25.0})
    assert r.status_code == 400 and "declared values" in r.json()["detail"]


def test_audit_records_normalized_hold_and_registered_observed_only():
    s = client.post("/api/v1/sessions", json={"ttl_hours": 1}).json()["session_id"]
    try:
        for v in (
            {"name": "x1", "type": "real", "min": 0, "max": 10},
            {"name": "c", "type": "categorical", "categories": ["1", "2"], "changes": "hard"},
            {"name": "z", "type": "real", "role": "observed"},
        ):
            client.post(f"/api/v1/sessions/{s}/variables", json=v).raise_for_status()
        rng = np.random.default_rng(1)
        for i in range(12):
            c = "12"[i % 2]
            x1, z = float(rng.uniform(0, 10)), float(rng.uniform(20, 30))
            client.post(f"/api/v1/sessions/{s}/experiments",
                        json={"inputs": {"x1": x1, "c": c, "z": z},
                              "output": -(x1 - 6) ** 2 + (c == "2") + 0.1 * z}).raise_for_status()
        client.post(f"/api/v1/sessions/{s}/model/train",
                    json={"backend": "botorch", "kernel": "Matern"}).raise_for_status()
        r = _suggest(s, hold={"c": 2}, observed={"z": 25.0, "stray": 1.0})
        assert r.status_code == 200, r.text
        entries = client.get(f"/api/v1/sessions/{s}/audit",
                             params={"entry_type": "acquisition_locked"}).json()["entries"]
        params = entries[-1]["parameters"]["parameters"]
        assert params["hold"] == {"c": "2"}
        assert params["observed"] == {"z": 25.0}
    finally:
        client.delete(f"/api/v1/sessions/{s}")
