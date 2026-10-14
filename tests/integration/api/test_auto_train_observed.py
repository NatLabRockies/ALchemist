"""Auto-train picks the BoTorch backend when observed variables exist.

Observed variables are only supported by the BoTorch backend, so the sklearn
default (or a previous sklearn config) would make every auto-train fail.
"""
import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routers.experiments import resolve_auto_train_config
from api.services import session_store

client = TestClient(app)


@pytest.fixture
def sid():
    s = client.post("/api/v1/sessions", json={"ttl_hours": 1}).json()["session_id"]
    for v in (
        {"name": "x1", "type": "real", "min": 0, "max": 10},
        {"name": "x2", "type": "real", "min": 0, "max": 10},
        {"name": "z", "type": "real", "role": "observed"},
    ):
        client.post(f"/api/v1/sessions/{s}/variables", json=v).raise_for_status()
    yield s
    client.delete(f"/api/v1/sessions/{s}")


def _entries(n=6, seed=2):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        x1, x2, z = (float(v) for v in rng.uniform(0, 10, 3))
        out.append({"inputs": {"x1": x1, "x2": x2, "z": z}, "output": x1 - x2 + z / 10})
    return out


def test_record_batch_auto_train_with_observed_and_no_prior_training(sid):
    r = client.post(f"/api/v1/sessions/{sid}/experiments/record-batch",
                    params={"auto_train": "true"}, json={"entries": _entries()})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_trained"] is True
    assert body["training_metrics"]["backend"] == "botorch"


def test_previous_sklearn_config_switches_to_botorch_with_a_log(sid, caplog):
    session = session_store.get(sid)
    session._last_training_config = {"backend": "sklearn", "kernel": "RBF"}
    with caplog.at_level(logging.INFO, logger="api.routers.experiments"):
        config = resolve_auto_train_config(session)
    assert config["backend"] == "botorch"
    assert any("observed" in r.getMessage() for r in caplog.records)


def test_explicit_backend_override_wins(sid):
    session = session_store.get(sid)
    assert resolve_auto_train_config(session, training_backend="sklearn")["backend"] == "sklearn"
