"""`POST /acquisition/suggest` leaves `xi` and `kappa` to the core when the
request does not set them (19), and the object-form `/variables/load`
refuses a variable the recorded rows have no value for (20).

`kappa` defaulted to 2.0 and was forwarded whenever not None; with the core
treating `kappa` as an alias for BoTorch's `beta`, every REST UCB/qUCB
request would have run at β = 2.0 instead of the documented 0.5.
"""
import io
import json

import pytest
from fastapi.testclient import TestClient

import alchemist_core.acquisition.botorch_acquisition as botorch_acq
import alchemist_core.session as core_session
from api.main import app
from api.models.requests import AcquisitionRequest

client = TestClient(app)


@pytest.fixture
def sid():
    s = client.post("/api/v1/sessions", json={}).json()["session_id"]
    assert client.post(f"/api/v1/sessions/{s}/variables",
                       json={"name": "x", "type": "real", "min": 0, "max": 1}).status_code == 200
    rows = [{"inputs": {"x": i / 10}, "output": float(i)} for i in range(6)]
    assert client.post(f"/api/v1/sessions/{s}/experiments/batch",
                       json={"experiments": rows}).status_code == 200
    yield s
    client.delete(f"/api/v1/sessions/{s}")


# 19 -----------------------------------------------------------------------

def test_request_model_has_no_exploration_defaults():
    r = AcquisitionRequest(strategy="UCB")
    assert r.xi is None and r.kappa is None


def test_unset_xi_and_kappa_are_not_forwarded(sid, monkeypatch):
    import pandas as pd
    seen = {}

    def fake_suggest(self, *args, **kwargs):
        seen.update(kwargs)
        return pd.DataFrame([{"x": 0.5}])

    monkeypatch.setattr(core_session.OptimizationSession, "suggest_next", fake_suggest)
    assert client.post(f"/api/v1/sessions/{sid}/model/train",
                       json={"backend": "sklearn", "kernel": "RBF"}).status_code == 200
    r = client.post(f"/api/v1/sessions/{sid}/acquisition/suggest", json={"strategy": "UCB"})
    assert r.status_code == 200, r.text
    assert "kappa" not in seen and "xi" not in seen
    seen.clear()
    r = client.post(f"/api/v1/sessions/{sid}/acquisition/suggest",
                    json={"strategy": "UCB", "kappa": 1.2, "xi": 0.05})
    assert r.status_code == 200, r.text
    assert seen["kappa"] == 1.2 and seen["xi"] == 0.05


def test_botorch_ucb_without_kappa_runs_at_beta_0_5(sid, monkeypatch):
    betas = []
    real = botorch_acq.UpperConfidenceBound

    def recording(*args, **kwargs):
        betas.append(kwargs.get("beta"))
        return real(*args, **kwargs)

    monkeypatch.setattr(botorch_acq, "UpperConfidenceBound", recording)
    assert client.post(f"/api/v1/sessions/{sid}/model/train",
                       json={"backend": "botorch", "kernel": "RBF"}).status_code == 200
    r = client.post(f"/api/v1/sessions/{sid}/acquisition/suggest", json={"strategy": "UCB"})
    assert r.status_code == 200, r.text
    assert betas == [pytest.approx(0.5)]


def test_audit_records_only_the_exploration_parameters_sent(sid):
    assert client.post(f"/api/v1/sessions/{sid}/model/train",
                       json={"backend": "sklearn", "kernel": "RBF"}).status_code == 200
    assert client.post(f"/api/v1/sessions/{sid}/acquisition/suggest",
                       json={"strategy": "EI"}).status_code == 200
    entries = client.get(f"/api/v1/sessions/{sid}/audit").json()["entries"]
    locked = [e for e in entries if e["entry_type"] == "acquisition_locked"][-1]
    assert "kappa" not in locked["parameters"] and "xi" not in locked["parameters"]


# 20 -----------------------------------------------------------------------

def _load(sid, payload):
    return client.post(f"/api/v1/sessions/{sid}/variables/load",
                       files={"file": ("v.json", io.BytesIO(json.dumps(payload).encode()),
                                       "application/json")})


def test_object_form_load_refuses_a_variable_the_rows_lack(sid):
    r = _load(sid, {"variables": [{"name": "x", "type": "real", "min": 0, "max": 1},
                                  {"name": "y", "type": "real", "min": 0, "max": 1}],
                    "constraints": []})
    assert r.status_code == 400, r.text
    assert "already has 6 recorded experiments" in r.json()["detail"]
    assert "'y'" in r.json()["detail"]
    names = [v["name"] for v in client.get(f"/api/v1/sessions/{sid}/variables").json()["variables"]]
    assert names == ["x"]
    assert client.get(f"/api/v1/sessions/{sid}").status_code == 200


def test_object_form_load_may_replace_the_same_variables(sid):
    r = _load(sid, {"variables": [{"name": "x", "type": "real", "min": 0, "max": 2}],
                    "constraints": []})
    assert r.status_code == 200, r.text
    (var,) = client.get(f"/api/v1/sessions/{sid}/variables").json()["variables"]
    assert var["bounds"] == [0, 2]
