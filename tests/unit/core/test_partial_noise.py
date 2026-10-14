"""A6: a Noise column that is set on only some rows."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from alchemist_core import OptimizationSession
from api.main import app


def _session(noise_for, multi=False):
    rng = np.random.default_rng(1)
    s = OptimizationSession()
    s.add_variable("x1", "real", bounds=(0.0, 10.0))
    s.add_variable("x2", "real", bounds=(0.0, 10.0))
    for i, (a, b) in enumerate(rng.uniform(0, 10, size=(10, 2))):
        s.add_experiment({"x1": float(a), "x2": float(b)}, float(a + b), noise=noise_for(i))
    if multi:
        em = s.experiment_manager
        em.df = em.df.rename(columns={"Output": "y1"})
        em.df["y2"] = em.df["x1"] - em.df["x2"]
        em.target_columns = ["y1", "y2"]
    return s


@pytest.mark.parametrize("backend", ["botorch", "sklearn"])
def test_partial_noise_raises_clear_error(backend):
    s = _session(lambda i: 0.5 if i < 4 else None)
    with pytest.raises(ValueError, match="noise value or none") as exc:
        s.train_model(backend=backend, kernel="Matern")
    assert "6 of 10" in str(exc.value)
    assert "4, 5, 6, 7, 8 ..." in str(exc.value)


def test_partial_noise_raises_for_multi_objective():
    s = _session(lambda i: 0.5 if i < 4 else None, multi=True)
    with pytest.raises(ValueError, match="noise value or none"):
        s.train_model(backend="botorch", kernel="Matern")


@pytest.mark.parametrize("backend", ["botorch", "sklearn"])
def test_full_noise_trains(backend):
    s = _session(lambda i: 0.5)
    assert s.train_model(backend=backend, kernel="Matern")["success"]


@pytest.mark.parametrize("backend", ["botorch", "sklearn"])
def test_all_nan_noise_column_trains_as_no_noise(backend):
    s = _session(lambda i: None)
    s.experiment_manager.df["Noise"] = np.nan
    assert s.train_model(backend=backend, kernel="Matern")["success"]
    # The column stays in the table; only the model ignores it.
    assert "Noise" in s.experiment_manager.df.columns


def test_all_nan_noise_column_trains_as_no_noise_multi_objective():
    s = _session(lambda i: None, multi=True)
    s.experiment_manager.df["Noise"] = np.nan
    assert s.train_model(backend="botorch", kernel="Matern")["success"]


# ---- REST -------------------------------------------------------------

client = TestClient(app)


def _rest_session_with_partial_noise() -> str:
    sid = client.post("/api/v1/sessions", json={"ttl_hours": 1}).json()["session_id"]
    for name in ("x1", "x2"):
        client.post(f"/api/v1/sessions/{sid}/variables",
                    json={"name": name, "type": "real", "min": 0.0, "max": 10.0}).raise_for_status()
    rng = np.random.default_rng(2)
    pts = [{"x1": float(a), "x2": float(b)} for a, b in rng.uniform(0, 10, size=(10, 2))]
    client.post(f"/api/v1/sessions/{sid}/experiments",
                json={"inputs": pts[0], "output": 1.0, "noise": 1.2}).raise_for_status()
    client.post(f"/api/v1/sessions/{sid}/experiments/batch",
                json={"experiments": [{"inputs": p, "output": p["x1"] + p["x2"]} for p in pts[1:]]}
                ).raise_for_status()
    return sid


def test_rest_train_partial_noise_is_400():
    sid = _rest_session_with_partial_noise()
    try:
        r = client.post(f"/api/v1/sessions/{sid}/model/train",
                        json={"backend": "sklearn", "kernel": "rbf"})
        assert r.status_code == 400
        assert "noise value or none" in r.text
    finally:
        client.delete(f"/api/v1/sessions/{sid}")


def test_rest_record_batch_with_auto_train_still_records_on_partial_noise():
    sid = _rest_session_with_partial_noise()
    try:
        r = client.post(
            f"/api/v1/sessions/{sid}/experiments/record-batch?auto_train=true",
            json={"entries": [{"inputs": {"x1": 1.0, "x2": 2.0}, "output": 3.0}]},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["n_added"] == 1
        assert body["n_experiments"] == 11
        assert body["model_trained"] is False
    finally:
        client.delete(f"/api/v1/sessions/{sid}")
