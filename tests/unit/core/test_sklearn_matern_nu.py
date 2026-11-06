"""F2: session.train_model(backend='sklearn', kernel_params={'nu': ...}) ignored ν.

The web app, the desktop app and the docs all send ``nu``; SklearnModel reads
``matern_nu``, so the fitted Matérn was always ν = 1.5. BoTorch mapped it already.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from alchemist_core.session import OptimizationSession
from api.main import app


def _session():
    s = OptimizationSession()
    s.add_variable("x1", "real", min=0.0, max=1.0)
    s.add_variable("x2", "real", min=0.0, max=1.0)
    rng = np.random.default_rng(1)
    for x1, x2 in rng.random((10, 2)):
        s.add_experiment({"x1": float(x1), "x2": float(x2)}, output=float(np.sin(3 * x1) + x2))
    return s


def _fitted_nu(s):
    return s.model.model.kernel_.k2.nu


@pytest.mark.parametrize("nu", [0.5, 2.5])
def test_nu_reaches_the_fitted_kernel(nu):
    s = _session()
    s.train_model(backend="sklearn", kernel="Matern", kernel_params={"nu": nu})
    assert _fitted_nu(s) == nu


def test_cross_validation_folds_use_the_same_nu():
    s = _session()
    s.train_model(backend="sklearn", kernel="Matern", kernel_params={"nu": 2.5})
    assert s.model.kernel.k2.nu == 2.5
    assert s.get_model_summary()["hyperparameters"]["matern_nu"] == 2.5


def test_matern_nu_spelling_still_works():
    s = _session()
    s.train_model(backend="sklearn", kernel="Matern", kernel_params={"matern_nu": 0.5})
    assert _fitted_nu(s) == 0.5


def test_default_is_still_one_point_five():
    s = _session()
    s.train_model(backend="sklearn", kernel="Matern")
    assert _fitted_nu(s) == 1.5


def test_saved_config_replays_the_same_nu():
    s = _session()
    s.train_model(backend="sklearn", kernel="Matern", kernel_params={"nu": 2.5})
    r = _session()
    r.train_model(**s.get_last_training_config())
    assert _fitted_nu(r) == 2.5


def test_null_nu_is_a_clear_error_and_keeps_the_previous_model():
    """The web app's `inf` choice reaches the server as null (T21). It used to fit ν = 1.5 silently."""
    s = _session()
    s.train_model(backend="sklearn", kernel="Matern")
    previous = s.model
    with pytest.raises(ValueError, match="Matern nu"):
        s.train_model(backend="sklearn", kernel="Matern", kernel_params={"nu": None})
    assert s.model is previous


def test_rest_request_like_the_web_apps_fits_that_nu():
    client = TestClient(app)
    sid = client.post("/api/v1/sessions", json={}).json()["session_id"]
    for name in ("x1", "x2"):
        client.post(f"/api/v1/sessions/{sid}/variables", json={"name": name, "type": "real", "min": 0, "max": 1})
    rng = np.random.default_rng(1)
    for x1, x2 in rng.random((10, 2)):
        client.post(f"/api/v1/sessions/{sid}/experiments",
                    json={"inputs": {"x1": float(x1), "x2": float(x2)}, "output": float(x1 + x2)}).raise_for_status()
    body = {"backend": "sklearn", "kernel": "Matern", "kernel_params": {"nu": 2.5}}
    trained = client.post(f"/api/v1/sessions/{sid}/model/train", json=body)
    assert trained.status_code == 200, trained.text
    assert trained.json()["hyperparameters"]["k2__nu"] == 2.5
    # JSON cannot carry inf: the web app's choice arrives as null.
    refused = client.post(f"/api/v1/sessions/{sid}/model/train", json={**body, "kernel_params": {"nu": None}})
    assert refused.status_code == 400
    assert "Matern nu" in refused.json()["detail"]
    client.delete(f"/api/v1/sessions/{sid}")
