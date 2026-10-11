"""
Add Point's "Retrain model" (auto_train=true) must refit the model the user
trained from the Model tab: same backend, kernel, kernel params, transforms
and calibration flag. It used to retrain with backend + kernel="rbf" only.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from alchemist_core.session import OptimizationSession
from api.main import app
from api.services import session_store

client = TestClient(app)

# What the Model tab (GPRPanel.tsx) sends for BoTorch, Matern nu=2.5,
# normalize/standardize, "calibrate uncertainty" ticked.
MODEL_TAB_REQUEST = {
    "backend": "botorch",
    "kernel": "Matern",
    "kernel_params": {"nu": 2.5},
    "input_transform": "normalize",
    "output_transform": "standardize",
    "calibration_enabled": True,
}
EXPECTED_CONFIG = {
    "backend": "botorch",
    "kernel": "Matern",
    "kernel_params": {"nu": 2.5},
    "input_transform_type": "normalize",
    "output_transform_type": "standardize",
    "calibration_enabled": True,
}

_rng = np.random.default_rng(7)
POINTS = [
    {"temperature": float(t), "pressure": float(p)}
    for t, p in zip(_rng.uniform(100, 500, 12), _rng.uniform(1, 10, 12))
]


def _output(point):
    return float((point["temperature"] / 500.0) ** 2 + point["pressure"] / 10.0)


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


def _add(sid, point, auto_train=False):
    resp = client.post(
        f"/api/v1/sessions/{sid}/experiments",
        params={"auto_train": str(auto_train).lower()},
        json={"inputs": point, "output": _output(point)},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _train_from_model_tab(sid, request=MODEL_TAB_REQUEST):
    resp = client.post(f"/api/v1/sessions/{sid}/model/train", json=request)
    assert resp.status_code == 200, resp.text


def _session(sid) -> OptimizationSession:
    return session_store.get(sid)


def _assert_user_model(session):
    model = session.model
    assert session.model_backend == "botorch"
    assert model.cont_kernel_type == "Matern"
    assert model.model.covar_module.base_kernel.nu == 2.5
    assert type(model.model.input_transform).__name__ == "Normalize"
    assert type(model.model.outcome_transform).__name__ == "Standardize"
    assert session.get_last_training_config() == EXPECTED_CONFIG


def test_add_point_retrain_keeps_model_tab_settings(sid):
    for point in POINTS[:8]:
        _add(sid, point)
    _train_from_model_tab(sid)
    first_model = _session(sid).model

    body = _add(sid, POINTS[8], auto_train=True)
    assert body["model_trained"] is True
    assert body["training_metrics"]["backend"] == "botorch"

    session = _session(sid)
    assert session.model is not first_model, "auto_train did not refit the model"
    _assert_user_model(session)


def test_queue_completion_retrain_keeps_model_tab_settings(sid):
    for point in POINTS[:8]:
        _add(sid, point)
    _train_from_model_tab(sid)
    first_model = _session(sid).model

    staged = client.post(
        f"/api/v1/sessions/{sid}/experiments/queue",
        json={"items": [{"inputs": POINTS[8], "reason": "qEI"}]},
    )
    assert staged.status_code == 200, staged.text
    item_id = [it for it in staged.json()["items"] if it["status"] == "pending"][-1]["id"]
    done = client.post(
        f"/api/v1/sessions/{sid}/experiments/queue/{item_id}/complete?auto_train=true",
        json={"outputs": [_output(POINTS[8])], "actual_inputs": POINTS[8]},
    )
    assert done.status_code == 200, done.text

    session = _session(sid)
    assert session.model is not first_model, "auto_train did not refit the model"
    _assert_user_model(session)


def test_retrain_keeps_explicit_no_transforms(sid):
    # Model tab with transforms left at "none" omits them from the request, so
    # the API passes None, and BoTorch fits without transforms. The old
    # auto-train dropped that and fell back to normalize/standardize.
    for point in POINTS[:8]:
        _add(sid, point)
    _train_from_model_tab(sid, {"backend": "botorch", "kernel": "RBF"})
    assert getattr(_session(sid).model.model, "input_transform", None) is None

    _add(sid, POINTS[8], auto_train=True)
    model = _session(sid).model
    assert model.cont_kernel_type == "RBF"
    assert getattr(model.model, "input_transform", None) is None
    assert getattr(model.model, "outcome_transform", None) is None


def test_initial_design_one_by_one_then_model_tab(sid):
    # Points 1-5 one by one with "Retrain model" ticked: the first auto-train
    # (5th point) has no user config yet and uses the Model tab defaults.
    for point in POINTS[:5]:
        _add(sid, point, auto_train=True)
    session = _session(sid)
    assert session.model_backend == "sklearn"
    assert session.get_last_training_config() == {
        "backend": "sklearn", "kernel": "RBF", "kernel_params": None}

    # The user trains from the Model tab, then keeps adding points 6-8.
    _train_from_model_tab(sid)
    for point in POINTS[5:8]:
        body = _add(sid, point, auto_train=True)
        assert body["model_trained"] is True
    assert len(_session(sid).experiment_manager.df) == 8
    _assert_user_model(_session(sid))


def test_query_overrides_still_win(sid):
    for point in POINTS[:8]:
        _add(sid, point)
    _train_from_model_tab(sid)
    resp = client.post(
        f"/api/v1/sessions/{sid}/experiments",
        params={"auto_train": "true", "training_backend": "sklearn", "training_kernel": "RBF"},
        json={"inputs": POINTS[8], "output": _output(POINTS[8])},
    )
    assert resp.status_code == 200, resp.text
    session = _session(sid)
    assert session.model_backend == "sklearn"
    assert session.get_last_training_config() == {
        "backend": "sklearn", "kernel": "RBF", "kernel_params": None}


def test_batch_retrains_once_with_session_config(sid, monkeypatch):
    for point in POINTS[:8]:
        _add(sid, point)
    _train_from_model_tab(sid)

    calls = []
    real_train = OptimizationSession.train_model

    def spy(self, *args, **kwargs):
        calls.append(kwargs)
        return real_train(self, *args, **kwargs)

    monkeypatch.setattr(OptimizationSession, "train_model", spy)
    resp = client.post(
        f"/api/v1/sessions/{sid}/experiments/batch",
        params={"auto_train": "true"},
        json={"experiments": [{"inputs": p, "output": _output(p)} for p in POINTS[8:11]]},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["model_trained"] is True
    assert calls == [EXPECTED_CONFIG]
    _assert_user_model(_session(sid))
