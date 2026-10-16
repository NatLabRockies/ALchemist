"""
GET /sessions/{id}/model returns the session's saved training config, so the
web app's Model tab can show the backend, kernel and transforms a joined or
loaded session was trained with instead of its own defaults (W31).
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

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

_rng = np.random.default_rng(11)
POINTS = [
    {"temperature": float(t), "pressure": float(p)}
    for t, p in zip(_rng.uniform(100, 500, 8), _rng.uniform(1, 10, 8))
]


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
    for point in POINTS:
        output = float((point["temperature"] / 500.0) ** 2 + point["pressure"] / 10.0)
        resp = client.post(f"/api/v1/sessions/{session_id}/experiments", json={"inputs": point, "output": output})
        assert resp.status_code == 200, resp.text
    yield session_id
    client.delete(f"/api/v1/sessions/{session_id}")


def test_model_info_has_no_config_before_training(sid):
    data = client.get(f"/api/v1/sessions/{sid}/model").json()
    assert data["is_trained"] is False
    assert data["training_config"] is None


def test_model_info_returns_the_training_config(sid):
    resp = client.post(f"/api/v1/sessions/{sid}/model/train", json=MODEL_TAB_REQUEST)
    assert resp.status_code == 200, resp.text

    data = client.get(f"/api/v1/sessions/{sid}/model").json()
    assert data["is_trained"] is True
    assert data["backend"] == "botorch"
    assert data["training_config"] == EXPECTED_CONFIG
