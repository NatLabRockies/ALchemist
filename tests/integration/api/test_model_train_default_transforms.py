"""W2: a train request without transforms gets BoTorch's normalize/standardize."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.services import session_store

client = TestClient(app)


@pytest.fixture
def sid():
    session_id = client.post("/api/v1/sessions", json={"ttl_hours": 1}).json()["session_id"]
    for name in ("x1", "x2"):
        client.post(
            f"/api/v1/sessions/{session_id}/variables",
            json={"name": name, "type": "real", "min": 0.0, "max": 1000.0},
        ).raise_for_status()
    rng = np.random.default_rng(0)
    for a, b in rng.uniform(0, 1000, size=(15, 2)):
        client.post(
            f"/api/v1/sessions/{session_id}/experiments",
            json={
                "inputs": {"x1": float(a), "x2": float(b)},
                "output": float(1e4 - (a - 300) ** 2 / 10 - (b - 600) ** 2 / 10),
            },
        ).raise_for_status()
    yield session_id
    client.delete(f"/api/v1/sessions/{session_id}")


def _train(sid, **body):
    response = client.post(f"/api/v1/sessions/{sid}/model/train", json=body)
    response.raise_for_status()
    return response


def _transforms(sid):
    """The scaling on the live trained model, not an echoed request."""
    live = session_store.get(sid)
    return live.model.input_transform_type, live.model.output_transform_type


def test_omitted_transforms_default_to_normalize_standardize(sid):
    _train(sid, backend="botorch", kernel="Matern")
    assert _transforms(sid) == ("normalize", "standardize")


def test_explicit_none_still_means_none(sid):
    _train(sid, backend="botorch", kernel="Matern", input_transform="none", output_transform="none")
    assert _transforms(sid) == ("none", "none")


def test_only_one_transform_given_other_defaults(sid):
    _train(sid, backend="botorch", kernel="Matern", input_transform="none")
    assert _transforms(sid) == ("none", "standardize")


def test_sklearn_default_unchanged(sid):
    _train(sid, backend="sklearn", kernel="Matern")
    assert _transforms(sid)[0] in (None, "none")
