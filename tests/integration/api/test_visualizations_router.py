"""
Integration tests for visualization endpoints.
"""

from fastapi.testclient import TestClient
import pytest

from alchemist_core.data.experiment_manager import PROVENANCE_COL
from api.main import app

client = TestClient(app)


def _create_trained_session(via_queue: bool = False, backend: str = "sklearn") -> str:
    response = client.post("/api/v1/sessions", json={"ttl_hours": 1})
    response.raise_for_status()
    session_id = response.json()["session_id"]

    # Define search space
    variables = [
        {
            "name": "temperature",
            "type": "real",
            "min": 100.0,
            "max": 500.0,
        },
        {
            "name": "pressure",
            "type": "real",
            "min": 1.0,
            "max": 10.0,
        },
    ]
    for payload in variables:
        add_response = client.post(f"/api/v1/sessions/{session_id}/variables", json=payload)
        add_response.raise_for_status()

    experiments = [
        {"inputs": {"temperature": 200, "pressure": 3}, "output": 0.65},
        {"inputs": {"temperature": 250, "pressure": 5}, "output": 0.85},
        {"inputs": {"temperature": 300, "pressure": 7}, "output": 0.92},
        {"inputs": {"temperature": 350, "pressure": 4}, "output": 0.78},
        {"inputs": {"temperature": 400, "pressure": 6}, "output": 0.88},
        {"inputs": {"temperature": 450, "pressure": 8}, "output": 0.81},
    ]
    if via_queue:
        # Record results the way production does (stage -> start -> complete).
        # Every queue completion stamps a ProvenanceId column into the table.
        stage_response = client.post(
            f"/api/v1/sessions/{session_id}/experiments/queue",
            json={"items": [{"inputs": e["inputs"]} for e in experiments]},
        )
        stage_response.raise_for_status()
        for item, experiment in zip(stage_response.json()["items"], experiments):
            item_url = f"/api/v1/sessions/{session_id}/experiments/queue/{item['id']}"
            client.post(f"{item_url}/start").raise_for_status()
            client.post(
                f"{item_url}/complete", json={"outputs": [experiment["output"]]}
            ).raise_for_status()
    else:
        batch_response = client.post(
            f"/api/v1/sessions/{session_id}/experiments/batch",
            json={"experiments": experiments},
        )
        batch_response.raise_for_status()

    train_response = client.post(
        f"/api/v1/sessions/{session_id}/model/train",
        json={
            "backend": backend,
            "kernel": "rbf",
            "output_transform": "standardize",
        },
    )
    train_response.raise_for_status()
    assert train_response.json()["success"] is True

    return session_id


def _cleanup_session(session_id: str) -> None:
    client.delete(f"/api/v1/sessions/{session_id}")


def test_contour_visualization_success_and_validation():
    session_id = _create_trained_session()
    try:
        contour_payload = {
            "x_var": "temperature",
            "y_var": "pressure",
            "grid_resolution": 20,
            "fixed_values": {},
            "include_experiments": True,
            "include_suggestions": False,
        }
        response = client.post(
            f"/api/v1/sessions/{session_id}/visualizations/contour",
            json=contour_payload,
        )
        assert response.status_code == 200
        body = response.json()
        assert body["x_var"] == "temperature"
        assert body["y_var"] == "pressure"
        assert len(body["predictions"]) == contour_payload["grid_resolution"]
        assert body["experiments"] is not None

        error_payload = {
            "x_var": "temperature",
            "y_var": "temperature",
            "grid_resolution": 10,
        }
        error_response = client.post(
            f"/api/v1/sessions/{session_id}/visualizations/contour",
            json=error_payload,
        )
        assert error_response.status_code == 400
        assert "must be different" in error_response.json()["detail"].lower()
    finally:
        _cleanup_session(session_id)


@pytest.mark.parametrize("backend", ["sklearn", "botorch"])
def test_contour_on_queue_built_session_ignores_provenance_column(backend: str):
    """ProvenanceId is metadata, not a model input. If the contour grid keeps
    it as a feature, it becomes an all-NaN extra input and predict fails
    (500): a dimension error on BoTorch, a NaN error on sklearn."""
    session_id = _create_trained_session(via_queue=True, backend=backend)
    try:
        # Premise: the queue stamped a ProvenanceId on every row.
        listing = client.get(f"/api/v1/sessions/{session_id}/experiments")
        rows = listing.json()["experiments"]
        assert rows and all(row.get(PROVENANCE_COL) for row in rows)

        grid_resolution = 20
        response = client.post(
            f"/api/v1/sessions/{session_id}/visualizations/contour",
            json={
                "x_var": "temperature",
                "y_var": "pressure",
                "grid_resolution": grid_resolution,
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        for key in ("predictions", "uncertainties"):
            assert len(body[key]) == grid_resolution
            assert all(len(row) == grid_resolution for row in body[key])
    finally:
        _cleanup_session(session_id)


def test_visualization_analysis_endpoints():
    session_id = _create_trained_session()
    try:
        parity_response = client.get(
            f"/api/v1/sessions/{session_id}/visualizations/parity"
        )
        assert parity_response.status_code == 200
        parity_body = parity_response.json()
        assert parity_body["metrics"]["rmse"] >= 0

        calibrated_response = client.get(
            f"/api/v1/sessions/{session_id}/visualizations/parity",
            params={"use_calibrated": "true"},
        )
        assert calibrated_response.status_code == 200

        metrics_response = client.get(
            f"/api/v1/sessions/{session_id}/visualizations/metrics",
            params={"cv_splits": 3},
        )
        assert metrics_response.status_code == 200
        metrics_body = metrics_response.json()
        assert metrics_body["training_sizes"][0] == 5

        qq_response = client.get(
            f"/api/v1/sessions/{session_id}/visualizations/qq-plot",
            params={"use_calibrated": "true"},
        )
        assert qq_response.status_code == 200
        qq_body = qq_response.json()
        assert qq_body["n_samples"] >= 5

        calibration_response = client.get(
            f"/api/v1/sessions/{session_id}/visualizations/calibration-curve"
        )
        assert calibration_response.status_code == 200
        calibration_body = calibration_response.json()
        assert calibration_body["n_samples"] >= 5

        hyperparams_response = client.get(
            f"/api/v1/sessions/{session_id}/visualizations/hyperparameters"
        )
        assert hyperparams_response.status_code == 200
        hyperparams_body = hyperparams_response.json()
        assert hyperparams_body["backend"] == "sklearn"
        assert "hyperparameters" in hyperparams_body
    finally:
        _cleanup_session(session_id)
