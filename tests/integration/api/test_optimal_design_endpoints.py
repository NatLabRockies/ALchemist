"""
Integration tests for optimal design API endpoints.
Tests POST /optimal-design/info (preview) and POST /optimal-design (generate).
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


@pytest.fixture
def session_id():
    response = client.post("/api/v1/sessions", json={"ttl_hours": 1})
    response.raise_for_status()
    sid = response.json()["session_id"]
    yield sid
    client.delete(f"/api/v1/sessions/{sid}")


def _add_variables(sid: str) -> None:
    """Add 3 continuous variables for a well-defined design space."""
    variables = [
        {"name": "Temperature", "type": "real", "min": 300.0, "max": 500.0},
        {"name": "Pressure", "type": "real", "min": 1.0, "max": 10.0},
        {"name": "Flow_Rate", "type": "real", "min": 0.5, "max": 5.0},
    ]
    for payload in variables:
        response = client.post(f"/api/v1/sessions/{sid}/variables", json=payload)
        response.raise_for_status()


# ============================================================
# POST /optimal-design/info tests
# ============================================================

class TestOptimalDesignInfo:
    """Tests for the model preview endpoint."""

    def test_info_requires_variables(self, session_id):
        response = client.post(
            f"/api/v1/sessions/{session_id}/optimal-design/info",
            json={"model_type": "linear"},
        )
        assert response.status_code == 400
        assert "no variables" in response.json()["detail"].lower()

    def test_info_linear(self, session_id):
        _add_variables(session_id)
        response = client.post(
            f"/api/v1/sessions/{session_id}/optimal-design/info",
            json={"model_type": "linear"},
        )
        assert response.status_code == 200
        body = response.json()
        assert "model_terms" in body
        assert "p_columns" in body
        assert "n_points_minimum" in body
        assert "n_points_recommended" in body
        # Linear: intercept + 3 main effects = 4 columns
        assert body["p_columns"] == 4
        assert body["n_points_minimum"] == 4
        assert body["n_points_recommended"] == 8
        assert "Intercept" in body["model_terms"]

    def test_info_quadratic(self, session_id):
        _add_variables(session_id)
        response = client.post(
            f"/api/v1/sessions/{session_id}/optimal-design/info",
            json={"model_type": "quadratic"},
        )
        assert response.status_code == 200
        body = response.json()
        # Quadratic: intercept + 3 main + 3 interactions + 3 squared = 10
        assert body["p_columns"] == 10
        assert body["n_points_recommended"] == 20

    def test_info_with_effects(self, session_id):
        _add_variables(session_id)
        response = client.post(
            f"/api/v1/sessions/{session_id}/optimal-design/info",
            json={"effects": ["Temperature", "Pressure", "Temperature*Pressure"]},
        )
        assert response.status_code == 200
        body = response.json()
        # Intercept + 3 effects = 4 columns
        assert body["p_columns"] == 4

    def test_info_rejects_both_model_type_and_effects(self, session_id):
        _add_variables(session_id)
        response = client.post(
            f"/api/v1/sessions/{session_id}/optimal-design/info",
            json={"model_type": "linear", "effects": ["Temperature"]},
        )
        assert response.status_code in (400, 422, 500)

    def test_info_rejects_neither(self, session_id):
        _add_variables(session_id)
        response = client.post(
            f"/api/v1/sessions/{session_id}/optimal-design/info",
            json={},
        )
        assert response.status_code in (400, 422, 500)


# ============================================================
# POST /optimal-design tests
# ============================================================

class TestOptimalDesignGenerate:
    """Tests for the design generation endpoint."""

    def test_generate_requires_variables(self, session_id):
        response = client.post(
            f"/api/v1/sessions/{session_id}/optimal-design",
            json={"model_type": "linear", "p_multiplier": 2.0},
        )
        assert response.status_code == 400
        assert "no variables" in response.json()["detail"].lower()

    def test_generate_with_model_type_and_multiplier(self, session_id):
        _add_variables(session_id)
        response = client.post(
            f"/api/v1/sessions/{session_id}/optimal-design",
            json={
                "model_type": "linear",
                "p_multiplier": 2.0,
                "criterion": "D",
                "algorithm": "fedorov",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert "points" in body
        assert "n_points" in body
        assert "design_info" in body
        assert body["n_points"] == len(body["points"])
        assert body["n_points"] >= 4  # at least p columns
        # Check design_info has expected keys
        info = body["design_info"]
        assert "criterion" in info
        assert "D_eff" in info
        assert info["criterion"] == "D"

    def test_generate_with_n_points(self, session_id):
        _add_variables(session_id)
        response = client.post(
            f"/api/v1/sessions/{session_id}/optimal-design",
            json={
                "model_type": "quadratic",
                "n_points": 15,
                "criterion": "D",
                "algorithm": "sequential",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["n_points"] == 15

    def test_generate_with_effects(self, session_id):
        _add_variables(session_id)
        response = client.post(
            f"/api/v1/sessions/{session_id}/optimal-design",
            json={
                "effects": ["Temperature", "Pressure", "Temperature**2"],
                "p_multiplier": 2.0,
                "criterion": "A",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["design_info"]["criterion"] == "A"
        # Check points have the right variable names
        if body["points"]:
            point = body["points"][0]
            assert "Temperature" in point
            assert "Pressure" in point
            assert "Flow_Rate" in point

    def test_generate_i_optimal(self, session_id):
        _add_variables(session_id)
        response = client.post(
            f"/api/v1/sessions/{session_id}/optimal-design",
            json={
                "model_type": "interaction",
                "p_multiplier": 2.0,
                "criterion": "I",
                "algorithm": "detmax",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["design_info"]["criterion"] == "I"

    def test_generate_rejects_both_npoints_and_multiplier(self, session_id):
        _add_variables(session_id)
        response = client.post(
            f"/api/v1/sessions/{session_id}/optimal-design",
            json={
                "model_type": "linear",
                "n_points": 10,
                "p_multiplier": 2.0,
            },
        )
        assert response.status_code in (400, 422, 500)

    def test_generate_rejects_neither_npoints_nor_multiplier(self, session_id):
        _add_variables(session_id)
        response = client.post(
            f"/api/v1/sessions/{session_id}/optimal-design",
            json={"model_type": "linear"},
        )
        assert response.status_code in (400, 422, 500)

    def test_generate_with_seed_is_reproducible(self, session_id):
        _add_variables(session_id)
        request = {
            "model_type": "linear",
            "p_multiplier": 2.0,
            "criterion": "D",
            "algorithm": "fedorov",
            "random_seed": 42,
        }
        r1 = client.post(
            f"/api/v1/sessions/{session_id}/optimal-design", json=request
        )
        r2 = client.post(
            f"/api/v1/sessions/{session_id}/optimal-design", json=request
        )
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.json()["points"] == r2.json()["points"]
