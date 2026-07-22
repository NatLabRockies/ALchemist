"""
Integration tests for acquisition router endpoints.
"""

from fastapi.testclient import TestClient
import pytest

from api.main import app

client = TestClient(app)


def _create_session() -> str:
    response = client.post("/api/v1/sessions", json={"ttl_hours": 1})
    response.raise_for_status()
    return response.json()["session_id"]


def _add_variables(session_id: str) -> None:
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
        response = client.post(f"/api/v1/sessions/{session_id}/variables", json=payload)
        response.raise_for_status()


def _add_experiments(session_id: str) -> None:
    experiments = [
        {"inputs": {"temperature": 200, "pressure": 3}, "output": 0.65},
        {"inputs": {"temperature": 250, "pressure": 5}, "output": 0.85},
        {"inputs": {"temperature": 300, "pressure": 7}, "output": 0.92},
        {"inputs": {"temperature": 350, "pressure": 4}, "output": 0.78},
        {"inputs": {"temperature": 400, "pressure": 6}, "output": 0.88},
        {"inputs": {"temperature": 450, "pressure": 8}, "output": 0.81},
    ]
    response = client.post(
        f"/api/v1/sessions/{session_id}/experiments/batch",
        json={"experiments": experiments},
    )
    response.raise_for_status()


def _train_model(session_id: str, backend: str) -> None:
    payload = {
        "backend": backend,
        "kernel": "rbf",
    }
    if backend == "sklearn":
        payload["output_transform"] = "standardize"
    response = client.post(
        f"/api/v1/sessions/{session_id}/model/train",
        json=payload,
    )
    response.raise_for_status()
    assert response.json()["success"] is True


def _prepare_trained_session(backend: str = "sklearn") -> str:
    session_id = _create_session()
    _add_variables(session_id)
    _add_experiments(session_id)
    _train_model(session_id, backend)
    return session_id


def _cleanup_session(session_id: str) -> None:
    client.delete(f"/api/v1/sessions/{session_id}")


def test_suggest_requires_trained_model():
    session_id = _create_session()
    try:
        _add_variables(session_id)
        response = client.post(
            f"/api/v1/sessions/{session_id}/acquisition/suggest",
            json={
                "strategy": "ei",
                "goal": "maximize",
                "n_suggestions": 1,
            },
        )
        assert response.status_code == 400
        body = response.json()
        assert body["error_type"] == "NoModelError"
    finally:
        _cleanup_session(session_id)


def test_suggest_records_audit_log_with_parameters():
    session_id = _prepare_trained_session(backend="sklearn")
    try:
        request_payload = {
            "strategy": "ucb",
            "goal": "minimize",
            "n_suggestions": 2,
            "xi": 0.05,
            "kappa": 1.2,
        }
        response = client.post(
            f"/api/v1/sessions/{session_id}/acquisition/suggest",
            json=request_payload,
        )
        assert response.status_code == 200
        body = response.json()
        assert body["n_suggestions"] >= 1
        assert len(body["suggestions"]) == body["n_suggestions"]

        audit_response = client.get(
            f"/api/v1/sessions/{session_id}/audit",
            params={"entry_type": "acquisition_locked"},
        )
        assert audit_response.status_code == 200
        audit_body = audit_response.json()
        assert audit_body["n_entries"] >= 1
        latest_entry = audit_body["entries"][-1]
        assert latest_entry["entry_type"] == "acquisition_locked"
        locked_parameters = latest_entry["parameters"]
        assert locked_parameters["strategy"].lower() == request_payload["strategy"]
        assert locked_parameters["parameters"]["n_suggestions"] == request_payload["n_suggestions"]
        assert locked_parameters["parameters"]["kappa"] == pytest.approx(request_payload["kappa"])
        assert locked_parameters["parameters"]["xi"] == pytest.approx(request_payload["xi"])
        assert len(locked_parameters["suggestions"]) == body["n_suggestions"]
    finally:
        _cleanup_session(session_id)


@pytest.mark.parametrize(
    "backend, goal",
    [
        ("sklearn", "minimize"),
        ("botorch", "maximize"),
    ],
)
def test_find_optimum_supports_multiple_backends(backend: str, goal: str):
    session_id = _prepare_trained_session(backend=backend)
    try:
        response = client.post(
            f"/api/v1/sessions/{session_id}/acquisition/find-optimum",
            json={"goal": goal},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["goal"] == goal
        assert isinstance(body["optimum"], dict)
        assert body["predicted_value"] is not None
        assert set(body["optimum"].keys()) == {"temperature", "pressure"}
        # predicted_std may be None if the backend does not report uncertainty
        assert "predicted_std" in body
    finally:
        _cleanup_session(session_id)


@pytest.mark.parametrize("backend", ["sklearn", "botorch"])
def test_find_optimum_endpoint_respects_input_constraint(backend: str):
    """POST /acquisition/find-optimum returns a constraint-feasible optimum."""
    from api.services import session_store

    session_id = _prepare_trained_session(backend=backend)
    try:
        # Register an input constraint directly on the session object
        # (no HTTP endpoint exists for input constraints yet).
        session = session_store.get(session_id)
        session.add_input_constraint(
            'inequality', {'temperature': 1.0, 'pressure': 1.0}, rhs=250.0
        )

        response = client.post(
            f"/api/v1/sessions/{session_id}/acquisition/find-optimum",
            json={"goal": "maximize"},
        )
        assert response.status_code == 200
        opt = response.json()["optimum"]
        assert (opt["temperature"] + opt["pressure"]) <= 250.0 + 1.0
    finally:
        _cleanup_session(session_id)
