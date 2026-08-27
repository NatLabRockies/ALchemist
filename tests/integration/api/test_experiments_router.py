"""
Integration tests for experiments router covering error handling and
auto-training/initial-design success paths.
"""

import io
import json

import pytest
from fastapi.testclient import TestClient

from alchemist_core.session import OptimizationSession
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
    variables = [
        {"name": "temperature", "type": "real", "min": 100.0, "max": 500.0},
        {"name": "pressure", "type": "real", "min": 1.0, "max": 10.0},
    ]
    for payload in variables:
        response = client.post(f"/api/v1/sessions/{sid}/variables", json=payload)
        response.raise_for_status()


def test_batch_requires_variables(session_id):
    response = client.post(
        f"/api/v1/sessions/{session_id}/experiments/batch",
        json={"experiments": []},
    )
    assert response.status_code == 400
    assert "no variables" in response.json()["detail"].lower()


def test_upload_csv_requires_variables(session_id):
    csv_content = """temperature,pressure,Output\n200,3,0.5\n"""
    files = {"file": ("data.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    response = client.post(
        f"/api/v1/sessions/{session_id}/experiments/upload",
        files=files,
    )
    assert response.status_code == 400
    assert "no variables" in response.json()["detail"].lower()


def test_add_experiment_requires_variables(session_id):
    response = client.post(
        f"/api/v1/sessions/{session_id}/experiments",
        json={"inputs": {"temperature": 200}, "output": 0.5},
    )
    assert response.status_code == 400
    assert "no variables" in response.json()["detail"].lower()


def test_auto_train_handles_failure(session_id):
    _add_variables(session_id)

    exp_payload = {
        "inputs": {"temperature": 200, "pressure": 3},
        "output": 0.5,
    }
    response = client.post(
        f"/api/v1/sessions/{session_id}/experiments",
        params={"auto_train": "true", "training_backend": "unknown_backend"},
        json=exp_payload,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model_trained"] is False


def test_add_experiment_auto_trains_and_logs(session_id, monkeypatch):
    _add_variables(session_id)

    # Seed session with enough experiments to meet the auto-train threshold.
    for idx in range(4):
        payload = {
            "inputs": {"temperature": 200 + idx * 5, "pressure": 3 + idx * 0.5},
            "output": 0.5 + idx * 0.01,
        }
        resp = client.post(f"/api/v1/sessions/{session_id}/experiments", json=payload)
        resp.raise_for_status()

    train_call = {}

    def fake_train_model(self, backend="sklearn", kernel="rbf"):
        train_call["backend"] = backend
        train_call["kernel"] = kernel
        return {
            "metrics": {"rmse": 0.12, "r2": 0.9},
            "hyperparameters": {"kernel": kernel},
        }

    monkeypatch.setattr(OptimizationSession, "train_model", fake_train_model)

    locked = {}

    def fake_lock_model(self, backend, kernel, hyperparameters, cv_metrics, iteration, notes):
        locked.update(
            {
                "backend": backend,
                "kernel": kernel,
                "iteration": iteration,
                "notes": notes,
            }
        )

    monkeypatch.setattr("alchemist_core.audit_log.AuditLog.lock_model", fake_lock_model)

    final_payload = {
        "inputs": {"temperature": 225, "pressure": 4.5},
        "output": 0.61,
        "iteration": 2,
    }
    response = client.post(
        f"/api/v1/sessions/{session_id}/experiments",
        params={"auto_train": "true", "training_backend": "sklearn", "training_kernel": "matern"},
        json=final_payload,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model_trained"] is True
    assert body["training_metrics"] == {"rmse": 0.12, "r2": 0.9, "backend": "sklearn"}
    assert train_call == {"backend": "sklearn", "kernel": "matern"}
    assert locked["iteration"] == 2
    assert locked["backend"] == "sklearn"


def test_csv_upload_success(session_id):
    _add_variables(session_id)

    csv_content = """temperature,pressure,Output\n200,3,0.5\n210,4,0.6\n"""
    files = {"file": ("good.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    response = client.post(
        f"/api/v1/sessions/{session_id}/experiments/upload",
        files=files,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["n_experiments"] == 2

    summary = client.get(f"/api/v1/sessions/{session_id}/experiments").json()
    assert summary["n_experiments"] == 2

    stats = client.get(f"/api/v1/sessions/{session_id}/experiments/summary")
    stats.raise_for_status()
    summary_body = stats.json()
    assert summary_body["n_experiments"] == 2


def test_batch_auto_train_returns_metrics(session_id, monkeypatch):
    _add_variables(session_id)

    train_call = {}

    def fake_train_model(self, backend="sklearn", kernel="rbf"):
        train_call["backend"] = backend
        train_call["kernel"] = kernel
        return {
            "metrics": {"rmse": 0.2, "r2": 0.85},
        }

    monkeypatch.setattr(OptimizationSession, "train_model", fake_train_model)

    experiments = [
        {"inputs": {"temperature": 200 + i * 10, "pressure": 3 + i}, "output": 0.5 + i * 0.05}
        for i in range(5)
    ]

    response = client.post(
        f"/api/v1/sessions/{session_id}/experiments/batch",
        params={"auto_train": "true", "training_backend": "sklearn", "training_kernel": "rbf"},
        json={"experiments": experiments},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model_trained"] is True
    assert body["training_metrics"] == {"rmse": 0.2, "r2": 0.85, "backend": "sklearn"}
    assert train_call == {"backend": "sklearn", "kernel": "rbf"}


def test_initial_design_requires_variables(session_id):
    response = client.post(
        f"/api/v1/sessions/{session_id}/initial-design",
        json={"method": "lhs", "n_points": 3, "lhs_criterion": "maximin"},
    )
    assert response.status_code == 400
    assert "no variables" in response.json()["detail"].lower()


def test_initial_design_generates_points(session_id):
    _add_variables(session_id)

    response = client.post(
        f"/api/v1/sessions/{session_id}/initial-design",
        json={
            "method": "lhs",
            "n_points": 4,
            "lhs_criterion": "maximin",
            "random_seed": 123,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["n_points"] == 4
    assert body["method"] == "lhs"
    assert len(body["points"]) == 4


# ==========================================================================
# POST /initial-design returned 400 for any space containing an integer
# variable: the space-filling samplers handed back np.int64, which is the one
# numpy scalar that does not subclass its Python counterpart, so the JSON
# encoder rejected it with "Unable to serialize unknown type". np.float64 and
# np.str_ leaked through the same path but serialized by accident.
#
# Types are asserted with `type(v) is T`, never isinstance: np.float64 passes
# isinstance(v, float) and np.str_ passes isinstance(v, str).
# ==========================================================================

SPACE_FILLING = ["random", "lhs", "sobol", "halton", "hammersly"]

# REST spells a categorical's values as `categories`, and types them as
# List[str], so the int-categorical case is reachable only from the core API.
_VAR_PAYLOADS = {
    "real": {"name": "x1", "type": "real", "min": 0.0, "max": 10.0},
    "integer": {"name": "x2", "type": "integer", "min": 0, "max": 10},
    "discrete": {"name": "x3", "type": "discrete", "allowed_values": [1.0, 2.0, 4.0]},
    "categorical": {"name": "x4", "type": "categorical", "categories": ["a", "b", "c"]},
}
_EXPECTED_TYPE = {"x1": float, "x2": int, "x3": float, "x4": str}


def _add_typed_variables(sid, kinds):
    for kind in kinds:
        r = client.post(f"/api/v1/sessions/{sid}/variables", json=_VAR_PAYLOADS[kind])
        r.raise_for_status()


def _post_design(sid, method, n_points=8):
    return client.post(
        f"/api/v1/sessions/{sid}/initial-design",
        json={"method": method, "n_points": n_points, "random_seed": 7},
    )


@pytest.mark.parametrize("method", SPACE_FILLING)
def test_initial_design_integer_variable_is_200(session_id, method):
    """The regression itself: an integer-only space used to 400."""
    _add_typed_variables(session_id, ["integer"])
    r = _post_design(session_id, method)
    assert r.status_code == 200, r.text
    points = r.json()["points"]
    assert len(points) == 8
    for p in points:
        assert type(p["x2"]) is int


# Every mixed combination of the four types, not just the full set: the leak
# was per-type, so a combination could reintroduce one type's leak alone.
_COMBOS = [
    ("real", "integer"),
    ("real", "discrete"),
    ("real", "categorical"),
    ("integer", "discrete"),
    ("integer", "categorical"),
    ("discrete", "categorical"),
    ("real", "integer", "discrete"),
    ("real", "integer", "categorical"),
    ("real", "discrete", "categorical"),
    ("integer", "discrete", "categorical"),
    ("real", "integer", "discrete", "categorical"),
]


@pytest.mark.parametrize("kinds", _COMBOS, ids=lambda k: "+".join(k))
def test_initial_design_mixed_types_are_200(session_id, kinds):
    _add_typed_variables(session_id, kinds)
    r = _post_design(session_id, "lhs")
    assert r.status_code == 200, r.text
    expected_names = {_VAR_PAYLOADS[k]["name"] for k in kinds}
    for p in r.json()["points"]:
        # The key set, not just the values found. Iterating p.items() alone
        # says nothing about a variable that is *missing* -- a design that
        # silently dropped one passes a values-only loop, which is exactly the
        # hole a context variable used to open here (see Task 12E).
        assert set(p) == expected_names, (
            f"design point carries {sorted(p)}, requested {sorted(expected_names)}"
        )
        for name, value in p.items():
            assert type(value) is _EXPECTED_TYPE[name], (
                f"{name} came back as {type(value).__name__} for {kinds}"
            )


@pytest.mark.parametrize("method", SPACE_FILLING)
def test_initial_design_all_types_json_native_over_the_wire(session_id, method):
    _add_typed_variables(session_id, list(_VAR_PAYLOADS))
    r = _post_design(session_id, method)
    assert r.status_code == 200, r.text
    expected_names = {v["name"] for v in _VAR_PAYLOADS.values()}
    for p in r.json()["points"]:
        # See the note on test_initial_design_mixed_types_are_200: a
        # values-only loop is blind to an omitted variable.
        assert set(p) == expected_names, (
            f"design point carries {sorted(p)}, registered {sorted(expected_names)}"
        )
        for name, value in p.items():
            assert type(value) is _EXPECTED_TYPE[name]


def test_initial_design_with_constraint_and_integer_variable(session_id):
    """DoE call + registered constraint + integer variable, over REST.

    This combination existed nowhere in the suite before, because the endpoint
    400'd on any integer variable -- which is why the branch's headline
    capability was end-to-end unverified for integers.
    """
    _add_typed_variables(session_id, ["real", "integer"])
    r = client.post(
        f"/api/v1/sessions/{session_id}/constraints",
        json={
            "constraint_type": "inequality",
            "coefficients": {"x1": 1.0, "x2": 1.0},
            "rhs": 10.0,
            "name": "budget",
        },
    )
    assert r.status_code == 200, r.text

    r = _post_design(session_id, "lhs")
    assert r.status_code == 200, r.text
    points = r.json()["points"]
    assert len(points) == 8
    for p in points:
        assert type(p["x1"]) is float
        assert type(p["x2"]) is int
        assert p["x1"] + p["x2"] <= 10.0 + 1e-9


# ==========================================================================
# Task 12E: a `context` variable in a non-final position made the space-filling
# design zip a 2-value sample against 3 names -- dropping the real variable
# behind it and labelling that variable's value onto the context one. With a
# constraint registered, `filter_feasible` then half-evaluated the constraint
# (it sums only the terms whose column is present) and the endpoint returned
# points violating the constraint while reporting 200.
#
# `POST /variables` has no context request model, so the only REST route to a
# context variable is an upload -- which is how the desktop and web frontends
# put one in a session too.
# ==========================================================================

_CONTEXT_POSITIONS = {
    "first": ["c1", "x1", "x5"],
    "middle": ["x1", "c1", "x5"],
    "last": ["x1", "x5", "c1"],
}


def _load_space_with_context(sid, position, constraints=()):
    order = _CONTEXT_POSITIONS[position]
    variables = []
    for name in order:
        if name == "c1":
            variables.append({"name": "c1", "type": "context"})
        else:
            variables.append({"name": name, "type": "real", "min": 0.0, "max": 5.0})
    payload = {"variables": variables, "constraints": list(constraints)}
    buf = io.BytesIO(json.dumps(payload).encode())
    r = client.post(
        f"/api/v1/sessions/{sid}/variables/load",
        files={"file": ("space.json", buf, "application/json")},
    )
    assert r.status_code == 200, r.text


@pytest.mark.parametrize("position", sorted(_CONTEXT_POSITIONS))
@pytest.mark.parametrize("method", SPACE_FILLING)
def test_initial_design_omits_context_variables_over_the_wire(
    session_id, position, method
):
    _load_space_with_context(session_id, position)
    r = _post_design(session_id, method, n_points=6)
    assert r.status_code == 200, r.text
    points = r.json()["points"]
    assert len(points) == 6
    for p in points:
        # The key set: the defect's signature is a missing key, and the design
        # that dropped x5 still answered every question about x1 correctly.
        assert set(p) == {"x1", "x5"}, (
            f"method={method} context={position}: point carries {sorted(p)}"
        )


@pytest.mark.parametrize("position", sorted(_CONTEXT_POSITIONS))
def test_constrained_initial_design_with_context_is_feasible_over_the_wire(
    session_id, position
):
    """The severe face, end to end: 200 with points that violated the constraint.

    The constraint is summed here in the test rather than asserted through
    `SearchSpace.filter_feasible`, which is the function that half-evaluated it.
    """
    _load_space_with_context(
        session_id,
        position,
        constraints=[
            {"type": "inequality", "coefficients": {"x1": 1.0, "x5": 1.0}, "rhs": 5.0}
        ],
    )
    r = _post_design(session_id, "lhs", n_points=6)
    assert r.status_code == 200, r.text
    points = r.json()["points"]
    assert len(points) == 6
    for p in points:
        assert set(p) == {"x1", "x5"}, sorted(p)
        assert p["x1"] + p["x5"] <= 5.0 + 1e-9, (
            f"context={position}: 200 with an infeasible point {p} "
            f"(x1 + x5 = {p['x1'] + p['x5']})"
        )
