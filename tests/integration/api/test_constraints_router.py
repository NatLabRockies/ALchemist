"""Constraint CRUD over REST.

Before this, constraints could only be set from Python, so a non-Python
consumer could not use the constraint feature at all.

Routers mount under /api/v1 (api/main.py:61-68). Setup mirrors
tests/integration/api/test_optimal_design_endpoints.py.
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


def _add_variables(sid, names=("x1", "x2")):
    for name in names:
        r = client.post(
            f"/api/v1/sessions/{sid}/variables",
            json={"name": name, "type": "real", "min": 0.0, "max": 10.0},
        )
        r.raise_for_status()


class TestConstraintCRUD:
    def test_add_constraint(self, session_id):
        _add_variables(session_id)
        r = client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": {"x1": 1.0, "x2": 1.0},
            "rhs": 10.0,
            "name": "half_plane_1",
        })
        assert r.status_code == 200
        assert r.json()["constraint"]["name"] == "half_plane_1"

    def test_list_constraints(self, session_id):
        _add_variables(session_id)
        client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": {"x1": 1.0, "x2": 1.0},
            "rhs": 10.0, "name": "c_a",
        })
        r = client.get(f"/api/v1/sessions/{session_id}/constraints")
        assert r.status_code == 200
        body = r.json()
        assert body["n_constraints"] == 1
        assert body["constraints"][0]["name"] == "c_a"

    def test_delete_constraint_by_name(self, session_id):
        _add_variables(session_id)
        client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "inequality", "coefficients": {"x1": 1.0},
            "rhs": 5.0, "name": "c_a",
        })
        client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "inequality", "coefficients": {"x2": 1.0},
            "rhs": 5.0, "name": "c_b",
        })
        r = client.delete(f"/api/v1/sessions/{session_id}/constraints/c_a")
        assert r.status_code == 200
        remaining = client.get(f"/api/v1/sessions/{session_id}/constraints").json()
        assert [c["name"] for c in remaining["constraints"]] == ["c_b"]

    def test_delete_unknown_constraint_is_404(self, session_id):
        _add_variables(session_id)
        client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": {"x1": 3.0, "x2": -2.0}, "rhs": 4.0, "name": "c_a",
        })
        r = client.delete(f"/api/v1/sessions/{session_id}/constraints/nope")
        assert r.status_code == 404
        # An absent route is also a 404, so pin the handler's own body: without
        # these the test passes against no implementation at all.
        detail = r.json()["detail"]
        assert "nope" in detail
        assert "c_a" in detail

    def test_constraint_on_unknown_variable_is_400(self, session_id):
        _add_variables(session_id)
        r = client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": {"nope": 1.0}, "rhs": 5.0,
        })
        assert r.status_code == 400

    def test_constraint_on_categorical_is_400(self, session_id):
        _add_variables(session_id)
        client.post(f"/api/v1/sessions/{session_id}/variables",
                    json={"name": "cat", "type": "categorical",
                          "categories": ["a", "b"]})
        r = client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": {"cat": 1.0}, "rhs": 5.0,
        })
        assert r.status_code == 400
        assert "not numeric" in r.json()["detail"]

    def test_registered_constraint_is_honored_by_a_design(self, session_id):
        _add_variables(session_id)
        client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": {"x1": 1.0, "x2": 1.0}, "rhs": 8.0,
        })
        r = client.post(f"/api/v1/sessions/{session_id}/initial-design",
                        json={"method": "lhs", "n_points": 6, "random_seed": 7})
        assert r.status_code == 200
        for pt in r.json()["points"]:
            assert pt["x1"] + pt["x2"] <= 8.0 + 1e-6


def _add_mixed_variables(sid):
    """A real, an integer and a discrete variable -- all constraint-eligible.

    The plan's own tests are all-`real` apart from the categorical rejection
    case. A real-only suite cannot see a bug that only bites the integer or
    discrete DoE path, so the mixed space below is what the success cases use.
    """
    payloads = [
        {"name": "x1", "type": "real", "min": 0.0, "max": 10.0},
        {"name": "x2", "type": "integer", "min": 0, "max": 10},
        {"name": "x3", "type": "discrete", "allowed_values": [0.0, 2.0, 4.0, 6.0, 8.0]},
    ]
    for payload in payloads:
        r = client.post(f"/api/v1/sessions/{sid}/variables", json=payload)
        r.raise_for_status()


class TestConstraintVariableTypes:
    """Constraints over integer and discrete variables, not just real ones."""

    def test_constraint_over_integer_and_discrete_is_accepted(self, session_id):
        _add_mixed_variables(session_id)
        r = client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": {"x2": 3.0, "x3": -2.0},
            "rhs": 6.0,
            "name": "int_disc",
        })
        assert r.status_code == 200, r.json()
        assert r.json()["constraint"]["name"] == "int_disc"

    def test_empty_coefficients_is_rejected(self, session_id):
        """A constraint over no variables constrains nothing; reject the shape."""
        _add_mixed_variables(session_id)
        r = client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "inequality", "coefficients": {}, "rhs": 5.0,
        })
        assert r.status_code == 422
        assert "too_short" in str(r.json()["detail"])
        assert client.get(
            f"/api/v1/sessions/{session_id}/constraints"
        ).json()["n_constraints"] == 0

    def test_unknown_constraint_type_is_rejected(self, session_id):
        _add_mixed_variables(session_id)
        r = client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "nonsense",
            "coefficients": {"x1": 3.0}, "rhs": 5.0,
        })
        assert r.status_code == 422

    def test_constraint_coefficients_round_trip_unchanged(self, session_id):
        """Asymmetric, non-unit coefficients: unit ones hide magnitude bugs."""
        _add_mixed_variables(session_id)
        client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "equality",
            "coefficients": {"x1": 3.0, "x2": -2.0, "x3": 0.5},
            "rhs": -4.5,
            "name": "asym",
        })
        body = client.get(f"/api/v1/sessions/{session_id}/constraints").json()
        stored = body["constraints"][0]
        assert stored["coefficients"] == {"x1": 3.0, "x2": -2.0, "x3": 0.5}
        assert stored["rhs"] == -4.5
        assert stored["type"] == "equality"

    def test_discrete_constraint_is_honored_by_a_design(self, session_id):
        """The registered constraint must reach the DoE for non-real types too.

        The space is real + discrete rather than real + integer: POST
        /initial-design returns 400 "Unable to serialize unknown type:
        <class 'numpy.int64'>" for ANY search space holding an integer
        variable, with no constraints registered at all. That is a
        pre-existing defect in the design-response encoder, unrelated to
        constraints, and out of scope here -- see the Task 11 report.
        Integer coefficients are still covered by
        test_constraint_over_integer_and_discrete_is_accepted and by
        tests/unit/core/data/test_constraints.py.
        """
        for payload in (
            {"name": "x1", "type": "real", "min": 0.0, "max": 10.0},
            {"name": "x3", "type": "discrete",
             "allowed_values": [0.0, 2.0, 4.0, 6.0, 8.0]},
        ):
            client.post(f"/api/v1/sessions/{session_id}/variables",
                        json=payload).raise_for_status()
        client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": {"x1": 3.0, "x3": -2.0}, "rhs": 0.0,
        })
        r = client.post(f"/api/v1/sessions/{session_id}/initial-design",
                        json={"method": "lhs", "n_points": 6, "random_seed": 11})
        assert r.status_code == 200, r.json()
        points = r.json()["points"]
        assert len(points) == 6
        for pt in points:
            assert 3.0 * pt["x1"] - 2.0 * pt["x3"] <= 1e-6, pt


class TestConstraintNameUniquenessOverRest:
    """Ruling 28: one DELETE must remove exactly one constraint.

    Auto-names used to be ``constraint_{len(constraints)}``, which repeats an
    index as soon as anything is removed. This endpoint is the first removal
    path in the codebase, so it is the first thing able to trigger it: three
    unnamed adds, one delete, one more unnamed add produced two constraints
    called ``constraint_2``, and deleting that name then silently dropped both
    while reporting a single name and a 200.
    """

    def _names(self, sid):
        body = client.get(f"/api/v1/sessions/{sid}/constraints").json()
        return [c["name"] for c in body["constraints"]]

    def _add_unnamed(self, sid, coefficients, rhs):
        r = client.post(f"/api/v1/sessions/{sid}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": coefficients, "rhs": rhs,
        })
        r.raise_for_status()
        return r.json()["constraint"]["name"]

    def test_add_delete_add_keeps_auto_names_unique(self, session_id):
        _add_mixed_variables(session_id)
        self._add_unnamed(session_id, {"x1": 3.0}, 5.0)
        self._add_unnamed(session_id, {"x2": -2.0}, 4.0)
        self._add_unnamed(session_id, {"x3": 0.5}, 3.0)
        assert self._names(session_id) == [
            "constraint_0", "constraint_1", "constraint_2",
        ]

        assert client.delete(
            f"/api/v1/sessions/{session_id}/constraints/constraint_0"
        ).status_code == 200

        fourth = self._add_unnamed(session_id, {"x1": 3.0, "x2": -2.0}, 7.0)
        names = self._names(session_id)
        assert len(set(names)) == len(names), names
        assert fourth == "constraint_3"
        assert names == ["constraint_1", "constraint_2", "constraint_3"]

    def test_delete_removes_exactly_one_constraint(self, session_id):
        _add_mixed_variables(session_id)
        for coefficients, rhs in [
            ({"x1": 3.0}, 5.0), ({"x2": -2.0}, 4.0), ({"x3": 0.5}, 3.0),
        ]:
            self._add_unnamed(session_id, coefficients, rhs)
        client.delete(f"/api/v1/sessions/{session_id}/constraints/constraint_0")
        self._add_unnamed(session_id, {"x1": 3.0, "x2": -2.0}, 7.0)

        before = len(self._names(session_id))
        r = client.delete(f"/api/v1/sessions/{session_id}/constraints/constraint_2")
        assert r.status_code == 200
        after = self._names(session_id)
        assert len(after) == before - 1, after
        assert after == ["constraint_1", "constraint_3"]

    def test_duplicate_explicit_name_is_400(self, session_id):
        _add_mixed_variables(session_id)
        first = client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": {"x1": 3.0}, "rhs": 5.0, "name": "half_plane",
        })
        assert first.status_code == 200
        second = client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": {"x2": -2.0}, "rhs": 4.0, "name": "half_plane",
        })
        assert second.status_code == 400
        assert "already registered" in second.json()["detail"]
        # The rejected constraint must not have been stored.
        assert self._names(session_id) == ["half_plane"]

    def test_explicit_name_does_not_collide_with_a_later_auto_name(self, session_id):
        _add_mixed_variables(session_id)
        r = client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": {"x1": 3.0}, "rhs": 5.0, "name": "constraint_0",
        })
        assert r.status_code == 200
        auto = self._add_unnamed(session_id, {"x2": -2.0}, 4.0)
        assert auto == "constraint_1"
        names = self._names(session_id)
        assert len(set(names)) == len(names), names
