"""Constraint CRUD over REST.

Before this, constraints could only be set from Python, so a non-Python
consumer could not use the constraint feature at all.

Routers mount under /api/v1 (api/main.py:61-68). Setup mirrors
tests/integration/api/test_optimal_design_endpoints.py.
"""

import io
import json
import os
import tempfile
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.services import session_store

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


class TestDeleteRemovesExactlyOneConstraint:
    """Fix 1: the DELETE filter removed *every* constraint sharing a name.

    ``add_constraint`` rejects a duplicate explicit name (Ruling 28), but that
    is not the only way a constraint reaches ``search_space.constraints``.
    ``SearchSpace.load_from_json`` assigns the list straight from the file and
    never calls ``add_constraint``, so a loaded search space can hold
    duplicates that no guard ever saw. Against that list the shipped filter
    ``[c for c in existing if c["name"] != name]`` dropped both while reporting
    a single deletion and a 200.
    """

    @staticmethod
    def _load_constraints_bypassing_add(session_id, constraints):
        """Install constraints via load_from_json, bypassing add_constraint."""
        space = session_store.get(session_id).search_space
        fd, filepath = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            space.save_to_json(filepath)
            with open(filepath) as f:
                raw = json.load(f)
            raw["constraints"] = constraints
            with open(filepath, "w") as f:
                json.dump(raw, f)
            space.load_from_json(filepath)
        finally:
            os.unlink(filepath)

    def test_delete_removes_one_of_two_constraints_sharing_a_name(self, session_id):
        _add_mixed_variables(session_id)
        # Asymmetric, non-unit coefficients over three different variable
        # types: the two 'dup' entries differ in every field but the name, so
        # the survivor identifies which one was removed.
        self._load_constraints_bypassing_add(session_id, [
            {"type": "inequality", "coefficients": {"x1": 3.0}, "rhs": 5.0,
             "name": "dup"},
            {"type": "equality", "coefficients": {"x2": -2.0}, "rhs": 4.0,
             "name": "dup"},
            {"type": "inequality", "coefficients": {"x3": 0.5}, "rhs": 3.0,
             "name": "keep"},
        ])
        assert [c["name"] for c in client.get(
            f"/api/v1/sessions/{session_id}/constraints"
        ).json()["constraints"]] == ["dup", "dup", "keep"]

        r = client.delete(f"/api/v1/sessions/{session_id}/constraints/dup")
        assert r.status_code == 200

        remaining = client.get(
            f"/api/v1/sessions/{session_id}/constraints"
        ).json()["constraints"]
        # Exactly one removed, not both.
        assert [c["name"] for c in remaining] == ["dup", "keep"], remaining
        # The *first* match went; the second is the one still standing.
        assert remaining[0]["type"] == "equality"
        assert remaining[0]["coefficients"] == {"x2": -2.0}
        assert remaining[0]["rhs"] == 4.0

    def test_second_delete_removes_the_remaining_duplicate(self, session_id):
        """The survivor is still addressable: two calls remove two."""
        _add_mixed_variables(session_id)
        self._load_constraints_bypassing_add(session_id, [
            {"type": "inequality", "coefficients": {"x1": 3.0}, "rhs": 5.0,
             "name": "dup"},
            {"type": "equality", "coefficients": {"x2": -2.0}, "rhs": 4.0,
             "name": "dup"},
        ])
        assert client.delete(
            f"/api/v1/sessions/{session_id}/constraints/dup"
        ).status_code == 200
        assert client.delete(
            f"/api/v1/sessions/{session_id}/constraints/dup"
        ).status_code == 200
        assert client.get(
            f"/api/v1/sessions/{session_id}/constraints"
        ).json()["n_constraints"] == 0
        # Third call has nothing left to address.
        assert client.delete(
            f"/api/v1/sessions/{session_id}/constraints/dup"
        ).status_code == 404


# A raw body is required for these: TestClient(json=...) refuses to serialize
# NaN/Infinity client-side, so the values would never reach the endpoint. A
# real HTTP client sends exactly this.
_JSON_HEADERS = {"Content-Type": "application/json"}


class TestNonFiniteConstraintValues:
    """Fix 2: non-finite rhs and coefficients were accepted over REST.

    Two consequences. The resource stopped being round-trippable -- the API
    emitted ``null`` for a value it had accepted, and that output cannot be
    POSTed back. And a NaN rhs makes every point infeasible, which drives the
    DoE into a pathological resampling path with no infeasible geometry
    involved at all.
    """

    def _post_raw(self, session_id, body):
        return client.post(
            f"/api/v1/sessions/{session_id}/constraints",
            content=body, headers=_JSON_HEADERS,
        )

    @pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e400"])
    def test_non_finite_rhs_is_rejected(self, session_id, literal):
        _add_mixed_variables(session_id)
        r = self._post_raw(session_id, (
            '{"constraint_type":"inequality","coefficients":{"x1":3.0},'
            f'"rhs":{literal}}}'
        ))
        assert r.status_code == 400, r.text
        assert "finite" in r.json()["detail"]
        assert client.get(
            f"/api/v1/sessions/{session_id}/constraints"
        ).json()["n_constraints"] == 0

    @pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e400"])
    @pytest.mark.parametrize("variable", ["x1", "x2", "x3"])
    def test_non_finite_coefficient_is_rejected(self, session_id, literal, variable):
        """Every constraint-eligible variable type, not just real."""
        _add_mixed_variables(session_id)
        r = self._post_raw(session_id, (
            '{"constraint_type":"equality","coefficients":'
            f'{{"{variable}":{literal}}},"rhs":5.0}}'
        ))
        assert r.status_code == 400, r.text
        assert "finite" in r.json()["detail"]
        assert variable in r.json()["detail"]
        assert client.get(
            f"/api/v1/sessions/{session_id}/constraints"
        ).json()["n_constraints"] == 0

    def test_one_non_finite_coefficient_rejects_the_whole_constraint(self, session_id):
        """A partly-finite coefficient map must not be stored in part."""
        _add_mixed_variables(session_id)
        r = self._post_raw(session_id, (
            '{"constraint_type":"inequality",'
            '"coefficients":{"x1":3.0,"x2":NaN,"x3":0.5},"rhs":-4.5}'
        ))
        assert r.status_code == 400, r.text
        assert client.get(
            f"/api/v1/sessions/{session_id}/constraints"
        ).json()["n_constraints"] == 0

    def test_large_but_finite_values_are_still_accepted(self, session_id):
        """The check rejects non-finite, not merely large. 1e308 is finite."""
        _add_mixed_variables(session_id)
        r = self._post_raw(session_id, (
            '{"constraint_type":"inequality","coefficients":{"x1":1e308},'
            '"rhs":-1e308,"name":"huge"}'
        ))
        assert r.status_code == 200, r.text
        stored = client.get(
            f"/api/v1/sessions/{session_id}/constraints"
        ).json()["constraints"][0]
        assert stored["coefficients"] == {"x1": 1e308}
        assert stored["rhs"] == -1e308

    def test_accepted_constraints_round_trip_through_the_api(self, session_id):
        """What GET emits must be POSTable back -- the property NaN broke."""
        _add_mixed_variables(session_id)
        client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "equality",
            "coefficients": {"x1": 3.0, "x2": -2.0, "x3": 0.5},
            "rhs": -4.5, "name": "asym",
        }).raise_for_status()
        stored = client.get(
            f"/api/v1/sessions/{session_id}/constraints"
        ).json()["constraints"][0]
        assert stored["rhs"] is not None
        assert None not in stored["coefficients"].values()

        # Feed the emitted representation straight back into a fresh session.
        second = client.post("/api/v1/sessions", json={"ttl_hours": 1}).json()["session_id"]
        try:
            _add_mixed_variables(second)
            replay = client.post(f"/api/v1/sessions/{second}/constraints", json={
                "constraint_type": stored["type"],
                "coefficients": stored["coefficients"],
                "rhs": stored["rhs"],
                "name": stored["name"],
            })
            assert replay.status_code == 200, replay.text
        finally:
            client.delete(f"/api/v1/sessions/{second}")


class TestConstraintNameIsAddressable:
    """Fix 3: every accepted name must be reachable by DELETE.

    ``name`` carried no validation at all, so names that cannot survive a URL
    path segment were accepted and produced permanently undeletable
    constraints. ``..`` was worse than undeletable: clients and proxies apply
    RFC 3986 dot-segment removal, so ``DELETE .../constraints/..`` resolves
    one level up onto the session endpoint and destroys the whole session.
    """

    @pytest.mark.parametrize("name", ["", "a/b", "/leading", "trailing/", ".", ".."])
    def test_unaddressable_name_is_rejected(self, session_id, name):
        _add_mixed_variables(session_id)
        r = client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": {"x2": -2.0}, "rhs": 4.0, "name": name,
        })
        assert r.status_code == 422, r.text
        assert client.get(
            f"/api/v1/sessions/{session_id}/constraints"
        ).json()["n_constraints"] == 0

    @pytest.mark.parametrize("name", [
        "half plane 1",      # spaces
        "   ",               # whitespace only -- ugly but addressable
        "c-1_x.2(+)",        # punctuation
        "purity 95%",        # percent sign
        "x1 <= 3 & x2 >= 1", # operators a user would actually type
        "αβ ≤ 3",            # unicode
        "...",               # not a dot segment; three dots is a normal name
        ".hidden",           # leading dot, not a dot segment
    ])
    def test_addressable_name_is_accepted_and_deletable(self, session_id, name):
        """The rule must not over-restrict: these all round-trip correctly."""
        _add_mixed_variables(session_id)
        r = client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": {"x1": 3.0, "x3": 0.5}, "rhs": 7.0, "name": name,
        })
        assert r.status_code == 200, r.text
        assert client.get(
            f"/api/v1/sessions/{session_id}/constraints"
        ).json()["constraints"][0]["name"] == name

        # The invariant: what POST accepted, DELETE can address. Percent-encode
        # the segment, which is what a correct HTTP client does.
        d = client.delete(
            f"/api/v1/sessions/{session_id}/constraints/{quote(name, safe='')}"
        )
        assert d.status_code == 200, d.text
        assert client.get(
            f"/api/v1/sessions/{session_id}/constraints"
        ).json()["n_constraints"] == 0

    def test_omitted_name_is_still_allowed(self, session_id):
        """The validator must not reject None; auto-naming still applies."""
        _add_mixed_variables(session_id)
        r = client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": {"x2": -2.0}, "rhs": 4.0,
        })
        assert r.status_code == 200, r.text
        assert r.json()["constraint"]["name"] == "constraint_0"

    def test_explicit_null_name_is_still_allowed(self, session_id):
        """An omitted name and an explicit ``null`` are different code paths.

        Pydantic does not run field validators over a field's default, so
        omitting ``name`` never reaches the validator at all. Sending
        ``"name": null`` does, and the validator has to short-circuit on None
        rather than fall through to ``"/" in value`` -- which raises TypeError
        inside validation and surfaces as a 500, not a 422.
        """
        _add_mixed_variables(session_id)
        r = client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": {"x2": -2.0}, "rhs": 4.0, "name": None,
        })
        assert r.status_code == 200, r.text
        assert r.json()["constraint"]["name"] == "constraint_0"

    def test_rejected_name_returns_a_serializable_body(self, session_id):
        """The 422 body must render.

        This is the first field_validator in the API. The app's
        RequestValidationError handler JSON-encodes ``exc.errors()``, and a
        plain ValueError raised from a validator lands in the error ``ctx`` as
        a live exception object, which is not JSON serializable -- turning the
        422 into a 500. The validator raises PydanticCustomError to avoid it.
        """
        _add_mixed_variables(session_id)
        r = client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": {"x1": 3.0}, "rhs": 5.0, "name": "a/b",
        })
        assert r.status_code == 422
        body = r.json()
        assert body["errors"][0]["type"] == "constraint_name_not_addressable"
        assert "a/b" in str(body)


# ============================================================
# Task 12 -- /variables/load accepts the {variables, constraints} format
# ============================================================

# Four variable types, not one. A load suite that only exercised `real` cannot
# see a defect that bites the categorical branch (which `from_dict` reaches
# through a different key and which constraints may not reference at all) or
# the discrete branch (whose allowed_values are sorted and float-coerced on
# the way in). Nine defects in this plan came from single-type coverage.
_FOUR_TYPE_VARIABLES = [
    {"name": "x1", "type": "real", "min": 0.0, "max": 10.0},
    {"name": "x2", "type": "integer", "min": 0, "max": 10},
    {"name": "x3", "type": "discrete", "allowed_values": [0.0, 2.0, 4.0, 6.0, 8.0]},
    {"name": "x4", "type": "categorical", "values": ["A", "B", "C"]},
]

# Asymmetric, non-unit coefficients on purpose: {x1: 1.0, x2: 1.0} is
# invariant under a swap and under a sign error, so magnitude-blind code
# passes it.
_LOADED_CONSTRAINTS = [
    {"type": "inequality", "coefficients": {"x1": 3.0, "x2": -2.0},
     "rhs": 8.0, "name": "c_a"},
    {"type": "equality", "coefficients": {"x3": 0.5, "x1": -1.5},
     "rhs": -4.0, "name": "c_b"},
]


def _upload(sid, payload):
    """POST a JSON payload to /variables/load as a file upload.

    ``json.dumps`` emits bare ``NaN`` / ``Infinity`` / ``null`` literals, which
    is exactly what a real file written by a non-Python producer contains, so
    the non-finite and null cases below need no special client handling.
    """
    buf = io.BytesIO(json.dumps(payload).encode())
    return client.post(
        f"/api/v1/sessions/{sid}/variables/load",
        files={"file": ("space.json", buf, "application/json")},
    )


def _dict_payload(variables=None, constraints=None):
    return {
        "variables": [dict(v) for v in (variables if variables is not None
                                        else _FOUR_TYPE_VARIABLES)],
        "constraints": [dict(c) for c in (constraints if constraints is not None
                                          else _LOADED_CONSTRAINTS)],
    }


def _variables_of(sid):
    return client.get(f"/api/v1/sessions/{sid}/variables").json()


def _constraints_of(sid):
    return client.get(f"/api/v1/sessions/{sid}/constraints").json()


class TestVariablesLoadDictFormat:
    """The endpoint accepts both the legacy bare list and the dict format.

    Before this, ``load_variables_from_file`` iterated the payload directly.
    Handed a dict it iterated the *keys*, so ``var.pop("type")`` ran against
    the string ``"variables"`` and raised ``AttributeError`` -- a 500, not a
    400. Constraints in an uploaded search space were unreachable.
    """

    def test_bare_list_format_still_works(self, session_id):
        r = _upload(session_id, [dict(v) for v in _FOUR_TYPE_VARIABLES])
        assert r.status_code == 200, r.text
        listed = _variables_of(session_id)
        assert listed["n_variables"] == 4
        assert {v["name"] for v in listed["variables"]} == {"x1", "x2", "x3", "x4"}
        assert {v["type"] for v in listed["variables"]} == {
            "real", "integer", "discrete", "categorical"
        }

    def test_bare_list_reports_zero_constraints(self, session_id):
        r = _upload(session_id, [dict(v) for v in _FOUR_TYPE_VARIABLES])
        assert r.json()["n_constraints"] == 0
        assert _constraints_of(session_id)["n_constraints"] == 0

    def test_dict_format_registers_variables_and_constraints(self, session_id):
        r = _upload(session_id, _dict_payload())
        assert r.status_code == 200, r.text
        assert r.json()["n_variables"] == 4
        assert r.json()["n_constraints"] == 2

        listed = _variables_of(session_id)
        assert listed["n_variables"] == 4
        assert {v["name"] for v in listed["variables"]} == {"x1", "x2", "x3", "x4"}
        # The categorical survives the dict path with its values intact.
        x4 = next(v for v in listed["variables"] if v["name"] == "x4")
        assert x4["categories"] == ["A", "B", "C"]
        # The discrete keeps its allowed values (sorted, float-coerced).
        x3 = next(v for v in listed["variables"] if v["name"] == "x3")
        assert x3["allowed_values"] == [0.0, 2.0, 4.0, 6.0, 8.0]

        got = _constraints_of(session_id)
        assert got["n_constraints"] == 2
        by_name = {c["name"]: c for c in got["constraints"]}
        assert set(by_name) == {"c_a", "c_b"}
        # Coefficients survive by value and by sign, not merely by count.
        assert by_name["c_a"]["type"] == "inequality"
        assert by_name["c_a"]["coefficients"] == {"x1": 3.0, "x2": -2.0}
        assert by_name["c_a"]["rhs"] == 8.0
        assert by_name["c_b"]["type"] == "equality"
        assert by_name["c_b"]["coefficients"] == {"x3": 0.5, "x1": -1.5}
        assert by_name["c_b"]["rhs"] == -4.0

    def test_dict_format_without_a_constraints_key_loads_variables(self, session_id):
        r = _upload(session_id, {"variables": [dict(v) for v in _FOUR_TYPE_VARIABLES]})
        assert r.status_code == 200, r.text
        assert r.json()["n_constraints"] == 0
        assert _variables_of(session_id)["n_variables"] == 4

    def test_dict_format_replaces_the_existing_search_space(self, session_id):
        """The dict path is a *load*, not a merge -- as SearchSpace.load_from_json is.

        Constraints must be replaced along with the variables. Keeping the old
        ones would leave constraints referencing variables that no longer
        exist, which no validation downstream would catch.
        """
        _add_mixed_variables(session_id)
        seed = client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": {"x1": 4.0, "x2": -1.0}, "rhs": 9.0, "name": "old",
        })
        assert seed.status_code == 200, seed.text

        r = _upload(session_id, _dict_payload(
            variables=[{"name": "y1", "type": "real", "min": -5.0, "max": 5.0},
                       {"name": "y2", "type": "integer", "min": 1, "max": 4}],
            constraints=[{"type": "inequality",
                          "coefficients": {"y1": 2.5, "y2": -0.75},
                          "rhs": 3.0, "name": "new"}],
        ))
        assert r.status_code == 200, r.text
        assert {v["name"] for v in _variables_of(session_id)["variables"]} == {"y1", "y2"}
        got = _constraints_of(session_id)
        assert got["n_constraints"] == 1
        assert got["constraints"][0]["name"] == "new"

    def test_export_default_is_still_a_bare_list_and_drops_constraints(self, session_id):
        """Ruling 29's residual gap, pinned rather than left implicit.

        ``/variables/export`` keeps its bare-list shape by default -- three
        existing tests and the desktop loader depend on it. The cost is that
        the default export silently loses constraints. That is a real gap and
        it is asserted here so it cannot regress into a surprise.
        """
        assert _upload(session_id, _dict_payload()).status_code == 200
        exported = client.get(
            f"/api/v1/sessions/{session_id}/variables/export"
        ).json()
        assert isinstance(exported, list)
        assert len(exported) == 4
        assert not any("constraints" in v for v in exported)

    def test_load_export_load_round_trips_constraints(self, session_id):
        """spec 9.6: load -> export -> load round-trips constraints.

        Via the opt-in dict export (Ruling 29). The second session is loaded
        from the *exported* bytes, so anything export drops is observable here.
        """
        assert _upload(session_id, _dict_payload()).status_code == 200

        exported = client.get(
            f"/api/v1/sessions/{session_id}/variables/export",
            params={"include_constraints": "true"},
        ).json()
        assert set(exported) == {"variables", "constraints"}
        assert len(exported["variables"]) == 4
        assert len(exported["constraints"]) == 2

        second = client.post("/api/v1/sessions", json={"ttl_hours": 1}).json()
        sid2 = second["session_id"]
        try:
            r = _upload(sid2, exported)
            assert r.status_code == 200, r.text
            assert _variables_of(sid2)["n_variables"] == 4
            assert {v["type"] for v in _variables_of(sid2)["variables"]} == {
                "real", "integer", "discrete", "categorical"
            }
            got = _constraints_of(sid2)
            assert got["n_constraints"] == 2
            by_name = {c["name"]: c for c in got["constraints"]}
            assert by_name["c_a"]["coefficients"] == {"x1": 3.0, "x2": -2.0}
            assert by_name["c_a"]["rhs"] == 8.0
            assert by_name["c_b"]["type"] == "equality"
            assert by_name["c_b"]["coefficients"] == {"x3": 0.5, "x1": -1.5}
            assert by_name["c_b"]["rhs"] == -4.0
        finally:
            client.delete(f"/api/v1/sessions/{sid2}")

    def test_round_tripped_constraints_are_honored_by_a_design(self, session_id):
        """The point of carrying constraints through a load is that the DoE obeys them.

        A load that registers constraints the design ignores would pass every
        listing assertion above and still be useless.
        """
        assert _upload(session_id, _dict_payload(
            variables=[{"name": "x1", "type": "real", "min": 0.0, "max": 10.0},
                       {"name": "x2", "type": "real", "min": 0.0, "max": 10.0}],
            constraints=[{"type": "inequality",
                          "coefficients": {"x1": 3.0, "x2": -2.0},
                          "rhs": 6.0, "name": "c_a"}],
        )).status_code == 200

        r = client.post(
            f"/api/v1/sessions/{session_id}/initial-design",
            json={"method": "lhs", "n_points": 12, "random_seed": 7},
        )
        assert r.status_code == 200, r.text
        points = r.json()["points"]
        assert points
        for p in points:
            assert 3.0 * p["x1"] - 2.0 * p["x2"] <= 6.0 + 1e-6, p


class TestLoadedConstraintsAreValidated:
    """A constraint arriving in a file goes through the same gate as POST /constraints.

    /variables/load is a REST write path into ``search_space.constraints``.
    Assigning the file's list straight across -- which is what
    ``SearchSpace.load_from_json`` does -- would let this endpoint register
    exactly what ``POST /constraints`` rejects. Every rejection below is
    whole-file and leaves the session untouched.
    """

    def test_constraint_on_a_categorical_variable_is_rejected(self, session_id):
        r = _upload(session_id, _dict_payload(constraints=[
            {"type": "inequality", "coefficients": {"x4": 2.0}, "rhs": 1.0, "name": "bad"},
        ]))
        assert r.status_code == 400, r.text
        assert "x4" in r.json()["detail"]
        assert _variables_of(session_id)["n_variables"] == 0

    def test_constraint_on_an_unknown_variable_is_rejected(self, session_id):
        r = _upload(session_id, _dict_payload(constraints=[
            {"type": "inequality", "coefficients": {"nope": 2.5}, "rhs": 1.0},
        ]))
        assert r.status_code == 400, r.text
        assert "nope" in r.json()["detail"]
        assert _variables_of(session_id)["n_variables"] == 0

    def test_unknown_constraint_type_is_rejected(self, session_id):
        r = _upload(session_id, _dict_payload(constraints=[
            {"type": "greater_than", "coefficients": {"x1": 3.0}, "rhs": 1.0},
        ]))
        assert r.status_code == 400, r.text
        assert _variables_of(session_id)["n_variables"] == 0

    def test_duplicate_constraint_name_in_the_file_is_rejected(self, session_id):
        """One of the two inputs where the routing decision is observable.

        ``load_from_json`` assigns the list raw, so a file with two ``c_a``
        entries loads silently and leaves two constraints sharing one delete
        identity. Routing through ``add_constraint`` rejects the file instead.
        """
        r = _upload(session_id, _dict_payload(constraints=[
            {"type": "inequality", "coefficients": {"x1": 3.0, "x2": -2.0},
             "rhs": 8.0, "name": "c_a"},
            {"type": "equality", "coefficients": {"x3": 0.5}, "rhs": 2.0, "name": "c_a"},
        ]))
        assert r.status_code == 400, r.text
        assert "c_a" in r.json()["detail"]
        assert _variables_of(session_id)["n_variables"] == 0
        assert _constraints_of(session_id)["n_constraints"] == 0

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_rhs_in_the_file_is_rejected(self, session_id, value):
        """The other observable input. A NaN rhs makes every point infeasible."""
        r = _upload(session_id, _dict_payload(constraints=[
            {"type": "inequality", "coefficients": {"x1": 3.0, "x2": -2.0},
             "rhs": value, "name": "c_a"},
        ]))
        assert r.status_code == 400, r.text
        assert "finite" in r.json()["detail"].lower()
        assert _variables_of(session_id)["n_variables"] == 0

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_coefficient_in_the_file_is_rejected(self, session_id, value):
        r = _upload(session_id, _dict_payload(constraints=[
            {"type": "inequality", "coefficients": {"x1": 3.0, "x2": value},
             "rhs": 8.0, "name": "c_a"},
        ]))
        assert r.status_code == 400, r.text
        assert "finite" in r.json()["detail"].lower()
        assert _variables_of(session_id)["n_variables"] == 0

    @pytest.mark.parametrize("value", [None, "abc", [1.0], {"a": 1}])
    def test_non_numeric_rhs_in_the_file_is_a_400_not_a_500(self, session_id, value):
        """Ruling 33 Item B, reached through the loader.

        ``np.isfinite(None)`` raises ``TypeError``, which ``add_constraint``
        documents as ``ValueError`` and which the router's 400 handler does not
        catch. ``"rhs": null`` is an entirely plausible thing for a file to
        contain, and before the guard it produced a 500.
        """
        r = _upload(session_id, _dict_payload(constraints=[
            {"type": "inequality", "coefficients": {"x1": 3.0, "x2": -2.0},
             "rhs": value, "name": "c_a"},
        ]))
        assert r.status_code == 400, r.text
        assert _variables_of(session_id)["n_variables"] == 0

    @pytest.mark.parametrize("value", [None, "abc", [1.0]])
    def test_non_numeric_coefficient_in_the_file_is_a_400_not_a_500(self, session_id, value):
        r = _upload(session_id, _dict_payload(constraints=[
            {"type": "inequality", "coefficients": {"x1": 3.0, "x2": value},
             "rhs": 8.0, "name": "c_a"},
        ]))
        assert r.status_code == 400, r.text
        assert _variables_of(session_id)["n_variables"] == 0

    @pytest.mark.parametrize("missing", ["type", "coefficients", "rhs"])
    def test_constraint_missing_a_required_key_is_rejected(self, session_id, missing):
        constraint = {"type": "inequality", "coefficients": {"x1": 3.0},
                      "rhs": 8.0, "name": "c_a"}
        constraint.pop(missing)
        r = _upload(session_id, _dict_payload(constraints=[constraint]))
        assert r.status_code == 400, r.text
        assert missing in r.json()["detail"]
        assert _variables_of(session_id)["n_variables"] == 0

    def test_every_missing_constraint_key_is_reported_at_once(self, session_id):
        """Not just the first one a KeyError would happen to hit.

        Letting the constraint fall through to ``add_constraint`` and reporting
        whatever KeyError comes back names one key per upload, so fixing a file
        with three missing keys takes three round trips. The check runs up front
        precisely so it can report all of them, and it names *which constraint*.
        """
        r = _upload(session_id, _dict_payload(constraints=[{"name": "c_a"}]))
        assert r.status_code == 400, r.text
        detail = r.json()["detail"]
        for key in ("type", "coefficients", "rhs"):
            assert key in detail, detail
        assert "constraints[0]" in detail, detail
        assert _variables_of(session_id)["n_variables"] == 0

    def test_a_rejected_file_leaves_an_existing_search_space_untouched(self, session_id):
        """Atomicity. The variables load first; a constraint failing afterwards
        must not leave the session holding half the file."""
        _add_mixed_variables(session_id)
        client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": {"x1": 4.0, "x2": -1.0}, "rhs": 9.0, "name": "keep_me",
        }).raise_for_status()

        r = _upload(session_id, _dict_payload(constraints=[
            {"type": "inequality", "coefficients": {"x1": 3.0}, "rhs": None},
        ]))
        assert r.status_code == 400, r.text

        assert client.get(f"/api/v1/sessions/{session_id}").status_code == 200
        listed = _variables_of(session_id)
        assert listed["n_variables"] == 3
        assert {v["name"] for v in listed["variables"]} == {"x1", "x2", "x3"}
        got = _constraints_of(session_id)
        assert got["n_constraints"] == 1
        assert got["constraints"][0]["name"] == "keep_me"

    def test_an_omitted_constraint_name_is_auto_generated(self, session_id):
        r = _upload(session_id, _dict_payload(constraints=[
            {"type": "inequality", "coefficients": {"x1": 3.0, "x2": -2.0}, "rhs": 8.0},
            {"type": "equality", "coefficients": {"x3": 0.5}, "rhs": 2.0},
        ]))
        assert r.status_code == 200, r.text
        names = [c["name"] for c in _constraints_of(session_id)["constraints"]]
        assert names == ["constraint_0", "constraint_1"]

    def test_loaded_constraints_are_deletable_one_at_a_time(self, session_id):
        """Whatever the loader registers, DELETE must be able to address."""
        assert _upload(session_id, _dict_payload()).status_code == 200
        d = client.delete(f"/api/v1/sessions/{session_id}/constraints/c_a")
        assert d.status_code == 200, d.text
        remaining = _constraints_of(session_id)
        assert remaining["n_constraints"] == 1
        assert remaining["constraints"][0]["name"] == "c_b"


# Reused from the POST-route suites: the rule is one rule, so the data that
# proves it must be the same data.
_LOAD_UNADDRESSABLE = ["", "a/b", "/leading", "trailing/", ".", ".."]

_LOAD_ADDRESSABLE = [
    "flow rate 1",       # spaces
    "purity 95%",        # percent sign
    "αβ ≤ 3",            # unicode and an operator a user would type
    "...",               # not a dot segment; three dots is a normal name
    ".hidden",           # leading dot, not a dot segment
    "x1(+)-2.0",         # punctuation
]

_LOAD_SHAPES = {
    "real": {"type": "real", "min": 0.0, "max": 10.0},
    "integer": {"type": "integer", "min": 0, "max": 10},
    "categorical": {"type": "categorical", "values": ["A", "B"]},
    "discrete": {"type": "discrete", "allowed_values": [1.0, 2.0]},
}


class TestLoadedNamesAreAddressable:
    """/variables/load creates exactly the variable POST /variables rejects.

    The four variable request models gained a validator rejecting names the
    DELETE route cannot address -- ``''``, anything with ``/``, and ``.`` /
    ``..``. ``..`` is not merely undeletable: clients apply RFC 3986 dot-segment
    removal before sending, so ``DELETE .../variables/..`` is rewritten onto
    the session route and destroys the entire session while returning 204.
    ``/variables/load`` parses raw JSON and never constructs those models, so
    it bypassed the validator entirely: POST returned 422 for ``name='..'``
    while load returned 200 and registered it.
    """

    @pytest.mark.parametrize("var_type", list(_LOAD_SHAPES))
    @pytest.mark.parametrize("name", _LOAD_UNADDRESSABLE)
    def test_bare_list_rejects_an_unaddressable_variable_name(
        self, session_id, var_type, name
    ):
        r = _upload(session_id, [{"name": name, **_LOAD_SHAPES[var_type]}])
        assert r.status_code == 400, r.text
        assert _variables_of(session_id)["n_variables"] == 0

    @pytest.mark.parametrize("var_type", list(_LOAD_SHAPES))
    @pytest.mark.parametrize("name", _LOAD_UNADDRESSABLE)
    def test_dict_format_rejects_an_unaddressable_variable_name(
        self, session_id, var_type, name
    ):
        r = _upload(session_id, _dict_payload(
            variables=[{"name": name, **_LOAD_SHAPES[var_type]}],
            constraints=[],
        ))
        assert r.status_code == 400, r.text
        assert _variables_of(session_id)["n_variables"] == 0

    @pytest.mark.parametrize("name", _LOAD_UNADDRESSABLE)
    def test_dict_format_rejects_an_unaddressable_constraint_name(
        self, session_id, name
    ):
        r = _upload(session_id, _dict_payload(constraints=[
            {"type": "inequality", "coefficients": {"x1": 3.0, "x2": -2.0},
             "rhs": 8.0, "name": name},
        ]))
        assert r.status_code == 400, r.text
        assert _variables_of(session_id)["n_variables"] == 0
        assert _constraints_of(session_id)["n_constraints"] == 0

    @pytest.mark.parametrize("shape", ["bare_list", "dict_variable", "dict_constraint"])
    def test_a_rejected_name_leaves_the_session_intact(self, session_id, shape):
        """The property that matters is not the status code.

        Before the fix the load returned 200 and registered ``..``; the user's
        next DELETE took the whole session with it. So: the session still
        exists, and everything already in it survives.
        """
        _add_mixed_variables(session_id)
        client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": {"x1": 4.0, "x2": -1.0}, "rhs": 9.0, "name": "keep_me",
        }).raise_for_status()

        if shape == "bare_list":
            payload = [{"name": "..", "type": "real", "min": 0.0, "max": 1.0}]
        elif shape == "dict_variable":
            payload = _dict_payload(
                variables=[{"name": "..", "type": "real", "min": 0.0, "max": 1.0}],
                constraints=[],
            )
        else:
            payload = _dict_payload(constraints=[
                {"type": "inequality", "coefficients": {"x1": 3.0},
                 "rhs": 8.0, "name": ".."},
            ])

        r = _upload(session_id, payload)
        assert r.status_code == 400, r.text

        assert client.get(f"/api/v1/sessions/{session_id}").status_code == 200
        listed = _variables_of(session_id)
        assert listed["n_variables"] == 3
        assert {v["name"] for v in listed["variables"]} == {"x1", "x2", "x3"}
        got = _constraints_of(session_id)
        assert got["n_constraints"] == 1
        assert got["constraints"][0]["name"] == "keep_me"

    # A JSON file can carry any type in "name". skopt happens to reject a
    # non-string dimension name for real/integer/categorical/discrete, which
    # masks most of this -- but a 'context' variable has no skopt dimension at
    # all, and a constraint name has no backstop whatever. Both would register
    # and then be permanently undeletable: DELETE compares the path segment,
    # always a str, against a name that is not one.
    _NON_STRING_NAMES = [7, 1.5, True, None, ["x1"], {"a": 1}]

    @pytest.mark.parametrize("name", _NON_STRING_NAMES)
    def test_a_non_string_variable_name_is_rejected(self, session_id, name):
        r = _upload(session_id, _dict_payload(
            variables=[{"name": name, "type": "context"}], constraints=[],
        ))
        assert r.status_code == 400, r.text
        assert _variables_of(session_id)["n_variables"] == 0

    @pytest.mark.parametrize("name", _NON_STRING_NAMES)
    def test_a_non_string_variable_name_is_rejected_in_a_bare_list(
        self, session_id, name
    ):
        r = _upload(session_id, [{"name": name, "type": "context"}])
        assert r.status_code == 400, r.text
        assert _variables_of(session_id)["n_variables"] == 0

    # None is excluded: an omitted or null constraint name means
    # "auto-generate", which is legal and covered separately.
    @pytest.mark.parametrize("name", [7, 1.5, True, ["c_a"], {"a": 1}])
    def test_a_non_string_constraint_name_is_rejected(self, session_id, name):
        r = _upload(session_id, _dict_payload(constraints=[
            {"type": "inequality", "coefficients": {"x1": 3.0, "x2": -2.0},
             "rhs": 8.0, "name": name},
        ]))
        assert r.status_code == 400, r.text
        assert _variables_of(session_id)["n_variables"] == 0
        assert _constraints_of(session_id)["n_constraints"] == 0

    def test_a_null_constraint_name_still_auto_generates(self, session_id):
        """The one non-string that is legal, so the guard must not over-reach."""
        r = _upload(session_id, _dict_payload(constraints=[
            {"type": "inequality", "coefficients": {"x1": 3.0, "x2": -2.0},
             "rhs": 8.0, "name": None},
        ]))
        assert r.status_code == 200, r.text
        assert _constraints_of(session_id)["constraints"][0]["name"] == "constraint_0"

    @pytest.mark.parametrize("var_type", list(_LOAD_SHAPES))
    @pytest.mark.parametrize("name", _LOAD_ADDRESSABLE)
    def test_legal_but_unusual_variable_names_still_load(
        self, session_id, var_type, name
    ):
        """The rule must not over-restrict. Acceptance is not enough either --
        what load accepts, DELETE has to be able to address."""
        r = _upload(session_id, [{"name": name, **_LOAD_SHAPES[var_type]}])
        assert r.status_code == 200, r.text
        assert _variables_of(session_id)["variables"][0]["name"] == name

        d = client.delete(
            f"/api/v1/sessions/{session_id}/variables/{quote(name, safe='')}"
        )
        assert d.status_code == 200, d.text
        assert _variables_of(session_id)["n_variables"] == 0

    @pytest.mark.parametrize("name", _LOAD_ADDRESSABLE)
    def test_legal_but_unusual_constraint_names_still_load(self, session_id, name):
        r = _upload(session_id, _dict_payload(constraints=[
            {"type": "inequality", "coefficients": {"x1": 3.0, "x2": -2.0},
             "rhs": 8.0, "name": name},
        ]))
        assert r.status_code == 200, r.text
        assert _constraints_of(session_id)["constraints"][0]["name"] == name

        d = client.delete(
            f"/api/v1/sessions/{session_id}/constraints/{quote(name, safe='')}"
        )
        assert d.status_code == 200, d.text
        assert _constraints_of(session_id)["n_constraints"] == 0


class TestMalformedLoadPayloads:
    """A malformed file is the client's error, not a 500."""

    @pytest.mark.parametrize("payload", [
        {"variables": "not a list"},
        {"variables": [], "constraints": "not a list"},
        {"variables": ["not a dict"]},
        {"variables": [{"type": "real", "min": 0.0, "max": 1.0}]},   # no name
        {"variables": [{"name": "x1", "min": 0.0, "max": 1.0}]},     # no type
        {"variables": [{"name": "x1", "type": 7}]},                  # non-string type
        {"variables": [{"name": "x1", "type": "real"}]},             # no bounds
        {"variables": [{"name": "x1", "type": "wat"}]},              # unknown type
        {"variables": [], "constraints": ["not a dict"]},
        ["not a dict either"],
        [{"name": "x1", "type": "real"}],                            # bare list, no bounds
        42,
    ])
    def test_malformed_payload_is_a_400(self, session_id, payload):
        r = _upload(session_id, payload)
        assert r.status_code == 400, r.text
        assert _variables_of(session_id)["n_variables"] == 0

    def test_a_duplicate_variable_name_within_the_file_is_a_400(self, session_id):
        r = _upload(session_id, _dict_payload(
            variables=[{"name": "x1", "type": "real", "min": 0.0, "max": 10.0},
                       {"name": "x1", "type": "integer", "min": 0, "max": 3}],
            constraints=[],
        ))
        assert r.status_code == 400, r.text
        assert _variables_of(session_id)["n_variables"] == 0

    def test_invalid_json_is_a_400_that_says_so(self, session_id):
        """The status alone does not discriminate here.

        ``json.JSONDecodeError`` subclasses ``ValueError``, and the app
        registers a global ``ValueError`` handler returning 400
        (api/middleware/error_handlers.py). So an uncaught decode error is
        already a 400 -- with the bare decoder message, which does not tell the
        caller that it was the *uploaded file* that would not parse. The
        message is the part worth pinning.
        """
        buf = io.BytesIO(b"{not json at all")
        r = client.post(
            f"/api/v1/sessions/{session_id}/variables/load",
            files={"file": ("space.json", buf, "application/json")},
        )
        assert r.status_code == 400, r.text
        assert "not valid JSON" in r.json()["detail"], r.text
        assert _variables_of(session_id)["n_variables"] == 0


# ============================================================
# Ruling 35 -- fix round 1
# ============================================================

# Surfaces a server-side crash as its status code instead of re-raising it, so
# a regression reads as "500 != 400" rather than as an exception in the test.
_lenient_client = TestClient(app, raise_server_exceptions=False)


def _upload_lenient(sid, payload):
    buf = io.BytesIO(json.dumps(payload).encode())
    return _lenient_client.post(
        f"/api/v1/sessions/{sid}/variables/load",
        files={"file": ("space.json", buf, "application/json")},
    )


class TestEmptyCategoricalIsAClientError:
    """An empty category list was a 500 on an endpoint documenting 400.

    ``Categorical([])`` computes ``1.0 / len(self.categories)`` to build its
    prior, so skopt raised ``ZeroDivisionError`` -- outside the
    ``(ValueError, KeyError, TypeError)`` tuple this router catches on either
    branch. The dict branch's dry run kept the session intact, so the damage
    was confined to the status code, but the status was wrong on both branches
    and ``TestMalformedLoadPayloads`` did not cover it.

    Fixed in ``SearchSpace.add_variable`` rather than by widening the catch
    tuple: the bare-list branch has no dry run to widen, and the desktop loader
    reaches the same call.
    """

    @pytest.mark.parametrize("key", ["values", "categories"])
    @pytest.mark.parametrize("shape", ["bare", "dict"])
    def test_an_empty_category_list_is_a_400(self, session_id, shape, key):
        var = {"name": "x4", "type": "categorical", key: []}
        payload = [var] if shape == "bare" else {"variables": [var], "constraints": []}
        r = _upload_lenient(session_id, payload)
        assert r.status_code == 400, r.text
        assert "x4" in r.json()["detail"]

    @pytest.mark.parametrize("shape", ["bare", "dict"])
    def test_the_session_is_left_intact(self, session_id, shape):
        _add_variables(session_id, names=("x1", "x2"))
        before = client.get(f"/api/v1/sessions/{session_id}/variables").json()
        var = {"name": "x4", "type": "categorical", "values": []}
        payload = [var] if shape == "bare" else {"variables": [var], "constraints": []}
        assert _upload_lenient(session_id, payload).status_code == 400
        assert client.get(f"/api/v1/sessions/{session_id}/variables").json() == before

    def test_an_empty_list_beside_good_variables_rejects_the_whole_file(
        self, session_id
    ):
        r = _upload_lenient(session_id, _dict_payload(
            variables=[
                {"name": "x1", "type": "real", "min": 0.0, "max": 10.0},
                {"name": "x4", "type": "categorical", "values": []},
            ],
            constraints=[],
        ))
        assert r.status_code == 400, r.text
        assert _variables_of(session_id)["n_variables"] == 0

    def test_a_non_empty_category_list_still_loads(self, session_id):
        r = _upload_lenient(session_id, _dict_payload(constraints=[]))
        assert r.status_code == 200, r.text
        assert _variables_of(session_id)["n_variables"] == 4


class TestLoadErrorsNameTheVariableAndTheKey:
    """``_load_error_detail`` claimed to name the file's fault; it did not.

    ``{"min": null}`` reported "'<=' not supported between instances of 'float'
    and 'NoneType'" and ``{"min": "0.0"}`` reported "unsupported operand
    type(s) for -: 'str' and 'str'" -- raw skopt TypeErrors naming neither the
    variable nor the key. Both now fail in ``add_variable``'s own bounds guard,
    which raises ValueError carrying both, and the docstring no longer claims
    more than the function does.
    """

    @pytest.mark.parametrize("bad", [None, "0.0", [1.0]])
    @pytest.mark.parametrize("key", ["min", "max"])
    @pytest.mark.parametrize("shape", ["bare", "dict"])
    def test_a_bad_bound_names_the_variable_and_the_key(
        self, session_id, shape, key, bad
    ):
        var = {"name": "x1", "type": "real", "min": 0.0, "max": 10.0}
        var[key] = bad
        payload = [var] if shape == "bare" else {"variables": [var], "constraints": []}
        r = _upload_lenient(session_id, payload)
        assert r.status_code == 400, r.text
        detail = r.json()["detail"]
        assert "x1" in detail, detail
        assert key in detail, detail
        assert "must be a finite number" in detail, detail

    @pytest.mark.parametrize("shape", ["bare", "dict"])
    def test_an_integer_variable_reports_the_same_way(self, session_id, shape):
        var = {"name": "x2", "type": "integer", "min": None, "max": 8}
        payload = [var] if shape == "bare" else {"variables": [var], "constraints": []}
        detail = _upload_lenient(session_id, payload).json()["detail"]
        assert "x2" in detail and "min" in detail, detail

    @pytest.mark.parametrize("shape", ["bare", "dict"])
    def test_a_missing_bound_still_names_the_key(self, session_id, shape):
        """The KeyError branch, which is the one that always worked."""
        var = {"name": "x1", "type": "real", "max": 10.0}
        payload = [var] if shape == "bare" else {"variables": [var], "constraints": []}
        r = _upload_lenient(session_id, payload)
        assert r.status_code == 400, r.text
        assert "'min'" in r.json()["detail"], r.text

    def test_the_docstring_does_not_promise_localization(self):
        """Ruling 27: the docstring and the code must not disagree.

        The function attributes a failure to the uploaded file; it does not
        localize one to a variable, and only the KeyError branch identifies
        anything at all. Whatever it claims must stay inside that.
        """
        from api.routers.variables import _load_error_detail

        doc = _load_error_detail.__doc__
        assert doc is not None
        lowered = doc.lower()
        assert "keyerror" in lowered, "the one branch that identifies a fault"
        assert "typeerror" in lowered, "the branch that does not"
        # The claim that was false: naming the fault, unqualified.
        assert "names the file's fault" not in lowered


class TestBareListIsTheUnprotectedLoadPath:
    """Fix 3, at the level the corrected comment describes.

    The comment in ``add_variable`` attributed the atomicity defect's
    reachability to the dict branch. The dict branch runs the file against a
    throwaway ``SearchSpace`` first and only touches the session once that has
    passed, so it cannot desync the session whatever ``add_variable`` does. The
    bare-list branch appends straight into the live session with no dry run,
    and that is where a rejected file used to leave a fragment behind.

    The bare list still keeps the entries that loaded before the failure --
    ``[x1, x2-inverted]`` leaves ``x1`` registered against a 400. That is a
    separate defect (branch item B8) and is not in scope here. What D5 fixed,
    and what these tests pin, is that whatever survives is *paired*: the
    half-registered ``x2`` with no dimension is gone.
    """

    DESYNCING = [
        {"name": "x1", "type": "real", "min": 0.0, "max": 10.0},
        {"name": "x2", "type": "real", "min": 9.0, "max": 1.0},   # inverted
    ]

    def test_the_bare_list_leaves_no_half_registered_variable(self, session_id):
        """Reached the live session directly; this is the path that carried it."""
        r = _upload_lenient(session_id, [dict(v) for v in self.DESYNCING])
        assert r.status_code == 400, r.text
        space = session_store.get(session_id).search_space
        # x1 survives (B8). x2 must not, in either list.
        assert [v["name"] for v in space.variables] == ["x1"]
        assert [d.name for d in space.skopt_dimensions] == ["x1"]

    def test_the_dict_branch_is_protected_by_its_dry_run(self, session_id):
        """Would hold even without the atomicity fix -- which is the point."""
        r = _upload_lenient(session_id, {
            "variables": [dict(v) for v in self.DESYNCING], "constraints": [],
        })
        assert r.status_code == 400, r.text
        space = session_store.get(session_id).search_space
        assert space.variables == []
        assert space.skopt_dimensions == []

    def test_a_desynced_session_would_break_delete_and_export(self, session_id):
        """The consequences the fragment had, pinned as absent.

        With ``x2`` present in ``variables`` but not in ``skopt_dimensions``,
        DELETE removed the dimension belonging to a different variable, the
        export emitted a file that would not load, and the next design died on
        an AssertionError.
        """
        assert _upload_lenient(
            session_id, [dict(v) for v in self.DESYNCING]
        ).status_code == 400
        client.post(
            f"/api/v1/sessions/{session_id}/variables",
            json={"name": "x3", "type": "discrete", "allowed_values": [0.5, 7.25]},
        ).raise_for_status()

        d = client.delete(f"/api/v1/sessions/{session_id}/variables/x1")
        assert d.status_code == 200, d.text
        space = session_store.get(session_id).search_space
        assert [v["name"] for v in space.variables] == ["x3"]
        assert [dim.name for dim in space.skopt_dimensions] == ["x3"], (
            "DELETE removed the wrong dimension -- the lists were desynced"
        )

        export = client.get(f"/api/v1/sessions/{session_id}/variables/export")
        assert export.status_code == 200, export.text
        assert [v["name"] for v in export.json()] == ["x3"]

    def test_the_export_of_the_surviving_fragment_reloads(self, session_id):
        """A desynced space exported a file that would not load back."""
        assert _upload_lenient(
            session_id, [dict(v) for v in self.DESYNCING]
        ).status_code == 400
        exported = client.get(
            f"/api/v1/sessions/{session_id}/variables/export"
        ).json()
        fresh = client.post("/api/v1/sessions", json={"ttl_hours": 1}).json()["session_id"]
        try:
            r = _upload_lenient(fresh, exported)
            assert r.status_code == 200, r.text
            assert _variables_of(fresh)["n_variables"] == 1
        finally:
            client.delete(f"/api/v1/sessions/{fresh}")


class TestFeasibilityReporting:
    """The ``feasibility`` block on /initial-design and /optimal-design.

    Constraint provenance used to exist only in a core log line, so a REST
    caller could not tell a constrained design from an unconstrained one.

    ``estimability`` reports the gate in ``alchemist_core/utils/doe.py``,
    which applies to exactly one of the three method classes::

        if (method in CLASSICAL_METHODS and method != "optimal"
                and getattr(search_space, 'constraints', None)):

    - classical, non-``optimal`` (``ccd``, ``full_factorial``, ...): the gate
      runs, and a 200 only exists if the remnant survived it -> ``"passed"``.
    - ``optimal``: exempt by design (its candidate set is already constrained
      and its model is user-specified) -> ``"not_applicable"``.
    - space-filling (``lhs``, ``sobol``, ...): the gate does not apply ->
      ``"not_applicable"``.

    Reporting ``"not_applicable"`` for the first class would hide the single
    most informative thing the response can say about a constrained classical
    design.
    """

    KEYS = {
        "constraints_applied", "n_candidates_total", "n_candidates_feasible",
        "n_boundary_added", "n_vertices_added", "vertex_enumeration_skipped",
        "n_points_dropped", "estimability",
    }

    @staticmethod
    def _constrain(sid, coefficients, rhs):
        r = client.post(f"/api/v1/sessions/{sid}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": coefficients, "rhs": rhs,
        })
        r.raise_for_status()

    # ---- optimal design ------------------------------------------------

    def test_unconstrained_optimal_design_reports_null_feasibility(self, session_id):
        _add_variables(session_id, names=("x1", "x2", "x3"))
        r = client.post(f"/api/v1/sessions/{session_id}/optimal-design", json={
            "n_points": 10, "model_type": "quadratic",
            "criterion": "D", "algorithm": "fedorov", "random_seed": 7,
        })
        assert r.status_code == 200, r.text
        assert r.json()["feasibility"] is None

    def test_constrained_optimal_design_reports_candidate_provenance(self, session_id):
        _add_variables(session_id, names=("x1", "x2", "x3"))
        self._constrain(session_id, {"x1": 1.0, "x2": 1.0}, 12.0)
        r = client.post(f"/api/v1/sessions/{session_id}/optimal-design", json={
            "n_points": 10, "model_type": "quadratic",
            "criterion": "D", "algorithm": "fedorov", "random_seed": 7,
        })
        assert r.status_code == 200, r.text
        feas = r.json()["feasibility"]
        assert set(feas) == self.KEYS
        assert feas["constraints_applied"] == ["constraint_0"]
        assert feas["n_candidates_feasible"] < feas["n_candidates_total"]
        assert feas["n_boundary_added"] > 0
        assert feas["vertex_enumeration_skipped"] is False
        # 'optimal' is the one class the gate exempts.
        assert feas["estimability"] == "not_applicable"
        assert feas["n_points_dropped"] is None

    def test_optimal_design_keeps_feasibility_out_of_design_info(self, session_id):
        """It is surfaced as its own field, not buried in design_info."""
        _add_variables(session_id, names=("x1", "x2", "x3"))
        self._constrain(session_id, {"x1": 1.0, "x2": 1.0}, 12.0)
        r = client.post(f"/api/v1/sessions/{session_id}/optimal-design", json={
            "n_points": 10, "model_type": "quadratic",
            "criterion": "D", "algorithm": "fedorov", "random_seed": 7,
        })
        assert r.status_code == 200, r.text
        assert "feasibility" not in r.json()["design_info"]
        assert r.json()["design_info"]["model_terms"]

    def test_reading_the_response_does_not_strip_the_session_cache(self, session_id):
        """The router must not pop out of the dict the session cached."""
        _add_variables(session_id, names=("x1", "x2", "x3"))
        self._constrain(session_id, {"x1": 1.0, "x2": 1.0}, 12.0)
        body = {"n_points": 10, "model_type": "quadratic",
                "criterion": "D", "algorithm": "fedorov", "random_seed": 7}
        client.post(f"/api/v1/sessions/{session_id}/optimal-design",
                    json=body).raise_for_status()
        cached = session_store.get(session_id)._last_optimal_design_info
        assert cached is not None
        assert cached.get("feasibility") is not None, (
            "the endpoint mutated the session's cached info dict"
        )

    # ---- space-filling -------------------------------------------------

    def test_unconstrained_initial_design_reports_null_feasibility(self, session_id):
        _add_variables(session_id, names=("x1", "x2", "x3"))
        r = client.post(f"/api/v1/sessions/{session_id}/initial-design",
                        json={"method": "lhs", "n_points": 6, "random_seed": 7})
        assert r.status_code == 200, r.text
        assert r.json()["feasibility"] is None

    def test_constrained_initial_design_reports_constraints(self, session_id):
        _add_variables(session_id, names=("x1", "x2", "x3"))
        self._constrain(session_id, {"x1": 1.0, "x2": 1.0}, 12.0)
        r = client.post(f"/api/v1/sessions/{session_id}/initial-design",
                        json={"method": "lhs", "n_points": 6, "random_seed": 7})
        assert r.status_code == 200, r.text
        feas = r.json()["feasibility"]
        assert set(feas) == self.KEYS
        assert feas["constraints_applied"] == ["constraint_0"]
        # A space-filling design reject-and-resamples to exactly n_points;
        # the gate never sees it, and nothing was dropped from a structure.
        assert feas["estimability"] == "not_applicable"
        assert feas["n_points_dropped"] is None
        assert r.json()["n_points"] == 6

    def test_a_second_space_filling_method_reports_the_same_way(self, session_id):
        """sobol, not lhs -- the verdict is the method class, not the method."""
        _add_variables(session_id, names=("x1", "x2", "x3"))
        self._constrain(session_id, {"x1": 1.0, "x2": 1.0}, 12.0)
        r = client.post(f"/api/v1/sessions/{session_id}/initial-design",
                        json={"method": "sobol", "n_points": 8, "random_seed": 7})
        assert r.status_code == 200, r.text
        feas = r.json()["feasibility"]
        assert feas["estimability"] == "not_applicable"
        assert feas["constraints_applied"] == ["constraint_0"]

    # ---- classical -----------------------------------------------------

    def test_constrained_ccd_reports_that_it_passed_the_estimability_gate(self, session_id):
        """The row the plan's own tests never exercised.

        16 structural runs, 4 cut away by the constraint, and the implied
        quadratic model still estimable from the remaining 12 -- which is
        exactly what the caller needs to know and cannot infer from a count.
        """
        _add_variables(session_id, names=("x1", "x2", "x3"))
        self._constrain(session_id, {"x1": 1.0, "x2": 1.0}, 11.0)
        r = client.post(f"/api/v1/sessions/{session_id}/initial-design",
                        json={"method": "ccd", "random_seed": 7})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["n_points"] == 12
        assert body["design_info"]["total_runs"] == 16
        feas = body["feasibility"]
        assert set(feas) == self.KEYS
        assert feas["estimability"] == "passed"
        assert feas["n_points_dropped"] == 4
        assert feas["constraints_applied"] == ["constraint_0"]
        # Candidate-set provenance belongs to the optimal-design augmenter;
        # a classical design has none of it.
        assert feas["n_candidates_total"] is None
        assert feas["n_candidates_feasible"] is None
        assert feas["n_boundary_added"] is None
        assert feas["n_vertices_added"] is None
        assert feas["vertex_enumeration_skipped"] is None

    def test_a_second_classical_method_over_a_discrete_variable_also_passes(self, session_id):
        """full_factorial, and a discrete factor rather than three reals."""
        _add_variables(session_id, names=("x1", "x2"))
        client.post(f"/api/v1/sessions/{session_id}/variables", json={
            "name": "x3", "type": "discrete", "allowed_values": [0.0, 5.0, 10.0],
        }).raise_for_status()
        self._constrain(session_id, {"x1": 1.0, "x2": 1.0}, 12.0)
        r = client.post(f"/api/v1/sessions/{session_id}/initial-design",
                        json={"method": "full_factorial", "n_levels": 2,
                              "random_seed": 7})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["n_points"] == 10
        assert body["design_info"]["total_runs"] == 13
        assert body["feasibility"]["estimability"] == "passed"
        assert body["feasibility"]["n_points_dropped"] == 3

    def test_a_classical_design_that_fails_the_gate_never_reports_passed(self, session_id):
        """"passed" must be earned, not emitted for every classical method.

        Unequal coefficients: a symmetric pair cuts a symmetric corner off a
        symmetric design and leaves it estimable.
        """
        _add_variables(session_id, names=("x1", "x2", "x3"))
        self._constrain(session_id, {"x1": 1.0, "x2": 0.8}, 9.3)
        r = client.post(f"/api/v1/sessions/{session_id}/initial-design",
                        json={"method": "ccd", "random_seed": 7})
        assert r.status_code == 400, r.text
        assert "feasibility" not in r.json()

    def test_an_unconstrained_classical_design_reports_null_feasibility(self, session_id):
        """The block is gated on constraints, not on the method class."""
        _add_variables(session_id, names=("x1", "x2", "x3"))
        r = client.post(f"/api/v1/sessions/{session_id}/initial-design",
                        json={"method": "ccd", "random_seed": 7})
        assert r.status_code == 200, r.text
        assert r.json()["n_points"] == 16
        assert r.json()["feasibility"] is None


class TestConstraintErrorMapping:
    """Both constraint exceptions surface under their own ``error_type``.

    The status code was never the gap. ``DesignNotEstimableError`` and
    ``InfeasibleRegionError`` both subclass ``ValueError``, so the generic
    handler (``error_handlers.py``, ``@app.exception_handler(ValueError)``)
    already returned 400 for them. What a client could not do was tell them
    apart: every one arrived as ``"error_type": "ValueError"``,
    indistinguishable from a malformed bound or an unknown method. These two
    are the only 400s on the design endpoints that are about the *constraint
    set* rather than the request, and they carry different remedies -- relax
    the region, versus switch to a method whose structure survives it.

    Starlette resolves a handler by walking ``type(exc).__mro__`` and taking
    the most specific registered match, so registering the subclasses is
    sufficient and registration order relative to the generic ``ValueError``
    handler does not matter (verified both orders).
    """

    @staticmethod
    def _constrain(sid, coefficients, rhs):
        r = client.post(f"/api/v1/sessions/{sid}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": coefficients, "rhs": rhs,
        })
        r.raise_for_status()

    def test_non_estimable_classical_design_is_400(self, session_id):
        """Unequal coefficients: a symmetric pair on a symmetric CCD cuts a
        symmetric corner and leaves the quadratic model estimable, so it
        returns a design rather than raising.
        """
        _add_variables(session_id, names=("x1", "x2", "x3"))
        self._constrain(session_id, {"x1": 1.0, "x2": 0.8}, 9.3)
        r = client.post(f"/api/v1/sessions/{session_id}/initial-design",
                        json={"method": "ccd", "random_seed": 7})
        assert r.status_code == 400, r.text
        assert r.json()["error_type"] == "DesignNotEstimableError"
        # The remedy reaches the client, not just the log.
        assert "optimal" in r.json()["detail"]
        assert "allow_infeasible" in r.json()["detail"]

    def test_infeasible_region_optimal_design_is_400(self, session_id):
        _add_variables(session_id, names=("x1", "x2", "x3"))
        self._constrain(session_id, {"x1": 1.0, "x2": 1.0}, -1.0)
        r = client.post(f"/api/v1/sessions/{session_id}/optimal-design", json={
            "n_points": 10, "model_type": "linear",
            "criterion": "D", "algorithm": "fedorov", "random_seed": 7,
        })
        assert r.status_code == 400, r.text
        assert r.json()["error_type"] == "InfeasibleRegionError"

    def test_infeasible_region_space_filling_design_is_400(self, session_id):
        """The same exception on the other endpoint.

        The space-filling path raises it from its own up-front emptiness
        proof rather than from the candidate augmenter, so the mapping has to
        hold for a second raise site to be worth anything.
        """
        _add_variables(session_id, names=("x1", "x2", "x3"))
        self._constrain(session_id, {"x1": 1.0, "x2": 1.0}, -1.0)
        r = client.post(f"/api/v1/sessions/{session_id}/initial-design",
                        json={"method": "lhs", "n_points": 5, "random_seed": 7})
        assert r.status_code == 400, r.text
        assert r.json()["error_type"] == "InfeasibleRegionError"

    def test_an_ordinary_value_error_still_reports_value_error(self, session_id):
        """The two new handlers are specific, not a blanket rename.

        A design rejected for a reason that has nothing to do with
        constraints must keep arriving as a plain ``ValueError``; otherwise
        the new ``error_type`` values carry no information.
        """
        _add_variables(session_id, names=("x1",))
        r = client.post(f"/api/v1/sessions/{session_id}/initial-design",
                        json={"method": "ccd", "random_seed": 7})
        assert r.status_code == 400, r.text
        assert r.json()["error_type"] == "ValueError"


class TestRetypingAConstrainedVariableIsRefusedOverREST:
    """``PUT /variables/{name}`` could retype a constrained variable to
    ``categorical`` and return 200, reaching from the other side the exact
    state ``POST /constraints`` refuses. ``GET /constraints`` then displayed
    the constraint as if nothing had happened, and every design failed with
    ``could not convert string to float: 'B'`` -- naming neither the variable
    nor the constraint.

    ``context`` is the other non-numeric type, but ``PUT`` has no
    ``AddContextVariableRequest`` in its body union, so it 422s in validation
    and never reaches the core. Only ``categorical`` is live over REST; the
    core-level test class covers both.
    """

    def _constrained(self, sid, rhs=8.0):
        _add_variables(sid, names=("x1", "x2", "x3"))
        r = client.post(f"/api/v1/sessions/{sid}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": {"x1": 1.0, "x2": 1.0},
            "rhs": rhs, "name": "budget",
        })
        r.raise_for_status()

    def test_the_retype_is_a_labelled_400(self, session_id):
        self._constrained(session_id)
        r = client.put(f"/api/v1/sessions/{session_id}/variables/x2", json={
            "name": "x2", "type": "categorical", "categories": ["A", "B"],
        })
        assert r.status_code == 400, r.text
        detail = r.json()["detail"]
        assert "x2" in detail and "budget" in detail

    def test_the_variable_and_the_constraint_are_both_untouched(self, session_id):
        self._constrained(session_id)
        client.put(f"/api/v1/sessions/{session_id}/variables/x2", json={
            "name": "x2", "type": "categorical", "categories": ["A", "B"],
        })
        variables = client.get(f"/api/v1/sessions/{session_id}/variables").json()["variables"]
        x2 = next(v for v in variables if v["name"] == "x2")
        assert x2["type"] == "real"
        assert x2["bounds"] == [0.0, 10.0]
        assert x2["categories"] is None
        constraints = client.get(f"/api/v1/sessions/{session_id}/constraints").json()
        assert constraints["constraints"][0]["coefficients"] == {"x1": 1.0, "x2": 1.0}

    def test_the_designs_that_used_to_fail_still_succeed(self, session_id):
        """The three calls the refused retype used to break.

        The constraint is deliberately permissive: a tight one degrades the
        classical design on its own merits, which is a different 400 and would
        hide whether the retype was refused.
        """
        self._constrained(session_id, rhs=25.0)
        client.put(f"/api/v1/sessions/{session_id}/variables/x2", json={
            "name": "x2", "type": "categorical", "categories": ["A", "B"],
        })
        for payload in ({"method": "lhs", "n_points": 4, "random_seed": 1},
                        {"method": "full_factorial", "n_levels": 2}):
            r = client.post(f"/api/v1/sessions/{session_id}/initial-design", json=payload)
            assert r.status_code == 200, (payload, r.text)
        r = client.post(f"/api/v1/sessions/{session_id}/optimal-design", json={
            "n_points": 8, "model_type": "linear", "random_seed": 3,
        })
        assert r.status_code == 200, r.text

    def test_an_unconstrained_variable_still_retypes(self, session_id):
        self._constrained(session_id)
        r = client.put(f"/api/v1/sessions/{session_id}/variables/x3", json={
            "name": "x3", "type": "categorical", "categories": ["P", "Q"],
        })
        assert r.status_code == 200, r.text
        assert r.json()["variable"]["type"] == "categorical"

    def test_deleting_the_constraint_releases_the_variable(self, session_id):
        """The refusal is recoverable, which is what the message promises."""
        self._constrained(session_id)
        assert client.delete(
            f"/api/v1/sessions/{session_id}/constraints/budget"
        ).status_code == 200
        r = client.put(f"/api/v1/sessions/{session_id}/variables/x2", json={
            "name": "x2", "type": "categorical", "categories": ["A", "B"],
        })
        assert r.status_code == 200, r.text


class TestTheTotalWipeoutReportsItsOwnErrorType:
    """A constrained classical design with *no* surviving points raised a bare
    ``ValueError``, so a client switching on ``error_type`` could not recognize
    the most severe constraint failure -- while the partial loss beside it
    already reported ``DesignNotEstimableError``. Both subclass ``ValueError``,
    so no catch tuple widens.
    """

    def _impossible(self, sid):
        _add_variables(sid, names=("x1", "x2", "x3"))
        client.post(f"/api/v1/sessions/{sid}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": {"x1": 1.0, "x2": 1.0},
            "rhs": -1.0, "name": "impossible",
        }).raise_for_status()

    @pytest.mark.parametrize("payload", [
        {"method": "ccd", "random_seed": 7},
        {"method": "box_behnken", "random_seed": 7},
        {"method": "full_factorial", "n_levels": 2},
    ])
    def test_it_is_an_infeasible_region_error(self, session_id, payload):
        self._impossible(session_id)
        r = client.post(f"/api/v1/sessions/{session_id}/initial-design", json=payload)
        assert r.status_code == 400, r.text
        assert r.json()["error_type"] == "InfeasibleRegionError"

    def test_the_message_is_unchanged(self, session_id):
        """The message already said the right thing; only the type was wrong."""
        self._impossible(session_id)
        r = client.post(f"/api/v1/sessions/{session_id}/initial-design",
                        json={"method": "ccd", "random_seed": 7})
        detail = r.json()["detail"]
        assert "No 'ccd' design points satisfy" in detail
        assert "cannot be resampled" in detail

    def test_allow_infeasible_does_not_rescue_it(self, session_id):
        self._impossible(session_id)
        r = client.post(f"/api/v1/sessions/{session_id}/initial-design",
                        json={"method": "ccd", "random_seed": 7,
                              "allow_infeasible": True})
        assert r.status_code == 400, r.text
        assert r.json()["error_type"] == "InfeasibleRegionError"

    def test_the_partial_loss_beside_it_keeps_its_own_type(self, session_id):
        """The two must stay distinguishable, which is the point of both."""
        _add_variables(session_id, names=("x1", "x2", "x3"))
        client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": {"x1": 1.0, "x2": 0.8}, "rhs": 9.3,
        }).raise_for_status()
        r = client.post(f"/api/v1/sessions/{session_id}/initial-design",
                        json={"method": "ccd", "random_seed": 7})
        assert r.status_code == 400, r.text
        assert r.json()["error_type"] == "DesignNotEstimableError"


class TestAllowInfeasibleIsReachableOverREST:
    """``CHANGELOG.md`` and the 400's own body both named ``allow_infeasible``
    as the escape hatch, through a parameter no REST caller could reach. A
    constrained classical design that returned 200 before this branch had no
    opt-out at all: ``method="optimal"`` and the space-filling methods are
    different designs, not the same one.
    """

    def _degrading(self, sid):
        _add_variables(sid, names=("x1", "x2", "x3"))
        client.post(f"/api/v1/sessions/{sid}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": {"x1": 1.0, "x2": 0.8}, "rhs": 9.3,
            "name": "budget",
        }).raise_for_status()

    def test_unset_is_the_unchanged_400(self, session_id):
        self._degrading(session_id)
        r = client.post(f"/api/v1/sessions/{session_id}/initial-design",
                        json={"method": "ccd", "random_seed": 7})
        assert r.status_code == 400, r.text
        assert r.json()["error_type"] == "DesignNotEstimableError"

    def test_explicit_false_is_the_same_400(self, session_id):
        self._degrading(session_id)
        r = client.post(f"/api/v1/sessions/{session_id}/initial-design",
                        json={"method": "ccd", "random_seed": 7,
                              "allow_infeasible": False})
        assert r.status_code == 400, r.text
        assert r.json()["error_type"] == "DesignNotEstimableError"

    def test_true_returns_the_remnant(self, session_id):
        self._degrading(session_id)
        r = client.post(f"/api/v1/sessions/{session_id}/initial-design",
                        json={"method": "ccd", "random_seed": 7,
                              "allow_infeasible": True})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["n_points"] > 0
        assert body["n_points"] < body["design_info"]["total_runs"], (
            "the remnant must be smaller than the design, or nothing was dropped"
        )
        assert body["feasibility"]["n_points_dropped"] == (
            body["design_info"]["total_runs"] - body["n_points"]
        )

    def test_every_returned_point_still_satisfies_the_constraint(self, session_id):
        """``allow_infeasible`` waives the estimability gate, not feasibility."""
        self._degrading(session_id)
        r = client.post(f"/api/v1/sessions/{session_id}/initial-design",
                        json={"method": "ccd", "random_seed": 7,
                              "allow_infeasible": True})
        for point in r.json()["points"]:
            assert point["x1"] + 0.8 * point["x2"] <= 9.3 + 1e-9, point

    def test_a_waived_gate_is_not_reported_as_passed(self, session_id):
        """The gate did not pass, it was suppressed, and the route cannot tell
        a design that would have passed from one that would not.
        """
        self._degrading(session_id)
        r = client.post(f"/api/v1/sessions/{session_id}/initial-design",
                        json={"method": "ccd", "random_seed": 7,
                              "allow_infeasible": True})
        assert r.json()["feasibility"]["estimability"] == "waived"

    def test_an_unwaived_gate_still_reports_passed(self, session_id):
        """A constraint that drops nothing structural: the gate runs and
        passes, and ``allow_infeasible`` left unset must not disturb that.
        """
        _add_variables(session_id, names=("x1", "x2", "x3"))
        client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": {"x1": 1.0, "x2": 1.0}, "rhs": 25.0,
        }).raise_for_status()
        r = client.post(f"/api/v1/sessions/{session_id}/initial-design",
                        json={"method": "ccd", "random_seed": 7})
        assert r.status_code == 200, r.text
        assert r.json()["feasibility"]["estimability"] == "passed"

    def test_a_space_filling_method_is_unaffected_by_the_flag(self, session_id):
        self._degrading(session_id)
        seeded = []
        for flag in (False, True):
            r = client.post(f"/api/v1/sessions/{session_id}/initial-design",
                            json={"method": "lhs", "n_points": 5,
                                  "random_seed": 11, "allow_infeasible": flag})
            assert r.status_code == 200, r.text
            assert r.json()["feasibility"]["estimability"] == "not_applicable"
            seeded.append(r.json()["points"])
        assert seeded[0] == seeded[1], "the flag must not perturb the sampler"

    def test_the_flag_is_absent_from_the_optimal_design_request(self, session_id):
        """The gate exempts ``optimal``, so the hatch has nothing to open
        there. Pinned so the field is not copied across by symmetry.
        """
        from api.models.requests import OptimalDesignRequest
        assert "allow_infeasible" not in OptimalDesignRequest.model_fields
