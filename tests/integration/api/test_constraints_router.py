"""Constraint CRUD over REST.

Before this, constraints could only be set from Python, so a non-Python
consumer could not use the constraint feature at all.

Routers mount under /api/v1 (api/main.py:61-68). Setup mirrors
tests/integration/api/test_optimal_design_endpoints.py.
"""

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
