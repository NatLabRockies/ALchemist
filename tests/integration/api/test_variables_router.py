"""
Integration tests for the variables router endpoints.
"""

import io
import json
import typing
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routers import variables as variables_router

client = TestClient(app)


@pytest.fixture
def session_id():
    """Create a fresh optimization session for each test and clean it up afterwards."""
    response = client.post("/api/v1/sessions", json={"ttl_hours": 1})
    assert response.status_code == 201
    session_id = response.json()["session_id"]
    yield session_id
    client.delete(f"/api/v1/sessions/{session_id}")


def test_add_variable_rejects_duplicates(session_id):
    payload = {
        "name": "temperature",
        "type": "real",
        "min": 250.0,
        "max": 500.0,
        "unit": "degC",
    }

    first_response = client.post(f"/api/v1/sessions/{session_id}/variables", json=payload)
    assert first_response.status_code == 200

    duplicate_response = client.post(f"/api/v1/sessions/{session_id}/variables", json=payload)
    assert duplicate_response.status_code == 400
    assert "already exists" in duplicate_response.json()["detail"].lower()


def test_update_variable_success_and_mismatched_name(session_id):
    payload = {
        "name": "pressure",
        "type": "real",
        "min": 1.0,
        "max": 10.0,
        "unit": "bar",
    }
    add_response = client.post(f"/api/v1/sessions/{session_id}/variables", json=payload)
    assert add_response.status_code == 200

    update_payload = {
        "name": "pressure",
        "type": "real",
        "min": 2.0,
        "max": 12.0,
        "unit": "bar",
    }
    update_response = client.put(
        f"/api/v1/sessions/{session_id}/variables/pressure",
        json=update_payload,
    )
    assert update_response.status_code == 200
    updated_variable = update_response.json()["variable"]
    assert updated_variable["min"] == pytest.approx(2.0)
    assert updated_variable["max"] == pytest.approx(12.0)

    mismatch_payload = {
        "name": "pressure-renamed",
        "type": "real",
        "min": 2.0,
        "max": 12.0,
    }
    mismatch_response = client.put(
        f"/api/v1/sessions/{session_id}/variables/pressure",
        json=mismatch_payload,
    )
    assert mismatch_response.status_code == 400
    assert "must match" in mismatch_response.json()["detail"].lower()

    missing_response = client.put(
        f"/api/v1/sessions/{session_id}/variables/does-not-exist",
        json={
            "name": "does-not-exist",
            "type": "real",
            "min": 0.0,
            "max": 1.0,
        },
    )
    assert missing_response.status_code == 404


def test_load_and_export_variables(session_id):
    variables_payload = [
        {
            "name": "temperature",
            "type": "real",
            "min": 100.0,
            "max": 200.0,
        },
        {
            "name": "catalyst",
            "type": "categorical",
            "categories": ["A", "B", "C"],
        },
    ]

    json_buffer = io.BytesIO(json.dumps(variables_payload).encode("utf-8"))
    load_response = client.post(
        f"/api/v1/sessions/{session_id}/variables/load",
        files={"file": ("variables.json", json_buffer, "application/json")},
    )
    assert load_response.status_code == 200
    assert load_response.json()["n_variables"] == 2

    list_response = client.get(f"/api/v1/sessions/{session_id}/variables")
    assert list_response.status_code == 200
    listed = list_response.json()
    assert listed["n_variables"] == 2
    names = {var["name"] for var in listed["variables"]}
    assert names == {"temperature", "catalyst"}

    export_response = client.get(f"/api/v1/sessions/{session_id}/variables/export")
    assert export_response.status_code == 200
    exported = export_response.json()
    assert len(exported) == 2
    assert any(var["name"] == "catalyst" for var in exported)
    assert export_response.headers["Content-Disposition"].startswith("attachment; filename=")


def test_export_uses_canonical_values_field_for_categorical(session_id):
    """Categorical exports must use 'values' (canonical SearchSpace.from_dict schema)
    so the resulting file can be re-loaded by the desktop GUI, core Python API,
    or the API's own /variables/load endpoint without translation."""
    client.post(
        f"/api/v1/sessions/{session_id}/variables",
        json={"name": "catalyst", "type": "categorical", "categories": ["A", "B"]},
    )

    export_response = client.get(f"/api/v1/sessions/{session_id}/variables/export")
    assert export_response.status_code == 200
    exported = export_response.json()
    catalyst = next(v for v in exported if v["name"] == "catalyst")
    assert "values" in catalyst, (
        "Categorical export must include 'values' (canonical SearchSpace schema). "
        "Found keys: " + ", ".join(sorted(catalyst.keys()))
    )
    assert catalyst["values"] == ["A", "B"]


def test_exported_variables_roundtrip_through_searchspace(session_id):
    """End-to-end: variables added via API can be exported and the resulting JSON
    is directly consumable by SearchSpace.from_dict (the desktop loader)."""
    from alchemist_core.data.search_space import SearchSpace

    client.post(
        f"/api/v1/sessions/{session_id}/variables",
        json={"name": "temperature", "type": "real", "min": 100.0, "max": 200.0},
    )
    client.post(
        f"/api/v1/sessions/{session_id}/variables",
        json={"name": "catalyst", "type": "categorical", "categories": ["A", "B"]},
    )
    client.post(
        f"/api/v1/sessions/{session_id}/variables",
        json={"name": "SAR", "type": "discrete", "allowed_values": [80, 280]},
    )

    export_response = client.get(f"/api/v1/sessions/{session_id}/variables/export")
    exported = export_response.json()

    # SearchSpace.from_dict is the canonical loader used by desktop GUI
    ss = SearchSpace().from_dict(exported)
    names = {v["name"] for v in ss.variables}
    assert names == {"temperature", "catalyst", "SAR"}
    assert "catalyst" in ss.get_categorical_variables()
    assert "SAR" in ss.get_discrete_variables()


def test_delete_variable(session_id):
    payload = {
        "name": "cycles",
        "type": "integer",
        "min": 1,
        "max": 5,
    }
    add_response = client.post(f"/api/v1/sessions/{session_id}/variables", json=payload)
    assert add_response.status_code == 200

    delete_response = client.delete(f"/api/v1/sessions/{session_id}/variables/cycles")
    assert delete_response.status_code == 200
    body = delete_response.json()
    assert body["n_variables"] == 0

    get_response = client.get(f"/api/v1/sessions/{session_id}/variables")
    assert get_response.status_code == 200
    assert get_response.json()["n_variables"] == 0


# ============================================================
# Variable names must be addressable by the DELETE route
# ============================================================

_VARIABLE_SHAPES = {
    "real": {"type": "real", "min": 0.0, "max": 10.0},
    "integer": {"type": "integer", "min": 0, "max": 10},
    "categorical": {"type": "categorical", "categories": ["A", "B"]},
    "discrete": {"type": "discrete", "allowed_values": [1.0, 2.0]},
}

# Every test in this section runs against all four variable types on purpose.
# Coverage that exercised only 'real' is what let this defect through.
_VARIABLE_TYPES = list(_VARIABLE_SHAPES)

_UNADDRESSABLE = ["", "a/b", "/leading", "trailing/", ".", ".."]

_ADDRESSABLE = [
    "flow rate 1",   # spaces
    "x1(+)-2.0",     # punctuation
    "purity 95%",    # percent sign
    "αβ ≤ 3",        # unicode and an operator a user would actually type
    "...",           # not a dot segment; three dots is a normal name
    ".hidden",       # leading dot, not a dot segment
]


def _variable_payload(var_type, name):
    return {"name": name, **_VARIABLE_SHAPES[var_type]}


def _n_variables(sid):
    return client.get(f"/api/v1/sessions/{sid}/variables").json()["n_variables"]


class TestVariableNameIsAddressable:
    """A variable name that cannot survive a URL path segment is rejected.

    ``name`` carried no validation, and a variable is addressed by name:
    ``DELETE /sessions/{id}/variables/{name}``. ``..`` was not merely
    undeletable -- RFC 3986 section 5.2.4 dot-segment removal is applied by
    clients and proxies before the request is sent, so the DELETE was
    rewritten onto the *session* endpoint and destroyed the whole session
    (every variable, every experiment, the trained model) while returning a
    success code. Reproduced end to end against a live uvicorn server.
    """

    def test_dot_segment_delete_is_rewritten_onto_the_session_route(self):
        """Pin the mechanism, so the rule is not removed as over-cautious.

        This is client-side URL normalization, not anything the router does;
        no server is involved and none can defend against it. It is the whole
        reason ``..`` has to be rejected at creation time.
        """
        import httpx

        rewritten = httpx.URL(
            "http://testserver/api/v1/sessions/abc123/variables/.."
        ).path
        assert rewritten == "/api/v1/sessions/abc123"

    @pytest.mark.parametrize("var_type", _VARIABLE_TYPES)
    @pytest.mark.parametrize("name", _UNADDRESSABLE)
    def test_unaddressable_name_is_rejected(self, session_id, var_type, name):
        r = client.post(
            f"/api/v1/sessions/{session_id}/variables",
            json=_variable_payload(var_type, name),
        )
        assert r.status_code == 422, r.text
        assert _n_variables(session_id) == 0

    @pytest.mark.parametrize("var_type", _VARIABLE_TYPES)
    @pytest.mark.parametrize("name", _ADDRESSABLE)
    def test_addressable_name_is_accepted_and_deletable(
        self, session_id, var_type, name
    ):
        """The rule must not over-restrict, and acceptance is not enough.

        What POST accepts, DELETE has to be able to address -- otherwise the
        variable is just as stuck as the names above. The segment is
        percent-encoded, which is what a correct HTTP client does.
        """
        r = client.post(
            f"/api/v1/sessions/{session_id}/variables",
            json=_variable_payload(var_type, name),
        )
        assert r.status_code == 200, r.text
        assert client.get(
            f"/api/v1/sessions/{session_id}/variables"
        ).json()["variables"][0]["name"] == name

        d = client.delete(
            f"/api/v1/sessions/{session_id}/variables/{quote(name, safe='')}"
        )
        assert d.status_code == 200, d.text
        assert _n_variables(session_id) == 0

    @pytest.mark.parametrize("var_type", _VARIABLE_TYPES)
    def test_rejected_name_returns_a_serializable_body(self, session_id, var_type):
        """The 422 body must render, for every variable type.

        The app's RequestValidationError handler JSON-encodes ``exc.errors()``.
        A plain ValueError raised from a field_validator lands in the error
        ``ctx`` as a live exception object, which is not JSON serializable and
        turns the 422 into a 500 -- for *every* 422 in the application, not
        just this one. The validator raises PydanticCustomError to avoid it,
        and ``r.json()`` below is what proves the body encodes at all.
        """
        r = client.post(
            f"/api/v1/sessions/{session_id}/variables",
            json=_variable_payload(var_type, "a/b"),
        )
        assert r.status_code == 422
        body = r.json()
        # The request body is a Union of the four variable models, so the
        # non-matching members contribute their own literal/missing errors.
        assert any(
            e["type"] == "variable_name_not_addressable" for e in body["errors"]
        ), body["errors"]
        assert "a/b" in str(body)
        json.dumps(body)

    @pytest.mark.parametrize("var_type", _VARIABLE_TYPES)
    def test_rejected_name_leaves_the_session_intact(self, session_id, var_type):
        """The regression this rule exists for: nothing is lost.

        Before the fix, POSTing '..' returned 200 and the follow-up DELETE
        took the session with it. The property that matters is not just the
        422 -- it is that the session and everything already in it survive.
        """
        seed = client.post(
            f"/api/v1/sessions/{session_id}/variables",
            json={"name": "x1", "type": "real", "min": 0.0, "max": 10.0},
        )
        assert seed.status_code == 200

        r = client.post(
            f"/api/v1/sessions/{session_id}/variables",
            json=_variable_payload(var_type, ".."),
        )
        assert r.status_code == 422, r.text

        assert client.get(f"/api/v1/sessions/{session_id}").status_code == 200
        listed = client.get(f"/api/v1/sessions/{session_id}/variables").json()
        assert listed["n_variables"] == 1
        assert listed["variables"][0]["name"] == "x1"

    @pytest.mark.parametrize("var_type", _VARIABLE_TYPES)
    def test_update_also_rejects_an_unaddressable_name(self, session_id, var_type):
        """PUT shares the same request models, so it inherits the same rule."""
        r = client.put(
            f"/api/v1/sessions/{session_id}/variables/x1",
            json=_variable_payload(var_type, ".."),
        )
        assert r.status_code == 422, r.text


# ============================================================
# Ruling 33 Item A -- the name guard is enforced, not merely conventional
# ============================================================

_VARIABLE_UNION_MEMBERS = [
    (route_name, model)
    for route_name in ("add_variable", "update_variable")
    for model in typing.get_args(
        getattr(variables_router, route_name).__annotations__["variable"]
    )
]


def _member_id(param):
    return getattr(param, "__name__", str(param))


class TestEveryVariableModelOnTheRouteCarriesTheNameGuard:
    """``VariableRequest`` helps, but a base class cannot enforce itself.

    The addressable-name rule lives on a fields-free ``VariableRequest`` base
    so all four variable models inherit it. That was recorded as meaning a
    fifth variable type "cannot silently reintroduce the hole" -- which is
    false. ``class X(BaseModel)`` is the shape every model in ``requests.py``
    had before the rule existed, so it is the shape a new one will be copied
    from, and such a model carries no validator at all: it was wired into the
    live Union and the whole Task 11 suite still passed, 169/169.

    These tests read the Union off the *route's own annotation* rather than a
    hand-maintained list, so a fifth member is picked up automatically and has
    to satisfy the rule to get in.
    """

    def test_the_route_annotation_is_a_union_of_at_least_the_four_types(self):
        for route_name in ("add_variable", "update_variable"):
            members = typing.get_args(
                getattr(variables_router, route_name).__annotations__["variable"]
            )
            assert len(members) >= 4, route_name

    @pytest.mark.parametrize(
        "route_name,model", _VARIABLE_UNION_MEMBERS, ids=_member_id
    )
    def test_every_union_member_declares_the_validator(self, route_name, model):
        validators = model.__pydantic_decorators__.field_validators
        assert "_name_must_be_addressable" in validators, (
            f"{model.__name__} is accepted by the {route_name} route but does "
            f"not carry the addressable-name validator. Inherit "
            f"VariableRequest (not BaseModel) so the rule applies. "
            f"Declared validators: {sorted(validators)}"
        )
        assert validators["_name_must_be_addressable"].info.fields == ("name",)

    @pytest.mark.parametrize(
        "route_name,model", _VARIABLE_UNION_MEMBERS, ids=_member_id
    )
    def test_every_union_member_actually_rejects_a_dot_segment(self, route_name, model):
        """Declaring the validator is necessary; firing is what matters.

        The payload shape is looked up by the model's own ``type`` literal, so
        a fifth variable type must also be added to ``_VARIABLE_SHAPES`` --
        which is the point: a new type cannot join the route unnoticed.
        """
        from pydantic import ValidationError

        var_type = typing.get_args(model.model_fields["type"].annotation)[0]
        assert var_type in _VARIABLE_SHAPES, (
            f"{model.__name__} declares type '{var_type}', which has no shape "
            f"in _VARIABLE_SHAPES. Add it so the guard is exercised."
        )
        with pytest.raises(ValidationError):
            model.model_validate(_variable_payload(var_type, ".."))


# ============================================================
# Ruling 29 -- opt-in dict export carries constraints
# ============================================================

def _seed_constrained_space(sid):
    """Two numeric variables and one asymmetric constraint."""
    for payload in (
        {"name": "x1", "type": "real", "min": 0.0, "max": 10.0},
        {"name": "x2", "type": "integer", "min": 0, "max": 10},
    ):
        client.post(f"/api/v1/sessions/{sid}/variables", json=payload).raise_for_status()
    client.post(f"/api/v1/sessions/{sid}/constraints", json={
        "constraint_type": "inequality",
        "coefficients": {"x1": 3.0, "x2": -2.0},
        "rhs": 8.0,
        "name": "c_a",
    }).raise_for_status()


class TestExportIncludeConstraints:
    """Export keeps its bare-list shape by default and gains an opt-in dict.

    spec 7.2 asserts export "already emits whatever ``to_dict`` produces, which
    includes constraints". Neither half is true: the endpoint builds its own
    list from the search-space summary and never calls ``to_dict``, and
    ``to_dict`` is a ``List[Dict]`` that carries no constraints. spec 9.6
    separately requires ``load -> export -> load`` to round-trip constraints.
    Changing the shape would break three tests in this file, one of which is a
    deliberate desktop-loader compatibility contract, so the requirement is met
    with an opt-in instead.
    """

    def test_default_export_is_unchanged_and_drops_constraints(self, session_id):
        _seed_constrained_space(session_id)
        r = client.get(f"/api/v1/sessions/{session_id}/variables/export")
        assert r.status_code == 200
        exported = r.json()
        assert isinstance(exported, list)
        assert {v["name"] for v in exported} == {"x1", "x2"}
        assert r.headers["Content-Disposition"].startswith("attachment; filename=")

    @pytest.mark.parametrize("flag", ["false", "False", "0"])
    def test_explicit_false_is_also_the_bare_list(self, session_id, flag):
        _seed_constrained_space(session_id)
        r = client.get(
            f"/api/v1/sessions/{session_id}/variables/export",
            params={"include_constraints": flag},
        )
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_opt_in_returns_variables_and_constraints(self, session_id):
        _seed_constrained_space(session_id)
        r = client.get(
            f"/api/v1/sessions/{session_id}/variables/export",
            params={"include_constraints": "true"},
        )
        assert r.status_code == 200
        exported = r.json()
        assert set(exported) == {"variables", "constraints"}
        assert {v["name"] for v in exported["variables"]} == {"x1", "x2"}
        assert len(exported["constraints"]) == 1
        c = exported["constraints"][0]
        assert c["name"] == "c_a"
        assert c["type"] == "inequality"
        assert c["coefficients"] == {"x1": 3.0, "x2": -2.0}
        assert c["rhs"] == 8.0
        assert r.headers["Content-Disposition"].startswith("attachment; filename=")

    def test_opt_in_variables_match_the_default_export_exactly(self, session_id):
        """The opt-in must wrap the same list, not build a second one."""
        _seed_constrained_space(session_id)
        client.post(
            f"/api/v1/sessions/{session_id}/variables",
            json={"name": "x3", "type": "categorical", "categories": ["A", "B"]},
        ).raise_for_status()
        bare = client.get(f"/api/v1/sessions/{session_id}/variables/export").json()
        wrapped = client.get(
            f"/api/v1/sessions/{session_id}/variables/export",
            params={"include_constraints": "true"},
        ).json()
        assert wrapped["variables"] == bare

    def test_opt_in_on_an_unconstrained_space_returns_an_empty_list(self, session_id):
        client.post(
            f"/api/v1/sessions/{session_id}/variables",
            json={"name": "x1", "type": "real", "min": 0.0, "max": 10.0},
        ).raise_for_status()
        exported = client.get(
            f"/api/v1/sessions/{session_id}/variables/export",
            params={"include_constraints": "true"},
        ).json()
        assert exported["constraints"] == []

    def test_opt_in_export_is_consumable_by_search_space_load(self, session_id, tmp_path):
        """Cross-surface: the opt-in payload is the same shape
        ``SearchSpace.save_to_json`` writes, so ``load_from_json`` reads it."""
        from alchemist_core.data.search_space import SearchSpace

        _seed_constrained_space(session_id)
        exported = client.get(
            f"/api/v1/sessions/{session_id}/variables/export",
            params={"include_constraints": "true"},
        ).json()

        path = tmp_path / "space.json"
        path.write_text(json.dumps(exported))
        ss = SearchSpace()
        ss.load_from_json(str(path))
        assert {v["name"] for v in ss.variables} == {"x1", "x2"}
        assert len(ss.get_constraints()) == 1
        assert ss.get_constraints()[0]["coefficients"] == {"x1": 3.0, "x2": -2.0}


# ============================================================
# Ruling 35 -- fix round 1
# ============================================================

def _upload_space(sid, payload):
    """POST a raw payload to /variables/load.

    ``json.dumps`` writes bare ``NaN``/``Infinity``/``-Infinity`` literals and
    ``json.load`` reads them back by default, so a file carrying a non-finite
    bound is an ordinary upload, not a crafted one.
    """
    buf = io.BytesIO(json.dumps(payload).encode())
    return client.post(
        f"/api/v1/sessions/{sid}/variables/load",
        files={"file": ("space.json", buf, "application/json")},
    )


_NON_FINITE = [float("nan"), float("inf"), float("-inf")]


class TestNonFiniteBoundsCannotPoisonTheExport:
    """A bound of NaN loaded with 200 and then broke both export shapes.

    skopt's ``low >= high`` guard is ``False`` for NaN, so ``Real(nan, 10.0)``
    built cleanly and the variable registered. Every later export then failed:
    ``JSONResponse`` serializes with ``allow_nan=False``, so both
    ``/variables/export`` and ``/variables/export?include_constraints=true``
    returned 400 "Out of range float values are not JSON compliant: nan". With
    no export shape able to emit the space, the session could not be recovered
    through the API at all.
    """

    @pytest.mark.parametrize("bad", _NON_FINITE)
    @pytest.mark.parametrize("shape", ["bare", "dict"])
    def test_a_non_finite_bound_is_rejected_on_both_load_shapes(
        self, session_id, shape, bad
    ):
        var = {"name": "x1", "type": "real", "min": bad, "max": 10.0}
        payload = [var] if shape == "bare" else {"variables": [var], "constraints": []}
        r = _upload_space(session_id, payload)
        assert r.status_code == 400, r.text
        assert "finite" in r.json()["detail"]
        assert "x1" in r.json()["detail"], "the message must name the variable"

    @pytest.mark.parametrize("bad", _NON_FINITE)
    @pytest.mark.parametrize("var", [
        {"name": "x1", "type": "real", "min": None, "max": 10.0},
        {"name": "x2", "type": "integer", "min": 0, "max": None},
        {"name": "x3", "type": "discrete", "allowed_values": [0.5, None, 7.25]},
    ])
    def test_every_numeric_variable_type_is_covered(self, session_id, var, bad):
        entry = {k: (bad if v is None else v) for k, v in var.items()}
        if entry["type"] == "discrete":
            entry["allowed_values"] = [0.5, bad, 7.25]
        r = _upload_space(session_id, [entry])
        assert r.status_code == 400, r.text
        assert entry["name"] in r.json()["detail"]

    @pytest.mark.parametrize("bad", _NON_FINITE)
    def test_both_export_shapes_still_work_after_the_rejection(self, session_id, bad):
        """The property that was lost: the session stays exportable."""
        _seed_constrained_space(session_id)
        r = _upload_space(session_id, [
            {"name": "x9", "type": "real", "min": bad, "max": 10.0},
        ])
        assert r.status_code == 400, r.text

        bare = client.get(f"/api/v1/sessions/{session_id}/variables/export")
        assert bare.status_code == 200, bare.text
        assert {v["name"] for v in bare.json()} == {"x1", "x2"}

        wrapped = client.get(
            f"/api/v1/sessions/{session_id}/variables/export",
            params={"include_constraints": "true"},
        )
        assert wrapped.status_code == 200, wrapped.text
        assert len(wrapped.json()["constraints"]) == 1

    @pytest.mark.parametrize("bad", _NON_FINITE)
    def test_the_rejected_file_leaves_the_session_exactly_as_it_was(
        self, session_id, bad
    ):
        _seed_constrained_space(session_id)
        before = client.get(f"/api/v1/sessions/{session_id}/variables").json()
        r = _upload_space(session_id, {
            "variables": [{"name": "x9", "type": "integer", "min": 0, "max": bad}],
            "constraints": [],
        })
        assert r.status_code == 400, r.text
        assert client.get(f"/api/v1/sessions/{session_id}/variables").json() == before

    def test_a_very_large_finite_bound_is_still_accepted(self, session_id):
        """The guard is about finiteness, not magnitude."""
        r = _upload_space(session_id, [
            {"name": "x1", "type": "real", "min": -1.5e300, "max": 2.5e300},
        ])
        assert r.status_code == 200, r.text
        export = client.get(f"/api/v1/sessions/{session_id}/variables/export")
        assert export.status_code == 200
        assert export.json()[0]["max"] == 2.5e300


class TestDictLoadPreservesVariableMetadata:
    """``unit`` and ``description`` survived the bare list but not the dict.

    ``from_dict`` forwarded only the dimension-building fields, so the shape
    this task newly advertised as *the* round-trip format -- and as what
    ``SearchSpace.save_to_json`` writes -- lost metadata that ``POST
    /variables`` had stored and the export had emitted. That made the new path
    strictly worse than the legacy one beside it.
    """

    ANNOTATED = [
        {"name": "x1", "type": "real", "min": 0.0, "max": 10.0,
         "unit": "kPa", "description": "first axis"},
        {"name": "x2", "type": "integer", "min": 0, "max": 8,
         "unit": "counts", "description": "second axis"},
        {"name": "x3", "type": "discrete", "allowed_values": [0.5, 7.25],
         "unit": "mm", "description": "third axis"},
        {"name": "x4", "type": "categorical", "categories": ["A", "B", "C"],
         "unit": "-", "description": "fourth axis"},
    ]

    def _seed(self, sid):
        for payload in self.ANNOTATED:
            client.post(
                f"/api/v1/sessions/{sid}/variables", json=payload
            ).raise_for_status()
        client.post(f"/api/v1/sessions/{sid}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": {"x1": 3.0, "x2": -2.5, "x3": 0.75},
            "rhs": 8.0,
            "name": "c_a",
        }).raise_for_status()

    @staticmethod
    def _metadata(exported):
        return {
            v["name"]: (v.get("unit"), v.get("description"))
            for v in exported
        }

    def test_the_full_advertised_chain_keeps_both_fields(self, session_id):
        """POST -> export(include_constraints) -> load -> export."""
        self._seed(session_id)
        first = client.get(
            f"/api/v1/sessions/{session_id}/variables/export",
            params={"include_constraints": "true"},
        )
        assert first.status_code == 200, first.text
        exported = first.json()
        assert self._metadata(exported["variables"]) == {
            "x1": ("kPa", "first axis"),
            "x2": ("counts", "second axis"),
            "x3": ("mm", "third axis"),
            "x4": ("-", "fourth axis"),
        }

        reload = _upload_space(session_id, exported)
        assert reload.status_code == 200, reload.text
        assert reload.json()["n_constraints"] == 1

        second = client.get(
            f"/api/v1/sessions/{session_id}/variables/export",
            params={"include_constraints": "true"},
        )
        assert second.status_code == 200, second.text
        assert second.json() == exported, "export -> load -> export is not a fixed point"

    def test_the_dict_path_matches_the_bare_list_path(self, session_id):
        """The comparison that made this a defect rather than a gap."""
        bare_sid = client.post("/api/v1/sessions", json={"ttl_hours": 1}).json()["session_id"]
        try:
            _upload_space(session_id, {
                "variables": [dict(v) for v in self.ANNOTATED], "constraints": [],
            }).raise_for_status()
            _upload_space(bare_sid, [dict(v) for v in self.ANNOTATED]).raise_for_status()
            via_dict = client.get(
                f"/api/v1/sessions/{session_id}/variables/export"
            ).json()
            via_list = client.get(
                f"/api/v1/sessions/{bare_sid}/variables/export"
            ).json()
            assert self._metadata(via_dict) == self._metadata(via_list)
            assert self._metadata(via_dict) != {
                v["name"]: (None, None) for v in self.ANNOTATED
            }
        finally:
            client.delete(f"/api/v1/sessions/{bare_sid}")

    def test_a_file_without_metadata_still_loads(self, session_id):
        """The desktop loader shares ``from_dict``; files that never carried
        these fields must be unaffected."""
        r = _upload_space(session_id, {
            "variables": [
                {"name": "x1", "type": "real", "min": 0.0, "max": 10.0},
                {"name": "x4", "type": "categorical", "values": ["A", "B"]},
            ],
            "constraints": [],
        })
        assert r.status_code == 200, r.text
        exported = client.get(
            f"/api/v1/sessions/{session_id}/variables/export"
        ).json()
        assert exported == [
            {"name": "x1", "type": "real", "min": 0.0, "max": 10.0},
            {"name": "x4", "type": "categorical", "values": ["A", "B"]},
        ]


def _session_with_a_derived_variable(tmp_path):
    """Upload a session file carrying a derived variable, return its id.

    There is no route that registers a derived variable, so the only way one
    reaches a REST session is ``POST /sessions/upload`` -- which is also how it
    comes back, and therefore why clearing it on a replace is recoverable.
    """
    from alchemist_core.session import OptimizationSession

    s = OptimizationSession()
    s.add_variable("x1", "real", min=0.0, max=10.0)
    s.add_variable("x2", "integer", min=1, max=8)
    s.add_derived_variable(
        "ratio", lambda row: row["x1"] / row["x2"], ["x1", "x2"], "x1 over x2"
    )
    path = tmp_path / "session.json"
    s.save_session(str(path))
    with open(path, "rb") as fh:
        r = client.post(
            "/api/v1/sessions/upload",
            files={"file": ("session.json", fh, "application/json")},
        )
    assert r.status_code == 201, r.text
    return r.json()["session_id"]


def _search_space_of(sid):
    from api.services.session_store import session_store
    return session_store._sessions[sid]["session"].search_space


class TestDictLoadReplacesDerivedVariables:
    """A "replace" that left derived variables behind left them dangling.

    The dict branch resets the variables, the three index lists and the
    constraints, but carried ``derived_variables`` across. A survivor's
    ``input_cols`` then names base variables the load has just deleted, and it
    can share a name with a newly loaded tunable variable -- a collision
    ``SearchSpace.add_derived_variable`` refuses outright, so this path was the
    only way to produce it.
    """

    def test_a_replace_clears_them(self, tmp_path):
        sid = _session_with_a_derived_variable(tmp_path)
        try:
            assert _search_space_of(sid).get_derived_variable_names() == ["ratio"]
            r = _upload_space(sid, {
                "variables": [{"name": "x7", "type": "real", "min": 0.0, "max": 3.5}],
                "constraints": [],
            })
            assert r.status_code == 200, r.text
            assert _search_space_of(sid).get_derived_variable_names() == []
        finally:
            client.delete(f"/api/v1/sessions/{sid}")

    def test_the_name_collision_is_no_longer_reachable(self, tmp_path):
        """Load a variable named exactly like the surviving derived one."""
        sid = _session_with_a_derived_variable(tmp_path)
        try:
            r = _upload_space(sid, {
                "variables": [{"name": "ratio", "type": "real", "min": 0.0, "max": 1.0}],
                "constraints": [],
            })
            assert r.status_code == 200, r.text
            space = _search_space_of(sid)
            tunable = {v["name"] for v in space.variables}
            derived = set(space.get_derived_variable_names())
            assert tunable & derived == set(), (
                "a name is registered as both tunable and derived -- the state "
                "add_derived_variable raises ValueError on"
            )
        finally:
            client.delete(f"/api/v1/sessions/{sid}")

    def test_no_survivor_references_a_variable_that_is_gone(self, tmp_path):
        sid = _session_with_a_derived_variable(tmp_path)
        try:
            _upload_space(sid, {
                "variables": [{"name": "x7", "type": "integer", "min": 0, "max": 8}],
                "constraints": [],
            }).raise_for_status()
            space = _search_space_of(sid)
            present = {v["name"] for v in space.variables}
            for dv in space.derived_variables_to_dict():
                assert set(dv["input_cols"]) <= present, dv
        finally:
            client.delete(f"/api/v1/sessions/{sid}")

    def test_the_bare_list_path_does_not_clear_them(self, tmp_path):
        """The bare list appends and never claimed to replace anything, so the
        derived variables it does not touch stay valid."""
        sid = _session_with_a_derived_variable(tmp_path)
        try:
            r = _upload_space(sid, [
                {"name": "x7", "type": "categorical", "values": ["A", "B"]},
            ])
            assert r.status_code == 200, r.text
            space = _search_space_of(sid)
            assert space.get_derived_variable_names() == ["ratio"]
            assert set(space.derived_variables_to_dict()[0]["input_cols"]) <= {
                v["name"] for v in space.variables
            }
        finally:
            client.delete(f"/api/v1/sessions/{sid}")

    def test_a_rejected_dict_file_leaves_them_alone(self, tmp_path):
        """Clearing happens with the rest of the replace, not before it."""
        sid = _session_with_a_derived_variable(tmp_path)
        try:
            r = _upload_space(sid, {
                "variables": [{"name": "x7", "type": "real", "min": 9.0, "max": 1.0}],
                "constraints": [],
            })
            assert r.status_code == 400, r.text
            assert _search_space_of(sid).get_derived_variable_names() == ["ratio"]
        finally:
            client.delete(f"/api/v1/sessions/{sid}")


# ============================================================
# Ruling 37 -- fix round 2
# ============================================================

# 2**64 is where numpy stops coercing; 2**2000 is past the float64 range,
# where a fix routing through ``float()`` (``math.isfinite``) would raise
# OverflowError instead -- the same defect one door further along.
_BEYOND_NUMPY = [2**63, 2**64, 2**70, 2**200, 2**2000]


class TestALargeIntegerBoundIsNotAServerError:
    """``POST /variables`` returned 500 for ``integer max=2**64``.

    Round 1's ``_validate_bound`` called ``np.isfinite(value)``. A Python
    ``int`` outside the uint64 range fits no numpy dtype, so numpy raises
    ``TypeError: ufunc 'isfinite' not supported for the input types`` rather
    than answering the question. ``POST /variables`` has no try/except and only
    ``ValueError`` has a global handler, so the TypeError became a bare 500 --
    on a request that returned 200 before the guard existed. ``2**63`` still
    worked, which is why nothing noticed: the boundary is uint64, not int64.

    ``real`` and ``discrete`` bounds never reached it, because their request
    models coerce through ``float`` first. Only ``integer`` preserves
    arbitrary-precision ints, so only ``integer`` could reach numpy with one.
    """

    @pytest.mark.parametrize("bound", _BEYOND_NUMPY)
    def test_post_variables_accepts_it(self, session_id, bound):
        r = client.post(f"/api/v1/sessions/{session_id}/variables", json={
            "name": "x2", "type": "integer", "min": 0, "max": bound,
        })
        assert r.status_code == 200, r.text
        assert r.json()["variable"]["max"] == bound

    @pytest.mark.parametrize("bound", _BEYOND_NUMPY)
    def test_it_is_never_a_500(self, session_id, bound):
        """Stated as itself -- the defect was the status code, not the value."""
        r = client.post(f"/api/v1/sessions/{session_id}/variables", json={
            "name": "x2", "type": "integer", "min": -bound, "max": bound,
        })
        assert r.status_code != 500, r.text
        assert r.json().get("error_type") != "TypeError"

    def test_the_variable_is_usable_afterwards(self, session_id):
        """Registering with 200 is not the property; the session still working
        is. A constrained space is built around the huge bound and exported."""
        client.post(f"/api/v1/sessions/{session_id}/variables", json={
            "name": "x2", "type": "integer", "min": 0, "max": 2**70,
        }).raise_for_status()
        client.post(f"/api/v1/sessions/{session_id}/variables", json={
            "name": "x1", "type": "real", "min": 0.0, "max": 10.0,
        }).raise_for_status()
        client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": {"x1": 3.0, "x2": -2.0},
            "rhs": 8.0,
            "name": "c_a",
        }).raise_for_status()

        listed = client.get(f"/api/v1/sessions/{session_id}/variables")
        assert listed.status_code == 200, listed.text
        assert listed.json()["n_variables"] == 2

        wrapped = client.get(
            f"/api/v1/sessions/{session_id}/variables/export",
            params={"include_constraints": "true"},
        )
        assert wrapped.status_code == 200, wrapped.text
        by_name = {v["name"]: v for v in wrapped.json()["variables"]}
        assert by_name["x2"]["max"] == 2**70
        assert len(wrapped.json()["constraints"]) == 1

    @pytest.mark.parametrize("bound", _BEYOND_NUMPY)
    @pytest.mark.parametrize("shape", ["bare", "dict"])
    def test_both_load_branches_accept_it_too(self, session_id, shape, bound):
        """The load branches did catch the TypeError, so they returned 400
        rather than 500 -- but a 400 for a legitimate bound is still wrong, and
        its text was a raw numpy ufunc string naming nothing."""
        var = {"name": "x2", "type": "integer", "min": -bound, "max": bound}
        payload = [var] if shape == "bare" else {
            "variables": [var], "constraints": [],
        }
        r = _upload_space(session_id, payload)
        assert r.status_code == 200, r.text
        export = client.get(f"/api/v1/sessions/{session_id}/variables/export")
        assert export.json()[0]["max"] == bound

    @pytest.mark.parametrize("shape", ["bare", "dict"])
    def test_no_response_carries_a_raw_numpy_ufunc_string(self, session_id, shape):
        var = {"name": "x2", "type": "integer", "min": 0, "max": 2**64}
        payload = [var] if shape == "bare" else {
            "variables": [var], "constraints": [],
        }
        r = _upload_space(session_id, payload)
        assert "isfinite" not in r.text
        assert "ufunc" not in r.text

    def test_a_huge_bound_round_trips_through_export_and_back(self, session_id):
        """``json.dumps`` emits a Python int at any width, so the export is
        readable and reloadable -- the property the guard defends."""
        client.post(f"/api/v1/sessions/{session_id}/variables", json={
            "name": "x2", "type": "integer", "min": 0, "max": 2**70,
        }).raise_for_status()
        exported = client.get(
            f"/api/v1/sessions/{session_id}/variables/export"
        ).json()
        second = client.post("/api/v1/sessions", json={"ttl_hours": 1}).json()
        try:
            r = _upload_space(second["session_id"], exported)
            assert r.status_code == 200, r.text
            again = client.get(
                f"/api/v1/sessions/{second['session_id']}/variables/export"
            )
            assert again.json() == exported
        finally:
            client.delete(f"/api/v1/sessions/{second['session_id']}")


class TestTheFinitenessRejectionStillNamesVariableAndKey:
    """Narrowing where the finiteness test runs must not narrow what it catches.

    The fix skips ``np.isfinite`` for the types that are finite by
    construction. If it skipped too much -- ``float``, say -- a NaN bound would
    load again and the round-1 defect would be back. Every numeric variable
    type is checked here, and the message is checked for *both* halves of its
    label, which the round-1 tests only did for the variable name.
    """

    @pytest.mark.parametrize("bad", _NON_FINITE)
    @pytest.mark.parametrize("var,key", [
        ({"name": "x1", "type": "real", "min": "BAD", "max": 10.0}, "min"),
        ({"name": "x1", "type": "real", "min": 0.0, "max": "BAD"}, "max"),
        ({"name": "x2", "type": "integer", "min": 0, "max": "BAD"}, "max"),
        (
            {"name": "x3", "type": "discrete",
             "allowed_values": [0.5, "BAD", 7.25]},
            "allowed_values[1]",
        ),
    ])
    @pytest.mark.parametrize("shape", ["bare", "dict"])
    def test_it_is_a_400_naming_both(self, session_id, shape, var, key, bad):
        entry = {
            k: (bad if v == "BAD" else v) for k, v in var.items()
        }
        if entry["type"] == "discrete":
            entry["allowed_values"] = [0.5, bad, 7.25]
        payload = [entry] if shape == "bare" else {
            "variables": [entry], "constraints": [],
        }
        r = _upload_space(session_id, payload)
        assert r.status_code == 400, r.text
        detail = r.json()["detail"]
        assert entry["name"] in detail, "the message must name the variable"
        assert key in detail, "the message must name the key"
        assert "must be finite" in detail

    def test_post_variables_still_rejects_a_non_finite_real_bound(self, session_id):
        """The endpoint that had no try/except: a genuine non-finite bound is
        still a 400 through the global ValueError handler, not a 500."""
        r = client.post(
            f"/api/v1/sessions/{session_id}/variables",
            content=json.dumps({
                "name": "x1", "type": "real", "min": 0.0, "max": float("inf"),
            }),
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code in (400, 422), r.text
        assert r.status_code != 500


class TestTheTypeErrorBackstopStillFires:
    """The branch ``_load_error_detail``'s docstring now names, exercised.

    Round 2 removed the guard as a source of TypeError, which left the
    ``(ValueError, KeyError, TypeError)`` catch on both load branches with no
    test that a real upload ever reaches its TypeError arm -- deleting
    ``TypeError`` from either tuple passed the whole suite. That is the same
    500-where-a-400-is-documented shape as the regression itself, so the
    backstop is pinned rather than left resting on a comment.

    ``"allowed_values": 5`` is the shape the docstring names: ``len()`` raises
    before any individual value is looked at, so the bounds guard never sees
    it. The message names the file and not the variable, which is exactly what
    the docstring says it can honestly claim.
    """

    @pytest.mark.parametrize("var", [
        {"name": "x3", "type": "discrete", "allowed_values": 5},
        {"name": "x4", "type": "categorical", "categories": 5},
    ])
    @pytest.mark.parametrize("shape", ["bare", "dict"])
    def test_a_non_list_collection_is_a_400_not_a_500(self, session_id, shape, var):
        payload = [var] if shape == "bare" else {
            "variables": [var], "constraints": [],
        }
        r = _upload_space(session_id, payload)
        assert r.status_code == 400, r.text
        assert r.json()["detail"].startswith(
            "Search space file could not be loaded:"
        )

    def test_the_session_is_untouched_by_it(self, session_id):
        _seed_constrained_space(session_id)
        before = client.get(f"/api/v1/sessions/{session_id}/variables").json()
        r = _upload_space(session_id, {
            "variables": [{"name": "x3", "type": "discrete", "allowed_values": 5}],
            "constraints": [],
        })
        assert r.status_code == 400, r.text
        assert client.get(f"/api/v1/sessions/{session_id}/variables").json() == before


# ============================================================
# Ruling 38 -- fix round 3
# ============================================================

# A raw Python int on a *real* bound through the load path: the one combination
# none of round 2's assertions exercised. ``POST /variables`` coerces through
# Pydantic's ``float`` first (an out-of-range int is a 422 there and never
# reaches the core), and every huge-int test written so far was integer-typed,
# so only this pairing could deliver an unconverted int to ``Real()``.
_REAL_BOUND_OUT_OF_RANGE = [2**1024, 2**2000]
_REAL_BOUND_IN_RANGE = [2**63, 2**64, 2**70, 2**200, 2**1023]


def _one_variable_payload(var, shape):
    return [var] if shape == "bare" else {"variables": [var], "constraints": []}


class TestARealBoundOutsideTheFloat64RangeIsNotAServerError:
    """``POST /variables/load`` returned 500 for ``real max=2**1024``.

    ``skopt.Real.__init__`` calls ``set_transformer("identity")``, which for
    the default uniform prior evaluates
    ``_uniform_inclusive(self.low, self.high - self.low)`` and so
    ``np.nextafter(scale, scale + 1.0)``. An int outside the float64 range
    raises ``OverflowError: int too large to convert to float`` from that
    ``scale + 1.0`` -- an ``ArithmeticError``, so outside the loaders'
    ``(ValueError, KeyError, TypeError)`` tuple and outside the global
    ``ValueError`` handler alike.

    This docstring said for two rounds that ``set_transformer`` builds
    ``Normalize(self.low, self.high)`` and converts *each bound*. Normalize is
    real but is built only under ``transform="normalize"``, which
    ``add_variable`` never passes, and the difference is not cosmetic: the
    quantity skopt converts is the span, so a per-bound reading produced a
    per-bound guard and the pair check that belonged beside it was not
    written. See ``TestARealVariablesSpanIsCheckedToo`` below.
    ``2**1023`` returned 200 and ``2**1024`` returned 500, on an endpoint that
    documents 400, and ``integer`` was unaffected at every magnitude because
    ``skopt.Integer`` converts nothing.

    That is the third exception type to leave this guard in three rounds --
    ZeroDivisionError, TypeError, OverflowError -- each revealed by removing
    the last. The fix asks whether the dimension about to be constructed can
    represent the bound, so there is no fourth exception to find.
    """

    @pytest.mark.parametrize("bound", _REAL_BOUND_OUT_OF_RANGE)
    @pytest.mark.parametrize("key", ["min", "max"])
    @pytest.mark.parametrize("shape", ["bare", "dict"])
    def test_it_is_a_400_naming_the_variable_and_the_key(
        self, session_id, shape, key, bound
    ):
        var = {"name": "x1", "type": "real", "min": 0.0, "max": 10.0}
        var[key] = -bound if key == "min" else bound
        r = _upload_space(session_id, _one_variable_payload(var, shape))
        assert r.status_code == 400, r.text
        detail = r.json()["detail"]
        assert "x1" in detail, "the message must name the variable"
        assert key in detail, "the message must name the key"
        assert "float64" in detail

    @pytest.mark.parametrize("bound", _REAL_BOUND_OUT_OF_RANGE)
    @pytest.mark.parametrize("key", ["min", "max"])
    @pytest.mark.parametrize("shape", ["bare", "dict"])
    def test_it_is_never_a_500(self, session_id, shape, key, bound):
        """Stated as the defect class, not as one exception name.

        Round 2's equivalent assertion pinned ``error_type != "TypeError"`` and
        passed the whole time ``OverflowError`` was being returned instead.
        """
        var = {"name": "x1", "type": "real", "min": 0.0, "max": 10.0}
        var[key] = -bound if key == "min" else bound
        r = _upload_space(session_id, _one_variable_payload(var, shape))
        assert r.status_code != 500, r.text
        assert "error_type" not in r.json(), r.text

    @pytest.mark.parametrize("bound", _REAL_BOUND_IN_RANGE)
    @pytest.mark.parametrize("key", ["min", "max"])
    @pytest.mark.parametrize("shape", ["bare", "dict"])
    def test_a_real_bound_inside_the_range_still_loads(
        self, session_id, shape, key, bound
    ):
        """The guard refuses unrepresentable, not large: ``2**1023`` was a 200
        before this round and stays one."""
        var = {"name": "x1", "type": "real", "min": -1.0, "max": 1.0}
        var[key] = -bound if key == "min" else bound
        r = _upload_space(session_id, _one_variable_payload(var, shape))
        assert r.status_code == 200, r.text
        exported = client.get(
            f"/api/v1/sessions/{session_id}/variables/export"
        ).json()
        assert exported[0][key] == var[key]

    @pytest.mark.parametrize(
        "bound", _REAL_BOUND_OUT_OF_RANGE + _REAL_BOUND_IN_RANGE
    )
    @pytest.mark.parametrize("key", ["min", "max"])
    @pytest.mark.parametrize("shape", ["bare", "dict"])
    def test_the_same_bound_on_an_integer_variable_is_still_a_200(
        self, session_id, shape, key, bound
    ):
        """The half of the rule a blanket magnitude limit would have broken."""
        var = {"name": "x2", "type": "integer", "min": -1, "max": 1}
        var[key] = -bound if key == "min" else bound
        r = _upload_space(session_id, _one_variable_payload(var, shape))
        assert r.status_code == 200, r.text
        exported = client.get(
            f"/api/v1/sessions/{session_id}/variables/export"
        ).json()
        assert exported[0][key] == var[key]

    @pytest.mark.parametrize("shape", ["bare", "dict"])
    def test_the_session_is_left_exactly_as_it_was(self, session_id, shape):
        """The bare-list branch appends with no dry run, so a file rejected
        partway must not leave a fragment of itself behind."""
        _seed_constrained_space(session_id)
        before = client.get(f"/api/v1/sessions/{session_id}/variables").json()
        payload = _one_variable_payload(
            {"name": "x9", "type": "real", "min": 0.0, "max": 2**1024}, shape
        )
        r = _upload_space(session_id, payload)
        assert r.status_code == 400, r.text
        assert client.get(f"/api/v1/sessions/{session_id}/variables").json() == before

    @pytest.mark.parametrize("shape", ["bare", "dict"])
    def test_a_discrete_value_outside_the_range_is_a_400_too(self, session_id, shape):
        """``allowed_values`` is coerced by ``float()`` into a Categorical, so
        it is float64-backed for the same reason and used to fail the same
        way -- one variable type over, which is where this defect class has
        hidden every time."""
        var = {"name": "x3", "type": "discrete", "allowed_values": [0.5, 2**1024]}
        r = _upload_space(session_id, _one_variable_payload(var, shape))
        assert r.status_code == 400, r.text
        detail = r.json()["detail"]
        assert "x3" in detail and "allowed_values[1]" in detail
        assert r.status_code != 500

    @pytest.mark.parametrize("shape", ["bare", "dict"])
    def test_a_discrete_value_inside_the_range_still_loads(self, session_id, shape):
        var = {"name": "x3", "type": "discrete", "allowed_values": [0.5, 2**70]}
        r = _upload_space(session_id, _one_variable_payload(var, shape))
        assert r.status_code == 200, r.text

    def test_a_constraint_value_of_any_magnitude_still_loads(self, session_id):
        """Constraints build no dimension, so the float64 rule does not reach
        them and must not be allowed to. ``rhs`` and coefficients were fine at
        every magnitude before this round and are unchanged by it."""
        payload = {
            "variables": [{"name": "x1", "type": "real", "min": 0.0, "max": 10.0}],
            "constraints": [{
                "type": "inequality",
                "coefficients": {"x1": 2**2000},
                "rhs": -(2**1024),
                "name": "c_a",
            }],
        }
        r = _upload_space(session_id, payload)
        assert r.status_code == 200, r.text
        exported = client.get(
            f"/api/v1/sessions/{session_id}/variables/export",
            params={"include_constraints": "true"},
        ).json()
        assert exported["constraints"][0]["rhs"] == -(2**1024)
        assert exported["constraints"][0]["coefficients"]["x1"] == 2**2000

    @pytest.mark.parametrize("shape", ["bare", "dict"])
    def test_no_response_carries_a_raw_conversion_message(self, session_id, shape):
        """"int too large to convert to float" is what escaped; it names
        neither the variable nor the key nor the file."""
        var = {"name": "x1", "type": "real", "min": 0.0, "max": 2**1024}
        r = _upload_space(session_id, _one_variable_payload(var, shape))
        assert "too large to convert" not in r.text
        assert "OverflowError" not in r.text


# ============================================================
# Ruling 38 -- fix round 4
# ============================================================

# Bound *pairs*. Every sweep above varies one bound and leaves the other a
# small literal, which is what hid this: ``test_a_real_bound_inside_the_range_
# still_loads[min-...2**1023]`` uploads ``min=-(2**1023), max=1.0`` and gets a
# 200. Move that ``1.0`` to ``2**1023`` -- one literal -- and it is a 500.
#
# Int, float and mixed spellings, because the load path carries all three and
# ``high - low`` is exact arithmetic in one, saturating arithmetic in another.
_SPAN_TOO_WIDE = [
    (-(2**1023), 2**1023),
    (-(10**308), 10**308),
    (-(2**1022), 2**1023 + 2**1022),
    (-1.7e308, 1.7e308),
    (-1.5e308, 0.9e308),
    (-(2**1023), 1.0e308),
    (-1.0e308, 2**1023),
]

_SPAN_FITS = [
    (-(10**307), 10**307),
    (-(2**1022), 2**1022),
    (-(10**308), 1),
    (-8.9e307, 8.9e307),
    (-1.7e308, 1.0),
    (-(2**1022), 8.0e307),
]


class TestARealVariablesSpanIsCheckedToo:
    """Two bounds each inside the float64 range, a span that is not.

    Round 3 refused a bound ``skopt.Real`` could not hold and stopped there.
    ``Real`` does not ask about a bound; ``set_transformer`` asks about
    ``self.high - self.low``. So the guard asserted a totality ``add_variable``
    did not have, and the pairs below walked past it:

    * ``min=-(2**1023) max=2**1023`` -> 500, ``OverflowError`` from
      ``scale + 1.0``, on the endpoint round 3 was making return 400.
    * ``min=-1.7e308 max=1.7e308`` -> 200, and a dimension with ``scale=inf``
      whose ``rvs`` returns the upper bound every time. A degenerate design
      reported as success, which is the worse of the two.

    The second face is pinned behaviourally in
    ``tests/unit/core/data/test_constraints.py``; here it is a status code and
    a message that names the variable, which is what the endpoint owes.
    """

    @pytest.mark.parametrize("low,high", _SPAN_TOO_WIDE)
    @pytest.mark.parametrize("shape", ["bare", "dict"])
    def test_it_is_a_400_naming_the_variable(self, session_id, shape, low, high):
        var = {"name": "x1", "type": "real", "min": low, "max": high}
        r = _upload_space(session_id, _one_variable_payload(var, shape))
        assert r.status_code == 400, r.text
        detail = r.json()["detail"]
        assert "x1" in detail, "the message must name the variable"
        assert "float64" in detail

    @pytest.mark.parametrize("low,high", _SPAN_TOO_WIDE)
    @pytest.mark.parametrize("shape", ["bare", "dict"])
    def test_it_is_never_a_500(self, session_id, shape, low, high):
        var = {"name": "x1", "type": "real", "min": low, "max": high}
        r = _upload_space(session_id, _one_variable_payload(var, shape))
        assert r.status_code != 500, r.text
        assert "error_type" not in r.json(), r.text

    @pytest.mark.parametrize("low,high", _SPAN_TOO_WIDE)
    @pytest.mark.parametrize("shape", ["bare", "dict"])
    def test_each_bound_alone_still_loads(self, session_id, shape, low, high):
        """What makes this a span rule rather than round 3's rule again: split
        the pair and both halves are ordinary 200s."""
        # Distinct names: the bare-list branch appends into the live session,
        # so uploading the second half under the first half's name would be
        # refused as a duplicate and prove nothing about the bound.
        for name, key, value in (("x1", "min", low), ("x2", "max", high)):
            var = {"name": name, "type": "real", "min": -1.0, "max": 1.0}
            var[key] = value
            r = _upload_space(session_id, _one_variable_payload(var, shape))
            assert r.status_code == 200, r.text
            exported = client.get(
                f"/api/v1/sessions/{session_id}/variables/export"
            ).json()
            loaded = [v for v in exported if v["name"] == name]
            assert loaded and loaded[0][key] == value, exported

    @pytest.mark.parametrize("low,high", _SPAN_FITS)
    @pytest.mark.parametrize("shape", ["bare", "dict"])
    def test_a_pair_whose_span_fits_still_loads(self, session_id, shape, low, high):
        var = {"name": "x1", "type": "real", "min": low, "max": high}
        r = _upload_space(session_id, _one_variable_payload(var, shape))
        assert r.status_code == 200, r.text
        exported = client.get(
            f"/api/v1/sessions/{session_id}/variables/export"
        ).json()
        assert exported[0]["min"] == low and exported[0]["max"] == high

    @pytest.mark.parametrize("low,high", _SPAN_TOO_WIDE)
    @pytest.mark.parametrize("shape", ["bare", "dict"])
    def test_the_same_pair_on_an_integer_variable_is_still_a_200(
        self, session_id, shape, low, high
    ):
        """``skopt.Integer`` computes no span, so the rule must not leak
        across -- the half a blanket limit would have broken."""
        var = {"name": "x2", "type": "integer", "min": int(low), "max": int(high)}
        r = _upload_space(session_id, _one_variable_payload(var, shape))
        assert r.status_code == 200, r.text
        exported = client.get(
            f"/api/v1/sessions/{session_id}/variables/export"
        ).json()
        assert exported[0]["min"] == int(low) and exported[0]["max"] == int(high)

    @pytest.mark.parametrize("low,high", _SPAN_TOO_WIDE)
    @pytest.mark.parametrize("shape", ["bare", "dict"])
    def test_a_discrete_variable_of_the_same_two_values_is_still_a_200(
        self, session_id, shape, low, high
    ):
        """``allowed_values`` becomes a ``Categorical``, which subtracts
        nothing. A third variable type, deliberately, because uniformity of
        variable type is what hid this defect on each of its four rounds."""
        var = {"name": "x3", "type": "discrete", "allowed_values": [low, high]}
        r = _upload_space(session_id, _one_variable_payload(var, shape))
        assert r.status_code == 200, r.text

    def test_a_constraint_spanning_the_same_range_is_still_a_200(self, session_id):
        """Constraints build no dimension, so no span rule reaches them."""
        payload = {
            "variables": [{"name": "x1", "type": "real", "min": 0.0, "max": 10.0}],
            "constraints": [{
                "type": "inequality",
                "coefficients": {"x1": -(10**308)},
                "rhs": 10**308,
                "name": "c_a",
            }],
        }
        r = _upload_space(session_id, payload)
        assert r.status_code == 200, r.text

    @pytest.mark.parametrize("shape", ["bare", "dict"])
    def test_the_session_is_left_exactly_as_it_was(self, session_id, shape):
        """The bare-list branch appends with no dry run."""
        _seed_constrained_space(session_id)
        before = client.get(f"/api/v1/sessions/{session_id}/variables").json()
        var = {"name": "x9", "type": "real", "min": -1.7e308, "max": 1.7e308}
        r = _upload_space(session_id, _one_variable_payload(var, shape))
        assert r.status_code == 400, r.text
        assert client.get(f"/api/v1/sessions/{session_id}/variables").json() == before

    @pytest.mark.parametrize("low,high", _SPAN_TOO_WIDE)
    @pytest.mark.parametrize("shape", ["bare", "dict"])
    def test_no_response_carries_a_raw_conversion_message(
        self, session_id, shape, low, high
    ):
        var = {"name": "x1", "type": "real", "min": low, "max": high}
        r = _upload_space(session_id, _one_variable_payload(var, shape))
        assert "too large to convert" not in r.text
        assert "OverflowError" not in r.text

    @pytest.mark.parametrize("low,high", _SPAN_TOO_WIDE)
    def test_the_direct_endpoint_refuses_the_same_pair(self, session_id, low, high):
        """``POST /variables`` coerces through Pydantic's ``float`` first, so
        the int spellings arrive as floats -- and the span is over the range
        either way. A 4xx, not a 500, whichever guard answers."""
        r = client.post(
            f"/api/v1/sessions/{session_id}/variables",
            json={"name": "x1", "type": "real", "min": low, "max": high},
        )
        assert r.status_code in (400, 422), r.text
        # A ValueError at a 400 is this endpoint's documented shape (the global
        # handler labels it). The assertion is that nothing else gets out --
        # OverflowError here was a 500, which is the whole defect.
        assert r.json().get("error_type") in (None, "ValueError"), r.text
