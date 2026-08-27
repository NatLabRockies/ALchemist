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
