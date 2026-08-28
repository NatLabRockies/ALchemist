"""Constraints must survive the REST save/reload round trip too.

``GET /sessions/{id}/download`` calls ``save_session`` and
``POST /sessions/upload`` calls ``load_session``, so the writer's missing
``constraints`` key emptied the constraint set of every session a user
downloaded and re-uploaded. The same pair backs ``session_store``'s on-disk
persistence, so an API restart lost them as well.

Routers mount under /api/v1 (api/main.py:61-68). Setup mirrors
tests/integration/api/test_constraints_router.py.
"""

import io
import json

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


@pytest.fixture
def session_id():
    r = client.post("/api/v1/sessions", json={"ttl_hours": 1})
    r.raise_for_status()
    sid = r.json()["session_id"]
    yield sid
    client.delete(f"/api/v1/sessions/{sid}")


def _seed(sid):
    """A real and an integer variable, an inequality and an equality."""
    for payload in (
        {"name": "x1", "type": "real", "min": 0.0, "max": 10.0},
        {"name": "x2", "type": "integer", "min": 0, "max": 8},
    ):
        client.post(f"/api/v1/sessions/{sid}/variables", json=payload).raise_for_status()
    for payload in (
        {"constraint_type": "inequality", "coefficients": {"x1": 1.0, "x2": 2.0},
         "rhs": 12.0, "name": "budget"},
        {"constraint_type": "equality", "coefficients": {"x1": 1.0, "x2": -1.0},
         "rhs": 0.0, "name": "balance"},
    ):
        client.post(f"/api/v1/sessions/{sid}/constraints", json=payload).raise_for_status()


def _download_and_reupload(sid):
    r = client.get(f"/api/v1/sessions/{sid}/download")
    assert r.status_code == 200, r.text
    r2 = client.post(
        "/api/v1/sessions/upload",
        files={"file": ("session.json", io.BytesIO(r.content), "application/json")},
    )
    assert r2.status_code == 201, r2.text
    new_sid = r2.json()["session_id"]
    return new_sid


class TestConstraintsSurviveDownloadAndUpload:
    def test_both_constraints_come_back(self, session_id):
        _seed(session_id)
        new_sid = _download_and_reupload(session_id)
        try:
            r = client.get(f"/api/v1/sessions/{new_sid}/constraints")
            assert r.status_code == 200, r.text
            restored = r.json()["constraints"]
            assert [c["name"] for c in restored] == ["budget", "balance"]
            assert [c["type"] for c in restored] == ["inequality", "equality"]
            assert restored[1]["coefficients"] == {"x1": 1.0, "x2": -1.0}
            assert restored[1]["rhs"] == 0.0
        finally:
            client.delete(f"/api/v1/sessions/{new_sid}")

    def test_a_restored_name_still_deletes(self, session_id):
        """Names are the delete identity; a reloaded session must keep it."""
        _seed(session_id)
        new_sid = _download_and_reupload(session_id)
        try:
            r = client.delete(f"/api/v1/sessions/{new_sid}/constraints/budget")
            assert r.status_code == 200, r.text
            remaining = client.get(
                f"/api/v1/sessions/{new_sid}/constraints"
            ).json()["constraints"]
            assert [c["name"] for c in remaining] == ["balance"]
        finally:
            client.delete(f"/api/v1/sessions/{new_sid}")


def _upload(constraints, sid):
    """Download ``sid``, hand-edit its constraint list, and upload the result."""
    r = client.get(f"/api/v1/sessions/{sid}/download")
    r.raise_for_status()
    data = json.loads(r.content)
    data["search_space"]["constraints"] = constraints
    body = json.dumps(data).encode()
    return client.post(
        "/api/v1/sessions/upload",
        files={"file": ("session.json", io.BytesIO(body), "application/json")},
    )


class TestAHandEditedConstraintCannotBeUploaded:
    """``POST /sessions/upload`` was the only constraint-bearing path on the
    branch with no validation: ``/variables/load`` validates the identical
    structure through ``_apply_search_space``. An uploaded session returned
    201, ``GET /constraints`` echoed the entries verbatim, every design 500'd
    -- and one entry with no ``name`` key made ``DELETE /constraints/{name}``
    a 500 for *every* constraint in the session, because the route evaluates
    ``c["name"]`` over the whole list to build its match. The repair route died
    on the thing it existed to repair, so the session was a dead end.
    """

    @pytest.mark.parametrize("entry", [
        {"type": "equality", "coefficients": {"x2": -1.0}, "rhs": None,
         "name": "c_null"},
        {"type": "sum_to", "coefficients": {"x1": 2.0}, "rhs": 4.0,
         "name": "c_type"},
        {"type": "inequality", "coefficients": {"x1": None}, "rhs": 6.0,
         "name": "c_coeff"},
        {"type": "inequality", "coefficients": ["x1"], "rhs": 6.0,
         "name": "c_list"},
    ])
    def test_the_upload_is_refused(self, session_id, entry):
        _seed(session_id)
        assert _upload([entry], session_id).status_code == 400

    def test_a_nameless_entry_uploads_and_stays_deletable(self, session_id):
        """The escalation, from the other end: every installed constraint must
        answer to ``c["name"]``, which is what the delete route iterates.
        """
        _seed(session_id)
        r = _upload([
            {"type": "inequality", "coefficients": {"x1": 1.0, "x2": 2.0},
             "rhs": 11.0},
            {"type": "equality", "coefficients": {"x1": -1.0}, "rhs": -2.0,
             "name": "explicit"},
        ], session_id)
        assert r.status_code == 201, r.text
        new_sid = r.json()["session_id"]
        try:
            listed = client.get(f"/api/v1/sessions/{new_sid}/constraints").json()
            assert [c["name"] for c in listed["constraints"]] == [
                "constraint_0", "explicit"
            ]
            for name in ("constraint_0", "explicit"):
                assert client.delete(
                    f"/api/v1/sessions/{new_sid}/constraints/{name}"
                ).status_code == 200, name
        finally:
            client.delete(f"/api/v1/sessions/{new_sid}")

    def test_a_dangling_constraint_still_uploads_and_is_repairable(self, session_id):
        """The state the loader deliberately preserves. It must stay loadable,
        and -- unlike before -- the repair route must work on it.
        """
        _seed(session_id)
        r = _upload([
            {"type": "inequality", "coefficients": {"x1": 1.0, "ghost": 2.0},
             "rhs": 5.0, "name": "c_ghost"},
            {"type": "equality", "coefficients": {"x2": -1.0}, "rhs": -3.0,
             "name": "c_kept"},
        ], session_id)
        assert r.status_code == 201, r.text
        new_sid = r.json()["session_id"]
        try:
            assert client.delete(
                f"/api/v1/sessions/{new_sid}/constraints/c_ghost"
            ).status_code == 200
            listed = client.get(f"/api/v1/sessions/{new_sid}/constraints").json()
            assert [c["name"] for c in listed["constraints"]] == ["c_kept"]
        finally:
            client.delete(f"/api/v1/sessions/{new_sid}")

    def test_a_well_formed_upload_still_designs(self, session_id):
        """The installed constraints must be live, not merely well-shaped."""
        _seed(session_id)
        r = _upload([
            {"type": "inequality", "coefficients": {"x1": 1.0, "x2": 1.0},
             "rhs": 6.0, "name": "budget"},
        ], session_id)
        assert r.status_code == 201, r.text
        new_sid = r.json()["session_id"]
        try:
            d = client.post(f"/api/v1/sessions/{new_sid}/initial-design",
                            json={"method": "lhs", "n_points": 6,
                                  "random_seed": 5})
            assert d.status_code == 200, d.text
            for point in d.json()["points"]:
                assert point["x1"] + point["x2"] <= 6.0 + 1e-9, point
            assert d.json()["feasibility"]["constraints_applied"] == ["budget"]
        finally:
            client.delete(f"/api/v1/sessions/{new_sid}")
