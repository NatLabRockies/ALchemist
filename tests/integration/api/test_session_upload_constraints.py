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
