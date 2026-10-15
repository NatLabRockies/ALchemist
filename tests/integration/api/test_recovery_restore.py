"""The recovery banner's round trip over HTTP.

Restore X's backup (a new session Y), save Y the way the web app does
(``DELETE /sessions/{Y}/backup`` after writing the file), and nothing is left
to offer. Before the fix X's backup survived and came back on every reload.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app
from api.services.session_store import session_store

client = TestClient(app)


def _recoveries():
    r = client.get("/api/v1/recovery/list")
    assert r.status_code == 200
    return [entry["session_id"] for entry in r.json()["recoveries"]]


def test_restore_then_save_leaves_nothing_to_recover():
    r = client.post("/api/v1/sessions", json={})
    old = r.json()["session_id"]
    r = client.post(f"/api/v1/sessions/{old}/variables",
                    json={"name": "x1", "type": "real", "min": 0, "max": 1})
    assert r.status_code == 200, r.text
    assert client.post(f"/api/v1/sessions/{old}/backup").status_code == 200
    try:
        assert old in _recoveries()

        r = client.post(f"/api/v1/recovery/{old}/restore")
        assert r.status_code == 201, r.text
        new = r.json()["session_id"]
        assert old not in _recoveries()
        assert new in _recoveries()

        assert client.delete(f"/api/v1/sessions/{new}/backup").status_code == 200
        assert old not in _recoveries() and new not in _recoveries()
    finally:
        for sid in (old, locals().get("new")):
            if sid:
                session_store.clear_recovery_backup(sid)
                client.delete(f"/api/v1/sessions/{sid}")
