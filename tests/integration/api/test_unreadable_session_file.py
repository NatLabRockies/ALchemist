"""A session file that exists but cannot be read answers 503, not 404.

The C1 controller reads an item-scoped 404 as "the queue item vanished" and
skips it. A session whose file is corrupt or locked is not gone; the server
says it cannot serve it right now, on every route that needs the session.
Deleting it still works: delete removes the file without loading it.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.services.session_store import session_store

SID = "corrupt-session"


@pytest.fixture
def corrupt():
    path = session_store.persist_dir / f"{SID}.json"
    path.write_text("{not json")
    session_store._sessions.pop(SID, None)
    yield SID
    path.unlink(missing_ok=True)


@pytest.fixture
def api():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.mark.parametrize("method, path", [
    ("get", "/api/v1/sessions/{sid}"),
    ("get", "/api/v1/sessions/{sid}/export"),
    ("post", "/api/v1/sessions/{sid}/save"),
    ("post", "/api/v1/sessions/{sid}/backup"),
    ("get", "/api/v1/sessions/{sid}/lock"),
    ("post", "/api/v1/sessions/{sid}/lock"),
    ("delete", "/api/v1/sessions/{sid}/lock"),
    ("get", "/api/v1/sessions/{sid}/variables"),
    ("get", "/api/v1/sessions/{sid}/experiments"),
])
def test_an_unreadable_file_is_a_503(api, corrupt, method, path):
    kwargs = {"json": {"locked_by": "test"}} if method == "post" and path.endswith("/lock") else {}
    r = getattr(api, method)(path.format(sid=corrupt), **kwargs)
    assert r.status_code == 503, (r.status_code, r.text)
    body = r.json()
    assert corrupt in body["detail"]
    assert "could not be read" in body["detail"]
    assert body["error_type"] == "SessionFileUnreadableError"


def test_an_unknown_session_is_still_a_404(api):
    assert api.get("/api/v1/sessions/no-such-session").status_code == 404
    assert api.get("/api/v1/sessions/no-such-session/variables").status_code == 404


def test_an_unreadable_session_can_still_be_deleted(api, corrupt):
    assert api.delete(f"/api/v1/sessions/{corrupt}").status_code == 204
    assert not (session_store.persist_dir / f"{corrupt}.json").exists()
    assert api.get(f"/api/v1/sessions/{corrupt}").status_code == 404


def test_an_in_memory_delete_that_cannot_unlink_is_a_500(api, monkeypatch):
    from pathlib import Path
    sid = api.post("/api/v1/sessions", json={}).json()["session_id"]
    real = Path.unlink

    def unlink(self, *args, **kwargs):
        if self.name == f"{sid}.json":
            raise PermissionError("held open")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda _s: None)
    assert api.delete(f"/api/v1/sessions/{sid}").status_code == 500
    monkeypatch.setattr(Path, "unlink", real)
    assert api.delete(f"/api/v1/sessions/{sid}").status_code == 204
