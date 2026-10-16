"""The recovery routes glob with the session ID; the ID must be a plain ID (NEW-2).

``DELETE /sessions/*/backup`` used to delete every user's crash backups and
``POST /recovery/*/restore`` moved and mangled them all: the routes built
``f"{session_id}_recovery_*.json"`` and globbed it with whatever the URL
carried. A ``*`` (or ``?``, ``[...]``) ID now touches nothing, through the
store and over HTTP.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.services.session_store import SessionStore, session_store


@pytest.fixture
def store_with_backups(tmp_path):
    store = SessionStore(persist_dir=str(tmp_path / "sessions"))
    store.recovery_dir = tmp_path / "recovery"
    store.recovery_dir.mkdir()
    ids = []
    for name in ("A", "B"):
        sid = store.create(name=name)
        store.get(sid).add_variable("x", "real", min=0.0, max=1.0)
        assert store.save_recovery_backup(sid)
        ids.append(sid)
    assert len(list(store.recovery_dir.glob("*_recovery_*.json"))) == 2
    return store, ids


def _backups(store):
    return sorted(p.name for p in store.recovery_dir.glob("*.json"))


@pytest.mark.parametrize("bad_id", ["*", "?", "[a-z]*", "*_recovery_", "", "../x"])
def test_clear_with_a_wildcard_id_deletes_nothing(store_with_backups, bad_id):
    store, _ = store_with_backups
    before = _backups(store)
    assert store.clear_recovery_backup(bad_id) is False
    assert _backups(store) == before


@pytest.mark.parametrize("bad_id", ["*", "?", "[a-z]*", "", "../x"])
def test_restore_with_a_wildcard_id_restores_and_moves_nothing(store_with_backups, bad_id):
    store, _ = store_with_backups
    before = _backups(store)
    assert store.restore_from_recovery(bad_id) is None
    assert _backups(store) == before
    assert store.count() == 2


def test_cleanup_with_a_wildcard_id_deletes_nothing(store_with_backups):
    store, _ = store_with_backups
    before = _backups(store)
    store._cleanup_old_recovery_files("*", keep_newest=0)
    assert _backups(store) == before


def test_move_with_a_wildcard_id_moves_nothing(store_with_backups):
    store, _ = store_with_backups
    before = _backups(store)
    store._move_recovery_files("*", "new-id")
    assert _backups(store) == before


def test_a_real_id_still_clears_only_its_own_backups(store_with_backups):
    store, (a, b) = store_with_backups
    assert store.clear_recovery_backup(a) is True
    names = _backups(store)
    assert len(names) == 1 and names[0].startswith(f"{b}_recovery_")


def test_a_real_id_still_restores_and_takes_its_backups(store_with_backups):
    store, (a, b) = store_with_backups
    new = store.restore_from_recovery(a)
    assert new is not None and new != a
    names = _backups(store)
    assert len(names) == 2
    assert any(n.startswith(f"{new}_recovery_") for n in names)
    assert any(n.startswith(f"{b}_recovery_") for n in names)
    assert not any(n.startswith(f"{a}_recovery_") for n in names)


def test_wildcard_ids_over_http_touch_nothing(tmp_path):
    client = TestClient(app)
    sid = client.post("/api/v1/sessions", json={}).json()["session_id"]
    assert client.post(f"/api/v1/sessions/{sid}/variables",
                       json={"name": "x1", "type": "real", "min": 0, "max": 1}).status_code == 200
    assert client.post(f"/api/v1/sessions/{sid}/backup").status_code == 200
    try:
        before = sorted(p.name for p in session_store.recovery_dir.glob(f"{sid}_recovery_*.json"))
        assert before
        r = client.delete("/api/v1/sessions/*/backup")
        assert r.status_code in (200, 404), r.text
        assert sorted(p.name for p in session_store.recovery_dir.glob(f"{sid}_recovery_*.json")) == before
        r = client.post("/api/v1/recovery/*/restore")
        assert r.status_code == 404, r.text
        assert sorted(p.name for p in session_store.recovery_dir.glob(f"{sid}_recovery_*.json")) == before
    finally:
        session_store.clear_recovery_backup(sid)
        client.delete(f"/api/v1/sessions/{sid}")
