"""How the session store reads and writes session files.

Two changes for 0.4.0:

* **Lazy loading.** The store used to read every ``*.json`` in its directory
  at startup, so a server that had ever run the test suite in its working
  directory loaded hundreds of sessions before answering a request. It now
  starts empty and reads a session's file the first time its ID is used, so a
  known ID still works after a restart and nothing else is loaded.
* **Honest, atomic saves.** ``_save_to_disk`` wrote the file in place and
  swallowed every error, so ``persist_session_to_disk`` (and ``POST /save``)
  reported success after a failed write, and a crash mid-write left a
  truncated file. It now writes beside the target and renames over it, and a
  failure reaches the caller.
"""

from __future__ import annotations

import json

import pytest

from api.services.session_store import SessionStore


def _store_with_one_session(tmp_path):
    store = SessionStore(persist_dir=str(tmp_path))
    sid = store.create(name="Campaign")
    store.get(sid).add_variable("temperature", "real", min=200.0, max=400.0)
    assert store.persist_session_to_disk(sid) is True
    return store, sid


class TestLazyLoading:
    def test_startup_loads_nothing(self, tmp_path):
        _, sid = _store_with_one_session(tmp_path)
        restarted = SessionStore(persist_dir=str(tmp_path))
        assert restarted.count() == 0

    def test_a_known_id_still_works_after_a_restart(self, tmp_path):
        _, sid = _store_with_one_session(tmp_path)
        restarted = SessionStore(persist_dir=str(tmp_path))
        session = restarted.get(sid)
        assert session is not None
        assert session.metadata.name == "Campaign"
        assert [v["name"] for v in session.search_space.variables] == ["temperature"]
        assert restarted.count() == 1

    def test_it_is_read_once(self, tmp_path):
        _, sid = _store_with_one_session(tmp_path)
        restarted = SessionStore(persist_dir=str(tmp_path))
        assert restarted.get(sid) is restarted.get(sid)

    def test_store_metadata_survives_the_round_trip(self, tmp_path):
        store, sid = _store_with_one_session(tmp_path)
        created = store.get_info(sid)["created_at"]
        restarted = SessionStore(persist_dir=str(tmp_path))
        assert restarted.get_info(sid)["created_at"] == created

    @pytest.mark.parametrize("method, args", [
        ("get_info", ()),
        ("export_session", ()),
        ("get_lock_status", ()),
        ("lock_session", ("controller",)),
    ])
    def test_every_entry_point_finds_an_unloaded_session(self, tmp_path, method, args):
        _, sid = _store_with_one_session(tmp_path)
        restarted = SessionStore(persist_dir=str(tmp_path))
        assert getattr(restarted, method)(sid, *args) is not None

    def test_persist_finds_an_unloaded_session(self, tmp_path):
        _, sid = _store_with_one_session(tmp_path)
        restarted = SessionStore(persist_dir=str(tmp_path))
        assert restarted.persist_session_to_disk(sid) is True

    def test_delete_removes_an_unloaded_sessions_file(self, tmp_path):
        _, sid = _store_with_one_session(tmp_path)
        restarted = SessionStore(persist_dir=str(tmp_path))
        assert restarted.delete(sid) is True
        assert not (tmp_path / f"{sid}.json").exists()
        assert SessionStore(persist_dir=str(tmp_path)).get(sid) is None

    def test_an_unknown_id_is_none(self, tmp_path):
        store = SessionStore(persist_dir=str(tmp_path))
        assert store.get("no-such-session") is None
        assert store.get_info("no-such-session") is None
        assert store.delete("no-such-session") is False

    @pytest.mark.parametrize("bad_id", ["../outside", "a/b", "..", "", "x.json"])
    def test_an_id_cannot_name_a_file_outside_the_directory(self, tmp_path, bad_id):
        inner = tmp_path / "sessions"
        _store_with_one_session(inner)
        (tmp_path / "outside.json").write_text("{}")
        store = SessionStore(persist_dir=str(inner))
        assert store.get(bad_id) is None

    def test_an_unreadable_file_is_none_not_a_crash(self, tmp_path):
        (tmp_path / "broken.json").write_text("{not json")
        store = SessionStore(persist_dir=str(tmp_path))
        assert store.get("broken") is None


class TestSaving:
    def test_the_file_is_complete_json_with_store_metadata(self, tmp_path):
        _, sid = _store_with_one_session(tmp_path)
        data = json.loads((tmp_path / f"{sid}.json").read_text())
        assert "_session_store_metadata" in data

    def test_no_temporary_file_is_left_behind(self, tmp_path):
        _store_with_one_session(tmp_path)
        assert [p.name for p in tmp_path.iterdir() if not p.name.endswith(".json")] == []
        assert all(not p.name.startswith(".") for p in tmp_path.iterdir())

    def test_a_failed_write_is_reported(self, tmp_path, monkeypatch):
        store, sid = _store_with_one_session(tmp_path)

        def boom(path):
            raise OSError("disk full")

        monkeypatch.setattr(store.get(sid), "save_session", boom)
        assert store.persist_session_to_disk(sid) is False

    def test_a_failed_write_leaves_the_previous_file_intact(self, tmp_path, monkeypatch):
        store, sid = _store_with_one_session(tmp_path)
        before = (tmp_path / f"{sid}.json").read_text()

        def half_written(path):
            with open(path, "w") as f:
                f.write('{"truncated": ')
            raise OSError("disk full")

        monkeypatch.setattr(store.get(sid), "save_session", half_written)
        assert store.persist_session_to_disk(sid) is False
        assert (tmp_path / f"{sid}.json").read_text() == before
        assert sorted(p.name for p in tmp_path.iterdir()) == [f"{sid}.json"]

    def test_create_still_returns_an_id_when_the_first_write_fails(self, tmp_path, monkeypatch):
        """Unchanged behaviour: the session exists in memory either way; the
        next autosave or explicit save retries the write."""
        store = SessionStore(persist_dir=str(tmp_path))

        def boom(self_, session_id):
            raise OSError("disk full")

        monkeypatch.setattr(SessionStore, "_save_to_disk", boom)
        sid = store.create()
        assert store.get(sid) is not None
