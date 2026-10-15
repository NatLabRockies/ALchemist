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

import importlib
import json
import os
import stat
import sys
import threading

import pytest

from api.services.session_store import SessionStore
from alchemist_core.session import OptimizationSession

# ``api.services.session_store`` the attribute is the singleton instance, so
# the module itself comes from importlib.
store_module = importlib.import_module("api.services.session_store")


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


class TestPerSessionLoadLock:
    """First loads serialise per session ID, not store-wide: with a retrain on
    load, one slow session must not hold up the others."""

    @staticmethod
    def _two_saved_sessions(tmp_path):
        store = SessionStore(persist_dir=str(tmp_path))
        a, b = store.create(name="A"), store.create(name="B")
        return a, b

    @staticmethod
    def _slow_loader(monkeypatch, slow_name):
        """Make loading the session named ``slow_name`` block until released."""
        entered, release = threading.Event(), threading.Event()
        real = OptimizationSession.load_session

        def load(path, *args, **kwargs):
            session = real(path, *args, **kwargs)
            if session.metadata.name == slow_name:
                entered.set()
                assert release.wait(10), "test never released the slow load"
            return session

        monkeypatch.setattr(store_module.OptimizationSession, "load_session", staticmethod(load))
        return entered, release

    def test_a_slow_load_does_not_block_another_sessions_load(self, tmp_path, monkeypatch):
        a, b = self._two_saved_sessions(tmp_path)
        restarted = SessionStore(persist_dir=str(tmp_path))
        entered, release = self._slow_loader(monkeypatch, "A")
        slow = threading.Thread(target=restarted.get, args=(a,))
        slow.start()
        try:
            assert entered.wait(10)
            other = {}
            t = threading.Thread(target=lambda: other.update(s=restarted.get(b)))
            t.start()
            t.join(5)
            assert not t.is_alive(), "loading B waited for A's slow load"
            assert other["s"].metadata.name == "B"
        finally:
            release.set()
            slow.join(10)

    def test_two_requests_for_one_unloaded_session_share_one_object(self, tmp_path, monkeypatch):
        a, _ = self._two_saved_sessions(tmp_path)
        restarted = SessionStore(persist_dir=str(tmp_path))
        entered, release = self._slow_loader(monkeypatch, "A")
        got = []
        threads = [threading.Thread(target=lambda: got.append(restarted.get(a))) for _ in range(2)]
        threads[0].start()
        assert entered.wait(10)
        threads[1].start()
        release.set()
        for t in threads:
            t.join(10)
        assert len(got) == 2 and got[0] is got[1]

    def test_a_concurrent_delete_waits_and_the_session_stays_deleted(self, tmp_path, monkeypatch):
        a, _ = self._two_saved_sessions(tmp_path)
        restarted = SessionStore(persist_dir=str(tmp_path))
        entered, release = self._slow_loader(monkeypatch, "A")
        loader = threading.Thread(target=restarted.get, args=(a,))
        loader.start()
        assert entered.wait(10)
        result = []
        deleter = threading.Thread(target=lambda: result.append(restarted.delete(a)))
        deleter.start()
        deleter.join(0.3)
        assert deleter.is_alive(), "delete did not wait for the load in progress"
        release.set()
        loader.join(10)
        deleter.join(10)
        assert result == [True]
        assert restarted.get(a) is None
        assert not (tmp_path / f"{a}.json").exists()
        assert a not in restarted._sessions


class TestDeleteWithoutLoading:
    def test_deleting_an_unloaded_session_does_not_load_it(self, tmp_path, monkeypatch):
        _, sid = _store_with_one_session(tmp_path)
        restarted = SessionStore(persist_dir=str(tmp_path))

        def no_load(*args, **kwargs):
            raise AssertionError("delete loaded the session")

        monkeypatch.setattr(store_module.OptimizationSession, "load_session", staticmethod(no_load))
        assert restarted.delete(sid) is True
        assert not (tmp_path / f"{sid}.json").exists()
        assert restarted.count() == 0

    def test_an_unloaded_session_with_an_unreadable_file_can_still_be_deleted(self, tmp_path):
        (tmp_path / "broken.json").write_text("{not json")
        store = SessionStore(persist_dir=str(tmp_path))
        assert store.delete("broken") is True
        assert not (tmp_path / "broken.json").exists()

    @pytest.mark.parametrize("bad_id", ["../outside", "a/b", "..", "", "x.json"])
    def test_delete_cannot_name_a_file_outside_the_directory(self, tmp_path, bad_id):
        inner = tmp_path / "sessions"
        inner.mkdir()
        (tmp_path / "outside.json").write_text("{}")
        (inner / "x.json.json").write_text("{}")
        store = SessionStore(persist_dir=str(inner))
        assert store.delete(bad_id) is False
        assert (tmp_path / "outside.json").exists()


class TestDurableWrite:
    def test_the_temp_file_is_fsynced_before_the_rename(self, tmp_path, monkeypatch):
        store, sid = _store_with_one_session(tmp_path)
        events = []
        real_fsync, real_replace = os.fsync, os.replace
        monkeypatch.setattr(store_module.os, "fsync",
                            lambda fd: (events.append("fsync"), real_fsync(fd))[1])
        monkeypatch.setattr(store_module.os, "replace",
                            lambda src, dst: (events.append("replace"), real_replace(src, dst))[1])
        assert store.persist_session_to_disk(sid) is True
        assert events == ["fsync", "replace"]

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
    def test_the_file_is_world_readable_like_the_old_in_place_write(self, tmp_path):
        _, sid = _store_with_one_session(tmp_path)
        assert stat.S_IMODE((tmp_path / f"{sid}.json").stat().st_mode) == 0o644


class TestReplaceRetry:
    """Windows raises PermissionError from os.replace while another process
    (an AV scanner, a copy in progress) has the target open."""

    @staticmethod
    def _flaky_replace(monkeypatch, failures):
        real = os.replace
        calls = {"n": 0}

        def replace(src, dst):
            if str(src).endswith(".tmp"):
                calls["n"] += 1
                if failures is None or calls["n"] <= failures:
                    raise PermissionError("target is open")
            return real(src, dst)

        monkeypatch.setattr(store_module.os, "replace", replace)
        monkeypatch.setattr(store_module.time, "sleep", lambda _s: None)
        return calls

    def test_a_transient_permission_error_is_retried(self, tmp_path, monkeypatch):
        store, sid = _store_with_one_session(tmp_path)
        store.get(sid).metadata.name = "renamed"
        calls = self._flaky_replace(monkeypatch, failures=2)
        assert store.persist_session_to_disk(sid) is True
        assert calls["n"] == 3
        assert json.loads((tmp_path / f"{sid}.json").read_text())["metadata"]["name"] == "renamed"

    def test_a_permanent_permission_error_is_reported_and_cleaned_up(self, tmp_path, monkeypatch):
        store, sid = _store_with_one_session(tmp_path)
        before = (tmp_path / f"{sid}.json").read_text()
        calls = self._flaky_replace(monkeypatch, failures=None)
        assert store.persist_session_to_disk(sid) is False
        assert calls["n"] == 5
        assert (tmp_path / f"{sid}.json").read_text() == before
        assert sorted(p.name for p in tmp_path.iterdir()) == [f"{sid}.json"]
