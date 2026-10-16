"""Load-time retrain events reach the instance session (A30); ``once`` does
not skip the next listener and ``off`` can remove a pending ``once`` (A32).

``_load_session_impl`` emitted ``model_retrained`` / ``model_retrain_failed``
on the session it builds internally, so a listener on the session that
called ``session.load_session(path)`` heard only ``session_loaded``. And
``once``'s wrapper called ``off`` while ``emit`` iterated the live listener
list, so the listener registered after it was skipped on that emit.
"""
import json

import pytest

from alchemist_core import OptimizationSession
from alchemist_core.events import EventEmitter


@pytest.fixture
def saved(tmp_path):
    s = OptimizationSession()
    s.add_variable("x", "real", bounds=(0.0, 1.0))
    for x in (0.1, 0.4, 0.7, 0.9, 0.3, 0.5):
        s.add_experiment({"x": x}, (x - 0.3) ** 2)
    s.train_model(backend="botorch")
    path = tmp_path / "s.json"
    s.save_session(str(path))
    return path


def test_instance_form_emits_model_retrained(saved):
    events = []
    s = OptimizationSession()
    s.on("model_retrained", lambda e: events.append(("retrained", e)))
    s.on("model_retrain_failed", lambda e: events.append(("failed", e)))
    s.on("session_loaded", lambda e: events.append(("loaded", e)))
    s.load_session(str(saved))
    assert [name for name, _ in events] == ["retrained", "loaded"]
    assert events[0][1]["backend"] == "botorch"


def test_instance_form_emits_model_retrain_failed(saved):
    data = json.loads(saved.read_text())
    data["model_config"]["training_config"] = {"backend": "botorch", "kernel": "NoSuchKernel"}
    saved.write_text(json.dumps(data))
    events = []
    s = OptimizationSession()
    s.on("model_retrain_failed", lambda e: events.append(e))
    s.load_session(str(saved))
    assert len(events) == 1 and "error" in events[0]
    assert s.model is None


def test_no_retrain_no_event(saved):
    events = []
    s = OptimizationSession()
    s.on("model_retrained", lambda e: events.append(e))
    s.load_session(str(saved), retrain_on_load=False)
    assert events == []


def test_once_does_not_skip_the_next_listener():
    em = EventEmitter()
    calls = []
    em.once("e", lambda d: calls.append("once"))
    em.on("e", lambda d: calls.append("second"))
    em.emit("e", {})
    assert calls == ["once", "second"]
    em.emit("e", {})
    assert calls == ["once", "second", "second"]


def test_off_removes_a_pending_once_listener():
    em = EventEmitter()
    calls = []
    cb = lambda d: calls.append("once")
    em.once("e", cb)
    assert em.listener_count("e") == 1
    em.off("e", cb)
    assert em.listener_count("e") == 0
    em.emit("e", {})
    assert calls == []


def test_listener_removed_during_emit_is_not_called_twice_or_skipped():
    em = EventEmitter()
    calls = []

    def first(d):
        calls.append("first")
        em.off("e", first)

    em.on("e", first)
    em.on("e", lambda d: calls.append("second"))
    em.emit("e", {})
    assert calls == ["first", "second"]
