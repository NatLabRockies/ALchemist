import json
import os
import tempfile

import pytest
from alchemist_core.session import OptimizationSession


def _session():
    s = OptimizationSession()
    s.add_variable("x", "real", bounds=(0.0, 10.0))
    return s


def test_default_control_record_is_run_and_idle():
    c = _session().get_control()
    assert c["requested"] == "run"
    assert c["reported"] == "idle"
    assert c["requested_at"] is None
    assert c["reported_at"] is None


def test_get_control_returns_a_copy():
    """Callers must not be able to mutate session state through the getter."""
    s = _session()
    s.get_control()["reported"] = "running"
    assert s.get_control()["reported"] == "idle"


def test_set_control_request_does_not_touch_reported_fields():
    """The disjointness property, at the core layer.

    This is what makes 'requested but not yet honored' representable. If a
    request write could set `reported`, the UI could claim the reactor is
    paused because someone clicked a button.
    """
    s = _session()
    s.set_control_report("running", reported_by="ctl")
    before = s.get_control()

    s.set_control_request("pause", requested_by="caleb")
    after = s.get_control()

    assert after["requested"] == "pause"
    assert after["reported"] == before["reported"] == "running"
    assert after["reported_at"] == before["reported_at"]
    assert after["reported_by"] == before["reported_by"] == "ctl"


def test_set_control_report_does_not_touch_requested_fields():
    s = _session()
    s.set_control_request("pause", requested_by="caleb")
    before = s.get_control()

    s.set_control_report("paused", reported_by="ctl", detail="held after q3")
    after = s.get_control()

    assert after["reported"] == "paused"
    assert after["detail"] == "held after q3"
    assert after["requested"] == before["requested"] == "pause"
    assert after["requested_at"] == before["requested_at"]
    assert after["requested_by"] == before["requested_by"] == "caleb"


@pytest.mark.parametrize("bad", ["stop", "PAUSE", "", None, "running"])
def test_invalid_control_request_is_rejected(bad):
    with pytest.raises(ValueError):
        _session().set_control_request(bad)


@pytest.mark.parametrize("bad", ["run", "pause", "PAUSED", "", None])
def test_invalid_control_report_is_rejected(bad):
    with pytest.raises(ValueError):
        _session().set_control_report(bad)


def test_a_changed_report_is_audited():
    s = _session()
    before = len(s.audit_log.get_entries("control_reported"))
    s.set_control_report("running", reported_by="ctl")
    assert len(s.audit_log.get_entries("control_reported")) == before + 1


def test_a_heartbeat_that_changes_nothing_is_not_audited():
    """The audit/broadcast asymmetry (spec section 4.3).

    At a 10 s heartbeat an 8-hour campaign produces ~2900 reports. Auditing
    each would bury the entries that matter under three thousand identical
    ones -- in the log that audit/export renders for a methods section.
    """
    s = _session()
    s.set_control_report("running", reported_by="ctl")
    after_first = len(s.audit_log.get_entries("control_reported"))

    s.set_control_report("running", reported_by="ctl")
    s.set_control_report("running", reported_by="ctl")

    assert len(s.audit_log.get_entries("control_reported")) == after_first


def test_a_heartbeat_still_refreshes_reported_at():
    """Not audited, but the timestamp MUST move -- the staleness display
    depends on it, and a frozen reported_at would read as a dead controller.
    """
    s = _session()
    s.set_control_report("running", reported_by="ctl")
    first = s.get_control()["reported_at"]
    s.control["reported_at"] = "2000-01-01T00:00:00"   # force a distinguishable value
    s.set_control_report("running", reported_by="ctl")
    assert s.get_control()["reported_at"] != "2000-01-01T00:00:00"
    assert first is not None


def test_a_changed_request_is_audited():
    s = _session()
    before = len(s.audit_log.get_entries("control_requested"))
    s.set_control_request("pause", requested_by="caleb")
    assert len(s.audit_log.get_entries("control_requested")) == before + 1


def test_repeating_the_same_request_is_not_audited():
    s = _session()
    s.set_control_request("pause", requested_by="caleb")
    after_first = len(s.audit_log.get_entries("control_requested"))
    s.set_control_request("pause", requested_by="caleb")
    assert len(s.audit_log.get_entries("control_requested")) == after_first


def test_control_survives_a_save_load_round_trip():
    """The plan named these to_dict/from_dict; the real persistence surface is
    save_session/load_session, so the round trip is exercised through that.
    """
    s = _session()
    s.set_control_request("pause", requested_by="caleb")
    s.set_control_report("paused", reported_by="ctl", detail="held after q3")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        s.save_session(path)
        assert json.load(open(path))["control"]["requested"] == "pause"

        restored = OptimizationSession.load_session(path, retrain_on_load=False)
        assert restored.get_control()["requested"] == "pause"
        assert restored.get_control()["reported"] == "paused"
        assert restored.get_control()["detail"] == "held after q3"
    finally:
        os.unlink(path)


def test_a_session_file_without_control_loads_with_defaults():
    """Historical session files predate this field."""
    s = _session()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        s.save_session(path)
        data = json.load(open(path))
        data.pop("control", None)
        json.dump(data, open(path, "w"))

        restored = OptimizationSession.load_session(path, retrain_on_load=False)
        assert restored.get_control()["requested"] == "run"
        assert restored.get_control()["reported"] == "idle"
    finally:
        os.unlink(path)


def test_load_into_an_existing_instance_restores_control():
    """load_session's instance form copies attributes one by one -- a site the
    static loader does not cover, and easy to miss.
    """
    s = _session()
    s.set_control_request("pause", requested_by="caleb")
    s.set_control_report("paused", reported_by="ctl")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        s.save_session(path)
        target = OptimizationSession()
        target.load_session(path, retrain_on_load=False)
        assert target.get_control()["requested"] == "pause"
        assert target.get_control()["reported"] == "paused"
    finally:
        os.unlink(path)
