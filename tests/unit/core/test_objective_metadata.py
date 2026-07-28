import pytest
from alchemist_core.session import OptimizationSession
from alchemist_core.audit_log import AuditLog


def test_audit_log_event_appends_entry():
    log = AuditLog()
    before = len(log.entries)
    entry = log.log_event("objective_label_changed", {"old": {}, "new": {"a": 1}}, notes="x")
    assert len(log.entries) == before + 1
    assert log.entries[-1].entry_type == "objective_label_changed"
    assert entry is log.entries[-1]


def _session():
    s = OptimizationSession()
    s.add_variable("x", "real", bounds=(0.0, 10.0))
    return s


def test_default_objective_metadata_empty():
    assert _session().objective_metadata == {}


def test_set_and_get_objective_metadata():
    s = _session()
    s.set_objective_metadata({"Output": {"label": "carbonyl_area", "unit": "a.u."}})
    assert s.objective_metadata["Output"]["label"] == "carbonyl_area"
    assert s.objective_metadata["Output"]["unit"] == "a.u."
    # get_objective_metadata returns a copy, not the internal dict
    got = s.get_objective_metadata()
    got["Output"]["label"] = "mutated"
    assert s.objective_metadata["Output"]["label"] == "carbonyl_area"


def test_set_objective_metadata_audits_change():
    s = _session()
    s.set_objective_metadata({"Output": {"label": "a"}})
    before = len(s.audit_log.entries)
    s.set_objective_metadata({"Output": {"label": "b"}})
    assert len(s.audit_log.entries) == before + 1
    assert s.audit_log.entries[-1].entry_type == "objective_label_changed"


def test_check_objective_label_match_ok():
    s = _session()
    s.set_objective_metadata({"Output": {"label": "a"}})
    s.check_objective_label({"Output": "a"})  # no raise


def test_check_objective_label_mismatch_raises():
    s = _session()
    s.set_objective_metadata({"Output": {"label": "a"}})
    with pytest.raises(ValueError):
        s.check_objective_label({"Output": "b"})


def test_check_objective_label_none_is_noop():
    s = _session()
    s.set_objective_metadata({"Output": {"label": "a"}})
    s.check_objective_label(None)   # no raise
    s.check_objective_label({})     # no raise


def test_check_objective_label_missing_objective_treated_as_none():
    # Guarding against a label the session has never set: current label is None,
    # so expecting any non-None label is a mismatch.
    s = _session()
    with pytest.raises(ValueError):
        s.check_objective_label({"Output": "a"})
