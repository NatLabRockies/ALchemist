"""Outcome constraints survive save_session -> load_session (A29).

``save_session`` wrote the linear input constraints but never
``_outcome_constraints``, and the loader had no reader for them, so a reloaded
session suggested unconstrained with no warning.
"""
import json

import pytest

from alchemist_core import OptimizationSession


def _session():
    s = OptimizationSession()
    s.add_variable("x", "real", bounds=(0.0, 1.0))
    s.add_experiment({"x": 0.2}, 0.5)
    s.add_experiment({"x": 0.8}, 0.9)
    s.add_outcome_constraint("Output", "upper", 0.7)
    s.add_outcome_constraint("Output", "lower", 0.1)
    return s


def test_save_writes_outcome_constraints(tmp_path):
    path = tmp_path / "s.json"
    _session().save_session(str(path))
    data = json.loads(path.read_text())
    assert data["outcome_constraints"] == [
        {"objective_name": "Output", "bound_type": "upper", "value": 0.7},
        {"objective_name": "Output", "bound_type": "lower", "value": 0.1},
    ]


def test_save_writes_an_empty_list_with_no_outcome_constraints(tmp_path):
    s = OptimizationSession()
    s.add_variable("x", "real", bounds=(0.0, 1.0))
    path = tmp_path / "s.json"
    s.save_session(str(path))
    assert json.loads(path.read_text())["outcome_constraints"] == []


@pytest.mark.parametrize("form", ["class", "instance"])
def test_load_restores_outcome_constraints(tmp_path, form):
    path = tmp_path / "s.json"
    original = _session()
    original.save_session(str(path))
    if form == "class":
        loaded = OptimizationSession.load_session(str(path), retrain_on_load=False)
    else:
        loaded = OptimizationSession()
        loaded.load_session(str(path), retrain_on_load=False)
    assert loaded.get_outcome_constraints() == original.get_outcome_constraints()
    # Live, not a display copy: the duplicate check sees them.
    with pytest.raises(ValueError, match="Duplicate"):
        loaded.add_outcome_constraint("Output", "upper", 0.5)


def test_file_without_the_key_loads_with_none(tmp_path):
    path = tmp_path / "s.json"
    _session().save_session(str(path))
    data = json.loads(path.read_text())
    del data["outcome_constraints"]
    path.write_text(json.dumps(data))
    loaded = OptimizationSession.load_session(str(path), retrain_on_load=False)
    assert loaded.get_outcome_constraints() == []
