"""``add_experiment`` checks its input keys and bounds (A4).

A key that is not a search-space variable used to become a column in the
data with no error, and a value outside a variable's bounds was taken
silently. Unknown keys are now refused (the API maps the ValueError to 400).
An out-of-bounds value is still accepted, since observed variables and
hardware-adjusted actuals may legitimately sit outside, but it is logged as a
warning. The bookkeeping columns (Iteration, Reason, ProvenanceId, Noise and
the split-plot columns) are not unknown keys: the queue and CSV paths carry
them.
"""
import logging

import pytest

from alchemist_core import OptimizationSession
from alchemist_core.data.experiment_manager import PLOT_COLUMNS, PROVENANCE_COL


@pytest.fixture
def session():
    s = OptimizationSession()
    s.add_variable("x", "real", bounds=(0.0, 1.0))
    s.add_variable("n", "integer", bounds=(1, 5))
    s.add_variable("c", "categorical", categories=["A", "B"])
    return s


def test_unknown_key_is_refused(session):
    with pytest.raises(ValueError, match="operator"):
        session.add_experiment({"x": 0.5, "n": 2, "c": "A", "operator": "cc"}, 1.0)
    assert len(session.experiment_manager.df) == 0


def test_bookkeeping_keys_are_not_unknown(session):
    inputs = {"x": 0.5, "n": 2, "c": "A", "Iteration": 3, "Reason": "LHS",
              PROVENANCE_COL: "abc", "Noise": 0.1, **{k: 1 for k in PLOT_COLUMNS}}
    session.add_experiment(inputs, 1.0)
    row = session.experiment_manager.df.iloc[-1]
    assert row["x"] == 0.5
    assert "operator" not in session.experiment_manager.df.columns


def test_out_of_bounds_value_is_kept_with_a_warning(session, caplog):
    with caplog.at_level(logging.WARNING):
        session.add_experiment({"x": 1.7, "n": 2, "c": "A"}, 1.0)
    assert session.experiment_manager.df.iloc[-1]["x"] == 1.7
    assert "x" in caplog.text and "1.7" in caplog.text and "bounds" in caplog.text.lower()


def test_integer_out_of_bounds_warns_too(session, caplog):
    with caplog.at_level(logging.WARNING):
        session.add_experiment({"x": 0.5, "n": 9, "c": "A"}, 1.0)
    assert "n" in caplog.text and "9" in caplog.text


def test_in_bounds_value_does_not_warn(session, caplog):
    with caplog.at_level(logging.WARNING):
        session.add_experiment({"x": 0.5, "n": 2, "c": "A"}, 1.0)
    assert "bounds" not in caplog.text.lower()


def test_queue_completion_still_records(session):
    session.add_staged_experiment({"x": 0.5, "n": 2, "c": "B", "WholePlot": 1,
                                   "Subplot": 1, "RunOrder": 1, "_reason": "LHS"})
    item = session.queue.pending_items()[0]
    session.complete_experiment(item.id, {"x": 0.5, "n": 2, "c": "B"}, output=2.0)
    row = session.experiment_manager.df.iloc[-1]
    assert row[PROVENANCE_COL] == item.id and row["WholePlot"] == 1


def test_csv_load_with_extra_columns_is_untouched(session, tmp_path):
    import pandas as pd
    csv = tmp_path / "r.csv"
    pd.DataFrame({"x": [0.1, 0.9], "n": [1, 2], "c": ["A", "B"],
                  "operator": ["cc", "cc"], "Output": [1.0, 2.0]}).to_csv(csv, index=False)
    session.load_data(str(csv))
    assert "operator" in session.experiment_manager.df.columns
