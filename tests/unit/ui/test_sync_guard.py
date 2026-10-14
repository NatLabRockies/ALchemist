"""D1/D2: what the desktop sync would lose, found before it rebuilds the session.

The desktop app rebuilds the session's search space and experiment table from
its own widgets. These tests pin the guard that lists what that rebuild would
drop, and (just as important) that a session the desktop app made itself is
never refused.

``ui.sync_guard`` imports no Tk, so this module runs headless.
"""
import pandas as pd

from alchemist_core import OptimizationSession
from ui.sync_guard import format_blockers, sync_blockers


def _session():
    s = OptimizationSession()
    s.add_variable("T", "real", bounds=(100.0, 300.0))
    s.add_variable("P", "real", bounds=(1.0, 10.0))
    return s


def _df(**extra):
    d = {"T": [150.0, 200.0], "P": [2.0, 3.0], "Output": [0.1, 0.2], **extra}
    return pd.DataFrame(d)


def _check(session, df, names=("T", "P"), targets=("Output",), **kw):
    return sync_blockers(session, df, list(names), list(targets), **kw)


def test_plain_session_is_safe():
    assert _check(_session(), _df()) == []


def test_empty_session_and_no_data_is_safe():
    assert _check(OptimizationSession(), None, names=()) == []
    assert _check(OptimizationSession(), pd.DataFrame(), names=()) == []


def test_desktop_native_exp_df_is_safe():
    """Every column the desktop app writes into exp_df: Iteration, Reason, Noise
    (ui.py save_new_point / toggle_noise / initial design), plus a numeric
    column of the user's own that the CSV loader keeps."""
    s = _session()
    s.add_variable("cat", "categorical", categories=["a", "b"])
    df = pd.DataFrame({
        "T": [150.0, 200.0], "P": [2.0, 3.0], "cat": ["a", "b"],
        "Output": [0.1, 0.2],
        "Noise": [1e-6, 1e-6],
        "Iteration": [0, 1],
        "Reason": ["Initial Design", "Manual"],
        "Replicate": [1, 2],
    })
    ui_vars = list(s.search_space.variables)
    assert _check(s, df, names=("T", "P", "cat"), ui_variables=ui_vars) == []


def test_desktop_native_blank_noise_and_reason_is_safe():
    """Noise and Reason may be empty; the cleaning keeps those rows."""
    df = _df(Noise=[None, 1e-6], Reason=[None, "Manual"], Iteration=[None, 1])
    assert _check(_session(), df) == []


def test_multi_objective_targets_are_known():
    df = _df(Output2=[1.0, 2.0])
    assert _check(_session(), df, targets=("Output", "Output2")) == []


def test_observed_variable_blocks():
    s = _session()
    s.add_variable("feed", "real", bounds=(0.0, 1.0), role="observed")
    out = _check(s, _df(feed=[0.2, 0.3]), names=("T", "P", "feed"))
    assert any("feed" in b for b in out)


def test_legacy_context_variable_blocks():
    s = _session()
    s.add_variable("ctx", "context")
    assert any("ctx" in b for b in _check(s, _df(), names=("T", "P", "ctx")))


def test_input_constraint_blocks():
    s = _session()
    s.add_input_constraint("inequality", {"T": 1.0, "P": 2.0}, rhs=250.0)
    assert any("constraint" in b for b in _check(s, _df()))


def test_outcome_constraint_blocks():
    s = _session()
    s.add_outcome_constraint("Output", "lower", 0.05)
    assert any("outcome" in b for b in _check(s, _df()))


def test_derived_variable_blocks():
    s = _session()
    s.search_space.add_derived_variable(
        "TP", lambda row: row["T"] * row["P"], ["T", "P"])
    assert any("derived" in b for b in _check(s, _df()))


def test_provenance_column_blocks():
    out = _check(_session(), _df(ProvenanceId=["a", "b"]))
    assert any("ProvenanceId" in b for b in out)


def test_split_plot_columns_block():
    out = _check(_session(), _df(WholePlot=[1, 1], RunOrder=[1, 2]))
    assert any("WholePlot" in b for b in out)
    assert any("RunOrder" in b for b in out)


def test_rows_that_would_be_dropped_block():
    df = _df()
    df.loc[1, "T"] = None
    out = _check(_session(), df)
    assert any("row" in b for b in out)


def test_empty_output_row_blocks():
    """The cleaning dropna()s targets too, so an unmeasured row is dropped."""
    df = _df()
    df.loc[0, "Output"] = None
    assert any("1 experiment row" in b for b in _check(_session(), df))


def test_non_numeric_feature_cell_blocks_but_categorical_text_does_not():
    s = _session()
    s.add_variable("cat", "categorical", categories=["a", "b"])
    ui_vars = list(s.search_space.variables)
    df = pd.DataFrame({"T": [1.0, 2.0], "P": [3.0, 4.0], "cat": ["a", "b"],
                       "Output": [1.0, 2.0]})
    assert _check(s, df, names=("T", "P", "cat"), ui_variables=ui_vars) == []
    df["T"] = df["T"].astype(object)
    df.loc[0, "T"] = "oops"        # pd.to_numeric(errors='coerce') -> NaN
    assert any("row" in b for b in
               _check(s, df, names=("T", "P", "cat"), ui_variables=ui_vars))


def test_text_in_an_unrelated_column_blocks():
    """A text column that is not a variable is coerced to NaN by the cleaning,
    so its rows would be dropped. The guard mirrors that."""
    df = _df(Sample=["a", "b"])
    assert any("row" in b for b in _check(_session(), df))


def test_hard_to_change_variable_blocks():
    s = OptimizationSession()
    s.add_variable("T", "real", bounds=(100.0, 300.0), changes="hard")
    out = _check(s, None, names=("T",))
    assert any("T" in b and "hard" in b for b in out)


def test_very_hard_to_change_variable_blocks():
    s = OptimizationSession()
    s.add_variable("T", "real", bounds=(100.0, 300.0), changes="very_hard")
    assert any("very hard" in b for b in _check(s, None, names=("T",)))


def test_variable_missing_from_ui_list_does_not_block():
    """Desktop load fills the variable list from the session, so a variable
    absent from it was deleted on purpose; blocking would make deleting one
    impossible."""
    assert _check(_session(), _df(), names=("T",)) == []


def test_format_blockers_lists_each_reason_and_points_to_the_web_app():
    text = format_blockers(["one thing", "another thing"])
    assert "- one thing" in text and "- another thing" in text
    assert "web app" in text and "nothing was changed" in text
