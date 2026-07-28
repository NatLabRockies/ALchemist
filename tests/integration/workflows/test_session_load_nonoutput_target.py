"""
Regression tests for restoring experiment data whose objective/target column is
NOT literally named 'Output'.

Real-world trigger: a SULI intern's inductive RCC session (objective column
"Methane umol/g/Wh") saved by ALchemist and reopened in the web app. The saved
file serializes the dataset with its real target column name, but the loader
historically hardcoded output = row.get('Output'), so every row failed
np.isfinite(None) and was silently skipped -> 0 experiments restored.

The loader must instead determine the real target column so historical files
(any version, no target_columns key) round-trip correctly.
"""

import json
import tempfile
from pathlib import Path

import pytest

from alchemist_core import OptimizationSession

FIXTURE = Path(__file__).parent.parent.parent / "fixtures" / "session_nonoutput_target.json"


def _make_nonoutput_session():
    """A session whose objective column is 'yield_pct', not 'Output'."""
    session = OptimizationSession()
    session.add_variable("temperature", "real", bounds=(100, 300))
    session.add_variable("catalyst", "categorical", categories=["A", "B"])
    # Give it experiment data with a non-'Output' target via load_data.
    import pandas as pd
    df = pd.DataFrame(
        {
            "temperature": [150.0, 200.0, 250.0],
            "catalyst": ["A", "B", "A"],
            "yield_pct": [0.0, 42.5, 88.1],
        }
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        csv_path = f.name
    df.to_csv(csv_path, index=False)
    try:
        session.load_data(csv_path, target_columns="yield_pct")
    finally:
        Path(csv_path).unlink(missing_ok=True)
    return session


# ---- the real intern fixture ----

@pytest.mark.skipif(not FIXTURE.exists(), reason="intern session fixture not present")
def test_load_real_nonoutput_session_restores_all_rows():
    """Anna's actual session file must restore all 15 experiments, not 0."""
    loaded = OptimizationSession.load_session(str(FIXTURE), retrain_on_load=False)
    df = loaded.experiment_manager.get_data()
    assert len(df) == 15, (
        f"expected 15 experiments restored, got {len(df)} — the target column "
        f"'Methane umol/g/Wh' was not recognized on load"
    )
    assert "Methane umol/g/Wh" in loaded.experiment_manager.target_columns
    assert "Methane umol/g/Wh" in df.columns


# ---- synthetic round-trip ----

def test_nonoutput_target_roundtrips():
    """A session with a non-'Output' objective column round-trips its data."""
    session = _make_nonoutput_session()
    assert len(session.experiment_manager.get_data()) == 3

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        session.save_session(path)
        loaded = OptimizationSession.load_session(path, retrain_on_load=False)
        df = loaded.experiment_manager.get_data()
        assert len(df) == 3, "non-'Output' target rows were dropped on load"
        assert "yield_pct" in loaded.experiment_manager.target_columns
        # values survived
        assert sorted(df["yield_pct"].tolist()) == [0.0, 42.5, 88.1]
    finally:
        Path(path).unlink(missing_ok=True)


def test_save_persists_target_columns():
    """save_session records the objective column name so files are self-describing."""
    session = _make_nonoutput_session()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        session.save_session(path)
        data = json.load(open(path))
        assert data["experiments"].get("target_columns") == ["yield_pct"]
    finally:
        Path(path).unlink(missing_ok=True)


def test_old_file_without_target_columns_key_still_infers():
    """Backward compat: files saved before target_columns persistence must still
    restore their data by inferring the target from the non-variable columns."""
    session = _make_nonoutput_session()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        session.save_session(path)
        # Simulate an OLD file: strip the new key.
        data = json.load(open(path))
        data["experiments"].pop("target_columns", None)
        json.dump(data, open(path, "w"))

        loaded = OptimizationSession.load_session(path, retrain_on_load=False)
        df = loaded.experiment_manager.get_data()
        assert len(df) == 3
        assert "yield_pct" in loaded.experiment_manager.target_columns
    finally:
        Path(path).unlink(missing_ok=True)


def test_output_named_target_still_works():
    """The common case (objective literally named 'Output') is unaffected."""
    session = OptimizationSession()
    session.add_variable("x", "real", bounds=(0, 1))
    import pandas as pd
    df = pd.DataFrame({"x": [0.1, 0.5, 0.9], "Output": [1.0, 2.0, 3.0]})
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        csv_path = f.name
    df.to_csv(csv_path, index=False)
    try:
        session.load_data(csv_path, target_columns="Output")
    finally:
        Path(csv_path).unlink(missing_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        session.save_session(path)
        loaded = OptimizationSession.load_session(path, retrain_on_load=False)
        assert len(loaded.experiment_manager.get_data()) == 3
        assert loaded.experiment_manager.target_columns == ["Output"]
    finally:
        Path(path).unlink(missing_ok=True)
