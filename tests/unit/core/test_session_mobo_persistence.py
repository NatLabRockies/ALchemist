"""A multi-objective session keeps every objective across save and load (NEW-1).

``_load_session_impl`` rebuilt the results table one row at a time through
``add_experiment`` with ``row[primary_target]`` only, so a 2-objective file
(whose ``experiments.data`` carries both columns and whose ``target_columns``
names both) came back with the second objective NaN on every row. The
load-time retrain then failed, and saving again wrote the NaNs to disk: a
researcher who saved, reopened and saved had lost ``y2``.
"""
import json
import logging

import numpy as np
import pandas as pd
import pytest

from alchemist_core import OptimizationSession


def _bowl(x, y):
    return (x - 0.3) ** 2 + (y - 0.7) ** 2


@pytest.fixture
def saved(tmp_path):
    """A trained 2-objective BoTorch session saved to disk."""
    s = OptimizationSession()
    s.add_variable("x", "real", bounds=(0.0, 1.0))
    s.add_variable("y", "real", bounds=(0.0, 1.0))
    pts = s.generate_initial_design(method="lhs", n_points=8, random_seed=2)
    rows = [{"x": p["x"], "y": p["y"], "y1": _bowl(p["x"], p["y"]),
             "y2": -_bowl(p["x"], p["y"]) + p["x"]} for p in pts]
    csv = tmp_path / "mo.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    s.load_data(str(csv), target_columns=["y1", "y2"])
    s.train_model(backend="botorch")
    path = tmp_path / "mo.json"
    s.save_session(str(path))
    return s, path


def _assert_both_targets_restored(original, loaded):
    df = loaded.experiment_manager.df
    assert loaded.experiment_manager.target_columns == ["y1", "y2"]
    assert len(df) == len(original.experiment_manager.df)
    assert df[["y1", "y2"]].notna().all().all()
    for col in ("y1", "y2"):
        np.testing.assert_allclose(df[col].to_numpy(),
                                   original.experiment_manager.df[col].to_numpy())


def test_class_form_restores_every_objective_and_retrains(saved, caplog):
    original, path = saved
    with caplog.at_level(logging.WARNING):
        loaded = OptimizationSession.load_session(str(path))
    _assert_both_targets_restored(original, loaded)
    assert "Failed to retrain" not in caplog.text
    assert loaded.model is not None


def test_instance_form_restores_every_objective(saved):
    original, path = saved
    loaded = OptimizationSession()
    loaded.load_session(str(path), retrain_on_load=False)
    _assert_both_targets_restored(original, loaded)
    loaded.train_model(backend="botorch")
    assert loaded.model is not None


def test_save_again_keeps_both_objectives(saved, tmp_path):
    original, path = saved
    loaded = OptimizationSession.load_session(str(path), retrain_on_load=False)
    again = tmp_path / "again.json"
    loaded.save_session(str(again))
    data = json.loads(again.read_text())
    assert data["experiments"]["target_columns"] == ["y1", "y2"]
    for row in data["experiments"]["data"]:
        assert row["y1"] is not None and row["y2"] is not None
        assert np.isfinite(row["y1"]) and np.isfinite(row["y2"])


def test_bookkeeping_columns_survive_the_multi_objective_restore(saved):
    original, path = saved
    original.add_staged_experiment({"x": 0.2, "y": 0.2})
    item = original.queue.pending_items()[0]
    original.complete_experiment(item.id, {"x": 0.2, "y": 0.2},
                                 output={"y1": 0.26, "y2": -0.06}, iteration=7)
    original.save_session(str(path))
    loaded = OptimizationSession.load_session(str(path), retrain_on_load=False)
    last = loaded.experiment_manager.df.iloc[-1]
    assert last["ProvenanceId"] == item.id
    assert int(last["Iteration"]) == 7
    assert last["y1"] == 0.26 and last["y2"] == -0.06
