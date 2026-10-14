"""Plot bookkeeping columns survive a save/load as metadata, not as inputs."""
import numpy as np
import pandas as pd
import pytest

from alchemist_core import OptimizationSession


def _session():
    s = OptimizationSession()
    s.add_variable("x1", "real", min=0, max=10)
    s.add_variable("c", "categorical", values=["a", "b"], changes="hard")
    s.add_experiment({"x1": 1.0, "c": "a"}, 1.0)
    s.add_experiment({"x1": 2.0, "c": "b"}, 2.0, plot={"WholePlot": 4, "RunOrder": 8})
    s.queue.stage({"x1": 3.0, "c": "a", "WholePlot": 3, "RunOrder": 7})
    return s


def test_plot_columns_round_trip_as_metadata(tmp_path):
    path = tmp_path / "s.json"
    _session().save_session(str(path))
    loaded = OptimizationSession.load_session(str(path), retrain_on_load=False)

    df = loaded.experiment_manager.df
    assert {"WholePlot", "RunOrder"} <= set(df.columns)
    assert pd.isna(df.loc[0, "WholePlot"]) and pd.isna(df.loc[0, "RunOrder"])
    assert df.loc[1, "WholePlot"] == 4 and df.loc[1, "RunOrder"] == 8
    assert len(df) == 2

    meta = loaded.experiment_manager.metadata_columns()
    assert "WholePlot" in meta and "RunOrder" in meta
    X, _ = loaded.experiment_manager.get_features_and_target()
    assert list(X.columns) == ["x1", "c"]

    pending = loaded.queue.pending_items()
    assert len(pending) == 1
    assert pending[0].plot == {"WholePlot": 3, "RunOrder": 7}
    assert pending[0].inputs == {"x1": 3.0, "c": "a"}


def test_target_inference_for_a_file_without_target_columns_ignores_plot_columns(tmp_path):
    import json

    path = tmp_path / "s.json"
    _session().save_session(str(path))
    data = json.loads(path.read_text())
    data["experiments"].pop("target_columns", None)
    path.write_text(json.dumps(data))

    loaded = OptimizationSession.load_session(str(path), retrain_on_load=False)
    assert loaded.experiment_manager.target_columns == ["Output"]
    df = loaded.experiment_manager.df
    assert df.loc[1, "WholePlot"] == 4 and len(df) == 2
