"""Categorical labels are indexed by their declared position, not by data order."""
import numpy as np
import pandas as pd
import pytest

from alchemist_core.session import OptimizationSession


def _session(values=("a", "b", "c"), labels_in_data=("c", "a"), n=18, seed=0):
    s = OptimizationSession()
    s.add_variable("x1", "real", min=0, max=10)
    s.add_variable("c", "categorical", values=list(values))
    rng = np.random.default_rng(seed)
    offset = {"a": 0.0, "b": 3.0, "c": -2.0}
    for i in range(n):
        lab = labels_in_data[i % len(labels_in_data)]
        x1 = float(rng.uniform(0, 10))
        s.add_experiment({"x1": x1, "c": lab}, float(-(x1 - 6) ** 2 / 10 + offset[lab]))
    return s


def test_encoding_follows_declared_order():
    s = _session()
    s.train_model(backend="botorch")
    assert s.model.categorical_encodings["c"] == {"a": 0, "b": 1, "c": 2}
    assert s.model.cat_dims == [1]


def test_unobserved_declared_level_is_suggestible():
    s = _session(labels_in_data=("a", "c"))
    s.train_model(backend="botorch")
    from alchemist_core.acquisition.botorch_acquisition import BoTorchAcquisition
    acq = BoTorchAcquisition(search_space=s.search_space, model=s.model, acq_func="ucb")
    bounds = acq._get_bounds_from_search_space()
    assert float(bounds[1, 1]) == 2.0


def test_undeclared_training_value_raises():
    s = _session()
    s.experiment_manager.df.loc[0, "c"] = "zzz"
    with pytest.raises(ValueError, match="zzz"):
        s.train_model(backend="botorch")


def test_numeric_labels_match_by_string_form():
    s = OptimizationSession()
    s.add_variable("x1", "real", min=0, max=1)
    s.add_variable("c", "categorical", values=["1", "2"])
    for i in range(8):
        s.add_experiment({"x1": i / 8, "c": "1" if i % 2 else "2"}, float(i))
    s.experiment_manager.df["c"] = s.experiment_manager.df["c"].astype(int)  # as a CSV upload yields
    s.train_model(backend="botorch")
    assert s.model.categorical_encodings["c"] == {"1": 0, "2": 1}


def test_permuting_declared_order_does_not_change_predictions():
    a = _session(values=("a", "b", "c"))
    b = _session(values=("c", "a", "b"))
    a.train_model(backend="botorch")
    b.train_model(backend="botorch")
    grid = pd.DataFrame({"x1": [1.0, 5.0, 9.0], "c": ["a", "c", "a"]})
    pa, _ = a.model.predict(grid, return_std=True)
    pb, _ = b.model.predict(grid, return_std=True)
    np.testing.assert_allclose(pa, pb, rtol=1e-3, atol=1e-4)


def test_save_load_retrain_keeps_declared_encoding(tmp_path):
    s = _session()
    s.train_model(backend="botorch")
    path = tmp_path / "s.json"
    s.save_session(str(path))
    s2 = OptimizationSession().load_session(str(path))
    assert s2.model is not None
    assert s2.model.categorical_encodings["c"] == {"a": 0, "b": 1, "c": 2}
