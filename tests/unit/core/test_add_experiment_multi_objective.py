"""``add_experiment`` on a multi-objective session takes every objective (A18).

It took one float and wrote it to the first target column, leaving the other
objectives NaN with no error, and the next ``train_model`` then failed on the
NaN. A dict keyed by objective name, or a sequence in target order, now fills
every target; a single float on a multi-objective session is refused with a
message naming the missing targets.
"""
import numpy as np
import pandas as pd
import pytest

from alchemist_core import OptimizationSession


@pytest.fixture
def mobo(tmp_path):
    s = OptimizationSession()
    s.add_variable("x", "real", bounds=(0.0, 1.0))
    csv = tmp_path / "mo.csv"
    pd.DataFrame({"x": [0.1, 0.5, 0.9], "y1": [1.0, 2.0, 3.0],
                  "y2": [3.0, 2.0, 1.0]}).to_csv(csv, index=False)
    s.load_data(str(csv), target_columns=["y1", "y2"])
    return s


def test_dict_output_fills_every_target(mobo):
    mobo.add_experiment({"x": 0.3}, {"y1": 1.5, "y2": 2.5})
    last = mobo.experiment_manager.df.iloc[-1]
    assert last["y1"] == 1.5 and last["y2"] == 2.5


def test_sequence_output_in_target_order(mobo):
    mobo.add_experiment({"x": 0.3}, [1.5, 2.5])
    last = mobo.experiment_manager.df.iloc[-1]
    assert last["y1"] == 1.5 and last["y2"] == 2.5


def test_single_float_on_multi_objective_names_missing_targets(mobo):
    with pytest.raises(ValueError, match="y2"):
        mobo.add_experiment({"x": 0.3}, 1.5)
    assert len(mobo.experiment_manager.df) == 3
    assert not mobo.experiment_manager.df[["y1", "y2"]].isna().any().any()


def test_dict_missing_a_target_names_it(mobo):
    with pytest.raises(ValueError, match="y2"):
        mobo.add_experiment({"x": 0.3}, {"y1": 1.5})


def test_dict_with_unknown_target_is_refused(mobo):
    with pytest.raises(ValueError, match="y3"):
        mobo.add_experiment({"x": 0.3}, {"y1": 1.5, "y2": 2.5, "y3": 0.0})


def test_sequence_of_wrong_length_is_refused(mobo):
    with pytest.raises(ValueError, match="2"):
        mobo.add_experiment({"x": 0.3}, [1.5])


def test_non_finite_objective_is_refused(mobo):
    with pytest.raises(ValueError, match="finite"):
        mobo.add_experiment({"x": 0.3}, {"y1": 1.5, "y2": float("nan")})


def test_single_objective_still_takes_a_float():
    s = OptimizationSession()
    s.add_variable("x", "real", bounds=(0.0, 1.0))
    s.add_experiment({"x": 0.3}, 1.5)
    s.add_experiment({"x": 0.4}, {"Output": 2.5})
    s.add_experiment({"x": 0.5}, [3.5])
    assert s.experiment_manager.df["Output"].tolist() == [1.5, 2.5, 3.5]
