"""``add_variable`` refuses a new variable once results are recorded (item 16).

Adding a variable to a session that already holds rows left every row
without a value for it; the next summary or training call then failed with
"Variable column(s) ['X9'] not found in experiment data", so the REST
session became unreadable (every GET 400) and the web app dropped it. The
variable is refused up front with a message that says what to do. A
variable whose column is already in the data (results loaded from a CSV,
then the variables defined) is still accepted.
"""
import pandas as pd
import pytest

from alchemist_core import OptimizationSession


def test_add_variable_is_refused_with_recorded_rows():
    s = OptimizationSession()
    s.add_variable("x", "real", bounds=(0.0, 1.0))
    for i in range(10):
        s.add_experiment({"x": i / 10}, float(i))
    with pytest.raises(ValueError, match=r"Cannot add variable 'X9'.*10 recorded experiments"):
        s.add_variable("X9", "real", bounds=(0.0, 1.0))
    assert [v["name"] for v in s.search_space.variables] == ["x"]
    s.get_data_summary()
    s.train_model(backend="sklearn")


def test_add_variable_works_with_no_rows():
    s = OptimizationSession()
    s.add_variable("x", "real", bounds=(0.0, 1.0))
    s.add_variable("X9", "real", bounds=(0.0, 1.0))
    assert [v["name"] for v in s.search_space.variables] == ["x", "X9"]


def test_csv_loaded_then_defined_still_works(tmp_path):
    csv = tmp_path / "r.csv"
    pd.DataFrame({"x": [0.1, 0.5, 0.9], "y": [0.2, 0.4, 0.8],
                  "Output": [1.0, 2.0, 3.0]}).to_csv(csv, index=False)
    s = OptimizationSession()
    s.load_data(str(csv))
    s.add_variable("x", "real", bounds=(0.0, 1.0))
    s.add_variable("y", "real", bounds=(0.0, 1.0))
    s.train_model(backend="sklearn")
    with pytest.raises(ValueError, match="Cannot add variable 'z'"):
        s.add_variable("z", "real", bounds=(0.0, 1.0))


def test_removing_a_variable_with_a_data_column_still_works():
    s = OptimizationSession()
    s.add_variable("x", "real", bounds=(0.0, 1.0))
    s.add_variable("y", "real", bounds=(0.0, 1.0))
    for i in range(4):
        s.add_experiment({"x": i / 4, "y": 1 - i / 4}, float(i))
    s.search_space.remove_variable("y")
    assert "y" in s.experiment_manager.df.columns
    s.get_data_summary()
    s.train_model(backend="sklearn")
    assert list(s.model.original_feature_names) == ["x"]
