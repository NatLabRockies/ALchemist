"""A categorical value outside the declared list is refused at data entry.

Accepting it would only fail later, at training, when the model cannot encode
the undeclared level. Matching is by value first and by string form second
(a categorical declared as "1" accepts the integer 1 from a CSV).
"""

import pandas as pd
import pytest

from alchemist_core.session import OptimizationSession


@pytest.fixture
def session():
    s = OptimizationSession()
    s.add_variable("x1", "real", bounds=(0.0, 10.0))
    s.add_variable("c", "categorical", categories=["a", "b"])
    s.add_variable("k", "categorical", categories=["p", "q"], role="observed")
    return s


def test_add_experiment_rejects_undeclared_controllable_value(session):
    with pytest.raises(ValueError) as err:
        session.add_experiment({"x1": 1.0, "c": "B", "k": "p"}, output=1.0)
    msg = str(err.value)
    assert "'c'" in msg and "B" in msg and "['a', 'b']" in msg
    assert len(session.experiment_manager.df) == 0


def test_add_experiment_rejects_undeclared_observed_value(session):
    with pytest.raises(ValueError, match="'k'"):
        session.add_experiment({"x1": 1.0, "c": "a", "k": "r"}, output=1.0)


def test_add_experiment_accepts_declared_values(session):
    session.add_experiment({"x1": 1.0, "c": "b", "k": "q"}, output=1.0)
    assert len(session.experiment_manager.df) == 1


def test_string_form_match_is_accepted():
    s = OptimizationSession()
    s.add_variable("x1", "real", bounds=(0.0, 10.0))
    s.add_variable("c", "categorical", categories=["1", "2"])
    s.add_experiment({"x1": 1.0, "c": 2}, output=1.0)
    assert len(s.experiment_manager.df) == 1


def test_load_data_rejects_undeclared_value_and_keeps_previous_data(session, tmp_path):
    session.add_experiment({"x1": 1.0, "c": "a", "k": "p"}, output=1.0)
    path = tmp_path / "data.csv"
    pd.DataFrame({"x1": [1.0, 2.0], "c": ["a", "B"], "k": ["p", "q"],
                  "Output": [1.0, 2.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="'c'"):
        session.load_data(str(path))
    assert len(session.experiment_manager.df) == 1
