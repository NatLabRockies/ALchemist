"""build_fixed_features: validation and encoding of held and observed values."""
import pytest

from alchemist_core.acquisition.fixed_features import build_fixed_features
from alchemist_core.data.search_space import SearchSpace


def _space():
    s = SearchSpace()
    s.add_variable("x1", "real", min=0, max=10)
    s.add_variable("n1", "integer", min=0, max=5)
    s.add_variable("d1", "discrete", allowed_values=[0.5, 1.0, 2.0])
    s.add_variable("c", "categorical", values=["a", "b", "c"], changes="hard")
    s.add_variable("z", "real", role="observed")
    s.add_variable("k", "categorical", values=["p", "q"], role="observed")
    return s


NAMES = ["x1", "n1", "d1", "c", "z", "k"]
ENC = {"c": {"a": 0, "b": 1, "c": 2}, "k": {"p": 0, "q": 1}}


def test_each_kind_encodes_to_its_column():
    ff = build_fixed_features(
        _space(), NAMES, ENC,
        hold={"x1": 2.5, "n1": 3, "d1": 1.0, "c": "c"},
        observed={"z": 31.2, "k": "q"},
    )
    assert ff.columns == {0: 2.5, 1: 3.0, 2: 1.0, 3: 2.0, 4: 31.2, 5: 1.0}
    assert ff.held == {"x1": 2.5, "n1": 3, "d1": 1.0, "c": "c"}
    assert ff.observed == {"z": 31.2, "k": "q"}


@pytest.mark.parametrize("hold, match", [
    ({"x1": 11.0}, "outside its bounds"),
    ({"n1": 2.5}, "integer"),
    ({"d1": 0.7}, "allowed"),
    ({"c": "d"}, "declared values"),
    ({"nope": 1.0}, "no such variable"),
    ({"z": 1.0}, "observed="),
    ({"x1": float("nan")}, "finite"),
    ({"x1": "abc"}, "not a number"),
])
def test_invalid_holds_raise(hold, match):
    with pytest.raises(ValueError, match=match):
        build_fixed_features(_space(), NAMES, ENC, hold=hold, observed={"z": 1.0, "k": "p"})


def test_observed_real_takes_any_finite_value():
    ff = build_fixed_features(_space(), NAMES, ENC, hold={}, observed={"z": -1e6, "k": "p"})
    assert ff.columns[4] == -1e6


def test_numeric_text_label_matches_declared_label():
    s = SearchSpace()
    s.add_variable("c", "categorical", values=["1", "2"])
    ff = build_fixed_features(s, ["c"], {"c": {"1": 0, "2": 1}}, hold={"c": 2}, observed={})
    assert ff.columns == {0: 1.0} and ff.held == {"c": "2"}
