"""Holding controllable variables and conditioning on observed ones, end to end."""
import numpy as np
import pytest

from alchemist_core import OptimizationSession

OFFSET = {"a": 0.0, "b": 8.0, "c": -4.0}


def _session(n=24, seed=0):
    rng = np.random.default_rng(seed)
    s = OptimizationSession()
    s.add_variable("x1", "real", min=0, max=10)
    s.add_variable("x2", "real", min=0, max=10)
    s.add_variable("c", "categorical", values=["a", "b", "c"], changes="hard")
    s.add_variable("k", "categorical", values=["p", "q"], role="observed")
    for i in range(n):
        c = "abc"[i % 3]
        k = "pq"[i % 2]
        x1, x2 = rng.uniform(0, 10, 2)
        y = -(x1 - 6) ** 2 - (x2 - 3) ** 2 + OFFSET[c] + (1.0 if k == "q" else 0.0)
        s.add_experiment({"x1": x1, "x2": x2, "c": c, "k": k}, float(y + rng.normal(0, 0.1)))
    s.train_model(backend="botorch")
    return s


def test_observed_categorical_enters_cat_dims_and_round_trips():
    s = _session()
    assert s.model.cat_dims == [2, 3]
    out = s.suggest_next(strategy="EI", observed={"k": "q"})
    assert list(out.columns) == ["x1", "x2", "c"]
    assert out.iloc[0]["c"] in {"a", "b", "c"}


@pytest.mark.parametrize("label", ["a", "b", "c"])
def test_held_categorical_is_returned_unchanged(label):
    s = _session()
    out = s.suggest_next(strategy="EI", hold={"c": label}, observed={"k": "p"})
    assert out.iloc[0]["c"] == label


def test_held_real_is_returned_unchanged_in_a_batch():
    s = _session()
    out = s.suggest_next(strategy="qEI", n_suggestions=3, hold={"x2": 7.5}, observed={"k": "p"})
    assert len(out) == 3
    assert np.allclose(out["x2"].astype(float), 7.5)


def test_context_keyword_still_works_and_cannot_be_combined():
    s = _session()
    assert len(s.suggest_next(strategy="EI", context={"k": "q"})) == 1
    with pytest.raises(ValueError, match="both"):
        s.suggest_next(strategy="EI", context={"k": "q"}, observed={"k": "q"})


def test_wrong_dictionary_errors_name_the_right_keyword():
    s = _session()
    with pytest.raises(ValueError, match="hold="):
        s.suggest_next(strategy="EI", observed={"k": "p", "c": "a"})
    with pytest.raises(ValueError, match="observed="):
        s.suggest_next(strategy="EI", hold={"k": "p"}, observed={"k": "p"})


def test_hold_is_recorded_for_provenance():
    s = _session()
    s.suggest_next(strategy="EI", hold={"c": "b"}, observed={"k": "p"})
    assert s._last_acquisition_info["hold"] == {"c": "b"}
    assert s._last_acquisition_info["observed"] == {"k": "p"}


def test_holding_everything_returns_the_held_point():
    s = _session()
    out = s.suggest_next(strategy="EI", hold={"x1": 1.0, "x2": 2.0, "c": "a"}, observed={"k": "p"})
    assert out.iloc[0].to_dict() == {"x1": 1.0, "x2": 2.0, "c": "a"}


def test_infeasible_hold_with_a_constraint_raises():
    from alchemist_core.utils.constrained_region import InfeasibleRegionError
    s = OptimizationSession()
    s.add_variable("x1", "real", min=0, max=10)
    s.add_variable("x2", "real", min=0, max=10)
    s.add_input_constraint("inequality", {"x1": 1.0, "x2": 1.0}, 5.0)
    rng = np.random.default_rng(1)
    for _ in range(10):
        x1 = rng.uniform(0, 2.5)
        x2 = rng.uniform(0, 2.5)
        s.add_experiment({"x1": x1, "x2": x2}, float(x1 + x2))
    s.train_model(backend="botorch")
    out = s.suggest_next(strategy="EI", hold={"x1": 2.0})
    assert out.iloc[0]["x1"] == 2.0 and out.iloc[0]["x2"] <= 3.0 + 1e-6
    with pytest.raises(InfeasibleRegionError, match="x1"):
        s.suggest_next(strategy="EI", hold={"x1": 9.0, "x2": 9.0})


def test_hold_rejected_for_qipv_and_sklearn():
    s = _session()
    with pytest.raises(ValueError, match="qIPV"):
        s.suggest_next(strategy="qIPV", hold={"c": "a"}, observed={"k": "p"})
    t = OptimizationSession()
    t.add_variable("x1", "real", min=0, max=1)
    t.add_variable("x2", "real", min=0, max=1)
    for i in range(8):
        t.add_experiment({"x1": i / 8, "x2": (i * 3 % 8) / 8}, float(i))
    t.train_model(backend="sklearn")
    with pytest.raises(ValueError, match="sklearn"):
        t.suggest_next(strategy="EI", hold={"x1": 0.5})
