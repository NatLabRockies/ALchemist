"""With something held, EI-family baselines are the best reachable in the held slice."""
import numpy as np
import pytest

from alchemist_core import OptimizationSession

OFFSET = {"a": 0.0, "b": 8.0, "c": -4.0}


def _session(n=30, seed=3):
    rng = np.random.default_rng(seed)
    s = OptimizationSession()
    s.add_variable("x1", "real", min=0, max=10)
    s.add_variable("x2", "real", min=0, max=10)
    s.add_variable("c", "categorical", values=["a", "b", "c"], changes="hard")
    for i in range(n):
        c = "abc"[i % 3]
        x1, x2 = rng.uniform(0, 10, 2)
        y = -(x1 - 6) ** 2 / 4 - (x2 - 3) ** 2 / 4 + OFFSET[c]
        s.add_experiment({"x1": x1, "x2": x2, "c": c}, float(y + rng.normal(0, 0.05)))
    s.train_model(backend="botorch")
    return s


def _best_f(session):
    return float(session.acquisition.acq_function.best_f)


@pytest.mark.parametrize("strategy", ["EI", "LogEI", "PI", "qEI"])
def test_held_slice_baseline_is_lower_than_global(strategy):
    s = _session()
    n = 2 if strategy == "qEI" else 1
    s.suggest_next(strategy=strategy, n_suggestions=n)
    global_best = _best_f(s)
    s.suggest_next(strategy=strategy, n_suggestions=n, hold={"c": "c"})
    slice_best = _best_f(s)
    assert slice_best < global_best - 6.0


def test_unheld_baseline_is_unchanged():
    s = _session()
    s.suggest_next(strategy="EI")
    first = _best_f(s)
    s.suggest_next(strategy="EI", hold={"c": "b"})
    s.suggest_next(strategy="EI")
    assert _best_f(s) == pytest.approx(first)


def test_held_slice_baseline_in_the_minimize_direction():
    s = _session()
    s.suggest_next(strategy="EI", goal="minimize")
    global_best = _best_f(s)
    s.suggest_next(strategy="EI", goal="minimize", hold={"c": "b"})
    slice_best = _best_f(s)
    assert slice_best > global_best + 6.0
