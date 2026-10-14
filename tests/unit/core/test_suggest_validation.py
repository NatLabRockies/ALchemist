"""A1 (analytic strategies with a batch) and A5 (unknown kwargs) in suggest_next."""
import numpy as np
import pytest

from alchemist_core import OptimizationSession


def _trained(backend):
    rng = np.random.default_rng(0)
    s = OptimizationSession()
    s.add_variable("x1", "real", bounds=(0.0, 10.0))
    s.add_variable("x2", "real", bounds=(0.0, 10.0))
    for a, b in rng.uniform(0, 10, size=(12, 2)):
        s.add_experiment({"x1": float(a), "x2": float(b)}, float(-(a - 3) ** 2 - (b - 6) ** 2))
    s.train_model(backend=backend, kernel="Matern")
    return s


@pytest.fixture(scope="module")
def botorch_session():
    return _trained("botorch")


@pytest.fixture(scope="module")
def sklearn_session():
    return _trained("sklearn")


@pytest.mark.parametrize("strategy", ["EI", "ei", "LogEI", "PI", "LogPI", "UCB", "ExpectedImprovement"])
def test_analytic_strategy_with_batch_raises_on_botorch(botorch_session, strategy):
    with pytest.raises(ValueError, match="qEI"):
        botorch_session.suggest_next(strategy=strategy, n_suggestions=3)


def test_batch_on_sklearn_raises(sklearn_session):
    with pytest.raises(ValueError, match="botorch"):
        sklearn_session.suggest_next(strategy="EI", n_suggestions=2)


def test_q_strategy_batch_still_works(botorch_session):
    out = botorch_session.suggest_next(strategy="qEI", n_suggestions=3)
    assert len(out) == 3


def test_analytic_single_point_still_works(botorch_session):
    assert len(botorch_session.suggest_next(strategy="EI", n_suggestions=1)) == 1


def test_unknown_kwarg_raises_typeerror(botorch_session):
    with pytest.raises(TypeError, match="n_candidates.*n_suggestions"):
        botorch_session.suggest_next(strategy="qEI", n_candidates=3)


def test_known_but_unused_kwarg_only_warns(botorch_session, caplog):
    # xi is a sklearn EI parameter; BoTorch EI ignores it with a warning (REST forwards xi).
    out = botorch_session.suggest_next(strategy="EI", xi=0.05)
    assert len(out) == 1
