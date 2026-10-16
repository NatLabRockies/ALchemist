"""``kappa`` is an alias for ``beta`` on BoTorch UCB/qUCB (T13).

The web app sends the exploration weight as ``kappa`` (the scikit-learn
name). BoTorch's UCB reads ``beta``, so the value was dropped and both
functions ran at the default 0.5. ``kappa`` now reaches ``beta`` when
``beta`` is not given.
"""
import logging
import math

import pytest

from alchemist_core import OptimizationSession


@pytest.fixture
def trained():
    s = OptimizationSession()
    s.add_variable("x", "real", bounds=(0.0, 1.0))
    s.add_variable("y", "real", bounds=(0.0, 1.0))
    for x, y in [(0.1, 0.2), (0.4, 0.8), (0.7, 0.3), (0.9, 0.9), (0.3, 0.6), (0.5, 0.1)]:
        s.add_experiment({"x": x, "y": y}, (x - 0.3) ** 2 + (y - 0.7) ** 2)
    s.train_model(backend="botorch")
    return s


def test_kappa_reaches_ucb_beta(trained, caplog):
    with caplog.at_level(logging.WARNING):
        trained.suggest_next(strategy="UCB", goal="maximize", kappa=9.0)
    assert float(trained.acquisition.acq_function.beta) == 9.0
    assert "Unsupported parameters" not in caplog.text


def test_kappa_reaches_qucb_beta_prime(trained):
    trained.suggest_next(strategy="qUCB", goal="maximize", n_suggestions=2, kappa=9.0)
    assert math.isclose(float(trained.acquisition.acq_function.beta_prime),
                        math.sqrt(9.0 * math.pi / 2), rel_tol=1e-6)


def test_beta_wins_when_both_are_given(trained):
    trained.suggest_next(strategy="UCB", goal="maximize", beta=2.0, kappa=9.0)
    assert float(trained.acquisition.acq_function.beta) == 2.0
