"""Every BoTorch acquisition honours goal='minimize' (T7, T20).

qEI and qUCB used to be built without a direction, so a minimisation campaign
using a batch acquisition was steered to the worst region. EI and LogEI with
an outcome constraint went through the same undirected qEI path, and EI
ignored the constraint altogether.
"""
import warnings

import numpy as np
import pytest
import torch

from alchemist_core import OptimizationSession


def bowl(x, y):
    # Minimum 0 at (0.3, 0.7); maximum 0.98 at (1, 0).
    return (x - 0.3) ** 2 + (y - 0.7) ** 2


@pytest.fixture
def session():
    torch.manual_seed(0)
    warnings.simplefilter("ignore")
    s = OptimizationSession()
    s.add_variable("x", "real", bounds=(0.0, 1.0))
    s.add_variable("y", "real", bounds=(0.0, 1.0))
    for p in s.generate_initial_design(method="lhs", n_points=12, random_seed=1):
        s.add_experiment(p, bowl(p["x"], p["y"]))
    s.train_model(backend="botorch", kernel="Matern")
    return s


def _values(df):
    return np.array([bowl(r.x, r.y) for r in df.itertuples()])


@pytest.mark.parametrize("strategy,q", [("qEI", 2), ("qUCB", 2), ("EI", 1), ("LogEI", 1), ("UCB", 1)])
def test_minimize_heads_to_the_minimum(session, strategy, q):
    vals = _values(session.suggest_next(strategy=strategy, goal="minimize", n_suggestions=q))
    assert vals.min() < 0.15, f"{strategy} minimize suggested {vals}"


@pytest.mark.parametrize("strategy,q", [("qEI", 2), ("qUCB", 2)])
def test_minimize_and_maximize_differ_for_batch(session, strategy, q):
    lo = _values(session.suggest_next(strategy=strategy, goal="minimize", n_suggestions=q))
    hi = _values(session.suggest_next(strategy=strategy, goal="maximize", n_suggestions=q))
    assert lo.min() < hi.min()
    assert hi.max() > 0.5, f"{strategy} maximize suggested {hi}"


@pytest.mark.parametrize("strategy", ["EI", "LogEI"])
def test_minimize_with_outcome_constraint(session, strategy):
    # Feasible region Output <= 0.3 contains the minimum, so the suggestion
    # must still head there rather than climb to the constraint boundary.
    session.add_outcome_constraint("Output", "upper", 0.3)
    vals = _values(session.suggest_next(strategy=strategy, goal="minimize"))
    assert vals.min() < 0.15, f"{strategy} minimize + constraint suggested {vals}"


def test_ei_honours_outcome_constraint(session):
    # Maximising with Output <= 0.3 must not run to the unconstrained maximum.
    session.add_outcome_constraint("Output", "upper", 0.3)
    vals = _values(session.suggest_next(strategy="EI", goal="maximize"))
    assert vals.max() < 0.6, f"EI ignored the outcome constraint: {vals}"
