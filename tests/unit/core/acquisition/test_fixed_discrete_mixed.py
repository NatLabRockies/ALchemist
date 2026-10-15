"""Held or observed discrete/categorical columns in the mixed optimizer.

A pinned column is fixed by ``fixed_features`` and its bounds, so it is not
also passed as a discrete or categorical dimension to the alternating
optimizer. An observed discrete value is measured and may sit off its grid;
in the enumeration fallback it is merged in without filtering, so the free
discrete variables keep their allowed-value enumeration.
"""
import logging

import numpy as np
import pytest
import torch

from alchemist_core import OptimizationSession
import alchemist_core.acquisition.botorch_acquisition as ba

D_ALLOWED = [1.0, 2.0, 4.0, 5.0]


def _session(observed_w=False, seed=4, n=20):
    # Seed torch as well as numpy: the GP fit starts from random restarts, and
    # an unseeded fit made this file flaky on CI (ModelFittingError).
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    s = OptimizationSession()
    s.add_variable("x1", "real", min=0, max=10)
    s.add_variable("d", "discrete", allowed_values=D_ALLOWED, changes="hard")
    s.add_variable("c", "categorical", values=["a", "b"])
    if observed_w:
        s.add_variable("w", "discrete", allowed_values=[0, 10], role="observed")
    for i in range(n):
        row = {"x1": float(rng.uniform(0, 10)), "d": D_ALLOWED[i % 4], "c": "ab"[i % 2]}
        y = -(row["x1"] - 6) ** 2 / 4 - 2 * (row["d"] - 3) ** 2 + (1.0 if row["c"] == "b" else 0.0)
        if observed_w:
            # (i // 4) % 2 is independent of c ("ab"[i % 2]) and d (i % 4):
            # each (c, d) combination sees both w levels. [0, 10][i % 2] made
            # w identical to c, so the fit was ill-conditioned.
            row["w"] = [0, 10][(i // 4) % 2]
            y += row["w"] / 10
        s.add_experiment(row, float(y + rng.normal(0, 0.05)))
    s.train_model(backend="botorch")
    return s


def _fallback_warnings(caplog):
    return [r for r in caplog.records
            if r.levelno >= logging.WARNING and "Falling back" in r.getMessage()]


def test_held_discrete_does_not_fall_back(caplog):
    s = _session()
    with caplog.at_level(logging.WARNING):
        out = s.suggest_next(strategy="EI", hold={"d": 2})
    assert float(out["d"].iloc[0]) == 2.0
    assert _fallback_warnings(caplog) == []


def test_held_categorical_does_not_fall_back(caplog):
    s = _session()
    with caplog.at_level(logging.WARNING):
        out = s.suggest_next(strategy="EI", hold={"c": "a"})
    assert out["c"].iloc[0] == "a"
    assert float(out["d"].iloc[0]) in D_ALLOWED
    assert _fallback_warnings(caplog) == []


def test_off_grid_observed_discrete_keeps_free_discrete_on_its_grid(caplog):
    s = _session(observed_w=True)
    with caplog.at_level(logging.WARNING):
        out = s.suggest_next(strategy="EI", observed={"w": 5})
    assert float(out["d"].iloc[0]) in D_ALLOWED
    assert _fallback_warnings(caplog) == []


def test_fallback_enumerates_free_discrete_with_off_grid_observed(monkeypatch):
    def unavailable(**kwargs):
        raise RuntimeError("forced fallback")

    monkeypatch.setattr(ba, "optimize_acqf_mixed_alternating", unavailable)
    s = _session(observed_w=True)
    out = s.suggest_next(strategy="EI", observed={"w": 5})
    assert float(out["d"].iloc[0]) in D_ALLOWED


def test_everything_held_returns_the_held_point_and_says_so(caplog):
    s = _session()
    with caplog.at_level(logging.INFO, logger="alchemist_core.acquisition.botorch_acquisition"):
        out = s.suggest_next(strategy="EI", hold={"x1": 3.0, "d": 4, "c": "b"})
    row = out.iloc[0]
    assert (float(row["x1"]), float(row["d"]), row["c"]) == (3.0, 4.0, "b")
    assert any("All variables are held" in r.getMessage() for r in caplog.records)


def test_fallback_offers_every_allowed_value_of_a_free_discrete(monkeypatch):
    def unavailable(**kwargs):
        raise RuntimeError("forced fallback")

    monkeypatch.setattr(ba, "optimize_acqf_mixed_alternating", unavailable)
    rng = np.random.default_rng(7)
    s = OptimizationSession()
    s.add_variable("x1", "real", min=0, max=10)
    s.add_variable("d", "discrete", allowed_values=[1, 2, 3, 4, 5])
    for i in range(30):
        d = float([1, 2, 3, 4, 5][i % 5])
        x1 = float(rng.uniform(0, 10))
        y = -((d - 3) ** 2) * 3 - (x1 - 6) ** 2 / 10 + rng.normal(0, 0.02)
        s.add_experiment({"x1": x1, "d": d}, float(y))
    s.train_model(backend="botorch")
    out = s.suggest_next(strategy="EI")
    assert float(out["d"].iloc[0]) == 3.0
