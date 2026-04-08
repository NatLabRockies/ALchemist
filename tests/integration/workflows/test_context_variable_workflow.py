"""
Integration tests for context variable support in OptimizationSession.

Uses a synthetic 2-tunable + 1-context setup. The context variable is a
measured batch quality score (continuous), known before each run but not
controlled by the optimizer.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from alchemist_core import OptimizationSession


def _make_session_with_data(n=20, seed=42):
    """
    Return a session with 2 tunable vars, 1 context var, n experiments.
    y = x1**2 + x2**2 + 0.3 * batch_quality + noise
    """
    rng = np.random.default_rng(seed)
    session = OptimizationSession()
    session.add_variable("x1", "real", min=0.0, max=1.0)
    session.add_variable("x2", "real", min=0.0, max=1.0)
    session.add_variable("batch_quality", "context")
    for _ in range(n):
        x1, x2 = rng.uniform(0, 1), rng.uniform(0, 1)
        bq = rng.uniform(0.5, 1.5)
        y = x1 ** 2 + x2 ** 2 + 0.3 * bq + rng.normal(0, 0.05)
        session.add_experiment({"x1": x1, "x2": x2, "batch_quality": bq}, float(y))
    return session


# ---- Registration ----

def test_context_variable_registered_on_search_space():
    session = _make_session_with_data()
    assert session.search_space.get_context_variable_names() == ["batch_quality"]


def test_context_variable_not_in_tunable_vars():
    session = _make_session_with_data()
    assert "batch_quality" not in session.search_space.get_tunable_variable_names()


# ---- train_model with context variable ----

def test_train_model_with_context_variable_succeeds():
    session = _make_session_with_data()
    result = session.train_model(backend="botorch")
    assert result["success"] is True


def test_gp_train_x_has_three_columns():
    """GP train_inputs should include tunable + context = 3 columns."""
    import torch
    session = _make_session_with_data()
    session.train_model(backend="botorch")
    train_X = session.model.model.train_inputs[0]
    assert train_X.shape[-1] == 3, f"Expected 3 columns, got {train_X.shape[-1]}"


def test_train_model_raises_missing_context_column():
    """train_model() raises if context variable column is absent from data."""
    # Add experiments without context var, then register it so train sees the gap
    rng = np.random.default_rng(0)
    session = OptimizationSession()
    session.add_variable("x1", "real", min=0.0, max=1.0)
    for _ in range(15):
        session.add_experiment({"x1": rng.uniform(0, 1)}, float(rng.uniform()))
    # Now register context var — data doesn't have this column
    session.add_variable("batch_quality", "context")
    with pytest.raises(ValueError, match="batch_quality"):
        session.train_model(backend="botorch")


def test_train_model_raises_for_sklearn_with_context_vars():
    """sklearn backend does not support context variables."""
    session = _make_session_with_data()
    with pytest.raises(ValueError, match="BoTorch"):
        session.train_model(backend="sklearn")


# ---- suggest_next with context ----

def test_suggest_next_returns_only_tunable_columns():
    """Suggestions contain only x1, x2 — not batch_quality."""
    session = _make_session_with_data()
    session.train_model(backend="botorch")
    suggestions = session.suggest_next(
        n_suggestions=1, strategy="EI", goal="maximize",
        context={"batch_quality": 1.0}
    )
    assert "batch_quality" not in suggestions.columns
    assert "x1" in suggestions.columns
    assert "x2" in suggestions.columns


def test_suggest_next_raises_without_context():
    """suggest_next() raises when context vars are registered but context is not provided."""
    session = _make_session_with_data()
    session.train_model(backend="botorch")
    with pytest.raises(ValueError, match="batch_quality"):
        session.suggest_next(n_suggestions=1, strategy="EI", goal="maximize")


def test_suggest_next_raises_with_missing_context_key():
    """suggest_next() raises when context dict is missing a registered context var."""
    session = _make_session_with_data()
    session.train_model(backend="botorch")
    with pytest.raises(ValueError, match="batch_quality"):
        session.suggest_next(
            n_suggestions=1, strategy="EI", goal="maximize",
            context={"wrong_name": 0.5}
        )


def test_suggest_next_extra_context_keys_ignored():
    """Extra keys in context dict beyond registered context vars are silently ignored."""
    session = _make_session_with_data()
    session.train_model(backend="botorch")
    suggestions = session.suggest_next(
        n_suggestions=1, strategy="EI", goal="maximize",
        context={"batch_quality": 1.0, "extra_key": 999.0}
    )
    assert "x1" in suggestions.columns


# ---- save / load round-trip ----

def test_save_session_persists_context_variable():
    """Context variable is saved in search_space.variables with type='context'."""
    session = _make_session_with_data()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    session.save_session(path)
    with open(path) as f:
        data = json.load(f)
    var_types = {v["name"]: v["type"] for v in data["search_space"]["variables"]}
    assert var_types.get("batch_quality") == "context"
    Path(path).unlink()


def test_load_session_restores_context_variable():
    """Loaded session has context variable in search space."""
    session = _make_session_with_data()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    session.save_session(path)

    loaded = OptimizationSession.load_session(path, retrain_on_load=False)
    assert loaded.search_space.get_context_variable_names() == ["batch_quality"]
    assert "batch_quality" not in loaded.search_space.get_tunable_variable_names()
    Path(path).unlink()


def test_load_and_train_after_save():
    """Round-trip: save → load → train_model → suggest_next works end-to-end."""
    session = _make_session_with_data()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    session.save_session(path)

    loaded = OptimizationSession.load_session(path, retrain_on_load=False)
    result = loaded.train_model(backend="botorch")
    assert result["success"] is True

    suggestions = loaded.suggest_next(
        n_suggestions=1, strategy="EI", goal="maximize",
        context={"batch_quality": 1.0}
    )
    assert "x1" in suggestions.columns
    assert "batch_quality" not in suggestions.columns
    Path(path).unlink()
