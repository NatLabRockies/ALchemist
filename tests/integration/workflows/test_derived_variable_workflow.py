"""Integration tests for derived variable support in OptimizationSession."""

import json
import pytest
import random
from pathlib import Path

from alchemist_core.session import OptimizationSession


def make_session_with_derived():
    """Helper: 2-variable session with 1 derived variable and 8 experiments."""
    session = OptimizationSession()
    session.add_variable("x", "real", bounds=(0.0, 1.0))
    session.add_variable("y", "real", bounds=(0.0, 1.0))
    session.add_derived_variable(
        name="xy_product",
        func=lambda row: row["x"] * row["y"],
        input_cols=["x", "y"],
        description="Product of x and y",
    )
    random.seed(42)
    for _ in range(8):
        x, y = random.random(), random.random()
        session.add_experiment({"x": x, "y": y}, output=x + y + random.gauss(0, 0.05))
    return session


class TestDerivedVariableRegistration:
    def test_add_derived_variable_registers_on_search_space(self):
        session = OptimizationSession()
        session.add_variable("x", "real", bounds=(0.0, 1.0))
        session.add_variable("y", "real", bounds=(0.0, 1.0))
        session.add_derived_variable(
            name="xy_product",
            func=lambda row: row["x"] * row["y"],
            input_cols=["x", "y"],
        )
        assert session.search_space.has_derived_variables() is True
        assert "xy_product" in session.search_space.get_derived_variable_names()


class TestFitModelWithDerivedVariables:
    def test_fit_model_with_derived_variable_succeeds(self):
        session = make_session_with_derived()
        session.train_model(backend="botorch")
        assert session.model is not None

    def test_gp_input_dim_includes_derived_features(self):
        session = make_session_with_derived()
        session.train_model(backend="botorch")
        # GP should have seen 3 dimensions: x, y, xy_product
        train_inputs = session.model.model.train_inputs[0]
        assert train_inputs.shape[-1] == 3

    def test_suggest_next_returns_only_base_variables(self):
        session = make_session_with_derived()
        session.train_model(backend="botorch")
        suggestions = session.suggest_next()
        # suggestions may be a list of dicts or a single dict
        if isinstance(suggestions, list):
            suggestion = suggestions[0]
        else:
            suggestion = suggestions
        assert "x" in suggestion
        assert "y" in suggestion
        assert "xy_product" not in suggestion

    def test_fit_model_raises_for_sklearn_with_derived_vars(self):
        session = make_session_with_derived()
        with pytest.raises(ValueError, match="BoTorch"):
            session.train_model(backend="sklearn")

    def test_fit_model_raises_for_unregistered_func(self):
        session = OptimizationSession()
        session.add_variable("x", "real", bounds=(0.0, 1.0))
        session.add_variable("y", "real", bounds=(0.0, 1.0))
        # Add stub directly, bypassing add_derived_variable
        session.search_space.add_derived_variable_stub("xy_product", ["x", "y"])
        random.seed(42)
        for _ in range(8):
            x, y = random.random(), random.random()
            session.add_experiment({"x": x, "y": y}, output=x + y + random.gauss(0, 0.05))
        with pytest.raises(ValueError, match="no callable registered"):
            session.train_model(backend="botorch")


class TestDerivedVariableSerialisation:
    def test_save_load_round_trip_restores_stubs(self, tmp_path):
        session = make_session_with_derived()
        save_path = str(tmp_path / "session.json")
        session.save_session(save_path)

        loaded = OptimizationSession.load_session(save_path, retrain_on_load=False)
        assert loaded.search_space.has_derived_variables() is True

        # Callable should be None after load (not serialized)
        dv = loaded.search_space.derived_variables[0]
        assert dv["func"] is None

        # Attempting fit_model without re-registering should raise
        with pytest.raises(ValueError, match="no callable registered"):
            loaded.train_model(backend="botorch")

    def test_save_load_reregister_fit_succeeds(self, tmp_path):
        session = make_session_with_derived()
        save_path = str(tmp_path / "session.json")
        session.save_session(save_path)

        loaded = OptimizationSession.load_session(save_path, retrain_on_load=False)
        loaded.register_derived_variable("xy_product", lambda row: row["x"] * row["y"])
        # Should not raise
        loaded.train_model(backend="botorch")
        assert loaded.model is not None
