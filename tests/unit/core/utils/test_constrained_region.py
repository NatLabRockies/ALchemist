"""Unit tests for the linear-constrained-region geometry helpers."""

import numpy as np
import pytest

from alchemist_core.data.search_space import SearchSpace
from alchemist_core.utils import constrained_region as cr


def _space():
    s = SearchSpace()
    s.add_variable("x1", "real", min=0.0, max=10.0)
    s.add_variable("x2", "real", min=0.0, max=10.0)
    return s


class TestVariableHelpers:
    def test_numeric_variables_excludes_categorical_and_context(self):
        s = _space()
        s.add_variable("c1", "categorical", values=["a", "b"])
        s.add_variable("ctx", "context")
        names = [v["name"] for v in cr.numeric_variables(s)]
        assert names == ["x1", "x2"]

    def test_numeric_variables_includes_integer_and_discrete(self):
        s = SearchSpace()
        s.add_variable("i1", "integer", min=0, max=5)
        s.add_variable("d1", "discrete", allowed_values=[1.0, 2.0, 4.0])
        names = [v["name"] for v in cr.numeric_variables(s)]
        assert names == ["i1", "d1"]

    def test_variable_bounds_real(self):
        s = _space()
        assert cr.variable_bounds(s.variables[0]) == (0.0, 10.0)

    def test_variable_bounds_discrete_uses_min_max_of_allowed(self):
        s = SearchSpace()
        s.add_variable("d1", "discrete", allowed_values=[4.0, 1.0, 2.0])
        assert cr.variable_bounds(s.variables[0]) == (1.0, 4.0)

    def test_variable_bounds_rejects_categorical(self):
        s = SearchSpace()
        s.add_variable("c1", "categorical", values=["a", "b"])
        with pytest.raises(ValueError, match="no numeric bounds"):
            cr.variable_bounds(s.variables[0])


class TestProjection:
    def test_projected_point_lies_on_the_hyperplane(self):
        c = {"type": "inequality", "coefficients": {"x1": 1.0, "x2": 1.0},
             "rhs": 5.0, "name": "c0"}
        out = cr.project_onto_constraint({"x1": 10.0, "x2": 10.0}, c)
        assert out["x1"] + out["x2"] == pytest.approx(5.0)

    def test_projection_is_idempotent(self):
        c = {"type": "inequality", "coefficients": {"x1": 1.0, "x2": 1.0},
             "rhs": 5.0, "name": "c0"}
        once = cr.project_onto_constraint({"x1": 10.0, "x2": 10.0}, c)
        twice = cr.project_onto_constraint(once, c)
        assert twice["x1"] == pytest.approx(once["x1"])
        assert twice["x2"] == pytest.approx(once["x2"])

    def test_projection_is_orthogonal(self):
        # From (10, 10) onto x1 + x2 == 5, the displacement is equal in both
        # coordinates because the normal is (1, 1).
        c = {"type": "inequality", "coefficients": {"x1": 1.0, "x2": 1.0},
             "rhs": 5.0, "name": "c0"}
        out = cr.project_onto_constraint({"x1": 10.0, "x2": 10.0}, c)
        assert out["x1"] == pytest.approx(2.5)
        assert out["x2"] == pytest.approx(2.5)

    def test_non_participating_keys_are_preserved(self):
        c = {"type": "inequality", "coefficients": {"x1": 1.0},
             "rhs": 3.0, "name": "c0"}
        out = cr.project_onto_constraint({"x1": 9.0, "cat": "a"}, c)
        assert out["cat"] == "a"
        assert out["x1"] == pytest.approx(3.0)

    def test_zero_coefficient_vector_returns_point_unchanged(self):
        c = {"type": "inequality", "coefficients": {"x1": 0.0},
             "rhs": 3.0, "name": "c0"}
        out = cr.project_onto_constraint({"x1": 9.0}, c)
        assert out["x1"] == 9.0
