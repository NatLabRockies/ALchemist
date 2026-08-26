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

    def test_variable_bounds_rejects_context(self):
        s = SearchSpace()
        s.add_variable("ctx", "context")
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

    def test_projection_pins_orthogonal_formula_with_asymmetric_coefficients(self):
        # 3*x1 + 4*x2 == 5, from (10, 10). With unequal coefficients, the
        # true orthogonal projection x - c*slack/||c||^2 diverges from an
        # even split x - slack/n, so this pins the actual formula rather
        # than just an on-hyperplane property that both formulas satisfy.
        c = {"type": "equality", "coefficients": {"x1": 3.0, "x2": 4.0},
             "rhs": 5.0, "name": "c0"}
        out = cr.project_onto_constraint({"x1": 10.0, "x2": 10.0}, c)
        assert out["x1"] == pytest.approx(2.2)
        assert out["x2"] == pytest.approx(-0.4)

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


class TestFeasibleInterval:
    def test_positive_coefficient_gives_upper_bound(self):
        # x1 + x2 <= 6, x2 fixed at 2  ->  x1 <= 4
        s = _space()
        s.add_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=6.0)
        assert cr.feasible_interval(s, "x1", {"x2": 2.0}) == pytest.approx((0.0, 4.0))

    def test_negative_coefficient_gives_lower_bound(self):
        # -x1 + x2 <= 1, x2 fixed at 5  ->  -x1 <= -4  ->  x1 >= 4
        s = _space()
        s.add_constraint("inequality", {"x1": -1.0, "x2": 1.0}, rhs=1.0)
        assert cr.feasible_interval(s, "x1", {"x2": 5.0}) == pytest.approx((4.0, 10.0))

    def test_equality_collapses_the_interval_to_a_point(self):
        # x1 + x2 == 7, x2 fixed at 3  ->  x1 == 4
        s = _space()
        s.add_constraint("equality", {"x1": 1.0, "x2": 1.0}, rhs=7.0)
        lo, hi = cr.feasible_interval(s, "x1", {"x2": 3.0})
        assert lo == pytest.approx(4.0)
        assert hi == pytest.approx(4.0)

    def test_zero_coefficient_contributes_nothing(self):
        s = _space()
        s.add_constraint("inequality", {"x1": 0.0, "x2": 1.0}, rhs=3.0)
        assert cr.feasible_interval(s, "x1", {"x2": 1.0}) == pytest.approx((0.0, 10.0))

    def test_constraint_not_naming_the_variable_is_ignored(self):
        s = _space()
        s.add_constraint("inequality", {"x2": 1.0}, rhs=3.0)
        assert cr.feasible_interval(s, "x1", {"x2": 1.0}) == pytest.approx((0.0, 10.0))

    def test_empty_intersection_returns_none(self):
        # x1 <= -5 is outside the variable's own [0, 10] bounds.
        s = _space()
        s.add_constraint("inequality", {"x1": 1.0}, rhs=-5.0)
        assert cr.feasible_interval(s, "x1", {"x2": 1.0}) is None

    def test_two_constraints_intersect(self):
        s = _space()
        s.add_constraint("inequality", {"x1": 1.0}, rhs=8.0)
        s.add_constraint("inequality", {"x1": -1.0}, rhs=-2.0)  # x1 >= 2
        assert cr.feasible_interval(s, "x1", {"x2": 0.0}) == pytest.approx((2.0, 8.0))

    def test_no_constraints_returns_variable_bounds(self):
        s = _space()
        assert cr.feasible_interval(s, "x1", {"x2": 1.0}) == pytest.approx((0.0, 10.0))

    def test_unknown_variable_raises(self):
        s = _space()
        with pytest.raises(ValueError, match="not found"):
            cr.feasible_interval(s, "nope", {})

    def test_equality_intersects_rather_than_overwrites_prior_bounds(self):
        # x1 <= 3, then x1 == 4: the two constraints conflict, so the
        # interval must be empty. A buggy implementation that lets an
        # equality *overwrite* the accumulated (lo, hi) instead of
        # intersecting with it would wrongly report (4.0, 4.0) here,
        # ignoring the earlier x1 <= 3 bound entirely.
        s = _space()
        s.add_constraint("inequality", {"x1": 1.0}, rhs=3.0)
        s.add_constraint("equality", {"x1": 1.0}, rhs=4.0)
        assert cr.feasible_interval(s, "x1", {}) is None
