"""Unit tests for the linear-constrained-region geometry helpers."""

import numpy as np
import pandas as pd
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

    def test_free_variable_non_unit_coefficient_pins_the_division(self):
        # 3*x1 <= 12  ->  x1 <= 4. Only x1 participates (rest == 0), so this
        # isolates the free-variable division: limit = (rhs - rest) / c_v.
        # A sign-only stand-in that never divides (limit = diff if c_v > 0
        # else -diff) would instead give limit = 12, i.e. hi = 10 (unbounded
        # by the variable's own range) -- a different, wrong answer.
        s = _space()
        s.add_constraint("inequality", {"x1": 3.0}, rhs=12.0)
        assert cr.feasible_interval(s, "x1", {}) == pytest.approx((0.0, 4.0))

    def test_free_variable_negative_non_unit_coefficient_pins_division_and_sign(self):
        # -2*x1 <= -3  ->  x1 >= 1.5. Exercises the sign-flip-to-lower-bound
        # path together with a non-unit magnitude. The sign-only stand-in
        # (limit = -diff = 3) would give lo = 3.0 instead of the correct 1.5.
        s = _space()
        s.add_constraint("inequality", {"x1": -2.0}, rhs=-3.0)
        assert cr.feasible_interval(s, "x1", {}) == pytest.approx((1.5, 10.0))

    def test_fixed_variable_non_unit_coefficient_pins_the_rest_multiplication(self):
        # x1 + 4*x2 <= 10, x2 fixed at 1  ->  rest = 4*1 = 4  ->  x1 <= 6.
        # The free variable's own coefficient is 1, so this isolates the
        # `coeff * fixed_value` term in `rest`. Dropping that multiplication
        # (summing raw fixed values instead) would give rest = 1 and
        # x1 <= 9 -- a different, wrong answer.
        s = _space()
        s.add_constraint("inequality", {"x1": 1.0, "x2": 4.0}, rhs=10.0)
        assert cr.feasible_interval(s, "x1", {"x2": 1.0}) == pytest.approx((0.0, 6.0))

    def test_asymmetric_free_and_fixed_coefficients_catch_swap_or_average_bugs(self):
        # 3*x1 + 4*x2 <= 26, x2 fixed at 2  ->  rest = 4*2 = 8
        #                                   ->  x1 <= (26 - 8) / 3 = 6.
        # Free and fixed coefficients are deliberately different (3 vs 4),
        # so a bug that swaps which coefficient divides vs. multiplies, or
        # that averages the two, would produce a value other than 6.0.
        s = _space()
        s.add_constraint("inequality", {"x1": 3.0, "x2": 4.0}, rhs=26.0)
        assert cr.feasible_interval(s, "x1", {"x2": 2.0}) == pytest.approx((0.0, 6.0))

    def test_negative_free_coefficient_with_non_unit_fixed_coefficient(self):
        # -2*x1 + 3*x2 <= 4, x2 fixed at 2  ->  rest = 3*2 = 6
        #                                   ->  -2*x1 <= 4 - 6 = -2
        #                                   ->  x1 >= (4 - 6) / -2 = 1.0.
        # Combines the sign-flip path with non-unit magnitude on both the
        # free and fixed coefficients, so it independently discriminates
        # both the division stand-in (would give lo = 2.0) and the
        # unmultiplied-rest bug (would give lo = 0.0, i.e. unbounded).
        s = _space()
        s.add_constraint("inequality", {"x1": -2.0, "x2": 3.0}, rhs=4.0)
        assert cr.feasible_interval(s, "x1", {"x2": 2.0}) == pytest.approx((1.0, 10.0))


class TestSnapToVariable:
    def test_real_clips_to_bounds_and_passes_through_interior_values(self):
        var = {"name": "x1", "type": "real", "min": 0.0, "max": 10.0}
        assert cr.snap_to_variable(-3.0, var) == pytest.approx(0.0)
        assert cr.snap_to_variable(15.0, var) == pytest.approx(10.0)
        assert cr.snap_to_variable(4.5, var) == pytest.approx(4.5)

    def test_integer_rounds_to_nearest_after_clipping(self):
        # Exercises rounding (not truncation) and rounding-after-clip, not
        # just clipping: a snap that clips but forgets to round would leave
        # 4.6 as 4.6 instead of 5.0.
        var = {"name": "i1", "type": "integer", "min": 0, "max": 10}
        assert cr.snap_to_variable(4.4, var) == 4.0
        assert cr.snap_to_variable(4.6, var) == 5.0
        assert cr.snap_to_variable(-3.2, var) == 0.0
        assert isinstance(cr.snap_to_variable(4.4, var), float)

    def test_discrete_snaps_to_nearest_allowed_value(self):
        # A snap that clips to [min(allowed), max(allowed)] but forgets the
        # nearest-allowed-value step would leave 4.0 or 5.5 unsnapped.
        var = {"name": "d1", "type": "discrete", "allowed_values": [0.0, 3.0, 7.0, 10.0]}
        assert cr.snap_to_variable(4.0, var) == 3.0
        assert cr.snap_to_variable(5.5, var) == 7.0
        assert cr.snap_to_variable(-5.0, var) == 0.0


class TestDedupe:
    def test_near_duplicate_rows_within_tolerance_collapse(self):
        # Raw-float comparison would keep both; rounding to a tolerance
        # grid must collapse them.
        df = pd.DataFrame({"x1": [3.0, 3.0 + 1e-10], "x2": [4.0, 4.0]})
        out = cr._dedupe(df, ["x1", "x2"])
        assert len(out) == 1

    def test_distinct_rows_are_kept(self):
        df = pd.DataFrame({"x1": [3.0, 3.1], "x2": [4.0, 4.0]})
        out = cr._dedupe(df, ["x1", "x2"])
        assert len(out) == 2

    def test_empty_frame_returns_empty(self):
        df = pd.DataFrame({"x1": [], "x2": []})
        out = cr._dedupe(df, ["x1", "x2"])
        assert out.empty


class TestFeasibleVertices:
    def test_triangle_vertices_are_found(self):
        # x1, x2 in [0, 10] with x1 + x2 <= 10 is the triangle
        # (0,0), (10,0), (0,10).
        s = _space()
        s.add_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=10.0)
        df = cr.feasible_vertices(s)
        found = {(round(r.x1, 6), round(r.x2, 6)) for r in df.itertuples()}
        assert {(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)} <= found

    def test_triangle_vertex_count_after_dedup(self):
        # Several plane-pair combinations solve to the same corner (e.g.
        # the constraint intersected with x1's bound face coincides with a
        # bound-only corner). A dedupe that doesn't collapse them would
        # leave duplicate rows for the same physical vertex.
        s = _space()
        s.add_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=10.0)
        df = cr.feasible_vertices(s)
        assert len(df) == 3

    def test_triangle_vertices_with_asymmetric_coefficients(self):
        # 2*x1 + 5*x2 <= 14 cuts a triangle whose non-origin vertices sit
        # at x2=0 -> x1=7 and x1=0 -> x2=2.8. Unit coefficients can't
        # distinguish a formula that divides by the wrong coefficient or
        # swaps which axis gets which intercept; this pins the real solve.
        s = _space()
        s.add_constraint("inequality", {"x1": 2.0, "x2": 5.0}, rhs=14.0)
        df = cr.feasible_vertices(s)
        found = {(round(r.x1, 6), round(r.x2, 6)) for r in df.itertuples()}
        assert {(0.0, 0.0), (7.0, 0.0), (0.0, 2.8)} <= found

    def test_every_returned_vertex_is_feasible(self):
        s = _space()
        s.add_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=7.0)
        df = cr.feasible_vertices(s)
        assert len(df) > 0
        assert s.filter_feasible(df, rtol=0.0, atol=1e-9).all()

    def test_no_constraints_returns_empty(self):
        s = _space()
        assert cr.feasible_vertices(s).empty

    def test_above_max_vars_returns_empty(self):
        s = _space()
        s.add_variable("x3", "real", min=0.0, max=10.0)
        s.add_constraint("inequality", {"x1": 1.0, "x2": 1.0, "x3": 1.0}, rhs=10.0)
        assert cr.feasible_vertices(s, max_vars=2).empty
        assert not cr.feasible_vertices(s, max_vars=3).empty

    def test_fixed_categorical_is_attached_to_every_vertex(self):
        s = _space()
        s.add_variable("cat", "categorical", values=["a", "b"])
        s.add_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=10.0)
        df = cr.feasible_vertices(s, fixed={"cat": "b"})
        assert len(df) > 0
        assert (df["cat"] == "b").all()

    def test_integer_vertices_are_rounded_and_still_feasible(self):
        s = SearchSpace()
        s.add_variable("i1", "integer", min=0, max=10)
        s.add_variable("i2", "integer", min=0, max=10)
        s.add_constraint("inequality", {"i1": 1.0, "i2": 1.0}, rhs=7.5)
        df = cr.feasible_vertices(s)
        assert len(df) > 0
        assert (df["i1"] == df["i1"].round()).all()
        assert s.filter_feasible(df, rtol=0.0, atol=1e-9).all()

    def test_integer_rounding_that_becomes_infeasible_is_excluded(self):
        # 3*i1 + 7*i2 <= 20, i1/i2 integer in [0, 10]. The continuous
        # vertices at the axes are (20/3, 0) ~= (6.667, 0) and
        # (0, 20/7) ~= (0, 2.857). Naive rounding gives (7, 0) and (0, 3),
        # both of which violate the constraint (21 > 20). An
        # implementation that skips the post-snap feasibility re-test
        # would ship these two infeasible points; the only vertex that
        # survives snapping is the origin.
        s = SearchSpace()
        s.add_variable("i1", "integer", min=0, max=10)
        s.add_variable("i2", "integer", min=0, max=10)
        s.add_constraint("inequality", {"i1": 3.0, "i2": 7.0}, rhs=20.0)
        df = cr.feasible_vertices(s)
        found = {(round(r.i1, 6), round(r.i2, 6)) for r in df.itertuples()}
        assert (7.0, 0.0) not in found
        assert (0.0, 3.0) not in found
        assert found == {(0.0, 0.0)}
        assert s.filter_feasible(df, rtol=0.0, atol=1e-9).all()

    def test_equality_constraint_vertices_lie_on_the_hyperplane(self):
        s = _space()
        s.add_constraint("equality", {"x1": 1.0, "x2": 1.0}, rhs=6.0)
        df = cr.feasible_vertices(s)
        assert len(df) > 0
        assert np.allclose(df["x1"] + df["x2"], 6.0, atol=1e-6)

    def test_equality_constraint_with_asymmetric_coefficients(self):
        s = _space()
        s.add_constraint("equality", {"x1": 3.0, "x2": -2.0}, rhs=4.0)
        df = cr.feasible_vertices(s)
        assert len(df) > 0
        assert np.allclose(3 * df["x1"] - 2 * df["x2"], 4.0, atol=1e-6)

    def test_axis_aligned_constraint_parallel_to_bound_face_is_handled(self):
        # 1*x1 + 0*x2 <= 6 is parallel to the x1=0/x1=10 bound faces, so
        # pairing it with either produces a singular 2x2 system that must
        # be skipped rather than solved into a spurious point. A filter
        # that is too loose would let a near-singular solve through and
        # produce garbage; one that is too tight could drop legitimate
        # vertices elsewhere. The correct feasible region is the rectangle
        # x1 in [0, 6], x2 in [0, 10], with exactly these four corners.
        s = _space()
        s.add_constraint("inequality", {"x1": 1.0}, rhs=6.0)
        df = cr.feasible_vertices(s)
        found = {(round(r.x1, 6), round(r.x2, 6)) for r in df.itertuples()}
        assert found == {(0.0, 0.0), (0.0, 10.0), (6.0, 0.0), (6.0, 10.0)}
