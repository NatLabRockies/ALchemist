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


def _lattice(s, n_levels=5):
    """Raw-space full-factorial lattice over the numeric variables."""
    import itertools as it
    numeric = cr.numeric_variables(s)
    axes = []
    for v in numeric:
        lo, hi = cr.variable_bounds(v)
        axes.append(np.linspace(lo, hi, n_levels))
    rows = [dict(zip([v["name"] for v in numeric], combo))
            for combo in it.product(*axes)]
    return pd.DataFrame(rows)


class TestAugmentWithBoundary:
    def test_every_returned_row_is_feasible(self):
        s = _space()
        s.add_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=7.0)
        out, _info = cr.augment_with_boundary(s, _lattice(s))
        assert len(out) > 0
        assert s.filter_feasible(out, rtol=0.0, atol=1e-9).all()

    def test_every_returned_row_is_feasible_asymmetric_coefficients(self):
        # Companion to the unit-coefficient case above: (3, 4) can't be
        # confused with an averaged or swapped-coefficient formula.
        s = _space()
        s.add_constraint("inequality", {"x1": 3.0, "x2": 4.0}, rhs=26.0)
        out, _info = cr.augment_with_boundary(s, _lattice(s))
        assert len(out) > 0
        assert s.filter_feasible(out, rtol=0.0, atol=1e-9).all()

    def test_boundary_points_exist_that_the_filtered_lattice_lacks(self):
        s = _space()
        s.add_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=7.0)
        lattice = _lattice(s)
        filtered = lattice[s.filter_feasible(lattice, rtol=0.0, atol=1e-9)]
        out, info = cr.augment_with_boundary(s, lattice)

        # Points lying ON the constraint (sum == 7) exist after augmentation.
        on_boundary = np.isclose(out["x1"] + out["x2"], 7.0, atol=1e-6).sum()
        assert on_boundary > 0
        assert len(out) > len(filtered)
        assert info["n_boundary_added"] + info["n_vertices_added"] > 0

    def test_boundary_points_exist_asymmetric_coefficients(self):
        # Same shape as above, but on 3*x1 + 4*x2 == 26 -- a boundary check
        # that only a real (not unit-coefficient) projection formula can
        # satisfy.
        s = _space()
        s.add_constraint("inequality", {"x1": 3.0, "x2": 4.0}, rhs=26.0)
        lattice = _lattice(s)
        filtered = lattice[s.filter_feasible(lattice, rtol=0.0, atol=1e-9)]
        out, info = cr.augment_with_boundary(s, lattice)

        on_boundary = np.isclose(3 * out["x1"] + 4 * out["x2"], 26.0, atol=1e-6).sum()
        assert on_boundary > 0
        assert len(out) > len(filtered)
        assert info["n_boundary_added"] + info["n_vertices_added"] > 0

    def test_info_dict_has_the_documented_keys(self):
        s = _space()
        s.add_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=7.0)
        _out, info = cr.augment_with_boundary(s, _lattice(s))
        assert set(info) == {
            "constraints_applied", "n_candidates_total", "n_candidates_feasible",
            "n_boundary_added", "n_vertices_added", "vertex_enumeration_skipped",
        }
        assert info["n_candidates_total"] == 25
        assert info["constraints_applied"] == ["constraint_0"]

    def test_no_constraints_returns_input_unchanged(self):
        s = _space()
        lattice = _lattice(s)
        out, info = cr.augment_with_boundary(s, lattice)
        pd.testing.assert_frame_equal(out, lattice)
        assert info["n_boundary_added"] == 0

    def test_empty_feasible_region_raises(self):
        s = _space()
        # x1 + x2 <= -1 is unreachable inside [0, 10]^2.
        s.add_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=-1.0)
        with pytest.raises(cr.InfeasibleRegionError, match="No feasible"):
            cr.augment_with_boundary(s, _lattice(s))

    def test_empty_feasible_region_raises_asymmetric_coefficients(self):
        # 3*x1 + 4*x2 <= -5 is unreachable inside [0, 10]^2 since the LHS is
        # always >= 0 there. Asymmetric coefficients rule out a sign-error
        # implementation that happens to still raise for the unit case.
        s = _space()
        s.add_constraint("inequality", {"x1": 3.0, "x2": 4.0}, rhs=-5.0)
        with pytest.raises(cr.InfeasibleRegionError, match="No feasible"):
            cr.augment_with_boundary(s, _lattice(s))

    def test_equality_constraint_yields_points_on_the_hyperplane(self):
        s = _space()
        s.add_constraint("equality", {"x1": 1.0, "x2": 1.0}, rhs=6.0)
        out, _info = cr.augment_with_boundary(s, _lattice(s))
        assert len(out) > 0
        assert np.allclose(out["x1"] + out["x2"], 6.0, atol=1e-6)

    def test_equality_constraint_asymmetric_coefficients(self):
        # 3*x1 - 2*x2 == 4. Asymmetric, signed coefficients: an
        # implementation that projects with an even split (rather than the
        # real weighted orthogonal formula) would not land exactly on this
        # hyperplane.
        s = _space()
        s.add_constraint("equality", {"x1": 3.0, "x2": -2.0}, rhs=4.0)
        out, _info = cr.augment_with_boundary(s, _lattice(s))
        assert len(out) > 0
        assert np.allclose(3 * out["x1"] - 2 * out["x2"], 4.0, atol=1e-6)

    def test_runs_per_categorical_combination(self):
        s = _space()
        s.add_variable("cat", "categorical", values=["a", "b"])
        s.add_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=7.0)
        lattice = _lattice(s)
        lattice = pd.concat([
            lattice.assign(cat="a"), lattice.assign(cat="b")
        ], ignore_index=True)
        out, _info = cr.augment_with_boundary(s, lattice)
        assert set(out["cat"]) == {"a", "b"}
        assert s.filter_feasible(out, rtol=0.0, atol=1e-9).all()

        # Strengthen: each categorical group must independently get its own
        # boundary points, not a merged/collapsed set. A grouping bug that
        # processes rows without holding the categorical fixed (e.g. drops
        # the group key, or only the first group's constraint work survives)
        # would leave one of these two empty.
        out_a = out[out["cat"] == "a"]
        out_b = out[out["cat"] == "b"]
        assert np.isclose(out_a["x1"] + out_a["x2"], 7.0, atol=1e-6).any()
        assert np.isclose(out_b["x1"] + out_b["x2"], 7.0, atol=1e-6).any()

    def test_runs_per_categorical_combination_asymmetric_coefficients(self):
        # Companion using 2*x1 + 5*x2 <= 14 (asymmetric, non-unit) so a
        # per-category boundary check can't be satisfied by coincidence.
        s = _space()
        s.add_variable("cat", "categorical", values=["a", "b"])
        s.add_constraint("inequality", {"x1": 2.0, "x2": 5.0}, rhs=14.0)
        lattice = _lattice(s)
        lattice = pd.concat([
            lattice.assign(cat="a"), lattice.assign(cat="b")
        ], ignore_index=True)
        out, _info = cr.augment_with_boundary(s, lattice)
        assert set(out["cat"]) == {"a", "b"}
        assert s.filter_feasible(out, rtol=0.0, atol=1e-9).all()

        out_a = out[out["cat"] == "a"]
        out_b = out[out["cat"] == "b"]
        assert np.isclose(2 * out_a["x1"] + 5 * out_a["x2"], 14.0, atol=1e-6).any()
        assert np.isclose(2 * out_b["x1"] + 5 * out_b["x2"], 14.0, atol=1e-6).any()

    def test_vertex_skip_is_reported_not_silent(self):
        s = SearchSpace()
        for i in range(6):
            s.add_variable(f"x{i}", "real", min=0.0, max=10.0)
        s.add_constraint("inequality", {"x0": 1.0, "x1": 1.0}, rhs=15.0)
        out, info = cr.augment_with_boundary(s, _lattice(s, n_levels=2),
                                             max_vertex_vars=5)
        assert info["vertex_enumeration_skipped"] is True
        assert info["n_vertices_added"] == 0
        assert len(out) > 0

    def test_projection_visits_every_constraint_not_just_the_first_violated(self):
        # A single hand-picked infeasible point that violates BOTH
        # constraints, where:
        #   - projecting onto c1 (registered first) lands on a point that
        #     STILL violates c2, so it must be rejected by the post-
        #     projection re-test;
        #   - projecting onto c2 (registered second) lands on a point that
        #     genuinely satisfies c1, so it must survive.
        # An implementation that projects only onto the first violated
        # constraint per row (rather than trying every constraint) would
        # never attempt the c2 projection for this row and would drop it
        # entirely, producing no output for this candidate set.
        #
        # Point P = (7.5, 0.5). c1: x1 + x2 <= 7. c2: 3*x1 - 2*x2 <= 5.
        #   P violates c1 (sum = 8 > 7) and c2 (21.5 > 5).
        #   project(P, c1) = (7.0, 0.0), which still has 3*7 - 2*0 = 21 > 5
        #   (violates c2) -- must be discarded.
        #   project(P, c2) = (48/13, 79/26) ~= (3.6923, 3.0385), which has
        #   x1 + x2 ~= 6.73 <= 7 (satisfies c1) -- must survive and land
        #   exactly on 3*x1 - 2*x2 == 5.
        #
        # max_vertex_vars=1 forces vertex_enumeration_skipped (2 numeric
        # variables > 1) so feasible_vertices() contributes nothing here --
        # the only possible source of output is the projection loop, which
        # isolates this check from that independent code path (vertex
        # enumeration would otherwise also happen to place a point on c2's
        # line, at its intersection with c1, masking this exact bug).
        s = _space()
        s.add_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=7.0, name="c1")
        s.add_constraint("inequality", {"x1": 3.0, "x2": -2.0}, rhs=5.0, name="c2")
        points = pd.DataFrame([{"x1": 7.5, "x2": 0.5}])

        out, info = cr.augment_with_boundary(s, points, max_vertex_vars=1)

        assert len(out) > 0
        assert s.filter_feasible(out, rtol=0.0, atol=1e-9).all()
        expected_b = np.isclose(out["x1"], 48 / 13, atol=1e-4) & np.isclose(out["x2"], 79 / 26, atol=1e-4)
        assert expected_b.any(), (
            "expected the hand-solved projection of P onto c2, "
            "(48/13, 79/26); an implementation that projects only onto the "
            "first violated constraint per row would drop this row entirely"
        )
        rejected_c1_only = np.isclose(out["x1"], 7.0, atol=1e-6) & np.isclose(out["x2"], 0.0, atol=1e-6)
        assert not rejected_c1_only.any(), (
            "the c1-only projection (7.0, 0.0) still violates c2 and must "
            "not appear in the output"
        )

    def test_dedup_counts_reflect_removed_duplicates_not_raw_additions(self):
        # A single infeasible point whose projection onto c1 lands EXACTLY
        # on a vertex that feasible_vertices() will also independently
        # produce, so the two additions collide and one is deduped away.
        #
        # Point P = (9, 2). c1: x1 + x2 <= 7 (feasible triangle over
        # [0,10]^2 has vertices (0,0), (7,0), (0,7)).
        # project(P, c1) = (7.0, 0.0) -- exactly the vertex where c1 meets
        # the x2 == 0 face.
        #
        # UPDATED to pin the CORRECT dedup-aware behavior. The implementation
        # tags each assembled row with its provenance (feasible / boundary /
        # vertex) and counts survivors by tag after dedup, with boundary
        # taking precedence over vertex on an exact collision (the surviving
        # (7, 0) row is attributed to the boundary projection, since parts
        # are concatenated feasible-then-boundary-then-vertex and
        # `duplicated()` keeps the first occurrence). This test previously
        # pinned the OLD, buggy global-delta formula
        # (`n_boundary_added = max(0, n_boundary - removed)` computed from a
        # single post-dedup row-count delta across the *entire* assembled
        # set), which asserted `n_boundary_added == 0` here -- wrong, because
        # it debited the collision from the boundary count even though the
        # projected point (7, 0) is genuinely present in the output. The
        # correct count is 1: the point survives and is boundary-provenance.
        s = _space()
        s.add_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=7.0)
        points = pd.DataFrame([{"x1": 9.0, "x2": 2.0}])

        out, info = cr.augment_with_boundary(s, points)

        assert len(out) > 0
        assert s.filter_feasible(out, rtol=0.0, atol=1e-9).all()
        vertex_present = np.isclose(out["x1"], 7.0, atol=1e-6) & np.isclose(out["x2"], 0.0, atol=1e-6)
        assert vertex_present.sum() == 1, "the collided point must appear exactly once after dedup"
        assert info["n_boundary_added"] == 1, (
            "the projected point (7, 0) genuinely survives in the output; "
            "it must be counted even though it coincides with a vertex that "
            "feasible_vertices() would also have produced"
        )
        assert info["n_vertices_added"] == 2, (
            "raw vertex enumeration finds 3 vertices ((0,0), (7,0), (0,7)), "
            "but (7,0) was deduped away against the boundary projection that "
            "landed on the same point, so only 2 vertex-tagged rows survive "
            "-- the count must reflect survival, not the raw pre-dedup total"
        )
        # The three counted categories must exactly reconstruct the frame.
        assert (info["n_candidates_feasible"] + info["n_boundary_added"]
                + info["n_vertices_added"]) == len(out)

    def test_unrelated_duplicate_collision_does_not_debit_boundary_count(self):
        # Reviewer's concrete counter-example for the original defect: two
        # duplicate already-feasible input rows (0, 0) plus one infeasible
        # point (10, 10) whose projection onto c1 lands on a boundary point
        # (3.5, 3.5) that is unique -- it does not coincide with any vertex
        # or other row. The dedup collision between the two (0, 0) rows is
        # entirely unrelated to the boundary addition, so it must not affect
        # n_boundary_added at all.
        s = _space()
        s.add_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=7.0)
        points = pd.DataFrame([
            {"x1": 0.0, "x2": 0.0},
            {"x1": 0.0, "x2": 0.0},
            {"x1": 10.0, "x2": 10.0},
        ])

        out, info = cr.augment_with_boundary(s, points)

        boundary_present = np.isclose(out["x1"], 3.5, atol=1e-6) & np.isclose(out["x2"], 3.5, atol=1e-6)
        assert boundary_present.sum() == 1, "the unique boundary point must survive in the output"
        assert info["n_boundary_added"] == 1, (
            "the boundary point (3.5, 3.5) is unique and survives -- the "
            "unrelated (0, 0)/(0, 0) duplicate collision among the "
            "pre-existing feasible rows must not be debited from it"
        )
        # The duplicate feasible rows collapse to one surviving feasible row.
        assert info["n_candidates_feasible"] == 1
        assert (info["n_candidates_feasible"] + info["n_boundary_added"]
                + info["n_vertices_added"]) == len(out)

    def test_returned_columns_exactly_match_input_columns(self):
        # The provenance tag used internally to compute dedup-aware counts
        # must never leak into the returned frame.
        s = _space()
        s.add_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=7.0)
        out, _info = cr.augment_with_boundary(s, _lattice(s))
        assert list(out.columns) == list(_lattice(s).columns)
