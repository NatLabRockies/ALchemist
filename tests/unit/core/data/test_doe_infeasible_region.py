"""A constrained space-filling design must not search for what cannot exist.

``generate_initial_design``'s space-filling branch discovered an infeasible
constraint set only by failing to sample its way out of it: it escalated the
oversampling factor 4 -> 16 -> 64 -> 256 -> 1024 -> 4096, drawing
``n_points * factor`` samples each round, and raised only after the last one.
The final round is a single ``n_points * 4096`` batch, and ``maximin`` LHS
costs time quadratic in the batch size, so ``n_points=6`` on an empty region
spent minutes proving something a linear program settles in milliseconds.

``generate_optimal_design`` has always refused the empty case up front
(``augment_with_boundary`` raises ``InfeasibleRegionError``). These tests pin
the space-filling path having learned the same check -- and, just as
importantly, pin the two ways the check must NOT overreach:

- a region that is merely **small** must still produce a design, because a
  spurious failure on a solvable design is worse than the slow path ever was;
- a region whose *continuous hull* is non-empty but whose **integer lattice**
  is empty must not be reported as empty, because the linear program only ever
  proves things about the hull.

Feasibility is asserted by hand (``sum(coeff * value)``) rather than through
``filter_feasible``, so a test cannot agree with a broken predicate.
"""

import time

import numpy as np
import pandas as pd
import pytest

from alchemist_core.data.search_space import SearchSpace
from alchemist_core.utils.doe import (
    SPACE_FILLING_METHODS,
    generate_initial_design,
    _HOPELESS_EXPECTED_HITS,
    _RESAMPLE_FACTORS,
)
from alchemist_core.utils.constrained_region import (
    DOE_ATOL,
    InfeasibleRegionError,
    region_is_provably_empty,
    region_is_provably_measure_zero,
)

# The bound generate_initial_design actually passes, derived the same way it
# derives it -- restating a literal here would let the test agree with a
# budget the production call no longer uses.
MIN_REACHABLE_FRACTION = _HOPELESS_EXPECTED_HITS / sum(_RESAMPLE_FACTORS)

# The equality span either side of which the verdict flips, from the same
# quantities: fraction ~ 2*atol/span, so the cut is at 2*atol/bound.
CUT_SPAN = (2.0 * DOE_ATOL) / MIN_REACHABLE_FRACTION


def _measure_zero(space):
    return region_is_provably_measure_zero(
        space, atol=DOE_ATOL, min_reachable_fraction=MIN_REACHABLE_FRACTION
    )

METHODS = sorted(SPACE_FILLING_METHODS)

# Generous enough that a loaded machine cannot trip it, tight enough that the
# escalation it replaces (minutes) could never pass. The measured cost of the
# checks themselves is ~1-2 ms.
FAST_SECONDS = 5.0

# n_points is a power of two throughout: skopt's Sobol sampler emits a
# UserWarning for any other count, and this branch tracks its warning count.
N = 8


def _lhs_by_hand(point, coefficients):
    return sum(float(c) * float(point[name]) for name, c in coefficients.items())


# ============================================================
# The region really is empty -- refuse immediately
# ============================================================

class TestProvablyEmptyRegionIsRefusedUpFront:

    @pytest.mark.parametrize("method", METHODS)
    def test_every_space_filling_method_fails_fast(self, method):
        """Not just 'it raises' -- it raises *quickly*, and says why."""
        space = SearchSpace()
        space.add_variable("x1", "real", min=0.0, max=10.0)
        space.add_variable("x2", "real", min=0.0, max=10.0)
        # Asymmetric, non-unit coefficients: 2.5*x1 + 1.5*x2 <= -8 cannot hold
        # anywhere in a box whose minimum for that expression is 0.
        space.add_constraint("inequality", {"x1": 2.5, "x2": 1.5}, -8.0)

        started = time.perf_counter()
        with pytest.raises(InfeasibleRegionError) as exc:
            generate_initial_design(space, method=method, n_points=N, random_seed=5)
        elapsed = time.perf_counter() - started

        assert elapsed < FAST_SECONDS, (
            f"method={method}: an empty region took {elapsed:.1f}s to refuse; "
            f"the escalation is still running"
        )
        assert "no feasible point" in str(exc.value)

    def test_the_message_blames_the_constraints_not_n_points(self):
        """The old message told the caller to 'reduce n_points', which cannot help."""
        space = SearchSpace()
        space.add_variable("x1", "real", min=0.0, max=10.0)
        space.add_variable("x2", "real", min=0.0, max=10.0)
        space.add_constraint("inequality", {"x1": 2.5, "x2": 1.5}, -8.0)

        with pytest.raises(InfeasibleRegionError) as exc:
            generate_initial_design(space, method="lhs", n_points=N, random_seed=5)

        message = str(exc.value)
        assert "relax the constraints" in message
        assert "reduce n_points" not in message
        # It must not hedge with the "may be very small" wording reserved for
        # the case the checks could not decide.
        assert "may be very small" not in message

    def test_emptiness_only_two_constraints_together_create(self):
        """Neither constraint is empty alone: an interval check cannot see this.

        ``2.5*x1 + 1.5*x2 <= 6`` caps ``2*x1 + 3*x2`` at 12 over the box
        (its maximum sits at the vertex ``x2 = 4``), so requiring
        ``2*x1 + 3*x2 >= 18`` as well leaves nothing. Both constraints have
        plenty of room on their own.
        """
        cap = {"x1": 2.5, "x2": 1.5}
        floor = {"x1": -2.0, "x2": -3.0}

        alone_cap = SearchSpace()
        alone_cap.add_variable("x1", "real", min=0.0, max=10.0)
        alone_cap.add_variable("x2", "real", min=0.0, max=10.0)
        alone_cap.add_constraint("inequality", cap, 6.0)
        assert not region_is_provably_empty(alone_cap)

        alone_floor = SearchSpace()
        alone_floor.add_variable("x1", "real", min=0.0, max=10.0)
        alone_floor.add_variable("x2", "real", min=0.0, max=10.0)
        alone_floor.add_constraint("inequality", floor, -18.0)
        assert not region_is_provably_empty(alone_floor)

        both = SearchSpace()
        both.add_variable("x1", "real", min=0.0, max=10.0)
        both.add_variable("x2", "real", min=0.0, max=10.0)
        both.add_constraint("inequality", cap, 6.0)
        both.add_constraint("inequality", floor, -18.0)
        assert region_is_provably_empty(both)

        started = time.perf_counter()
        with pytest.raises(InfeasibleRegionError):
            generate_initial_design(both, method="lhs", n_points=N, random_seed=2)
        assert time.perf_counter() - started < FAST_SECONDS

    def test_an_integer_space_with_an_empty_hull_is_refused(self):
        """The variable type is `integer`, not `real` -- same verdict."""
        space = SearchSpace()
        space.add_variable("x1", "integer", min=2, max=9)
        space.add_variable("x2", "integer", min=3, max=7)
        # Minimum of 3*x1 + 4*x2 over the box is 3*2 + 4*3 = 18 > 11.
        space.add_constraint("inequality", {"x1": 3.0, "x2": 4.0}, 11.0)

        started = time.perf_counter()
        with pytest.raises(InfeasibleRegionError):
            generate_initial_design(space, method="random", n_points=N, random_seed=8)
        assert time.perf_counter() - started < FAST_SECONDS

    def test_a_discrete_space_with_an_empty_hull_is_refused(self):
        space = SearchSpace()
        space.add_variable("x1", "discrete", allowed_values=[1.0, 2.5, 4.0])
        space.add_variable("x2", "discrete", allowed_values=[0.5, 1.5])
        # Minimum of 2*x1 + 6*x2 is 2*1 + 6*0.5 = 5 > 4.
        space.add_constraint("inequality", {"x1": 2.0, "x2": 6.0}, 4.0)

        with pytest.raises(InfeasibleRegionError):
            generate_initial_design(space, method="halton", n_points=N, random_seed=8)

    def test_a_categorical_alongside_does_not_confuse_the_check(self):
        """Categoricals carry a dimension but never appear in a constraint."""
        space = SearchSpace()
        space.add_variable("x1", "real", min=0.0, max=4.0)
        space.add_variable("x4", "categorical", values=["a", "b", "c"])
        space.add_variable("x2", "integer", min=0, max=6)
        space.add_constraint("inequality", {"x1": -1.5, "x2": -2.0}, -30.0)

        with pytest.raises(InfeasibleRegionError):
            generate_initial_design(space, method="sobol", n_points=N, random_seed=8)


# ============================================================
# The region is small, or only looks empty -- keep working
# ============================================================

class TestASolvableDesignIsStillSolved:

    @pytest.mark.parametrize("method", METHODS)
    def test_a_few_percent_of_the_box_still_produces_a_design(self, method):
        """~2 % of [0,10]^2. The escalation must still be allowed to run."""
        coefficients = {"x1": -1.0, "x2": -1.0}
        space = SearchSpace()
        space.add_variable("x1", "real", min=0.0, max=10.0)
        space.add_variable("x2", "real", min=0.0, max=10.0)
        space.add_constraint("inequality", coefficients, -18.0)  # x1 + x2 >= 18

        points = generate_initial_design(
            space, method=method, n_points=N, random_seed=17
        )

        assert len(points) == N
        for point in points:
            assert _lhs_by_hand(point, coefficients) <= -18.0 + 1e-9, point

    def test_a_thin_band_between_two_constraints_still_produces_a_design(self):
        """Feasible but narrow, and bounded on both sides."""
        lower = {"x1": -3.0, "x2": -1.0}
        upper = {"x1": 3.0, "x2": 1.0}
        space = SearchSpace()
        space.add_variable("x1", "real", min=0.0, max=10.0)
        space.add_variable("x2", "real", min=0.0, max=10.0)
        space.add_constraint("inequality", lower, -12.0)   # 3*x1 + x2 >= 12
        space.add_constraint("inequality", upper, 16.0)    # 3*x1 + x2 <= 16

        points = generate_initial_design(
            space, method="random", n_points=N, random_seed=23
        )

        assert len(points) == N
        for point in points:
            value = _lhs_by_hand(point, upper)
            assert 12.0 - 1e-9 <= value <= 16.0 + 1e-9, point

    def test_an_empty_integer_lattice_inside_a_non_empty_hull_is_not_called_empty(self):
        """The linear program reasons about the hull, so it must not claim more.

        ``4*(x1 + x2)`` must land in ``[0.8, 3.2]``. Over the continuous hull
        that is satisfiable (``x1 = 0.5, x2 = 0`` gives 2.0), but every integer
        pair gives ``x1 + x2`` an integer, so the product is 0 or >= 4 -- never
        inside the band. The check must return "not proven" and let the sampler
        reach its own, differently-worded, conclusion.
        """
        space = SearchSpace()
        space.add_variable("x1", "integer", min=0, max=6)
        space.add_variable("x2", "integer", min=0, max=6)
        space.add_constraint("inequality", {"x1": 4.0, "x2": 4.0}, 3.2)
        space.add_constraint("inequality", {"x1": -4.0, "x2": -4.0}, -0.8)

        assert region_is_provably_empty(space) is False

        with pytest.raises(ValueError) as exc:
            generate_initial_design(space, method="random", n_points=4, random_seed=9)

        # Not the up-front verdict: the sampler's own give-up message, which
        # hedges precisely because nothing was proved.
        assert not isinstance(exc.value, InfeasibleRegionError)
        assert "may be very small" in str(exc.value)

    @pytest.mark.parametrize("method", METHODS)
    def test_an_unconstrained_design_never_reaches_the_checks(self, method):
        space = SearchSpace()
        space.add_variable("x1", "real", min=0.0, max=10.0)
        space.add_variable("x2", "integer", min=1, max=9)

        points = generate_initial_design(
            space, method=method, n_points=N, random_seed=4
        )

        assert len(points) == N
        for point in points:
            assert 0.0 <= point["x1"] <= 10.0
            assert 1 <= point["x2"] <= 9


# ============================================================
# A zero-volume region is not an empty one
# ============================================================

class TestAnEqualityOverContinuousVariables:

    @pytest.mark.parametrize("method", METHODS)
    def test_is_refused_immediately_and_separately(self, method):
        space = SearchSpace()
        space.add_variable("x1", "real", min=0.0, max=10.0)
        space.add_variable("x2", "real", min=0.0, max=10.0)
        space.add_constraint("equality", {"x1": 2.0, "x2": 3.0}, 12.0)

        started = time.perf_counter()
        with pytest.raises(InfeasibleRegionError) as exc:
            generate_initial_design(space, method=method, n_points=N, random_seed=6)
        elapsed = time.perf_counter() - started

        assert elapsed < FAST_SECONDS, (
            f"method={method}: a zero-volume region took {elapsed:.1f}s"
        )
        message = str(exc.value)
        # The distinction the caller needs: the region exists, it just cannot
        # be sampled -- and there is a method that can reach it.
        assert "zero-volume" in message
        assert "not empty" in message
        assert "optimal" in message
        assert "no feasible point" not in message

    def test_a_discrete_equality_is_left_alone_and_still_succeeds(self):
        """The grid intersects the constraint, so the design is reachable."""
        coefficients = {"x3": 1.0}
        space = SearchSpace()
        space.add_variable("x1", "real", min=0.0, max=5.0)
        space.add_variable("x3", "discrete", allowed_values=[1.0, 2.0, 4.0])
        space.add_constraint("equality", coefficients, 2.0)

        assert _measure_zero(space) is False

        points = generate_initial_design(
            space, method="random", n_points=4, random_seed=13
        )
        assert len(points) == 4
        for point in points:
            assert abs(_lhs_by_hand(point, coefficients) - 2.0) <= 1e-9, point

    def test_an_integer_equality_is_left_alone_and_still_succeeds(self):
        coefficients = {"x2": 2.0}
        space = SearchSpace()
        space.add_variable("x2", "integer", min=0, max=8)
        space.add_variable("x1", "real", min=0.0, max=5.0)
        space.add_constraint("equality", coefficients, 6.0)  # x2 == 3

        assert _measure_zero(space) is False

        points = generate_initial_design(
            space, method="random", n_points=4, random_seed=21
        )
        assert len(points) == 4
        for point in points:
            assert point["x2"] == 3, point

    @pytest.mark.parametrize("coefficient,lo,hi", [
        # A coefficient far below tolerance over a full-width range...
        (1e-11, 0.0, 10.0),
        # ...and an ordinary coefficient over a range narrower than the slab.
        (1.0, 2.9999999995, 3.0000000005),
    ])
    def test_an_equality_the_box_satisfies_everywhere_still_produces_a_design(
        self, coefficient, lo, hi
    ):
        """Non-zero is not the same as *able to move lhs out of the slab*.

        ``filter_feasible`` accepts the whole band ``|lhs - rhs| <= atol``. If
        the equality's ``real`` terms cannot vary ``lhs`` by more than that
        band, every point in the box satisfies it: the feasible region has
        *full* volume, not zero. Declaring measure-zero here would refuse a
        design that is feasible everywhere. ``add_constraint`` checks only that
        coefficients are finite, so both shapes are reachable over REST.
        """
        space = SearchSpace()
        space.add_variable("x1", "real", min=lo, max=hi)
        space.add_variable("x2", "real", min=0.0, max=10.0)
        space.add_constraint("equality", {"x1": coefficient}, coefficient * lo)

        assert _measure_zero(space) is False
        points = generate_initial_design(
            space, method="random", n_points=N, random_seed=2
        )
        assert len(points) == N
        for point in points:
            assert lo <= point["x1"] <= hi, point

    @pytest.mark.parametrize("coefficient,expected_percent", [
        (1e-09, 9.7),   # slab covers ~10% of the box
        (4e-09, 2.7),   # ~2.5%
        (1e-10, 100.0),  # the equality holds everywhere
    ])
    @pytest.mark.parametrize("method", METHODS)
    def test_a_narrow_equality_the_loop_can_actually_sample_returns_a_design(
        self, method, coefficient, expected_percent
    ):
        """Span is large; the *reachable fraction* is not small. Sample it.

        These spans (1e-8, 4e-8) sit far above any small absolute threshold,
        yet each leaves several percent of the box feasible and every sampler
        found eight points in milliseconds before this predicate existed.
        Gating on span alone refused them outright -- a regression against the
        task's own base, and a contradiction of ``filter_feasible``, which is
        the sole definition of feasibility and has no "wide regions only"
        clause.
        """
        coefficients = {"x1": coefficient}
        space = SearchSpace()
        space.add_variable("x1", "real", min=0.0, max=10.0)
        space.add_variable("x2", "real", min=0.0, max=10.0)
        space.add_constraint("equality", coefficients, 0.0)

        # The claim about the box is measured here, not asserted from theory.
        probe = pd.DataFrame({
            "x1": np.linspace(0.0, 10.0, 4000),
            "x2": np.linspace(0.0, 10.0, 4000),
        })
        measured = 100.0 * space.filter_feasible(
            probe, rtol=0.0, atol=DOE_ATOL
        ).mean()
        assert measured == pytest.approx(expected_percent, abs=1.5), (
            f"the premise moved: {measured:.2f}% of the box is feasible"
        )

        assert _measure_zero(space) is False
        points = generate_initial_design(
            space, method=method, n_points=N, random_seed=4
        )
        assert len(points) == N
        for point in points:
            assert abs(_lhs_by_hand(point, coefficients)) <= DOE_ATOL, point

    def test_the_verdict_flips_at_the_reachability_cut_and_not_before(self):
        """Both sides of the boundary, at the cut the production call uses.

        Below the cut the predicate stays silent and the resample loop is left
        to try -- which is what the base behaviour did. Above it, the region is
        refused immediately.
        """
        def _space(span):
            space = SearchSpace()
            # One real variable of width 10, so coefficient = span / 10.
            space.add_variable("x1", "real", min=0.0, max=10.0)
            space.add_variable("x2", "real", min=0.0, max=10.0)
            space.add_constraint("equality", {"x1": span / 10.0}, 0.0)
            return space

        assert _measure_zero(_space(CUT_SPAN * 0.5)) is False
        assert _measure_zero(_space(CUT_SPAN * 2.0)) is True

        # And the two sides reach different endings, not merely different
        # predicate values.
        with pytest.raises(InfeasibleRegionError):
            generate_initial_design(_space(CUT_SPAN * 2.0), method="random",
                                    n_points=4, random_seed=5)
        with pytest.raises(ValueError) as exc:
            generate_initial_design(_space(CUT_SPAN * 0.5), method="random",
                                    n_points=4, random_seed=5)
        assert not isinstance(exc.value, InfeasibleRegionError)
        assert "may be very small" in str(exc.value)

    def test_the_cut_sits_well_below_what_the_resample_loop_can_reach(self):
        """The margin is the point, so pin that it exists and its direction."""
        loop_can_reach = 1.0 / sum(_RESAMPLE_FACTORS)
        assert MIN_REACHABLE_FRACTION < loop_can_reach, (
            "the cut must sit below the fraction the loop can assemble a "
            "design from, or it refuses designs the loop would have found"
        )
        assert loop_can_reach / MIN_REACHABLE_FRACTION >= 50

    def test_an_ordinary_real_equality_is_still_measure_zero(self):
        """The span gate must not have disarmed the check it guards."""
        space = SearchSpace()
        space.add_variable("x1", "real", min=0.0, max=10.0)
        space.add_variable("x2", "real", min=0.0, max=10.0)
        space.add_constraint("equality", {"x1": 2.0, "x2": 3.0}, 12.0)
        assert _measure_zero(space) is True

    def test_a_zero_coefficient_on_the_real_variable_does_not_trigger_it(self):
        """A term that contributes nothing constrains nothing."""
        space = SearchSpace()
        space.add_variable("x1", "real", min=0.0, max=5.0)
        space.add_variable("x2", "integer", min=0, max=8)
        space.add_constraint("equality", {"x1": 0.0, "x2": 2.0}, 6.0)

        assert _measure_zero(space) is False


# ============================================================
# The predicates themselves
# ============================================================

class TestThePredicatesRefuseToGuess:

    def test_no_constraints_is_not_empty(self):
        space = SearchSpace()
        space.add_variable("x1", "real", min=0.0, max=1.0)
        assert region_is_provably_empty(space) is False
        assert _measure_zero(space) is False

    def test_a_constraint_naming_an_unknown_variable_is_indeterminate(self):
        """Dropping the term would solve a different problem than filter_feasible.

        ``add_constraint`` rejects an unknown name, so this is reached only by
        a corrupted or hand-assembled constraint list -- exactly when guessing
        is most expensive.
        """
        space = SearchSpace()
        space.add_variable("x1", "real", min=0.0, max=10.0)
        space.add_constraint("inequality", {"x1": 1.0}, 5.0)
        # Would be provably empty if the unknown term were simply dropped.
        space.constraints.append({
            "name": "hand_assembled",
            "type": "inequality",
            "coefficients": {"x1": 1.0, "x_absent": 4.0},
            "rhs": -100.0,
        })

        assert region_is_provably_empty(space) is False

    def test_an_equality_slab_is_not_reported_empty(self):
        """filter_feasible accepts |lhs - rhs| <= atol, so the program must too.

        A program that modelled the equality as a hyperplane and then failed to
        place a point on it exactly would report an empty region for one that
        ``filter_feasible`` accepts.
        """
        space = SearchSpace()
        space.add_variable("x2", "integer", min=0, max=8)
        space.add_constraint("equality", {"x2": 2.0}, 6.0)
        assert region_is_provably_empty(space) is False

    def test_a_term_less_constraint_is_skipped_exactly_as_filter_feasible_skips_it(self):
        """An empty coefficients mapping leaves the whole box feasible.

        ``filter_feasible`` never judges such a constraint: ``any_col`` stays
        False and it is ``continue``d (``search_space.py:1289-1290``), so every
        point passes. Modelled as an all-zero LP row against ``rhs + atol`` a
        negative ``rhs`` makes the program infeasible, which would prove
        "empty" for a box that is entirely feasible.

        ``add_constraint`` performs no non-empty check -- its coefficient loop
        simply never runs -- so this arrives through
        ``POST /variables/load`` and through the library and desktop paths.
        """
        space = SearchSpace()
        space.add_variable("x1", "real", min=0.0, max=10.0)
        space.add_variable("x2", "real", min=0.0, max=10.0)
        space.constraints.append({
            "name": "term_less",
            "type": "inequality",
            "coefficients": {},
            "rhs": -1.0,
        })

        assert region_is_provably_empty(space) is False
        # The design must come back, not merely fail to be refused.
        points = generate_initial_design(
            space, method="random", n_points=N, random_seed=1
        )
        assert len(points) == N
        for point in points:
            assert 0.0 <= point["x1"] <= 10.0
            assert 0.0 <= point["x2"] <= 10.0

    def test_a_tiny_but_real_region_is_not_reported_empty(self):
        space = SearchSpace()
        space.add_variable("x1", "real", min=0.0, max=10.0)
        space.add_variable("x2", "real", min=0.0, max=10.0)
        space.add_constraint("inequality", {"x1": -1.0, "x2": -1.0}, -19.99)
        assert region_is_provably_empty(space) is False
