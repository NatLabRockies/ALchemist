"""A ``context`` variable must not shift a design's name->value pairing.

``generate_initial_design``'s space-filling branch drew samples from
``skopt_dimensions`` -- which omits ``context`` variables -- and zipped them
against names taken from ``search_space.variables``, which does not. Whenever a
context variable sat anywhere but *last*, the zip ran off the end: the real
variable behind the context one was dropped from the design entirely and its
sampled value was emitted under the context variable's name.

    variables       : [x1(real), c1(context), x2(integer)]
    skopt_dimensions: [x1, x2]
    point returned  : {'x1': <x1's value>, 'c1': <x2's value>}   # x2 missing

The condition is **position, not presence** -- a context variable in the last
position zips correctly by accident, which is why nothing caught this. Every
test here therefore runs the context variable in first, middle *and* last
position; the last-position rows pass against the old code and are here so a
future change cannot break that row silently.

With a constraint registered this is worse than a corrupt design.
``SearchSpace.filter_feasible`` sums only the constraint terms whose column is
present in the frame it is given, so the reject-and-resample loop screened the
truncated points against a constraint that had silently lost a term and
returned points violating the real one while reporting success.
``TestConstrainedDesignPointsAreGenuinelyFeasible`` computes the constraint by
hand for exactly that reason: an assertion routed through ``filter_feasible``
passes against the broken code.

The defect's signature is a **missing key**, so every assertion here compares
the *full key set* of a point. A test that only looks up the values it expects
to find passes against the broken code too.
"""

import pytest

from alchemist_core.data.search_space import SearchSpace
from alchemist_core.utils.doe import generate_initial_design
from alchemist_core.utils.doe import SPACE_FILLING_METHODS

METHODS = sorted(SPACE_FILLING_METHODS)
POSITIONS = ["first", "middle", "last"]

# Registration kwargs by short name. Kept deliberately heterogeneous so no test
# space is a sweep of one variable type: the misalignment is a positional
# defect and is blind to type, but a fix that special-cased one type would
# survive a single-type net.
_TUNABLE = {
    "x1": dict(var_type="real", min=0.0, max=5.0),
    "x2": dict(var_type="integer", min=100, max=200),
    "x3": dict(var_type="discrete", allowed_values=[1.0, 2.0, 4.0]),
    "x4": dict(var_type="categorical", values=["a", "b", "c"]),
}


def _space(position, tunable=("x1", "x2", "x3"), n_context=1):
    """A space with ``n_context`` context variables at ``position``.

    ``position`` places them relative to the tunable variables, so the same
    test body exercises the accidentally-correct last slot and the two broken
    ones.
    """
    context = [f"c{i + 1}" for i in range(n_context)]
    tunable = list(tunable)
    if position == "first":
        order = context + tunable
    elif position == "last":
        order = tunable + context
    elif position == "middle":
        mid = max(1, len(tunable) // 2)
        order = tunable[:mid] + context + tunable[mid:]
    else:  # pragma: no cover - guards a typo in a parametrize list
        raise AssertionError(f"unknown position {position!r}")

    space = SearchSpace()
    for name in order:
        if name in _TUNABLE:
            space.add_variable(name, **_TUNABLE[name])
        else:
            space.add_variable(name, "context")
    return space, tunable, context


class TestEveryDesignPointCarriesExactlyTheTunableVariables:
    """The full key set, on the whole method x position grid."""

    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("position", POSITIONS)
    def test_key_set_is_exactly_the_dimension_bearing_variables(self, method, position):
        space, tunable, context = _space(position)
        points = generate_initial_design(
            space, method=method, n_points=6, random_seed=11
        )
        assert len(points) == 6
        for point in points:
            assert set(point) == set(tunable), (
                f"method={method} context={position}: design point keys "
                f"{sorted(point)} != tunable variables {sorted(tunable)}"
            )

    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("position", POSITIONS)
    def test_no_context_name_appears_at_all(self, method, position):
        """Not a real value, not a placeholder, not a None -- absent."""
        space, _tunable, context = _space(position)
        points = generate_initial_design(
            space, method=method, n_points=4, random_seed=3
        )
        for point in points:
            for name in context:
                assert name not in point, (
                    f"method={method} context={position}: context variable "
                    f"{name!r} carries {point[name]!r} in a design point"
                )

    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("position", POSITIONS)
    def test_values_land_on_the_right_variable(self, method, position):
        """A shifted zip labels x2's integer onto c1 and drops x3 entirely.

        Checking each value against its *own* variable's domain is what
        separates "the right names are present" from "the right values are
        under them": with the shift, x2's 100..200 integer arrives under a name
        whose declared range is 0..5.
        """
        space, tunable, _context = _space(position)
        points = generate_initial_design(
            space, method=method, n_points=6, random_seed=5
        )
        for point in points:
            assert 0.0 <= point["x1"] <= 5.0 and isinstance(point["x1"], float)
            assert 100 <= point["x2"] <= 200 and isinstance(point["x2"], int)
            assert point["x3"] in {1.0, 2.0, 4.0}

    @pytest.mark.parametrize("method", METHODS)
    def test_two_context_variables_split_across_the_space(self, method):
        """No constant offset relates variables to skopt_dimensions here.

        A "fix" that merely subtracted one from every index would pass a
        single-context space and fail this one.
        """
        space = SearchSpace()
        space.add_variable("c1", "context")
        space.add_variable("x1", **_TUNABLE["x1"])
        space.add_variable("c2", "context")
        space.add_variable("x4", **_TUNABLE["x4"])
        space.add_variable("x2", **_TUNABLE["x2"])

        points = generate_initial_design(
            space, method=method, n_points=5, random_seed=17
        )
        for point in points:
            assert set(point) == {"x1", "x4", "x2"}, sorted(point)
            assert 0.0 <= point["x1"] <= 5.0
            assert point["x4"] in {"a", "b", "c"}
            assert 100 <= point["x2"] <= 200

    @pytest.mark.parametrize("method", METHODS)
    def test_a_context_only_space_never_invents_a_value_for_it(self, method):
        """Degenerate end of the range: no dimensions at all.

        Measured identical before and after this fix: ``random`` returns empty
        dicts, and the other four propagate the raw sampler's refusal to work
        in zero dimensions (``AssertionError`` from skopt's samplers,
        ``ValueError`` from Sobol). That refusal is pre-existing, is not this
        defect, and is deliberately left alone -- see the report. What is
        pinned here is the part this task owns: whichever way it goes, ``c1``
        never comes back carrying a value.
        """
        space = SearchSpace()
        space.add_variable("c1", "context")
        try:
            points = generate_initial_design(
                space, method=method, n_points=3, random_seed=2
            )
        except (AssertionError, ValueError):
            return
        assert all(point == {} for point in points)

    @pytest.mark.parametrize("position", POSITIONS)
    def test_a_single_tunable_variable_behind_a_context_one(self, position):
        """The smallest space where the shift can drop the only real variable."""
        space, tunable, _context = _space(position, tunable=("x4",))
        points = generate_initial_design(
            space, method="random", n_points=4, random_seed=9
        )
        for point in points:
            assert set(point) == {"x4"}
            assert point["x4"] in {"a", "b", "c"}


class TestUnconstrainedBehaviorIsUnchangedWithoutContextVariables:
    """The substitution must be a no-op on every space that has no context."""

    @pytest.mark.parametrize("method", METHODS)
    def test_names_match_the_registration_order(self, method):
        space = SearchSpace()
        for name in ("x1", "x2", "x3", "x4"):
            space.add_variable(name, **_TUNABLE[name])
        points = generate_initial_design(
            space, method=method, n_points=4, random_seed=7
        )
        for point in points:
            assert list(point) == ["x1", "x2", "x3", "x4"]

    def test_get_dimension_names_equals_the_variable_names(self):
        """Why the substitution is a no-op: the two lists coincide here."""
        space = SearchSpace()
        for name in ("x1", "x2", "x3", "x4"):
            space.add_variable(name, **_TUNABLE[name])
        assert space.get_dimension_names() == [v["name"] for v in space.variables]


def _lhs_by_hand(point, coefficients):
    """sum(coeff * value) for a design point, computed here rather than by
    ``SearchSpace.filter_feasible``.

    ``filter_feasible`` sums only the terms whose column is present in the
    frame it is given and skips a constraint entirely only when *none* of its
    columns are present. That is precisely the behavior that made the broken
    design look feasible, so an assertion routed through it passes against the
    broken code. Missing a column is a ``KeyError`` here, by design.
    """
    return sum(coeff * point[name] for name, coeff in coefficients.items())


def _two_real_space_with_context(position):
    """``x1`` and ``x5`` on [0, 5], with ``c1`` at ``position`` among them."""
    order = {
        "first": ["c1", "x1", "x5"],
        "middle": ["x1", "c1", "x5"],
        "last": ["x1", "x5", "c1"],
    }[position]
    space = SearchSpace()
    for name in order:
        if name == "c1":
            space.add_variable("c1", "context")
        else:
            space.add_variable(name, "real", min=0.0, max=5.0)
    return space


class TestConstrainedDesignPointsAreGenuinelyFeasible:
    """The severe face: a constrained design returned infeasible points.

    Executed at 839814b on ``[x1(real), c1(context), x2(real)]`` with
    ``x1 + x2 <= 5``: three of four returned points violated the constraint and
    the call reported success, because the loop screened ``{x1, c1}`` frames
    against a constraint that had silently degraded to ``x1 <= 5``.
    """

    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("position", POSITIONS)
    def test_an_inequality_holds_for_every_returned_point(self, method, position):
        # Two real variables on the same scale, so the constraint binds a
        # genuine two-variable half-plane rather than being satisfiable by
        # either variable alone.
        space = _two_real_space_with_context(position)
        coefficients = {"x1": 1.0, "x5": 1.0}
        space.add_constraint("inequality", coefficients, 5.0)

        points = generate_initial_design(
            space, method=method, n_points=4, random_seed=7
        )
        assert len(points) == 4
        for point in points:
            assert set(point) == {"x1", "x5"}, (
                f"method={method} context={position}: {sorted(point)} -- a "
                f"missing column is what let the constraint be half-evaluated"
            )
            assert _lhs_by_hand(point, coefficients) <= 5.0 + 1e-9, (
                f"method={method} context={position}: returned point {point} "
                f"violates x1 + x5 <= 5 (lhs = "
                f"{_lhs_by_hand(point, coefficients)})"
            )

    @pytest.mark.parametrize("position", ["first", "middle"])
    def test_the_constraint_binds_hard_enough_to_reject(self, position):
        """Guard against a vacuous pass: the region must exclude most draws.

        With ``x1 + x5 <= 2`` on ``[0,5]^2`` the feasible region is 8% of the
        box, so a design that skipped the filter would be overwhelmingly likely
        to return an infeasible point. This is what makes the test above a
        genuine check rather than one the sampler satisfies by luck.
        """
        space = _two_real_space_with_context(position)
        coefficients = {"x1": 1.0, "x5": 1.0}
        space.add_constraint("inequality", coefficients, 2.0)

        points = generate_initial_design(
            space, method="random", n_points=8, random_seed=23
        )
        for point in points:
            assert set(point) == {"x1", "x5"}
            assert _lhs_by_hand(point, coefficients) <= 2.0 + 1e-9, point

    @pytest.mark.parametrize("position", POSITIONS)
    def test_a_constraint_naming_the_variable_behind_the_context_one(self, position):
        """The dropped variable is the one the constraint is about.

        Before the fix, ``x2`` was the key missing from every point, so the
        constraint lost the only term that could have rejected anything.
        """
        space = SearchSpace()
        order = {
            "first": ["c1", "x1", "x2"],
            "middle": ["x1", "c1", "x2"],
            "last": ["x1", "x2", "c1"],
        }[position]
        for name in order:
            if name == "c1":
                space.add_variable("c1", "context")
            else:
                space.add_variable(name, "real", min=0.0, max=5.0)
        coefficients = {"x2": 1.0}
        space.add_constraint("inequality", coefficients, 1.5)

        points = generate_initial_design(
            space, method="lhs", n_points=5, random_seed=31
        )
        for point in points:
            assert set(point) == {"x1", "x2"}
            assert point["x2"] <= 1.5 + 1e-9, point

    @pytest.mark.parametrize("position", POSITIONS)
    def test_an_equality_constraint_holds_by_hand(self, position):
        """The other constraint type, on a discrete variable's grid."""
        space = SearchSpace()
        order = {
            "first": ["c1", "x3", "x1"],
            "middle": ["x3", "c1", "x1"],
            "last": ["x3", "x1", "c1"],
        }[position]
        for name in order:
            if name == "c1":
                space.add_variable("c1", "context")
            else:
                space.add_variable(name, **_TUNABLE[name])
        coefficients = {"x3": 1.0}
        space.add_constraint("equality", coefficients, 2.0)

        points = generate_initial_design(
            space, method="random", n_points=4, random_seed=13
        )
        for point in points:
            assert set(point) == {"x1", "x3"}
            assert abs(_lhs_by_hand(point, coefficients) - 2.0) <= 1e-9, point


class TestScalarsStayJsonNativeAfterTheRestructure:
    """Task 12A's ``_as_json_native`` lives in the comprehension this changed.

    ``np.int64`` does not subclass ``int``, so dropping the coercion turns
    every integer design back into an unserializable one. ``type(v) is T``,
    never ``isinstance``: ``np.float64`` passes ``isinstance(v, float)``.
    """

    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("position", POSITIONS)
    def test_types_are_exact_python_scalars(self, method, position):
        space, _tunable, _context = _space(position, tunable=("x1", "x2", "x3", "x4"))
        points = generate_initial_design(
            space, method=method, n_points=4, random_seed=7
        )
        for point in points:
            assert type(point["x1"]) is float
            assert type(point["x2"]) is int
            assert type(point["x3"]) is float
            assert type(point["x4"]) is str

    @pytest.mark.parametrize("method", METHODS)
    def test_a_design_with_a_context_variable_is_json_serializable(self, method):
        import json

        space, _tunable, _context = _space("middle", tunable=("x1", "x2", "x3", "x4"))
        points = generate_initial_design(
            space, method=method, n_points=4, random_seed=7
        )
        assert json.loads(json.dumps(points)) == points


# ==========================================================================
# The classical and optimal paths were NOT clean either. Measured at 7d377a9,
# on [x1(real), c1(context), x2(integer), x3(discrete)] -- and, unlike the
# space-filling defect, in the last position too, so there is no accidentally
# correct arrangement:
#
#   full_factorial  KeyError: 'min'      first, middle AND last
#   gsd             KeyError: 'min'      first, middle AND last
#   optimal         IndexError           first, middle
#   optimal         KeyError             last
#
# Same root cause: a list positionally paired with the *dimension-bearing*
# variables, indexed by a position taken from the full ``variables`` list.
# _full_factorial and _gsd reach for var['min'] on a context variable;
# optimal_design's candidate grid has one column per dimension-bearing
# variable while build_column_map numbered var_idx off all of them, and
# parse_model_spec gave the context variable a main effect nothing can set.
#
# fractional_factorial, ccd, box_behnken and plackett_burman were already
# correct: they route through _get_continuous_vars, which filters by type.
# They are pinned here anyway.
# ==========================================================================

CLASSICAL_METHODS_UNDER_TEST = [
    "full_factorial",
    "fractional_factorial",
    "ccd",
    "box_behnken",
    "plackett_burman",
    "gsd",
]


def _three_continuous_space(position):
    """``x1(real)``, ``x2(integer)``, ``x3(discrete)`` with ``c1`` at ``position``.

    Three continuous variables because ``box_behnken`` requires at least
    three, and three *different* types because the level lookups the defect
    lands in differ per type.
    """
    order = {
        "first": ["c1", "x1", "x2", "x3"],
        "middle": ["x1", "c1", "x2", "x3"],
        "last": ["x1", "x2", "x3", "c1"],
    }[position]
    space = SearchSpace()
    for name in order:
        if name == "c1":
            space.add_variable("c1", "context")
        else:
            space.add_variable(name, **_TUNABLE[name])
    return space


class TestClassicalDesignsSkipContextVariables:

    @pytest.mark.parametrize("method", CLASSICAL_METHODS_UNDER_TEST)
    @pytest.mark.parametrize("position", POSITIONS)
    def test_key_set_is_exactly_the_dimension_bearing_variables(self, method, position):
        space = _three_continuous_space(position)
        points = generate_initial_design(space, method=method, random_seed=3)
        assert points
        for point in points:
            assert set(point) == {"x1", "x2", "x3"}, (
                f"method={method} context={position}: {sorted(point)}"
            )

    @pytest.mark.parametrize("method", CLASSICAL_METHODS_UNDER_TEST)
    @pytest.mark.parametrize("position", POSITIONS)
    def test_values_stay_inside_their_own_variable_domain(self, method, position):
        space = _three_continuous_space(position)
        for point in generate_initial_design(space, method=method, random_seed=3):
            assert 0.0 <= point["x1"] <= 5.0
            assert 100 <= point["x2"] <= 200
            assert point["x3"] in {1.0, 2.0, 4.0}

    @pytest.mark.parametrize("position", POSITIONS)
    def test_a_categorical_level_grid_is_not_shifted(self, position):
        """full_factorial and gsd enumerate levels per variable, not per dimension.

        A categorical alongside a context variable is the case where a shifted
        level array would silently pick the wrong category rather than raise.
        """
        order = {
            "first": ["c1", "x4", "x1"],
            "middle": ["x4", "c1", "x1"],
            "last": ["x4", "x1", "c1"],
        }[position]
        space = SearchSpace()
        for name in order:
            if name == "c1":
                space.add_variable("c1", "context")
            else:
                space.add_variable(name, **_TUNABLE[name])

        for method in ("full_factorial", "gsd"):
            points = generate_initial_design(space, method=method, random_seed=3)
            seen = set()
            for point in points:
                assert set(point) == {"x4", "x1"}, (method, sorted(point))
                assert point["x4"] in {"a", "b", "c"}
                assert 0.0 <= point["x1"] <= 5.0
                seen.add(point["x4"])
            assert seen == {"a", "b", "c"}, (
                f"{method} context={position}: levels {sorted(seen)} -- a "
                f"shifted level array loses categories"
            )

    @pytest.mark.parametrize("position", POSITIONS)
    def test_a_constrained_classical_design_is_feasible_by_hand(self, position):
        """The classical constraint filter runs on the frame this builds.

        Computed here rather than through ``filter_feasible``: that is the
        function that half-evaluates a constraint whose column is missing.
        """
        space = _two_real_space_with_context(position)
        coefficients = {"x1": 1.0, "x5": 1.0}
        space.add_constraint("inequality", coefficients, 6.0)

        points = generate_initial_design(
            space, method="full_factorial", random_seed=3
        )
        assert points
        for point in points:
            assert set(point) == {"x1", "x5"}, sorted(point)
            assert _lhs_by_hand(point, coefficients) <= 6.0 + 1e-9, point


class TestOptimalDesignSkipsContextVariables:

    @pytest.mark.parametrize("position", POSITIONS)
    def test_key_set_is_exactly_the_dimension_bearing_variables(self, position):
        space = _three_continuous_space(position)
        points = generate_initial_design(
            space, method="optimal", model_type="linear", n_points=8, random_seed=3
        )
        assert len(points) == 8
        for point in points:
            assert set(point) == {"x1", "x2", "x3"}, sorted(point)

    @pytest.mark.parametrize("position", POSITIONS)
    @pytest.mark.parametrize("model_type", ["linear", "interaction", "quadratic"])
    def test_every_model_type(self, position, model_type):
        space = _three_continuous_space(position)
        points = generate_initial_design(
            space, method="optimal", model_type=model_type,
            n_points=14, random_seed=3,
        )
        for point in points:
            assert set(point) == {"x1", "x2", "x3"}, sorted(point)

    @pytest.mark.parametrize("position", POSITIONS)
    def test_a_context_variable_is_not_a_model_term(self, position):
        """It cannot be set, so it cannot be a factor the design optimizes over."""
        from alchemist_core.utils.optimal_design import (
            get_model_term_names,
            parse_model_spec,
        )

        space = _three_continuous_space(position)
        terms = parse_model_spec(space, model_type="linear")
        names = get_model_term_names(space, terms)
        assert "c1" not in names, names
        assert set(names) == {"Intercept", "x1", "x2", "x3"}, names

    @pytest.mark.parametrize("position", POSITIONS)
    def test_an_explicit_effects_list_cannot_name_a_context_variable(self, position):
        """It is not a factor, so naming it is an unknown-variable error.

        Pinned because the alternative -- accepting it and building a column
        for a variable that has none -- is the crash this fixed.
        """
        space = _three_continuous_space(position)
        with pytest.raises(ValueError, match="c1"):
            generate_initial_design(
                space, method="optimal", effects=["x1", "c1"],
                n_points=8, random_seed=3,
            )

    @pytest.mark.parametrize("position", POSITIONS)
    def test_a_constrained_optimal_design_is_feasible_by_hand(self, position):
        space = _two_real_space_with_context(position)
        coefficients = {"x1": 1.0, "x5": 1.0}
        space.add_constraint("inequality", coefficients, 5.0)

        points = generate_initial_design(
            space, method="optimal", model_type="linear", n_points=6, random_seed=3
        )
        assert len(points) == 6
        for point in points:
            assert set(point) == {"x1", "x5"}, sorted(point)
            assert _lhs_by_hand(point, coefficients) <= 5.0 + 1e-6, point

    @pytest.mark.parametrize("position", POSITIONS)
    def test_design_info_counts_only_the_real_factors(self, position):
        """session.get_optimal_design_info builds its own design matrix."""
        from alchemist_core import OptimizationSession

        session = OptimizationSession()
        session.search_space = _three_continuous_space(position)
        info = session.get_optimal_design_info(model_type="linear")
        assert "c1" not in info["model_terms"], info["model_terms"]
        assert info["p_columns"] == 4  # intercept + x1 + x2 + x3


class TestTheEstimabilityGateStillJudgesWithAContextVariable:
    """The constrained-classical gate must not degrade to "cannot judge".

    ``_inestimable_terms`` builds a design matrix to decide whether a design
    that lost points to a constraint can still estimate its implied model, and
    wraps that construction in ``except (ValueError, KeyError, IndexError)``
    so an unparseable model degrades to "no opinion" rather than crashing.
    That is the right behavior for its stated failure modes -- and it is also
    what hid this one: numbering its column map off ``search_space.variables``
    while ``parse_model_spec`` numbers terms off the dimension-bearing list
    raises straight into that catch, so a rank-deficient design was returned
    as if the gate had approved it.

    Measured, with only that one line reverted (mutation M10): context first
    and middle returned a 10-of-16 rank-deficient CCD; context last raised
    correctly. The scenario is ``test_ccd_losing_structural_points_raises``'s,
    with a context variable added.
    """

    @staticmethod
    def _session(position):
        from alchemist_core import OptimizationSession

        session = OptimizationSession()
        order = {
            "first": ["c1", "x1", "x2", "x3"],
            "middle": ["x1", "c1", "x2", "x3"],
            "last": ["x1", "x2", "x3", "c1"],
        }[position]
        for name in order:
            if name == "c1":
                session.add_variable("c1", "context")
            else:
                session.add_variable(name, "real", bounds=(0.0, 10.0))
        return session

    @pytest.mark.parametrize("position", POSITIONS)
    def test_a_rank_deficient_ccd_is_still_refused(self, position):
        from alchemist_core.utils.doe import DesignNotEstimableError

        session = self._session(position)
        session.add_input_constraint("inequality", {"x1": 1.0, "x2": 0.8}, rhs=9.3)
        with pytest.raises(DesignNotEstimableError, match="ccd"):
            session.generate_initial_design(method="ccd", random_seed=7)

    @pytest.mark.parametrize("position", POSITIONS)
    def test_a_harmless_drop_is_still_allowed_through(self, position):
        """The other side of the gate: it must not start refusing everything.

        A test that only pinned the raise would pass against a gate wired to
        raise unconditionally.
        """
        session = self._session(position)
        coefficients = {"x1": 1.0, "x2": 1.0}
        session.add_input_constraint("inequality", coefficients, rhs=11.0)

        points = session.generate_initial_design(method="ccd", random_seed=7)
        assert 0 < len(points) < 16  # structural points were genuinely dropped
        for point in points:
            assert set(point) == {"x1", "x2", "x3"}, sorted(point)
            assert _lhs_by_hand(point, coefficients) <= 11.0 + 1e-6, point
