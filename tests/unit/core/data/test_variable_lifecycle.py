"""Replacing and removing a registered variable without breaking the pairing.

``SearchSpace.variables`` and ``SearchSpace.skopt_dimensions`` are two lists of
different lengths that callers address positionally against each other. A
``context`` variable sits in the first and contributes nothing to the second,
so an index into one is not an index into the other -- and it is only
*accidentally* equal to one when no context variable precedes the target, which
is why every space built in this module puts one in a non-final position.

``replace_variable`` and ``remove_variable`` exist so that no caller has to
know any of that. These tests pin what they guarantee: position in both lists,
membership in the categorical/discrete name lists in both directions, and --
the point of the whole shape -- that a redefinition is validated by exactly the
guards ``add_variable`` enforces, because it is ``add_variable`` that runs.
"""

import math

import pytest
from skopt.space import Categorical, Integer, Real

from alchemist_core.data.search_space import SearchSpace


def _space():
    """A space whose context variable is not last and not the only one.

    ``c1`` sits at ``variables[0]`` and ``c2`` at ``variables[2]``, so every
    dimension-bearing variable after the first has a different index in
    ``variables`` than in ``skopt_dimensions``, and the offset is not a
    constant either. A fix that merely subtracts one would pass a
    single-context space and fail this one.
    """
    space = SearchSpace()
    space.add_variable("c1", "context")
    space.add_variable("x1", "real", min=0.0, max=10.0)
    space.add_variable("c2", "context")
    space.add_variable("x2", "categorical", values=["a", "b"])
    space.add_variable("x3", "integer", min=1, max=9)
    space.add_variable("x4", "discrete", allowed_values=[2.0, 4.0, 8.0])
    return space


def _describe(dimensions):
    """Name, class and payload of each dimension, in order."""
    described = []
    for dim in dimensions:
        if isinstance(dim, Categorical):
            described.append((dim.name, "Categorical", tuple(dim.categories)))
        else:
            described.append((dim.name, type(dim).__name__, (dim.low, dim.high)))
    return described


# ======================================================================
# The pairing primitive
# ======================================================================


class TestDimensionPairing:
    def test_dimension_names_are_positionally_paired_with_skopt_dimensions(self):
        space = _space()
        assert space.get_dimension_names() == ["x1", "x2", "x3", "x4"]
        assert space.get_dimension_names() == [d.name for d in space.skopt_dimensions]

    def test_the_two_lists_differ_in_length_so_the_pairing_is_not_index_for_index(self):
        """The condition the whole defect class needs, stated outright."""
        space = _space()
        assert len(space.variables) == 6
        assert len(space.skopt_dimensions) == 4
        # x2 is at a different position in each list -- this is the desync.
        assert [v["name"] for v in space.variables].index("x2") == 3
        assert space.get_dimension_index("x2") == 1

    @pytest.mark.parametrize(
        "var_type, kwargs",
        [
            ("real", {"min": 0.0, "max": 1.0}),
            ("integer", {"min": 0, "max": 5}),
            ("categorical", {"values": ["p", "q"]}),
            ("discrete", {"allowed_values": [1.0, 3.0]}),
            ("context", {}),
        ],
    )
    def test_the_two_name_lists_classify_every_accepted_type_identically(
        self, var_type, kwargs
    ):
        """``get_dimension_names`` and ``get_tunable_variable_names`` must agree.

        They are two answers to questions that are different in principle --
        "does this variable carry a skopt dimension" versus "is this variable
        not context" -- and identical in fact for every type the library
        accepts today. Kept as two methods because the distinction is real, but
        the agreement is load-bearing and nothing in the code binds them:
        ``get_tunable_variable_names`` is consumed *positionally* against
        model/candidate columns at ``botorch_acquisition.py:753-760`` (and
        ``:715``, ``:797``), so the moment the two diverge those sites become a
        fresh instance of exactly the bug this module closed.
        """
        space = SearchSpace()
        space.add_variable("v", var_type, **kwargs)
        assert space.get_dimension_names() == space.get_tunable_variable_names()

    def test_the_two_name_lists_agree_on_a_space_holding_every_accepted_type(self):
        """The same agreement in combination, and in registration order."""
        space = SearchSpace()
        space.add_variable("x1", "real", min=0.0, max=1.0)
        space.add_variable("c1", "context")
        space.add_variable("x2", "integer", min=0, max=5)
        space.add_variable("x3", "discrete", allowed_values=[1.0, 2.0])
        space.add_variable("x4", "categorical", values=["a", "b"])
        space.add_variable("c2", "context")

        assert space.get_dimension_names() == space.get_tunable_variable_names()
        assert space.get_dimension_names() == ["x1", "x2", "x3", "x4"]

    def test_the_two_name_lists_still_agree_after_replace_and_remove(self):
        """The agreement has to survive the operations this module added."""
        space = _space()
        space.replace_variable("x2", "context")
        space.replace_variable("c1", "integer", min=0, max=3)
        space.remove_variable("x4")

        assert space.get_dimension_names() == space.get_tunable_variable_names()
        assert space.get_dimension_names() == ["c1", "x1", "x3"]

    def test_the_dimension_bearing_set_is_the_non_context_half_of_the_taxonomy(self):
        """States the equivalence the two name lists rest on, as a set identity.

        A new *dimension-bearing* type has to join ``_DIMENSION_BEARING_TYPES``
        to be paired correctly, and that makes this fail until the taxonomy
        below is updated -- at which point whoever updates it is looking
        straight at the ``context`` exclusion that
        ``get_tunable_variable_names`` hard-codes.

        It does not catch a new type that is neither dimension-bearing nor
        ``context``. That is the one case where the two name lists would
        genuinely diverge, and no test can announce it, because nothing in the
        library enumerates its own accepted types. See the report: the complete
        close is to move the positional consumers onto
        ``get_dimension_names``.
        """
        accepted_types = {"real", "integer", "categorical", "discrete", "context"}
        assert SearchSpace._DIMENSION_BEARING_TYPES == accepted_types - {"context"}

    def test_dimension_names_report_the_intended_pairing_not_a_corrupted_one(self):
        """The primitive has to be able to *detect* a desync, not agree with it.

        ``get_dimension_names`` is derived from ``self.variables``, so it
        answers "which variables should hold a dimension, and in what order".
        Reading ``dim.name`` off ``skopt_dimensions`` instead would look
        equivalent -- the two agree on every space this module can build -- but
        it makes the comparison a tautology: a corrupted dimension list would
        describe itself as correct, and a consumer keying a sample by those
        names would emit plausible, wrongly-labelled values instead of a
        detectable mismatch. Half-registration is reachable over REST (see
        ``add_variable``'s comment), so this is not hypothetical.
        """
        space = _space()
        # Exactly the corruption update_variable used to produce: x2's
        # dimension overwritten by a second one carrying x1's name.
        space.skopt_dimensions[1] = Real(5.0, 6.0, name="x1")

        assert space.get_dimension_names() == ["x1", "x2", "x3", "x4"]
        assert space.get_dimension_names() != [d.name for d in space.skopt_dimensions]

    @pytest.mark.parametrize(
        "name, expected",
        [("x1", 0), ("x2", 1), ("x3", 2), ("x4", 3)],
    )
    def test_dimension_index_addresses_the_dimension_that_actually_bears_the_name(
        self, name, expected
    ):
        space = _space()
        assert space.get_dimension_index(name) == expected
        assert space.skopt_dimensions[expected].name == name

    @pytest.mark.parametrize("name", ["c1", "c2", "not-registered"])
    def test_dimension_index_is_none_when_there_is_no_dimension_slot(self, name):
        assert _space().get_dimension_index(name) is None

    @pytest.mark.parametrize(
        "var_type, kwargs",
        [
            ("real", {"min": 0.0, "max": 1.0}),
            ("integer", {"min": 0, "max": 5}),
            ("categorical", {"values": ["p", "q"]}),
            ("discrete", {"allowed_values": [1.0, 3.0]}),
            ("context", {}),
        ],
    )
    def test_dimension_bearing_set_agrees_with_what_add_variable_builds(
        self, var_type, kwargs
    ):
        """The invariant, made checkable rather than left to a comment.

        ``_DIMENSION_BEARING_TYPES`` is a hand-written set, and the whole
        pairing rests on it agreeing with what ``add_variable`` actually
        appends. This compares the two by execution for each type listed below,
        so a *wrong entry for a known type* -- ``context`` added to the set, or
        ``real`` dropped from it -- fails here.

        What it does not do is notice a sixth type. A type added to
        ``add_variable`` and to neither the set nor this parametrize list would
        simply not be exercised, and this test would pass. Closing that needs a
        source of truth this module can enumerate, which does not exist today;
        ``replace_variable``'s own agreement check is what catches the drift at
        runtime instead.
        """
        space = SearchSpace()
        before = len(space.skopt_dimensions)
        space.add_variable("v", var_type, **kwargs)
        grew = len(space.skopt_dimensions) == before + 1

        assert grew is (var_type in SearchSpace._DIMENSION_BEARING_TYPES)
        assert grew is (space.get_dimension_index("v") is not None)
        assert space.get_dimension_names() == [d.name for d in space.skopt_dimensions]


# ======================================================================
# replace_variable
# ======================================================================


class TestReplaceVariablePreservesPosition:
    def test_replacing_a_middle_variable_leaves_every_other_dimension_alone(self):
        space = _space()
        before = _describe(space.skopt_dimensions)

        space.replace_variable("x2", "categorical", values=["p", "q", "r"])

        after = _describe(space.skopt_dimensions)
        assert after[1] == ("x2", "Categorical", ("p", "q", "r"))
        # Neighbours untouched, in place.
        assert after[0] == before[0]
        assert after[2] == before[2]
        assert after[3] == before[3]

    def test_ordering_is_preserved_in_both_lists(self):
        """Not remove-then-add: that moves the variable to the end of both."""
        space = _space()
        space.replace_variable("x1", "real", min=5.0, max=6.0)

        assert [v["name"] for v in space.variables] == [
            "c1", "x1", "c2", "x2", "x3", "x4",
        ]
        assert space.get_dimension_names() == ["x1", "x2", "x3", "x4"]
        assert [d.name for d in space.skopt_dimensions] == ["x1", "x2", "x3", "x4"]

    def test_the_replaced_dimension_carries_the_new_definition(self):
        space = _space()
        space.replace_variable("x3", "integer", min=100, max=200)

        dim = space.skopt_dimensions[space.get_dimension_index("x3")]
        assert isinstance(dim, Integer)
        assert (dim.low, dim.high) == (100, 200)
        assert space.variables[4] == {
            "name": "x3", "type": "integer", "min": 100, "max": 200,
        }

    def test_exactly_one_dimension_bears_each_name_after_a_replace(self):
        """Defect 1b's signature: a replace that duplicated one name and lost another."""
        space = _space()
        space.replace_variable("x1", "real", min=5.0, max=6.0)

        names = [d.name for d in space.skopt_dimensions]
        assert names == sorted(set(names), key=names.index)
        assert len(names) == len(set(names))
        assert set(names) == {"x1", "x2", "x3", "x4"}


class TestReplaceVariableInheritsEveryGuard:
    """The redefinition is built by ``add_variable``, so its guards all apply."""

    @pytest.mark.parametrize(
        "var_type, kwargs, expected_fragment",
        [
            ("real", {"min": float("nan"), "max": 1.0}, "min must be finite"),
            ("real", {"min": 0.0, "max": float("inf")}, "max must be finite"),
            ("real", {"min": -1.7e308, "max": 1.7e308}, "span more than it"),
            ("integer", {"min": 5, "max": 4}, "lower bound"),
            ("categorical", {"values": []}, "at least 1 value"),
            ("discrete", {"allowed_values": [1.0]}, "at least 2 values"),
            ("discrete", {"allowed_values": [1.0, 1.0]}, "duplicate values"),
            ("nonsense", {}, "Unknown variable type"),
        ],
    )
    def test_a_rejected_redefinition_raises(self, var_type, kwargs, expected_fragment):
        space = _space()
        with pytest.raises(ValueError, match=expected_fragment):
            space.replace_variable("x1", var_type, **kwargs)

    @pytest.mark.parametrize(
        "var_type, kwargs",
        [
            ("real", {"min": float("nan"), "max": 1.0}),
            ("real", {"min": -1.7e308, "max": 1.7e308}),
            ("integer", {"min": 5, "max": 4}),
            ("categorical", {"values": []}),
            ("discrete", {"allowed_values": [1.0]}),
        ],
    )
    def test_a_rejected_redefinition_leaves_the_space_exactly_as_it_was(
        self, var_type, kwargs
    ):
        """Atomic: nothing is half-overwritten by a failure partway through."""
        space = _space()
        before_vars = [dict(v) for v in space.variables]
        before_dims = _describe(space.skopt_dimensions)
        before_cat = list(space.categorical_variables)
        before_disc = list(space.discrete_variables)

        with pytest.raises(ValueError):
            space.replace_variable("x1", var_type, **kwargs)

        assert space.variables == before_vars
        assert _describe(space.skopt_dimensions) == before_dims
        assert space.categorical_variables == before_cat
        assert space.discrete_variables == before_disc

    def test_the_duplicate_name_guard_does_not_fire_on_the_name_being_replaced(self):
        """The one guard a replace must not inherit."""
        space = _space()
        space.replace_variable("x1", "real", min=1.0, max=2.0)
        assert [v["name"] for v in space.variables].count("x1") == 1

    def test_replacing_an_unregistered_name_is_refused(self):
        space = _space()
        with pytest.raises(ValueError, match="not registered"):
            space.replace_variable("x9", "real", min=0.0, max=1.0)
        assert len(space.variables) == 6


class TestReplaceVariableAcrossTypes:
    @pytest.mark.parametrize(
        "var_type, kwargs, expect_cat, expect_disc",
        [
            ("real", {"min": 0.0, "max": 1.0}, False, False),
            ("integer", {"min": 0, "max": 4}, False, False),
            ("categorical", {"values": ["p", "q"]}, True, False),
            ("discrete", {"allowed_values": [3.0, 5.0]}, False, True),
            ("context", {}, False, False),
        ],
    )
    def test_membership_lists_track_the_type_out_of_categorical(
        self, var_type, kwargs, expect_cat, expect_disc
    ):
        space = _space()
        assert "x2" in space.categorical_variables

        space.replace_variable("x2", var_type, **kwargs)

        assert ("x2" in space.categorical_variables) is expect_cat
        assert ("x2" in space.discrete_variables) is expect_disc

    @pytest.mark.parametrize(
        "var_type, kwargs, expect_cat, expect_disc",
        [
            ("real", {"min": 0.0, "max": 1.0}, False, False),
            ("integer", {"min": 0, "max": 4}, False, False),
            ("categorical", {"values": ["p", "q"]}, True, False),
            ("discrete", {"allowed_values": [3.0, 5.0]}, False, True),
            ("context", {}, False, False),
        ],
    )
    def test_membership_lists_track_the_type_out_of_discrete(
        self, var_type, kwargs, expect_cat, expect_disc
    ):
        space = _space()
        assert "x4" in space.discrete_variables

        space.replace_variable("x4", var_type, **kwargs)

        assert ("x4" in space.categorical_variables) is expect_cat
        assert ("x4" in space.discrete_variables) is expect_disc

    @pytest.mark.parametrize(
        "target, var_type, kwargs, expect_cat, expect_disc",
        [
            ("x1", "categorical", {"values": ["p"]}, True, False),
            ("x1", "discrete", {"allowed_values": [1.0, 2.0]}, False, True),
            ("x3", "categorical", {"values": ["p"]}, True, False),
            ("x3", "discrete", {"allowed_values": [1.0, 2.0]}, False, True),
        ],
    )
    def test_membership_lists_track_the_type_into_categorical_and_discrete(
        self, target, var_type, kwargs, expect_cat, expect_disc
    ):
        space = _space()
        space.replace_variable(target, var_type, **kwargs)

        assert (target in space.categorical_variables) is expect_cat
        assert (target in space.discrete_variables) is expect_disc

    def test_a_variable_keeping_its_type_keeps_its_place_in_the_membership_list(self):
        """Remove-and-re-append would reorder the one-hot encoder's columns."""
        space = _space()
        space.add_variable("x5", "categorical", values=["m", "n"])
        assert space.categorical_variables == ["x2", "x5"]

        space.replace_variable("x2", "categorical", values=["p", "q"])

        assert space.categorical_variables == ["x2", "x5"]

    def test_context_gains_a_dimension_at_the_position_its_name_maps_to(self):
        space = _space()
        assert space.get_dimension_index("c2") is None

        space.replace_variable("c2", "real", min=-1.0, max=1.0)

        assert space.get_dimension_names() == ["x1", "c2", "x2", "x3", "x4"]
        assert [d.name for d in space.skopt_dimensions] == [
            "x1", "c2", "x2", "x3", "x4",
        ]
        dim = space.skopt_dimensions[1]
        assert isinstance(dim, Real) and (dim.low, dim.high) == (-1.0, 1.0)
        # Position in self.variables is unchanged -- only the type is.
        assert [v["name"] for v in space.variables] == [
            "c1", "x1", "c2", "x2", "x3", "x4",
        ]

    def test_a_variable_becoming_context_gives_up_its_dimension(self):
        space = _space()
        space.replace_variable("x2", "context")

        assert space.get_dimension_index("x2") is None
        assert space.get_dimension_names() == ["x1", "x3", "x4"]
        assert [d.name for d in space.skopt_dimensions] == ["x1", "x3", "x4"]
        assert "x2" not in space.categorical_variables
        assert [v["name"] for v in space.variables] == [
            "c1", "x1", "c2", "x2", "x3", "x4",
        ]

    def test_discrete_values_are_sorted_and_coerced_by_the_shared_path(self):
        space = _space()
        space.replace_variable("x4", "discrete", allowed_values=[9, 1, 5])

        assert space.variables[5]["allowed_values"] == [1.0, 5.0, 9.0]
        dim = space.skopt_dimensions[space.get_dimension_index("x4")]
        assert list(dim.categories) == [1.0, 5.0, 9.0]

    def test_optional_metadata_survives_a_replace(self):
        space = _space()
        space.replace_variable(
            "x1", "real", min=1.0, max=2.0, unit="u", description="d"
        )
        assert space.variables[1]["unit"] == "u"
        assert space.variables[1]["description"] == "d"


# ======================================================================
# remove_variable
# ======================================================================


class TestRemoveVariable:
    @pytest.mark.parametrize(
        "name, remaining_dims",
        [
            ("x1", ["x2", "x3", "x4"]),
            ("x2", ["x1", "x3", "x4"]),
            ("x3", ["x1", "x2", "x4"]),
            ("x4", ["x1", "x2", "x3"]),
        ],
    )
    def test_removing_a_variable_removes_its_own_dimension_and_no_other(
        self, name, remaining_dims
    ):
        space = _space()
        space.remove_variable(name)

        assert space.get_dimension_names() == remaining_dims
        assert [d.name for d in space.skopt_dimensions] == remaining_dims
        assert name not in [v["name"] for v in space.variables]

    @pytest.mark.parametrize("name", ["c1", "c2"])
    def test_removing_a_context_variable_removes_no_dimension_at_all(self, name):
        space = _space()
        before = _describe(space.skopt_dimensions)

        space.remove_variable(name)

        assert _describe(space.skopt_dimensions) == before
        assert name not in [v["name"] for v in space.variables]
        assert len(space.variables) == 5

    def test_removal_clears_membership_in_both_name_lists(self):
        space = _space()
        space.remove_variable("x2")
        space.remove_variable("x4")

        assert space.categorical_variables == []
        assert space.discrete_variables == []

    def test_removing_an_unregistered_name_is_refused(self):
        space = _space()
        with pytest.raises(ValueError, match="not registered"):
            space.remove_variable("x9")
        assert len(space.variables) == 6

    def test_the_pairing_survives_a_sequence_of_removals(self):
        space = _space()
        for name in ("c1", "x2", "x4", "c2"):
            space.remove_variable(name)
            assert space.get_dimension_names() == [
                d.name for d in space.skopt_dimensions
            ]
        assert space.get_dimension_names() == ["x1", "x3"]


# ======================================================================
# The two operations together
# ======================================================================


def test_replace_and_remove_keep_the_space_usable_end_to_end():
    """A sampled point has one value per dimension, keyed by the paired names."""
    space = _space()
    space.replace_variable("x2", "integer", min=0, max=3)
    space.remove_variable("c1")
    space.replace_variable("x1", "real", min=2.0, max=4.0)

    names = space.get_dimension_names()
    assert names == [d.name for d in space.skopt_dimensions]

    sample = [d.rvs(1)[0] for d in space.skopt_dimensions]
    point = dict(zip(names, sample))
    assert set(point) == {"x1", "x2", "x3", "x4"}
    assert 2.0 <= point["x1"] <= 4.0
    assert not math.isnan(point["x1"])
    assert 0 <= point["x2"] <= 3
    assert point["x4"] in (2.0, 4.0, 8.0)


class TestTheDimensionDecisionHasOneSourceOfTruth:
    """``replace_variable`` asks two things whether a type carries a dimension.

    ``staged`` answers by having built one; ``_DIMENSION_BEARING_TYPES``
    answers for ``get_dimension_index``, which decides where it goes. If they
    disagree the insert position is ``None`` and ``list.insert`` raises
    TypeError -- not a ValueError, so it leaves the API as an unlabelled 500
    rather than a 400 naming the problem.

    Both directions of the drift are forced here by monkeypatching the set,
    because no combination of real inputs can produce it while the two agree.
    """

    def test_a_type_that_builds_a_dimension_but_is_not_in_the_set(self, monkeypatch):
        space = _space()
        monkeypatch.setattr(
            SearchSpace,
            "_DIMENSION_BEARING_TYPES",
            frozenset({"integer", "categorical", "discrete"}),  # "real" dropped
        )
        with pytest.raises(ValueError, match="drifted"):
            space.replace_variable("x3", "real", min=0.0, max=1.0)

    def test_a_type_that_builds_no_dimension_but_is_in_the_set(self, monkeypatch):
        space = _space()
        monkeypatch.setattr(
            SearchSpace,
            "_DIMENSION_BEARING_TYPES",
            frozenset({"real", "integer", "categorical", "discrete", "context"}),
        )
        with pytest.raises(ValueError, match="drifted"):
            space.replace_variable("x1", "context")

    def test_the_drift_is_refused_before_the_space_is_touched(self, monkeypatch):
        """Atomicity survives the new check, because it runs against ``staged``."""
        space = _space()
        before_vars = [dict(v) for v in space.variables]
        before_dims = _describe(space.skopt_dimensions)

        monkeypatch.setattr(
            SearchSpace,
            "_DIMENSION_BEARING_TYPES",
            frozenset({"integer", "categorical", "discrete"}),
        )
        with pytest.raises(ValueError, match="drifted"):
            space.replace_variable("x1", "real", min=5.0, max=6.0)

        assert space.variables == before_vars
        assert _describe(space.skopt_dimensions) == before_dims

    def test_a_drift_never_surfaces_as_a_typeerror(self, monkeypatch):
        """The whole point: ValueError has a global 400 handler, TypeError does not."""
        space = _space()
        monkeypatch.setattr(
            SearchSpace,
            "_DIMENSION_BEARING_TYPES",
            frozenset({"integer", "categorical", "discrete"}),
        )
        with pytest.raises(ValueError):
            space.replace_variable("x3", "real", min=0.0, max=1.0)
