"""Round-trip tests for candidate coded/raw encoding.

The constrained-candidate pipeline decodes the coded lattice to raw variable
space, augments it there (so filter_feasible stays the single definition of
feasibility), then re-encodes.

That round trip is exact for real and discrete variables: both use the same
affine ``(value - mid) / half_range`` mapping in each direction (discrete
additionally snaps to its nearest allowed value, which is a no-op for values
already on the lattice). It is generally *not* exact for integer variables:
decode_candidates rounds a decoded coded value to the nearest integer
(optimal_design.py's ``int(round(actual))``), so an integer whose coded
lattice value maps to a non-integer actual value comes back re-encoded at a
different coded value than the one it was decoded from. See
test_encode_decode_round_trip_is_identity_on_the_lattice, which is exact only
because its fixture's n_levels happens to put every lattice point for the
integer variable on an exact integer, and its companion
test_integer_round_trip_is_not_identity_in_general below, which is not.
"""

import numpy as np
import pytest

from alchemist_core.data.search_space import SearchSpace
from alchemist_core.utils.optimal_design import (
    build_column_map,
    decode_candidates,
    encode_candidates,
    generate_mixed_candidate_set,
)


def _space():
    s = SearchSpace()
    s.add_variable("x1", "real", min=0.0, max=10.0)
    s.add_variable("x2", "integer", min=0, max=8)
    s.add_variable("d1", "discrete", allowed_values=[1.0, 2.0, 4.0])
    s.add_variable("cat", "categorical", values=["a", "b"])
    return s


def test_build_column_map_matches_generate_mixed_candidate_set():
    s = _space()
    _cand, column_map = generate_mixed_candidate_set(s, n_levels=3)
    assert build_column_map(s.variables) == column_map


def test_decode_all_rows_when_indices_omitted():
    s = _space()
    cand, column_map = generate_mixed_candidate_set(s, n_levels=3)
    points = decode_candidates(cand, column_map, s.variables)
    assert len(points) == cand.shape[0]


def test_decode_respects_explicit_indices():
    s = _space()
    cand, column_map = generate_mixed_candidate_set(s, n_levels=3)
    points = decode_candidates(cand, column_map, s.variables,
                               selected_indices=np.array([0, 2]))
    assert len(points) == 2


def test_encode_decode_round_trip_is_identity_on_the_lattice():
    s = _space()
    cand, column_map = generate_mixed_candidate_set(s, n_levels=3)
    points = decode_candidates(cand, column_map, s.variables)
    recoded = encode_candidates(points, column_map, s.variables)
    assert recoded.shape == cand.shape
    assert np.allclose(recoded, cand, atol=1e-9)


def test_encode_handles_categorical_one_hot():
    s = _space()
    column_map = build_column_map(s.variables)
    coded = encode_candidates(
        [{"x1": 10.0, "x2": 8, "d1": 4.0, "cat": "b"}], column_map, s.variables
    )
    onehot = [coded[0][i] for i, cm in enumerate(column_map) if cm["type"] == "onehot"]
    assert onehot == [0.0, 1.0]


def test_encode_maps_bounds_to_plus_minus_one():
    s = SearchSpace()
    s.add_variable("x1", "real", min=2.0, max=6.0)
    column_map = build_column_map(s.variables)
    coded = encode_candidates([{"x1": 2.0}, {"x1": 6.0}, {"x1": 4.0}],
                              column_map, s.variables)
    assert coded[0][0] == pytest.approx(-1.0)
    assert coded[1][0] == pytest.approx(1.0)
    assert coded[2][0] == pytest.approx(0.0)


def test_encode_degenerate_range_is_zero():
    s = SearchSpace()
    # NOTE: not s.add_variable(...) — SearchSpace routes "real" variables
    # through skopt.space.Real, which raises on low == high. That skopt
    # constraint is a pre-existing SearchSpace behavior unrelated to
    # encode_candidates (which only reads var["min"]/var["max"] off the
    # plain dict), so the degenerate variable is appended directly.
    s.variables.append({"name": "x1", "type": "real", "min": 5.0, "max": 5.0})
    column_map = build_column_map(s.variables)
    coded = encode_candidates([{"x1": 5.0}], column_map, s.variables)
    assert coded[0][0] == 0.0


# ============================================================
# Discriminating companions (not in the plan's verbatim list).
#
# This plan's earlier tasks shipped tests that passed against a wrong
# implementation because the chosen values happened to make the right and
# a plausible-wrong formula coincide. These pin down two spots where that
# could happen again for Task 6.
# ============================================================

def test_generate_mixed_candidate_set_candidate_values_match_hand_computed():
    """Pins the *array values*, not just column_map.

    A rewritten `coded_columns` loop (Step 3) could reproduce a correct
    column_map while assembling the wrong candidates array — e.g. sourcing a
    one-hot column from the wrong source column, or misordering columns.
    `test_build_column_map_matches_generate_mixed_candidate_set` would not
    catch that since it only compares the map.
    """
    s = SearchSpace()
    s.add_variable("x1", "real", min=-5.0, max=20.0)
    s.add_variable("cat", "categorical", values=["a", "b", "c"])
    candidates, column_map = generate_mixed_candidate_set(s, n_levels=2)

    # x1 levels: linspace(-1, 1, 2) -> [-1.0, 1.0]
    # cat levels (grid indices): arange(3) -> [0, 1, 2]
    # itertools.product iterates the last axis fastest.
    expected = np.array([
        [-1.0, 1.0, 0.0, 0.0],  # x1=-1, cat=a
        [-1.0, 0.0, 1.0, 0.0],  # x1=-1, cat=b
        [-1.0, 0.0, 0.0, 1.0],  # x1=-1, cat=c
        [1.0, 1.0, 0.0, 0.0],   # x1=1,  cat=a
        [1.0, 0.0, 1.0, 0.0],   # x1=1,  cat=b
        [1.0, 0.0, 0.0, 1.0],   # x1=1,  cat=c
    ])
    assert [cm["type"] for cm in column_map] == [
        "continuous", "onehot", "onehot", "onehot"
    ]
    assert np.array_equal(candidates, expected)


def test_encode_discrete_uses_allowed_value_bounds_not_var_min_max():
    """Pins encode_candidates' discrete branch to allowed_values, not var["min"]/["max"].

    SearchSpace.add_variable passes through unrecognized kwargs (min/max)
    onto the discrete variable dict alongside allowed_values, so a
    discrete var can legally carry both. If encode_candidates used
    var["min"]/var["max"] instead of min/max(allowed_values), this would
    compute a wildly different coded value from the asymmetric, non-round
    bounds below.
    """
    s = SearchSpace()
    s.add_variable("d1", "discrete", allowed_values=[9.0, 2.25, 5.75],
                   min=-100.0, max=100.0)
    column_map = build_column_map(s.variables)
    coded = encode_candidates([{"d1": 5.75}], column_map, s.variables)

    # Correct: low=2.25, high=9.0 (from allowed_values) -> mid=5.625, half_range=3.375
    expected = (5.75 - 5.625) / 3.375
    assert coded[0][0] == pytest.approx(expected)
    # Sanity: this is nowhere near what var["min"]/var["max"] (-100/100) would give.
    assert coded[0][0] != pytest.approx((5.75 - 0.0) / 100.0)


def test_decode_candidates_alias_matches_public_function():
    """_decode_candidates must keep behaving like decode_candidates with the
    original positional (candidates, selected_indices, column_map, variables)
    order — the sole internal call site depends on this.
    """
    from alchemist_core.utils.optimal_design import _decode_candidates

    s = _space()
    cand, column_map = generate_mixed_candidate_set(s, n_levels=3)
    idx = np.array([0, 3, 7])

    alias_result = _decode_candidates(cand, idx, column_map, s.variables)
    direct_result = decode_candidates(cand, column_map, s.variables,
                                      selected_indices=idx)
    assert alias_result == direct_result


# ============================================================
# Fix-round companions (review findings on the first pass).
# ============================================================

def test_integer_round_trip_is_not_identity_in_general():
    """Companion to test_encode_decode_round_trip_is_identity_on_the_lattice.

    That test is exact only by coincidence: at n_levels=3, x2's (integer,
    0-8) lattice values [-1, 0, 1] all decode to exact integers (0, 4, 8).
    At n_levels=4, the interior lattice values [-1/3, 1/3] decode to 2.667
    and 5.333, which decode_candidates rounds to 3 and 5 — and re-encoding
    3 and 5 does not reproduce -1/3 and 1/3. Integers are the one variable
    type this round trip is lossy for; reals and discrete are exact (see
    the module docstring).
    """
    s = _space()
    cand, column_map = generate_mixed_candidate_set(s, n_levels=4)
    points = decode_candidates(cand, column_map, s.variables)
    recoded = encode_candidates(points, column_map, s.variables)

    x2_col = next(i for i, cm in enumerate(column_map) if cm["var_name"] == "x2")
    assert not np.allclose(recoded[:, x2_col], cand[:, x2_col], atol=1e-9)

    # Pin the exact mechanism, using the specific values a reviewer verified:
    # coded -1/3 decodes to 3 (not 2.667) and re-encodes to -0.25 (not -1/3).
    decoded_x2 = sorted({p["x2"] for p in points})
    assert decoded_x2 == [0, 3, 5, 8]


def test_generate_mixed_candidate_set_handles_duplicate_categorical_values():
    """SearchSpace.add_variable does not reject duplicate categorical
    `values` (unlike discrete, which rejects duplicate `allowed_values`), so
    this is reachable through the public API. Each category *position* must
    still get its own one-hot column: cats.index(cat_val) would resolve both
    "a" columns to the same index, corrupting the encoding (one row with two
    columns set, one row with none). A local positional counter must
    reproduce the original enumerate(cats)-based encoding: the identity
    matrix.
    """
    s = SearchSpace()
    s.add_variable("cat", "categorical", values=["a", "a", "b"])
    candidates, column_map = generate_mixed_candidate_set(s, n_levels=2)

    assert np.array_equal(candidates, np.eye(3))
    # Every row is a valid one-hot indicator (exactly one column set).
    assert np.array_equal(candidates.sum(axis=1), np.ones(3))


def test_encode_raises_on_unknown_category_value():
    """A value that matches none of a variable's known categories must not
    silently become an all-zero one-hot row — decode_candidates' argmax
    would resolve an all-zero row to category index 0, silently corrupting
    the point into "the first category" instead of failing loudly.
    """
    s = SearchSpace()
    s.add_variable("c", "categorical", values=["a", "b"])
    column_map = build_column_map(s.variables)
    with pytest.raises(ValueError):
        encode_candidates([{"c": "not-a-known-category"}], column_map, s.variables)


def test_encode_raises_on_missing_categorical_key():
    """A point missing a categorical variable's key must fail loudly, the
    same way it already does for continuous variables (point[name] raises),
    rather than silently emitting an all-zero one-hot row.
    """
    s = SearchSpace()
    s.add_variable("c", "categorical", values=["a", "b"])
    column_map = build_column_map(s.variables)
    with pytest.raises(KeyError):
        encode_candidates([{}], column_map, s.variables)
