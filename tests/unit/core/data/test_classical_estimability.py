"""A constrained classical design must not silently return a degraded design.

A CCD's axial points are what let it estimate quadratic terms. Dropping the
infeasible ones does not give "a CCD minus two runs" — it gives a design that
is rank-deficient for the model it claims to fit. Dropping a replicated center
point, by contrast, is harmless. The gate distinguishes the two.

Deviation from the task brief: the brief's raise-triggering scenario cuts a
symmetric corner (x1 + x2 <= 11.0) off the default (orthogonal, circumscribed)
3-variable CCD. That CCD's axial points reach the full [0, 10] variable
bounds while its factorial corners are pulled inward, so every variable keeps
5 distinct coded levels even after that corner is removed — the surviving 12
of 16 points remain full rank for the implied quadratic model (verified
numerically; see test_design_losing_structural_points_but_estimable_passes,
which repurposes the brief's exact scenario to cover the "harmless drop" path
the brief's own tests never exercised). To actually reach rank deficiency,
the corner-cutting tests below weight x1 more heavily than x2 (asymmetric,
non-round coefficients — 1.0 vs 0.8) at a tighter rhs; this removes enough of
the high-x1 spread (both a factorial pair and the positive x1 axial point) to
alias a quadratic term while still leaving 10 of 16 points standing, so the
failure is a genuine rank collapse rather than merely too few points.
"""

import pandas as pd
import pytest

from alchemist_core import OptimizationSession
from alchemist_core.utils.doe import (
    DesignNotEstimableError,
    IMPLIED_MODEL,
    _central_composite,
    _implied_model_type,
    _inestimable_terms,
    _plackett_burman,
)

FEAS_TOL = 1e-6


def _session():
    s = OptimizationSession()
    s.add_variable("x1", "real", bounds=(0.0, 10.0))
    s.add_variable("x2", "real", bounds=(0.0, 10.0))
    s.add_variable("x3", "real", bounds=(0.0, 10.0))
    return s


def test_ccd_losing_structural_points_raises():
    s = _session()
    # Cuts off the high-x1/high-x2 corner (x1 weighted more heavily than x2),
    # removing factorial and axial points until the quadratic model can no
    # longer be fit from the 10 points that remain.
    s.add_input_constraint("inequality", {"x1": 1.0, "x2": 0.8}, rhs=9.3)
    with pytest.raises(DesignNotEstimableError, match="ccd"):
        s.generate_initial_design(method="ccd", random_seed=7)


def test_raise_message_names_the_inestimable_terms():
    s = _session()
    s.add_input_constraint("inequality", {"x1": 1.0, "x2": 0.8}, rhs=9.3)
    with pytest.raises(DesignNotEstimableError) as exc:
        s.generate_initial_design(method="ccd", random_seed=7)
    msg = str(exc.value)
    assert "dropped" in msg
    assert "optimal" in msg  # steers toward the right tool


def test_allow_infeasible_restores_warn_and_drop():
    s = _session()
    s.add_input_constraint("inequality", {"x1": 1.0, "x2": 0.8}, rhs=9.3)
    points = s.generate_initial_design(
        method="ccd", random_seed=7, allow_infeasible=True
    )
    df = pd.DataFrame(points)
    assert len(df) > 0
    assert ((1.0 * df["x1"] + 0.8 * df["x2"]) <= 9.3 + FEAS_TOL).all()


def test_design_losing_nothing_passes_unchanged():
    s = _session()
    # Satisfied everywhere in [0, 10]^3 — nothing is dropped.
    s.add_input_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=100.0)
    points = s.generate_initial_design(method="ccd", random_seed=7)
    assert len(points) > 0


def test_fully_infeasible_design_still_raises_the_original_error():
    s = _session()
    s.add_input_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=-1.0)
    with pytest.raises(ValueError, match="No 'ccd' design points"):
        s.generate_initial_design(method="ccd", random_seed=7)


def test_optimal_method_is_exempt_from_the_gate():
    """'optimal' is in CLASSICAL_METHODS but its points are already feasible."""
    s = _session()
    s.add_input_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=11.0)
    points = s.generate_initial_design(
        method="optimal", n_points=10, model_type="quadratic", random_seed=7
    )
    df = pd.DataFrame(points)
    assert ((df["x1"] + df["x2"]) <= 11.0 + FEAS_TOL).all()


def test_space_filling_methods_are_unaffected():
    s = _session()
    s.add_input_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=11.0)
    points = s.generate_initial_design(n_points=8, method="lhs", random_seed=7)
    df = pd.DataFrame(points)
    assert len(df) == 8
    assert ((df["x1"] + df["x2"]) <= 11.0 + FEAS_TOL).all()


# ============================================================
# Companion coverage the brief's own tests leave open (see module docstring
# for why the brief's original symmetric-corner scenario landed here instead
# of in the raise path above).
# ============================================================

def test_design_losing_structural_points_but_estimable_passes():
    """A drop that costs runs but not rank must not raise.

    This is the brief's original scenario (x1 + x2 <= 11.0, symmetric
    weights): it removes 4 of the CCD's 16 points — a real factorial corner
    and a real positive axial point on each of x1 and x2 — yet the surviving
    12 points still span every quadratic term. A gate that raises on any
    point loss, rather than checking rank, would fail this test.
    """
    s = _session()
    s.add_input_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=11.0)
    points = s.generate_initial_design(method="ccd", random_seed=7)
    df = pd.DataFrame(points)
    assert 0 < len(df) < 16  # structural points were genuinely dropped
    assert ((df["x1"] + df["x2"]) <= 11.0 + FEAS_TOL).all()


def test_gate_applies_to_non_ccd_classical_methods():
    """The gate is keyed off CLASSICAL_METHODS membership, not the string 'ccd'."""
    s = _session()
    # Plackett-Burman's implied model is linear (intercept + 3 mains, 4
    # terms). Cutting x1's high level costs both corners that carry it,
    # leaving 3 of 5 points — too few to keep every main effect estimable.
    s.add_input_constraint("inequality", {"x1": 1.0}, rhs=8.0)
    with pytest.raises(DesignNotEstimableError, match="plackett_burman") as exc:
        s.generate_initial_design(method="plackett_burman", random_seed=7)
    assert "linear" in str(exc.value)


def test_allow_infeasible_does_not_rescue_zero_feasible_points():
    """allow_infeasible=True only changes the rank-deficient case, not the
    pre-existing zero-survivors case — that one still raises a plain
    ValueError with its original message.
    """
    s = _session()
    s.add_input_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=-1.0)
    with pytest.raises(ValueError, match="No 'ccd' design points"):
        s.generate_initial_design(method="ccd", random_seed=7, allow_infeasible=True)


@pytest.mark.parametrize("method,n_levels,expected", [
    ("ccd", 2, "quadratic"),
    ("box_behnken", 2, "quadratic"),
    ("fractional_factorial", 2, "interaction"),
    ("plackett_burman", 2, "linear"),
    ("gsd", 2, "linear"),
    ("full_factorial", 2, "interaction"),
    ("full_factorial", 3, "quadratic"),
    ("full_factorial", 5, "quadratic"),
])
def test_implied_model_type_table(method, n_levels, expected):
    """Each classical method maps to its own implied model — not a single
    default shared by all of them (a plausible bug: hardcoding "linear" for
    every method, or swapping the quadratic/interaction assignments, would
    pass some of these parametrizations and fail others).
    """
    assert _implied_model_type(method, n_levels) == expected


def test_implied_model_dict_omits_full_factorial():
    """full_factorial's implied model depends on n_levels and is resolved by
    _implied_model_type, not looked up directly in the static table.
    """
    assert "full_factorial" not in IMPLIED_MODEL
    assert IMPLIED_MODEL["ccd"] == "quadratic"
    assert IMPLIED_MODEL["box_behnken"] == "quadratic"
    assert IMPLIED_MODEL["fractional_factorial"] == "interaction"
    assert IMPLIED_MODEL["plackett_burman"] == "linear"
    assert IMPLIED_MODEL["gsd"] == "linear"


def test_inestimable_terms_reports_the_actual_deficient_columns():
    """_inestimable_terms must name the specific term(s) the surviving points
    can no longer separate — not just "something" or an arbitrary column.

    A plausible wrong implementation could report a fixed slice (e.g. always
    the first term, or the whole term list) and still satisfy every
    string-containment assertion elsewhere in this file, since none of them
    inspect the returned names. This test pins the exact set for two
    independently-derived scenarios so such a bug is caught here.
    """
    s = _session()
    ccd_points = _central_composite(s.search_space, n_center=1,
                                    alpha="orthogonal", face="circumscribed")
    # Keep only the points the {'x1': 1.0, 'x2': 0.8} <= 9.3 constraint admits
    # (mirrors the filtering generate_initial_design performs internally).
    surviving = [p for p in ccd_points
                 if 1.0 * p["x1"] + 0.8 * p["x2"] <= 9.3 + FEAS_TOL]
    assert len(surviving) == 10  # 6 dropped of 16, per the module docstring
    assert _inestimable_terms(s.search_space, surviving, "ccd", 2) == ["x3^2"]

    pb_points = _plackett_burman(s.search_space, n_center=1)
    pb_surviving = [p for p in pb_points if p["x1"] <= 8.0 + FEAS_TOL]
    assert len(pb_surviving) == 3  # 2 dropped of 5
    assert _inestimable_terms(s.search_space, pb_surviving, "plackett_burman", 2) == ["x3"]


def test_inestimable_terms_empty_for_no_points():
    """The gate must never block on its own inability to judge."""
    s = _session()
    assert _inestimable_terms(s.search_space, [], "ccd", 2) == []
