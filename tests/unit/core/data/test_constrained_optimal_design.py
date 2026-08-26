"""Constrained D/A/I-optimal designs select from a feasible, boundary-aware set."""

import numpy as np
import pandas as pd
import pytest

from alchemist_core import OptimizationSession
from alchemist_core.utils.constrained_region import InfeasibleRegionError

FEAS_TOL = 1e-9


def _session():
    s = OptimizationSession()
    s.add_variable("x1", "real", bounds=(0.0, 10.0))
    s.add_variable("x2", "real", bounds=(0.0, 10.0))
    s.add_variable("x3", "real", bounds=(0.0, 10.0))
    return s


def test_every_point_of_a_constrained_optimal_design_is_feasible():
    s = _session()
    s.add_input_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=12.0)
    points, _info = s.generate_optimal_design(
        n_points=10, model_type="quadratic", random_seed=7
    )
    df = pd.DataFrame(points)
    assert ((df["x1"] + df["x2"]) <= 12.0 + FEAS_TOL).all()


def test_design_reaches_the_constraint_boundary():
    """Filter-only would leave a dead band; augmentation must not."""
    s = _session()
    s.add_input_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=12.0)
    points, _info = s.generate_optimal_design(
        n_points=12, model_type="quadratic", random_seed=7
    )
    df = pd.DataFrame(points)
    closest = (12.0 - (df["x1"] + df["x2"])).min()
    # A filtered 5-level lattice on [0,10] can get no closer than 2.0 here.
    assert closest < 0.5


def test_feasibility_info_is_reported():
    s = _session()
    s.add_input_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=12.0)
    _points, info = s.generate_optimal_design(
        n_points=10, model_type="quadratic", random_seed=7
    )
    feas = info["feasibility"]
    assert feas["constraints_applied"] == ["constraint_0"]
    assert feas["n_candidates_feasible"] < feas["n_candidates_total"]
    assert feas["n_boundary_added"] + feas["n_vertices_added"] > 0


def test_unconstrained_design_reports_no_feasibility_block():
    s = _session()
    _points, info = s.generate_optimal_design(
        n_points=10, model_type="quadratic", random_seed=7
    )
    assert info["feasibility"] is None


def test_equality_constrained_design_lies_on_the_hyperplane():
    s = _session()
    s.add_input_constraint("equality", {"x1": 1.0, "x2": 1.0}, rhs=8.0)
    # Deviation from the brief: model_type="linear" includes x1 and x2 as
    # separate main effects. On this hyperplane x2 = 8 - x1 exactly, so
    # [intercept, x1, x2] are exactly collinear and run_optimal_design's
    # pre-existing rank-deficiency guard (unrelated to this task) correctly
    # raises ValueError before an exchange algorithm ever runs. This is a
    # real, unavoidable property of any equality constraint tying two
    # variables that are both included as pure main effects — not a defect
    # in boundary augmentation. x1*x2 keeps x2 "in the model" (so the
    # unused-variable space-filling step does not overwrite it with values
    # that ignore the constraint) without reintroducing that collinearity.
    points, _info = s.generate_optimal_design(
        n_points=8, effects=["x1", "x3", "x1*x2"], random_seed=7
    )
    df = pd.DataFrame(points)
    assert np.allclose(df["x1"] + df["x2"], 8.0, atol=1e-5)


def test_empty_feasible_region_raises():
    s = _session()
    s.add_input_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=-1.0)
    with pytest.raises(InfeasibleRegionError):
        s.generate_optimal_design(n_points=8, model_type="linear", random_seed=7)


def test_constrained_design_beats_filter_only_on_d_efficiency():
    """The quantitative justification for augmenting rather than just filtering.

    Builds the real filter-only baseline a pre-Task-7 implementation would have
    optimized over -- the raw lattice with infeasible rows simply dropped, no
    boundary or vertex points added -- runs the same exchange algorithm over
    it, and asserts the augmented design's log|X'X| strictly beats it. This is
    the feature's actual headline claim (the 8->23-candidate expansion from the
    Task 5 worked example buys a better design), not merely "the matrix isn't
    singular".
    """
    from alchemist_core.utils.optimal_design import (
        build_custom_design_matrix, build_column_map, encode_candidates,
        decode_candidates, generate_mixed_candidate_set, parse_model_spec,
        _run_algorithm,
    )

    s = _session()
    s.add_input_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=12.0)
    terms = parse_model_spec(s.search_space, model_type="quadratic")
    column_map = build_column_map(s.search_space.variables)

    def _logdet(coded):
        X = build_custom_design_matrix(coded, terms, column_map,
                                       s.search_space.variables)
        sign, logabsdet = np.linalg.slogdet(X.T @ X)
        return logabsdet if sign > 0 else -np.inf

    # Filter-only baseline: same lattice, same exchange algorithm, same seed
    # -- only the candidate set differs (filtered, not augmented).
    lattice_coded, lattice_column_map = generate_mixed_candidate_set(
        s.search_space, n_levels=5
    )
    lattice_points = decode_candidates(
        lattice_coded, lattice_column_map, s.search_space.variables
    )
    lattice_df = pd.DataFrame(lattice_points)
    feasible_df = lattice_df[s.search_space.filter_feasible(lattice_df)]
    feasible_df = feasible_df.reset_index(drop=True)
    feasible_coded = encode_candidates(
        feasible_df.to_dict("records"), column_map, s.search_space.variables
    )
    filter_only_design_matrix = build_custom_design_matrix(
        feasible_coded, terms, column_map, s.search_space.variables
    )
    selected, _run_info = _run_algorithm(
        candidates_coded=feasible_coded,
        design_matrix_candidates=filter_only_design_matrix,
        n_points=12,
        criterion="D",
        algorithm="fedorov",
        max_iter=200,
        terms=terms,
        column_map=column_map,
        variables=s.search_space.variables,
        random_seed=7,
    )
    filter_only_logdet = _logdet(feasible_coded[selected])

    augmented, _info = s.generate_optimal_design(
        n_points=12, model_type="quadratic", random_seed=7
    )
    augmented_coded = encode_candidates(
        augmented, column_map, s.search_space.variables
    )
    augmented_logdet = _logdet(augmented_coded)

    assert augmented_logdet > filter_only_logdet


def test_equality_constraint_rank_failure_names_the_constraint():
    """The rank-deficiency error must name equality constraints as a cause.

    Any two-variable equality constraint makes model_type='linear' (and
    'interaction' and 'quadratic', which subsume it) unusable: on the
    hyperplane every feasible candidate satisfies the tied variables' relation
    exactly, so their coded main-effect columns are exactly collinear with the
    intercept. Today's rank-deficiency guard already catches this (correctly)
    but its "Likely causes" list didn't mention equality constraints -- a user
    hitting this would see three unrelated causes and have to reverse-engineer
    the real one themselves.
    """
    s = _session()
    s.add_input_constraint("equality", {"x1": 1.0, "x2": 1.0}, rhs=8.0)
    with pytest.raises(ValueError, match="equality constraint"):
        s.generate_optimal_design(n_points=8, model_type="linear", random_seed=7)
