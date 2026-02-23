"""
Tests for optimal experimental design (OED) functionality.

Tests cover:
- Model specification parsing (parse_model_spec)
- Custom design matrix building (build_custom_design_matrix)
- Mixed candidate set generation (generate_mixed_candidate_set)
- Main orchestrator (run_optimal_design)
- Integration through generate_initial_design(method="optimal")
- All 5 algorithms
- Edge cases and error handling
"""

import numpy as np
import pytest

from alchemist_core.data.search_space import SearchSpace
from alchemist_core.utils.optimal_design import (
    build_custom_design_matrix,
    generate_mixed_candidate_set,
    get_model_term_names,
    parse_model_spec,
    run_optimal_design,
)
from alchemist_core.utils.doe import generate_initial_design, get_design_info


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def continuous_space():
    """SearchSpace with 3 continuous variables."""
    ss = SearchSpace()
    ss.add_variable("Temperature", "real", min=200, max=400)
    ss.add_variable("Pressure", "real", min=1, max=10)
    ss.add_variable("Flow_Rate", "real", min=0.5, max=5.0)
    return ss


@pytest.fixture
def two_var_space():
    """SearchSpace with 2 continuous variables."""
    ss = SearchSpace()
    ss.add_variable("Temperature", "real", min=200, max=400)
    ss.add_variable("Pressure", "real", min=1, max=10)
    return ss


@pytest.fixture
def mixed_space():
    """SearchSpace with continuous and categorical variables."""
    ss = SearchSpace()
    ss.add_variable("Temperature", "real", min=200, max=400)
    ss.add_variable("Pressure", "real", min=1, max=10)
    ss.add_variable("Catalyst", "categorical", values=["Pt", "Pd", "Ru"])
    return ss


@pytest.fixture
def integer_space():
    """SearchSpace with an integer variable."""
    ss = SearchSpace()
    ss.add_variable("Temperature", "real", min=200, max=400)
    ss.add_variable("N_Layers", "integer", min=1, max=5)
    return ss


@pytest.fixture
def categorical_only_space():
    """SearchSpace with only categorical variables."""
    ss = SearchSpace()
    ss.add_variable("Catalyst", "categorical", values=["Pt", "Pd", "Ru"])
    ss.add_variable("Support", "categorical", values=["Al2O3", "SiO2"])
    return ss


# ============================================================
# parse_model_spec tests
# ============================================================

class TestParseModelSpec:
    """Tests for model specification parsing."""

    def test_linear_shortcut(self, continuous_space):
        terms = parse_model_spec(continuous_space, model_type="linear")
        # Intercept + 3 main effects = 4 terms
        assert len(terms) == 4
        assert () in terms  # intercept

    def test_interaction_shortcut(self, continuous_space):
        terms = parse_model_spec(continuous_space, model_type="interaction")
        # Intercept + 3 main + 3 interactions = 7
        assert len(terms) == 7

    def test_quadratic_shortcut(self, continuous_space):
        terms = parse_model_spec(continuous_space, model_type="quadratic")
        # Intercept + 3 main + 3 interactions + 3 quadratic = 10
        assert len(terms) == 10

    def test_quadratic_skips_categorical(self, mixed_space):
        """Quadratic terms should only be generated for continuous vars."""
        terms = parse_model_spec(mixed_space, model_type="quadratic")
        # Intercept + 3 main + 3 interactions + 2 quadratic (Temperature, Pressure only) = 9
        assert len(terms) == 9

    def test_custom_effects_main(self, continuous_space):
        terms = parse_model_spec(
            continuous_space,
            effects=["Temperature", "Pressure"],
        )
        # Intercept + 2 main effects = 3
        assert len(terms) == 3

    def test_custom_effects_interaction(self, continuous_space):
        terms = parse_model_spec(
            continuous_space,
            effects=["Temperature", "Pressure", "Temperature*Pressure"],
        )
        # Intercept + 2 main + 1 interaction = 4
        assert len(terms) == 4

    def test_custom_effects_quadratic(self, continuous_space):
        terms = parse_model_spec(
            continuous_space,
            effects=["Temperature", "Temperature**2"],
        )
        # Intercept + 1 main + 1 quadratic = 3
        assert len(terms) == 3

    def test_custom_effects_mixed(self, continuous_space):
        terms = parse_model_spec(
            continuous_space,
            effects=["Temperature", "Pressure", "Flow_Rate",
                     "Temperature*Pressure", "Temperature**2"],
        )
        # Intercept + 3 main + 1 interaction + 1 quadratic = 6
        assert len(terms) == 6

    def test_duplicate_effects_deduplicated(self, continuous_space):
        terms = parse_model_spec(
            continuous_space,
            effects=["Temperature", "Temperature", "Pressure"],
        )
        # Intercept + 2 unique main = 3
        assert len(terms) == 3

    def test_error_both_model_type_and_effects(self, continuous_space):
        with pytest.raises(ValueError, match="not both"):
            parse_model_spec(
                continuous_space,
                model_type="linear",
                effects=["Temperature"],
            )

    def test_error_neither_model_type_nor_effects(self, continuous_space):
        with pytest.raises(ValueError, match="Specify either"):
            parse_model_spec(continuous_space)

    def test_error_unknown_model_type(self, continuous_space):
        with pytest.raises(ValueError, match="Unknown model_type"):
            parse_model_spec(continuous_space, model_type="cubic")

    def test_error_unknown_variable(self, continuous_space):
        with pytest.raises(ValueError, match="Unknown variable"):
            parse_model_spec(
                continuous_space,
                effects=["Temperature", "NonexistentVar"],
            )

    def test_error_unknown_variable_in_interaction(self, continuous_space):
        with pytest.raises(ValueError, match="Unknown variable"):
            parse_model_spec(
                continuous_space,
                effects=["Temperature*NonexistentVar"],
            )

    def test_error_quadratic_on_categorical(self, mixed_space):
        with pytest.raises(ValueError, match="not valid for categorical"):
            parse_model_spec(
                mixed_space,
                effects=["Catalyst**2"],
            )

    def test_interaction_canonical_ordering(self, continuous_space):
        """Interaction terms should have indices in sorted order."""
        terms = parse_model_spec(
            continuous_space,
            effects=["Pressure*Temperature"],  # Reversed order
        )
        # Should be sorted: Temperature(0) < Pressure(1)
        interaction_term = [t for t in terms if len(t) == 2][0]
        assert interaction_term[0][0] < interaction_term[1][0]


# ============================================================
# get_model_term_names tests
# ============================================================

class TestGetModelTermNames:

    def test_term_names_linear(self, continuous_space):
        terms = parse_model_spec(continuous_space, model_type="linear")
        names = get_model_term_names(continuous_space, terms)
        assert names[0] == "Intercept"
        assert "Temperature" in names
        assert "Pressure" in names
        assert "Flow_Rate" in names

    def test_term_names_quadratic(self, two_var_space):
        terms = parse_model_spec(two_var_space, model_type="quadratic")
        names = get_model_term_names(two_var_space, terms)
        assert "Temperature^2" in names
        assert "Temperature*Pressure" in names


# ============================================================
# generate_mixed_candidate_set tests
# ============================================================

class TestGenerateMixedCandidateSet:

    def test_continuous_only(self, two_var_space):
        candidates, col_map = generate_mixed_candidate_set(two_var_space, n_levels=3)
        # 3 levels × 3 levels = 9 candidates, 2 columns
        assert candidates.shape == (9, 2)
        assert len(col_map) == 2
        assert all(cm["type"] == "continuous" for cm in col_map)

    def test_continuous_only_five_levels(self, two_var_space):
        candidates, col_map = generate_mixed_candidate_set(two_var_space, n_levels=5)
        assert candidates.shape == (25, 2)

    def test_mixed_space(self, mixed_space):
        candidates, col_map = generate_mixed_candidate_set(mixed_space, n_levels=3)
        # 3 levels × 3 levels × 3 categories = 27 candidates
        # Columns: Temperature(1) + Pressure(1) + Catalyst(3 onehot) = 5
        assert candidates.shape == (27, 5)
        assert len(col_map) == 5
        continuous_cols = [cm for cm in col_map if cm["type"] == "continuous"]
        onehot_cols = [cm for cm in col_map if cm["type"] == "onehot"]
        assert len(continuous_cols) == 2
        assert len(onehot_cols) == 3

    def test_categorical_only(self, categorical_only_space):
        candidates, col_map = generate_mixed_candidate_set(
            categorical_only_space, n_levels=3
        )
        # 3 Catalyst × 2 Support = 6 candidates
        # Columns: 3 onehot + 2 onehot = 5
        assert candidates.shape == (6, 5)

    def test_onehot_rows_sum_to_one(self, mixed_space):
        """Each row should have exactly one 1.0 per categorical variable."""
        candidates, col_map = generate_mixed_candidate_set(mixed_space, n_levels=3)
        # Catalyst is variable index 2, with 3 categories
        cat_cols = [i for i, cm in enumerate(col_map)
                    if cm["var_name"] == "Catalyst"]
        for row in candidates:
            assert np.sum(row[cat_cols]) == 1.0

    def test_continuous_values_in_range(self, two_var_space):
        candidates, _ = generate_mixed_candidate_set(two_var_space, n_levels=5)
        assert np.all(candidates >= -1.0)
        assert np.all(candidates <= 1.0)

    def test_integer_variable(self, integer_space):
        candidates, col_map = generate_mixed_candidate_set(integer_space, n_levels=3)
        assert candidates.shape[0] == 9
        assert candidates.shape[1] == 2


# ============================================================
# build_custom_design_matrix tests
# ============================================================

class TestBuildCustomDesignMatrix:

    def test_intercept_only(self, two_var_space):
        candidates, col_map = generate_mixed_candidate_set(two_var_space, n_levels=3)
        terms = [()]  # intercept only
        X = build_custom_design_matrix(
            candidates, terms, col_map, two_var_space.variables
        )
        assert X.shape == (9, 1)
        assert np.all(X[:, 0] == 1.0)

    def test_linear_model(self, two_var_space):
        candidates, col_map = generate_mixed_candidate_set(two_var_space, n_levels=3)
        terms = parse_model_spec(two_var_space, model_type="linear")
        X = build_custom_design_matrix(
            candidates, terms, col_map, two_var_space.variables
        )
        # Intercept + 2 main effects = 3 columns
        assert X.shape == (9, 3)
        assert np.all(X[:, 0] == 1.0)  # intercept

    def test_quadratic_model(self, two_var_space):
        candidates, col_map = generate_mixed_candidate_set(two_var_space, n_levels=5)
        terms = parse_model_spec(two_var_space, model_type="quadratic")
        X = build_custom_design_matrix(
            candidates, terms, col_map, two_var_space.variables
        )
        # Intercept + 2 main + 1 interaction + 2 quadratic = 6
        assert X.shape == (25, 6)

    def test_interaction_term_values(self, two_var_space):
        """Interaction column should be product of the two main effect columns."""
        candidates, col_map = generate_mixed_candidate_set(two_var_space, n_levels=3)
        terms = parse_model_spec(
            two_var_space,
            effects=["Temperature", "Pressure", "Temperature*Pressure"],
        )
        X = build_custom_design_matrix(
            candidates, terms, col_map, two_var_space.variables
        )
        # X[:, 3] should equal X[:, 1] * X[:, 2]
        np.testing.assert_allclose(X[:, 3], X[:, 1] * X[:, 2])

    def test_quadratic_term_values(self, two_var_space):
        """Quadratic column should be square of the main effect column."""
        candidates, col_map = generate_mixed_candidate_set(two_var_space, n_levels=5)
        terms = parse_model_spec(
            two_var_space,
            effects=["Temperature", "Temperature**2"],
        )
        X = build_custom_design_matrix(
            candidates, terms, col_map, two_var_space.variables
        )
        # X[:, 2] should equal X[:, 1] ** 2
        np.testing.assert_allclose(X[:, 2], X[:, 1] ** 2)

    def test_categorical_main_effect_expansion(self, mixed_space):
        """Categorical main effect should expand to k-1 dummy columns."""
        candidates, col_map = generate_mixed_candidate_set(mixed_space, n_levels=3)
        # Just intercept + Catalyst main effect
        terms = parse_model_spec(mixed_space, effects=["Catalyst"])
        X = build_custom_design_matrix(
            candidates, terms, col_map, mixed_space.variables
        )
        # Intercept (1) + Catalyst dummy (k-1 = 2) = 3 columns
        assert X.shape[1] == 3

    def test_categorical_interaction_expansion(self, mixed_space):
        """Catalyst*Temperature interaction should expand to k-1 columns."""
        candidates, col_map = generate_mixed_candidate_set(mixed_space, n_levels=3)
        terms = parse_model_spec(
            mixed_space,
            effects=["Temperature", "Catalyst", "Catalyst*Temperature"],
        )
        X = build_custom_design_matrix(
            candidates, terms, col_map, mixed_space.variables
        )
        # Intercept(1) + Temperature(1) + Catalyst(2) + Catalyst*Temperature(2) = 6
        assert X.shape[1] == 6


# ============================================================
# run_optimal_design tests
# ============================================================

class TestRunOptimalDesign:

    def test_basic_d_optimal(self, continuous_space):
        points, info = run_optimal_design(
            continuous_space, n_points=10, model_type="linear",
            criterion="D", algorithm="sequential", n_levels=3,
            random_seed=42,
        )
        assert len(points) == 10
        assert all(isinstance(p, dict) for p in points)
        assert all("Temperature" in p for p in points)
        assert info["criterion"] == "D"
        assert info["n_runs"] == 10
        assert "model_terms" in info

    def test_a_optimal(self, two_var_space):
        points, info = run_optimal_design(
            two_var_space, n_points=8, model_type="quadratic",
            criterion="A", algorithm="sequential", n_levels=3,
            random_seed=42,
        )
        assert len(points) == 8
        assert info["criterion"] == "A"

    def test_i_optimal(self, two_var_space):
        points, info = run_optimal_design(
            two_var_space, n_points=8, model_type="quadratic",
            criterion="I", algorithm="sequential", n_levels=3,
            random_seed=42,
        )
        assert len(points) == 8
        assert info["criterion"] == "I"

    def test_custom_effects(self, continuous_space):
        points, info = run_optimal_design(
            continuous_space, n_points=8,
            effects=["Temperature", "Pressure", "Temperature*Pressure"],
            criterion="D", algorithm="sequential", n_levels=3,
            random_seed=42,
        )
        assert len(points) == 8
        assert info["p_columns"] == 4  # intercept + 3 effects

    def test_mixed_space_generates_warning(self, mixed_space, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            points, info = run_optimal_design(
                mixed_space, n_points=6, model_type="linear",
                criterion="D", algorithm="sequential", n_levels=3,
                random_seed=42,
            )
        assert len(points) == 6
        assert "experimental" in caplog.text.lower() or "dummy" in caplog.text.lower()

    def test_values_within_bounds(self, continuous_space):
        points, _ = run_optimal_design(
            continuous_space, n_points=15, model_type="quadratic",
            criterion="D", algorithm="sequential", n_levels=5,
            random_seed=42,
        )
        for p in points:
            assert 200 <= p["Temperature"] <= 400
            assert 1 <= p["Pressure"] <= 10
            assert 0.5 <= p["Flow_Rate"] <= 5.0

    def test_integer_rounding(self, integer_space):
        points, _ = run_optimal_design(
            integer_space, n_points=6, model_type="linear",
            criterion="D", algorithm="sequential", n_levels=5,
            random_seed=42,
        )
        for p in points:
            assert isinstance(p["N_Layers"], int)
            assert 1 <= p["N_Layers"] <= 5

    def test_categorical_values_valid(self, mixed_space):
        points, _ = run_optimal_design(
            mixed_space, n_points=6, model_type="linear",
            criterion="D", algorithm="sequential", n_levels=3,
            random_seed=42,
        )
        for p in points:
            assert p["Catalyst"] in ["Pt", "Pd", "Ru"]

    def test_model_terms_in_info(self, continuous_space):
        _, info = run_optimal_design(
            continuous_space, n_points=10, model_type="quadratic",
            criterion="D", algorithm="sequential", n_levels=3,
            random_seed=42,
        )
        assert "Intercept" in info["model_terms"]
        assert "Temperature" in info["model_terms"]
        assert "Temperature*Pressure" in info["model_terms"]
        assert "Temperature^2" in info["model_terms"]

    def test_efficiency_metrics_in_info(self, two_var_space):
        _, info = run_optimal_design(
            two_var_space, n_points=10, model_type="quadratic",
            criterion="D", algorithm="sequential", n_levels=5,
            random_seed=42,
        )
        assert "D_eff" in info
        assert "A_eff" in info
        assert isinstance(info["D_eff"], float)
        assert isinstance(info["A_eff"], float)

    def test_error_empty_space(self):
        ss = SearchSpace()
        with pytest.raises(ValueError, match="no variables"):
            run_optimal_design(ss, n_points=5, model_type="linear")

    def test_error_unknown_criterion(self, two_var_space):
        with pytest.raises(ValueError, match="Unknown criterion"):
            run_optimal_design(
                two_var_space, n_points=5, model_type="linear",
                criterion="X",
            )

    def test_error_unknown_algorithm(self, two_var_space):
        with pytest.raises(ValueError, match="Unknown algorithm"):
            run_optimal_design(
                two_var_space, n_points=5, model_type="linear",
                algorithm="bogus",
            )

    def test_error_n_points_zero(self, two_var_space):
        with pytest.raises(ValueError, match="n_points must be >= 1"):
            run_optimal_design(
                two_var_space, n_points=0, model_type="linear",
            )

    def test_reproducibility_with_seed(self, continuous_space):
        points1, _ = run_optimal_design(
            continuous_space, n_points=8, model_type="linear",
            criterion="D", algorithm="sequential", n_levels=3,
            random_seed=123,
        )
        points2, _ = run_optimal_design(
            continuous_space, n_points=8, model_type="linear",
            criterion="D", algorithm="sequential", n_levels=3,
            random_seed=123,
        )
        assert points1 == points2


# ============================================================
# Algorithm tests
# ============================================================

class TestAlgorithms:
    """Test that all 5 algorithms produce valid designs."""

    @pytest.mark.parametrize("algorithm", [
        "sequential", "simple_exchange", "fedorov",
        "modified_fedorov", "detmax",
    ])
    def test_algorithm_produces_valid_design(self, two_var_space, algorithm):
        points, info = run_optimal_design(
            two_var_space, n_points=6, model_type="linear",
            criterion="D", algorithm=algorithm, n_levels=3,
            random_seed=42, max_iter=20,
        )
        assert len(points) == 6
        assert info["algorithm"] == algorithm
        assert info["n_runs"] == 6
        # All points should have valid variable values
        for p in points:
            assert 200 <= p["Temperature"] <= 400
            assert 1 <= p["Pressure"] <= 10

    @pytest.mark.parametrize("criterion", ["D", "A", "I"])
    def test_criterion_with_fedorov(self, two_var_space, criterion):
        points, info = run_optimal_design(
            two_var_space, n_points=6, model_type="linear",
            criterion=criterion, algorithm="fedorov", n_levels=3,
            random_seed=42, max_iter=20,
        )
        assert len(points) == 6
        assert info["criterion"] == criterion


# ============================================================
# Integration through generate_initial_design (doe.py)
# ============================================================

class TestDoeIntegration:

    def test_generate_initial_design_optimal(self, continuous_space):
        points = generate_initial_design(
            continuous_space, method="optimal", n_points=10,
            model_type="quadratic", criterion="D", algorithm="sequential",
            n_levels=5, random_seed=42,
        )
        assert len(points) == 10
        assert all("Temperature" in p for p in points)

    def test_generate_initial_design_optimal_custom_effects(self, continuous_space):
        points = generate_initial_design(
            continuous_space, method="optimal", n_points=8,
            effects=["Temperature", "Pressure", "Temperature*Pressure"],
            criterion="D", algorithm="sequential",
            n_levels=3, random_seed=42,
        )
        assert len(points) == 8

    def test_generate_initial_design_optimal_requires_n_points(self, continuous_space):
        with pytest.raises(ValueError, match="n_points is required"):
            generate_initial_design(
                continuous_space, method="optimal",
                model_type="linear",
            )

    def test_get_design_info_optimal(self, continuous_space):
        info = get_design_info(
            "optimal", continuous_space,
            model_type="quadratic", criterion="D",
            algorithm="fedorov", n_points=15,
        )
        assert info is not None
        assert info["p_columns"] == 10  # full quadratic for 3 vars
        assert "model_terms" in info
        assert info["criterion"] == "D"
        assert info["total_runs"] == 15

    def test_get_design_info_optimal_custom_effects(self, continuous_space):
        info = get_design_info(
            "optimal", continuous_space,
            effects=["Temperature", "Pressure"],
            criterion="A", algorithm="sequential",
        )
        assert info is not None
        assert info["p_columns"] == 3  # intercept + 2 main

    def test_n_levels_default_for_optimal(self, continuous_space):
        """When n_levels=2 (default for factorial), optimal should use 5."""
        points = generate_initial_design(
            continuous_space, method="optimal", n_points=8,
            model_type="linear", criterion="D", algorithm="sequential",
            random_seed=42,
        )
        assert len(points) == 8


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:

    def test_single_variable(self):
        ss = SearchSpace()
        ss.add_variable("Temperature", "real", min=100, max=500)
        points, info = run_optimal_design(
            ss, n_points=5, model_type="quadratic",
            criterion="D", algorithm="sequential", n_levels=5,
            random_seed=42,
        )
        assert len(points) == 5
        # Quadratic model for 1 var: intercept + main + quadratic = 3
        assert info["p_columns"] == 3

    def test_many_points_from_small_grid(self, two_var_space):
        """Requesting more points than candidates should still work
        (with potential duplicates)."""
        # 3 levels × 2 vars = 9 candidates, requesting 9
        points, _ = run_optimal_design(
            two_var_space, n_points=9, model_type="linear",
            criterion="D", algorithm="sequential", n_levels=3,
            random_seed=42,
        )
        assert len(points) == 9

    def test_n_points_equals_p(self, two_var_space):
        """Minimum viable design: n_points = number of model parameters."""
        # Linear: intercept + 2 main = 3 params
        points, info = run_optimal_design(
            two_var_space, n_points=3, model_type="linear",
            criterion="D", algorithm="sequential", n_levels=5,
            random_seed=42,
        )
        assert len(points) == 3
        assert info["p_columns"] == 3

    def test_categorical_only_space(self, categorical_only_space):
        points, info = run_optimal_design(
            categorical_only_space, n_points=5, model_type="linear",
            criterion="D", algorithm="sequential", n_levels=3,
            random_seed=42,
        )
        assert len(points) == 5
        for p in points:
            assert p["Catalyst"] in ["Pt", "Pd", "Ru"]
            assert p["Support"] in ["Al2O3", "SiO2"]

    def test_three_way_interaction(self, continuous_space):
        """Three-way interaction should work."""
        points, info = run_optimal_design(
            continuous_space, n_points=10,
            effects=["Temperature", "Pressure", "Flow_Rate",
                     "Temperature*Pressure*Flow_Rate"],
            criterion="D", algorithm="sequential", n_levels=3,
            random_seed=42,
        )
        assert len(points) == 10
        assert "Temperature*Pressure*Flow_Rate" in info["model_terms"]


# ============================================================
# Usability guardrail tests (warnings + spreading)
# ============================================================

class TestUsabilityGuardrails:
    """Tests for the warnings and non-model-variable spreading added in v0.3.3."""

    # ---- Unused-variable warning (#1) ----

    def test_unused_variable_warning_emitted(self, continuous_space, caplog):
        """Warning fires when a search-space variable is absent from effects."""
        import logging
        # continuous_space has Temperature, Pressure, Flow_Rate.
        # Only include Temperature and Pressure — Flow_Rate is "free".
        with caplog.at_level(logging.WARNING):
            run_optimal_design(
                continuous_space, n_points=8,
                effects=["Temperature", "Pressure"],
                criterion="D", algorithm="sequential", n_levels=3,
                random_seed=42,
            )
        assert "NOT included in any model term" in caplog.text
        assert "Flow_Rate" in caplog.text

    def test_unused_variable_warning_not_emitted_when_all_covered(
        self, continuous_space, caplog
    ):
        """No unused-variable warning when all variables are in the effects list."""
        import logging
        with caplog.at_level(logging.WARNING):
            run_optimal_design(
                continuous_space, n_points=8,
                effects=["Temperature", "Pressure", "Flow_Rate"],
                criterion="D", algorithm="sequential", n_levels=3,
                random_seed=42,
            )
        assert "NOT included in any model term" not in caplog.text

    def test_unused_variable_warning_not_emitted_with_model_type(
        self, continuous_space, caplog
    ):
        """model_type shortcuts include all variables — no unused warning."""
        import logging
        with caplog.at_level(logging.WARNING):
            run_optimal_design(
                continuous_space, n_points=8, model_type="linear",
                criterion="D", algorithm="sequential", n_levels=3,
                random_seed=42,
            )
        assert "NOT included in any model term" not in caplog.text

    # ---- Hierarchy (marginality) warning (#2) ----

    def test_hierarchy_warning_interaction_missing_main_effect(
        self, continuous_space, caplog
    ):
        """Warning fires when interaction is specified without its main effects."""
        import logging
        # Include interaction but omit Temperature and Pressure main effects.
        with caplog.at_level(logging.WARNING):
            run_optimal_design(
                continuous_space, n_points=8,
                effects=["Flow_Rate", "Temperature*Pressure"],
                criterion="D", algorithm="sequential", n_levels=3,
                random_seed=42,
            )
        assert "marginality" in caplog.text
        # Both missing variables should be named in the warning
        assert "Temperature" in caplog.text
        assert "Pressure" in caplog.text

    def test_hierarchy_warning_quadratic_missing_main_effect(
        self, continuous_space, caplog
    ):
        """Warning fires when quadratic is specified without the main effect."""
        import logging
        with caplog.at_level(logging.WARNING):
            run_optimal_design(
                continuous_space, n_points=8,
                effects=["Pressure", "Temperature**2"],
                criterion="D", algorithm="sequential", n_levels=3,
                random_seed=42,
            )
        assert "marginality" in caplog.text
        assert "Temperature" in caplog.text

    def test_hierarchy_warning_not_emitted_when_main_effects_present(
        self, continuous_space, caplog
    ):
        """No hierarchy warning when all higher-order terms have main effects."""
        import logging
        with caplog.at_level(logging.WARNING):
            run_optimal_design(
                continuous_space, n_points=10,
                effects=["Temperature", "Pressure",
                         "Temperature*Pressure", "Temperature**2"],
                criterion="D", algorithm="sequential", n_levels=3,
                random_seed=42,
            )
        assert "marginality" not in caplog.text

    def test_hierarchy_warning_not_emitted_with_model_type(
        self, continuous_space, caplog
    ):
        """model_type='interaction' always includes main effects — no hierarchy warning."""
        import logging
        with caplog.at_level(logging.WARNING):
            run_optimal_design(
                continuous_space, n_points=10, model_type="interaction",
                criterion="D", algorithm="sequential", n_levels=3,
                random_seed=42,
            )
        assert "marginality" not in caplog.text

    # ---- Non-model variable spreading (#3) ----

    def test_free_variable_spread_across_range(self, continuous_space):
        """Non-model variables should be spread across their range, not clustered."""
        # Only include Temperature — Pressure and Flow_Rate are free.
        points, _ = run_optimal_design(
            continuous_space, n_points=10,
            effects=["Temperature"],
            criterion="D", algorithm="sequential", n_levels=5,
            random_seed=42,
        )
        pressures = [p["Pressure"] for p in points]
        flow_rates = [p["Flow_Rate"] for p in points]
        # Range should span most of the variable domain (not all the same value)
        assert max(pressures) - min(pressures) > 0, "Pressure should be spread"
        assert max(flow_rates) - min(flow_rates) > 0, "Flow_Rate should be spread"

    def test_free_variable_within_bounds(self, continuous_space):
        """Spread values for non-model variables must stay within variable bounds."""
        points, _ = run_optimal_design(
            continuous_space, n_points=10,
            effects=["Temperature"],
            criterion="D", algorithm="sequential", n_levels=5,
            random_seed=42,
        )
        for p in points:
            assert 1.0 <= p["Pressure"] <= 10.0
            assert 0.5 <= p["Flow_Rate"] <= 5.0

    def test_no_spreading_when_all_vars_in_model(self, continuous_space):
        """When all variables are in the model, the output is unchanged by spreading."""
        points1, _ = run_optimal_design(
            continuous_space, n_points=8, model_type="linear",
            criterion="D", algorithm="sequential", n_levels=3,
            random_seed=42,
        )
        points2, _ = run_optimal_design(
            continuous_space, n_points=8, model_type="linear",
            criterion="D", algorithm="sequential", n_levels=3,
            random_seed=42,
        )
        # Reproducibility should still hold (spreading code path not triggered)
        assert points1 == points2

    # ---- Session-level get_optimal_design_info() (#4) ----

    def test_get_optimal_design_info_basic(self):
        """get_optimal_design_info() returns term names and counts."""
        from alchemist_core.session import OptimizationSession
        session = OptimizationSession()
        session.add_variable("Temperature", "real", min=200, max=400)
        session.add_variable("Pressure", "real", min=1, max=10)
        session.add_variable("Flow_Rate", "real", min=0.5, max=5.0)

        info = session.get_optimal_design_info(
            effects=["Temperature", "Pressure",
                     "Temperature*Pressure", "Temperature**2"]
        )
        assert "model_terms" in info
        assert "p_columns" in info
        assert "n_points_minimum" in info
        assert "n_points_recommended" in info
        assert "Intercept" in info["model_terms"]
        assert "Temperature" in info["model_terms"]
        assert "Temperature*Pressure" in info["model_terms"]
        assert "Temperature^2" in info["model_terms"]
        assert info["p_columns"] == 5  # intercept + 2 main + 1 interaction + 1 quad
        assert info["n_points_minimum"] == info["p_columns"]
        assert info["n_points_recommended"] == 2 * info["p_columns"]

    def test_get_optimal_design_info_model_type(self):
        """get_optimal_design_info() works with model_type shortcut."""
        from alchemist_core.session import OptimizationSession
        session = OptimizationSession()
        session.add_variable("Temperature", "real", min=200, max=400)
        session.add_variable("Pressure", "real", min=1, max=10)

        info = session.get_optimal_design_info(model_type="quadratic")
        # Intercept + 2 main + 1 interaction + 2 quadratic = 6
        assert info["p_columns"] == 6
        assert info["n_points_recommended"] == 12

    def test_get_optimal_design_info_empty_space_raises(self):
        """get_optimal_design_info() raises ValueError with empty search space."""
        from alchemist_core.session import OptimizationSession
        session = OptimizationSession()
        with pytest.raises(ValueError, match="No variables"):
            session.get_optimal_design_info(model_type="linear")


# ============================================================
# TestGenerateOptimalDesign
# ============================================================

class TestGenerateOptimalDesign:
    """Tests for OptimizationSession.generate_optimal_design()."""

    def _make_session(self, n_vars=3):
        """Helper: build a session with n continuous variables."""
        from alchemist_core.session import OptimizationSession
        session = OptimizationSession()
        names = ["Temperature", "Pressure", "Flow_Rate", "Concentration"]
        bounds = [(200, 400), (1, 10), (0.5, 5.0), (0.01, 1.0)]
        for i in range(n_vars):
            session.add_variable(names[i], "real",
                                 min=bounds[i][0], max=bounds[i][1])
        return session

    def test_n_points_returns_tuple(self):
        """generate_optimal_design() returns (List[Dict], Dict) of correct length."""
        session = self._make_session(n_vars=2)
        result = session.generate_optimal_design(
            model_type="linear", n_points=6, random_seed=42
        )
        assert isinstance(result, tuple) and len(result) == 2
        points, info = result
        assert isinstance(points, list)
        assert len(points) == 6
        assert all(isinstance(p, dict) for p in points)
        assert isinstance(info, dict)

    def test_p_multiplier_resolves_n_points(self):
        """p_multiplier=2.0 yields at least 2*p design points."""
        session = self._make_session(n_vars=2)
        # linear model: intercept + 2 main effects = 3 columns
        design_info = session.get_optimal_design_info(model_type="linear")
        p = design_info["p_columns"]

        points, info = session.generate_optimal_design(
            model_type="linear", p_multiplier=2.0, random_seed=0
        )
        import math
        expected_n = math.ceil(2.0 * p)
        assert len(points) == expected_n
        assert info["n_runs"] == expected_n

    def test_info_has_d_eff_and_model_terms(self):
        """info dict contains D_eff (float) and model_terms (list of str)."""
        session = self._make_session(n_vars=2)
        _, info = session.generate_optimal_design(
            model_type="linear", n_points=8, random_seed=1
        )
        assert "D_eff" in info
        assert isinstance(info["D_eff"], float)
        assert "model_terms" in info
        assert isinstance(info["model_terms"], list)
        assert "Intercept" in info["model_terms"]
        assert "Temperature" in info["model_terms"]

    def test_error_both_n_points_and_p_multiplier(self):
        """Passing both n_points and p_multiplier raises ValueError."""
        session = self._make_session(n_vars=2)
        with pytest.raises(ValueError, match="either n_points or p_multiplier, not both"):
            session.generate_optimal_design(
                model_type="linear", n_points=6, p_multiplier=2.0
            )

    def test_error_neither_n_points_nor_p_multiplier(self):
        """Passing neither n_points nor p_multiplier raises ValueError."""
        session = self._make_session(n_vars=2)
        with pytest.raises(ValueError, match="either n_points.*or p_multiplier"):
            session.generate_optimal_design(model_type="linear")

    def test_error_n_points_below_p(self):
        """n_points=1 with a multi-column model raises ValueError (singular)."""
        session = self._make_session(n_vars=3)
        with pytest.raises(ValueError, match="fewer than the number of model columns"):
            session.generate_optimal_design(
                model_type="linear", n_points=1, random_seed=0
            )

    def test_error_p_multiplier_below_one(self):
        """p_multiplier < 1.0 raises ValueError."""
        session = self._make_session(n_vars=2)
        with pytest.raises(ValueError, match="p_multiplier must be >= 1.0"):
            session.generate_optimal_design(
                model_type="linear", p_multiplier=0.5
            )

    def test_p_multiplier_adapts_to_model_size(self):
        """Bigger model (quadratic) → more runs than linear at same multiplier."""
        session = self._make_session(n_vars=3)
        _, info_lin = session.generate_optimal_design(
            model_type="linear", p_multiplier=2.0, random_seed=7
        )
        _, info_quad = session.generate_optimal_design(
            model_type="quadratic", p_multiplier=2.0, random_seed=7
        )
        assert info_quad["n_runs"] > info_lin["n_runs"]

    def test_shim_still_works(self):
        """generate_initial_design(method='optimal', n_points=8) returns List[Dict]."""
        session = self._make_session(n_vars=2)
        points = session.generate_initial_design(
            method="optimal", n_points=8, model_type="linear", random_seed=42
        )
        assert isinstance(points, list)
        assert len(points) == 8
        assert all(isinstance(p, dict) for p in points)

    def test_last_optimal_design_info_cached(self):
        """_last_optimal_design_info is populated after generate_optimal_design()."""
        session = self._make_session(n_vars=2)
        assert session._last_optimal_design_info is None
        session.generate_optimal_design(
            model_type="linear", n_points=6, random_seed=3
        )
        assert session._last_optimal_design_info is not None
        assert "D_eff" in session._last_optimal_design_info
        assert "model_terms" in session._last_optimal_design_info
