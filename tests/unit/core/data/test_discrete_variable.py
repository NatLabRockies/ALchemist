"""
Tests for the Discrete variable type across SearchSpace, DoE, and optimal design.

Discrete variables store a finite list of allowed numeric values (e.g. SAR ∈ {80, 280}).
They are treated as numeric (support squared/interaction terms) but the candidate
grid is restricted to exactly those values — no interior candidates are generated.
"""

import pytest
import numpy as np

from alchemist_core.data.search_space import SearchSpace
from alchemist_core.utils.doe import generate_initial_design, get_design_info
from alchemist_core.utils.optimal_design import (
    generate_mixed_candidate_set,
    run_optimal_design,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def discrete_two_values():
    """SearchSpace with one discrete variable (2 allowed values)."""
    ss = SearchSpace()
    ss.add_variable("SAR", "discrete", allowed_values=[80, 280])
    return ss


@pytest.fixture
def discrete_three_values():
    """SearchSpace with one discrete variable (3 allowed values)."""
    ss = SearchSpace()
    ss.add_variable("SAR", "discrete", allowed_values=[80, 180, 280])
    return ss


@pytest.fixture
def mixed_discrete_continuous():
    """SearchSpace with a continuous and a discrete variable."""
    ss = SearchSpace()
    ss.add_variable("Temperature", "real", min=200, max=400)
    ss.add_variable("SAR", "discrete", allowed_values=[80, 180, 280])
    return ss


@pytest.fixture
def mixed_discrete_categorical():
    """SearchSpace with continuous, discrete, and categorical variables."""
    ss = SearchSpace()
    ss.add_variable("Temperature", "real", min=200, max=400)
    ss.add_variable("SAR", "discrete", allowed_values=[80, 280])
    ss.add_variable("Catalyst", "categorical", values=["Pt", "Pd"])
    return ss


# ============================================================
# SearchSpace — basic discrete support
# ============================================================

class TestSearchSpaceDiscrete:

    def test_add_discrete_variable(self):
        ss = SearchSpace()
        ss.add_variable("SAR", "discrete", allowed_values=[80, 280])
        assert len(ss) == 1
        assert "SAR" in ss.get_variable_names()

    def test_get_discrete_variables_returns_names(self, discrete_two_values):
        assert "SAR" in discrete_two_values.get_discrete_variables()

    def test_discrete_not_in_categorical_variables(self, discrete_two_values):
        assert "SAR" not in discrete_two_values.get_categorical_variables()

    def test_discrete_not_in_integer_variables(self, discrete_two_values):
        assert "SAR" not in discrete_two_values.get_integer_variables()

    def test_discrete_variable_stores_allowed_values(self, discrete_two_values):
        var = discrete_two_values.variables[0]
        assert var["type"] == "discrete"
        assert set(var["allowed_values"]) == {80.0, 280.0}

    def test_allowed_values_sorted_ascending(self):
        ss = SearchSpace()
        ss.add_variable("X", "discrete", allowed_values=[280, 80, 180])
        var = ss.variables[0]
        assert var["allowed_values"] == sorted(var["allowed_values"])

    def test_discrete_validation_requires_at_least_two_values(self):
        ss = SearchSpace()
        with pytest.raises(ValueError, match="at least 2"):
            ss.add_variable("X", "discrete", allowed_values=[80])

    def test_discrete_validation_no_duplicates(self):
        ss = SearchSpace()
        with pytest.raises(ValueError, match="duplicate"):
            ss.add_variable("X", "discrete", allowed_values=[80, 80, 280])

    def test_from_dict_roundtrip(self, mixed_discrete_continuous):
        """Serialisation/deserialisation preserves discrete type and allowed_values."""
        d = mixed_discrete_continuous.to_dict()
        ss2 = SearchSpace().from_dict(d)
        assert "SAR" in ss2.get_discrete_variables()
        sar = next(v for v in ss2.variables if v["name"] == "SAR")
        assert set(sar["allowed_values"]) == {80.0, 180.0, 280.0}

    def test_bounds_span_allowed_values_range(self, discrete_three_values):
        """to_botorch_bounds uses min/max of allowed_values."""
        bounds = discrete_three_values.to_botorch_bounds()
        # Returns dict: {var_name: array([lower, upper])}
        assert bounds["SAR"][0] == pytest.approx(80.0)
        assert bounds["SAR"][1] == pytest.approx(280.0)


# ============================================================
# SearchSpace — discrete from_dict with lowercase/uppercase type
# ============================================================

class TestSearchSpaceFromDict:

    def test_from_dict_lowercase_discrete(self):
        data = [{"name": "SAR", "type": "discrete", "allowed_values": [80, 280]}]
        ss = SearchSpace().from_dict(data)
        assert "SAR" in ss.get_discrete_variables()

    def test_from_dict_uppercase_discrete(self):
        """Desktop GUI stores 'Discrete' (capitalised); from_dict must handle it."""
        data = [{"name": "SAR", "type": "Discrete", "allowed_values": [80, 280]}]
        ss = SearchSpace().from_dict(data)
        assert "SAR" in ss.get_discrete_variables()


# ============================================================
# DoE — space-filling methods with discrete variables
# ============================================================

class TestDoEWithDiscrete:

    def test_random_sampling_respects_allowed_values(self, mixed_discrete_continuous):
        points = generate_initial_design(
            mixed_discrete_continuous, method="random", n_points=20, random_seed=42
        )
        assert len(points) == 20
        for p in points:
            assert p["SAR"] in {80.0, 180.0, 280.0}, f"SAR={p['SAR']} not in allowed values"

    def test_lhs_respects_allowed_values(self, mixed_discrete_continuous):
        points = generate_initial_design(
            mixed_discrete_continuous, method="lhs", n_points=12, random_seed=42
        )
        for p in points:
            assert p["SAR"] in {80.0, 180.0, 280.0}

    def test_sobol_respects_allowed_values(self, mixed_discrete_continuous):
        points = generate_initial_design(
            mixed_discrete_continuous, method="sobol", n_points=8, random_seed=42
        )
        for p in points:
            assert p["SAR"] in {80.0, 180.0, 280.0}

    def test_discrete_continuous_bounds(self, mixed_discrete_continuous):
        points = generate_initial_design(
            mixed_discrete_continuous, method="lhs", n_points=15, random_seed=0
        )
        for p in points:
            assert 200 <= p["Temperature"] <= 400
            assert p["SAR"] in {80.0, 180.0, 280.0}

    def test_discrete_with_categorical(self, mixed_discrete_categorical):
        points = generate_initial_design(
            mixed_discrete_categorical, method="random", n_points=20, random_seed=1
        )
        for p in points:
            assert p["SAR"] in {80.0, 280.0}
            assert p["Catalyst"] in {"Pt", "Pd"}

    def test_full_factorial_with_discrete(self, mixed_discrete_continuous):
        """Full factorial should use only allowed_values for the discrete variable."""
        # n_center=0 avoids the default center-point replicate so run count is exact
        points = generate_initial_design(
            mixed_discrete_continuous, method="full_factorial", n_levels=2, n_center=0
        )
        sar_values = {p["SAR"] for p in points}
        # All SAR values must be from allowed_values
        assert sar_values.issubset({80.0, 180.0, 280.0})
        # Temperature: 2 levels, SAR: 3 discrete levels → 2 × 3 = 6 factorial runs
        assert len(points) == 6

    def test_get_design_info_discrete_levels(self, mixed_discrete_continuous):
        """get_design_info for full_factorial should count discrete levels correctly."""
        info = get_design_info(
            "full_factorial", mixed_discrete_continuous, n_levels=2, n_center=0
        )
        # Temperature: 2 levels, SAR: 3 allowed values → 2 × 3 = 6 runs
        assert info["total_runs"] == 6


# ============================================================
# Optimal design — candidate set restricted to allowed values
# ============================================================

class TestOptimalDesignDiscrete:

    def test_candidate_set_discrete_only_allowed_values(self, discrete_two_values):
        """Candidate grid for a discrete variable must only contain allowed values."""
        candidates, col_map = generate_mixed_candidate_set(
            discrete_two_values, n_levels=5
        )
        # Discrete variable is a single continuous column in the candidate matrix
        assert candidates.shape[1] == 1
        # The candidates (coded) should only be at -1 and +1 (the two allowed values coded)
        unique_coded = set(np.round(candidates[:, 0], 6))
        assert unique_coded == {-1.0, 1.0}

    def test_candidate_set_three_discrete_values(self, discrete_three_values):
        """Three allowed values → three coded candidate levels."""
        candidates, col_map = generate_mixed_candidate_set(
            discrete_three_values, n_levels=5
        )
        unique_coded = set(np.round(candidates[:, 0], 6))
        assert len(unique_coded) == 3

    def test_optimal_design_discrete_values_in_output(self, discrete_two_values):
        """run_optimal_design must decode discrete back to allowed values."""
        points, _ = run_optimal_design(
            discrete_two_values, n_points=4, model_type="linear",
            criterion="D", algorithm="sequential", n_levels=5,
            random_seed=42,
        )
        assert len(points) == 4
        for p in points:
            assert p["SAR"] in {80.0, 280.0}, f"SAR={p['SAR']} not in allowed set"

    def test_optimal_design_three_discrete_values(self, discrete_three_values):
        """run_optimal_design with 3 allowed values."""
        points, _ = run_optimal_design(
            discrete_three_values, n_points=6, model_type="linear",
            criterion="D", algorithm="sequential", n_levels=5,
            random_seed=42,
        )
        for p in points:
            assert p["SAR"] in {80.0, 180.0, 280.0}

    def test_optimal_design_mixed_discrete_continuous(self, mixed_discrete_continuous):
        """Mixed space: continuous values in bounds, discrete snapped to allowed."""
        points, _ = run_optimal_design(
            mixed_discrete_continuous, n_points=8, model_type="linear",
            criterion="D", algorithm="sequential", n_levels=5,
            random_seed=42,
        )
        for p in points:
            assert 200 <= p["Temperature"] <= 400
            assert p["SAR"] in {80.0, 180.0, 280.0}

    def test_quadratic_term_allowed_for_discrete(self, mixed_discrete_continuous):
        """Discrete variables support quadratic model terms (unlike categorical)."""
        from alchemist_core.utils.optimal_design import parse_model_spec
        terms = parse_model_spec(
            mixed_discrete_continuous,
            effects=["Temperature", "SAR", "SAR**2", "Temperature*SAR"],
        )
        # Should not raise; quadratic on discrete is valid
        term_names_tuples = [t for t in terms if len(t) == 1]
        # We got terms without error — that's the key assertion
        assert len(terms) > 1

    def test_optimal_design_with_all_three_types(self, mixed_discrete_categorical):
        """Optimal design works with continuous + discrete + categorical."""
        points, info = run_optimal_design(
            mixed_discrete_categorical, n_points=8, model_type="linear",
            criterion="D", algorithm="sequential", n_levels=3,
            random_seed=42,
        )
        assert len(points) == 8
        for p in points:
            assert 200 <= p["Temperature"] <= 400
            assert p["SAR"] in {80.0, 280.0}
            assert p["Catalyst"] in {"Pt", "Pd"}

    def test_no_intermediate_values_in_optimal_design(self):
        """Key requirement: optimal design never suggests SAR=173 for {80, 280}."""
        ss = SearchSpace()
        ss.add_variable("Temperature", "real", min=200, max=400)
        ss.add_variable("SAR", "discrete", allowed_values=[80, 280])

        points, _ = run_optimal_design(
            ss, n_points=10, model_type="linear",
            criterion="D", algorithm="fedorov", n_levels=5,
            random_seed=7,
        )
        for p in points:
            assert p["SAR"] in {80.0, 280.0}, (
                f"Intermediate SAR value {p['SAR']} generated — "
                "discrete candidate restriction is broken"
            )


# ============================================================
# Session-level integration
# ============================================================

class TestSessionDiscrete:

    def _make_session(self):
        from alchemist_core.session import OptimizationSession
        session = OptimizationSession()
        session.add_variable("Temperature", "real", min=200, max=400)
        session.add_variable("SAR", "discrete", allowed_values=[80, 280])
        return session

    def test_session_add_discrete_variable(self):
        session = self._make_session()
        assert "SAR" in session.search_space.get_discrete_variables()

    def test_session_generate_initial_design_discrete(self):
        session = self._make_session()
        points = session.generate_initial_design(
            method="lhs", n_points=10, random_seed=42
        )
        assert len(points) == 10
        for p in points:
            assert p["SAR"] in {80.0, 280.0}

    def test_session_get_optimal_design_info_discrete(self):
        """get_optimal_design_info allows SAR**2 for discrete (numerical treatment)."""
        session = self._make_session()
        info = session.get_optimal_design_info(
            effects=["Temperature", "SAR", "SAR**2", "Temperature*SAR"]
        )
        assert "SAR**2" in info["model_terms"] or "SAR^2" in info["model_terms"]
        assert info["p_columns"] == 5  # intercept + 2 main + 1 quad + 1 interaction

    def test_session_generate_optimal_design_discrete_values_valid(self):
        session = self._make_session()
        points, info = session.generate_optimal_design(
            effects=["Temperature", "SAR", "SAR**2", "Temperature*SAR"],
            n_points=10, random_seed=42
        )
        assert len(points) == 10
        for p in points:
            assert p["SAR"] in {80.0, 280.0}
