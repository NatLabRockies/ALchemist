import numpy as np
import pandas as pd
import pytest
from alchemist_core import OptimizationSession


def _session():
    s = OptimizationSession()
    s.add_variable('H2', 'real', bounds=(0.0, 100.0))
    s.add_variable('CO', 'real', bounds=(0.0, 100.0))
    s.add_variable('CO2', 'real', bounds=(0.0, 100.0))
    return s


# DOE sampling enforces STRICT feasibility (rtol=0), unlike grid-plot masking
# which uses a relative band. A design point must not exceed the stated bound.
FEAS_TOL = 1e-3


class TestInitialDesignFeasible:
    def test_random_design_all_feasible_inequality(self):
        s = _session()
        s.add_input_constraint('inequality', {'H2': 1.0, 'CO': 1.0, 'CO2': 1.0}, rhs=100.0)
        design = s.generate_initial_design(n_points=8, method='random', random_seed=42)
        df = design if isinstance(design, pd.DataFrame) else pd.DataFrame(design)
        totals = df['H2'] + df['CO'] + df['CO2']
        assert (totals <= 100.0 + FEAS_TOL).all()
        assert len(df) == 8

    def test_lhs_design_all_feasible_inequality(self):
        s = _session()
        s.add_input_constraint('inequality', {'H2': 1.0, 'CO': 1.0, 'CO2': 1.0}, rhs=100.0)
        design = s.generate_initial_design(n_points=8, method='lhs', random_seed=42)
        df = design if isinstance(design, pd.DataFrame) else pd.DataFrame(design)
        totals = df['H2'] + df['CO'] + df['CO2']
        assert (totals <= 100.0 + FEAS_TOL).all()

    def test_sobol_design_all_feasible_inequality(self):
        s = _session()
        s.add_input_constraint('inequality', {'H2': 1.0, 'CO': 1.0, 'CO2': 1.0}, rhs=100.0)
        design = s.generate_initial_design(n_points=8, method='sobol', random_seed=42)
        df = design if isinstance(design, pd.DataFrame) else pd.DataFrame(design)
        totals = df['H2'] + df['CO'] + df['CO2']
        assert (totals <= 100.0 + FEAS_TOL).all()

    def test_infeasible_constraint_raises(self):
        s = _session()
        # No point in [0,100]^3 can have sum <= -1
        s.add_input_constraint('inequality', {'H2': 1.0, 'CO': 1.0, 'CO2': 1.0}, rhs=-1.0)
        with pytest.raises(ValueError, match='(?i)feasible'):
            s.generate_initial_design(n_points=8, method='random', random_seed=42)

    def test_no_constraints_unchanged_count(self):
        s = _session()
        design = s.generate_initial_design(n_points=8, method='random', random_seed=42)
        df = design if isinstance(design, pd.DataFrame) else pd.DataFrame(design)
        assert len(df) == 8
