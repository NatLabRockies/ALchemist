"""Tests that registered linear input constraints are honored during acquisition.

Root cause covered: input constraints were translated into normalized [0, 1]
space, but ``optimize_acqf`` runs in raw variable space (the BoTorch model
applies its ``Normalize`` transform internally). The space mismatch produced
suggestions that satisfied the wrong-space constraint and violated the real one
(e.g. H2 + CO + CO2 == 1 instead of == 100).
"""

import numpy as np
import pandas as pd
import pytest
import torch

from alchemist_core import OptimizationSession
from alchemist_core.data.search_space import SearchSpace


TOL = 1e-2


def _syngas_session():
    """Single-objective session over three composition inputs in [0, 100]."""
    session = OptimizationSession()
    session.add_variable('H2', 'real', bounds=(0.0, 100.0))
    session.add_variable('CO', 'real', bounds=(0.0, 100.0))
    session.add_variable('CO2', 'real', bounds=(0.0, 100.0))

    np.random.seed(0)
    n = 20
    H2 = np.random.uniform(0, 100, n)
    CO = np.random.uniform(0, 100, n)
    CO2 = np.random.uniform(0, 100, n)
    y = 0.5 * H2 - 0.2 * CO + 0.1 * CO2 + np.random.normal(0, 1, n)
    df = pd.DataFrame({'H2': H2, 'CO': CO, 'CO2': CO2, 'yield': y})
    session.experiment_manager.target_columns = ['yield']
    session.experiment_manager.df = df
    session.train_model(backend='botorch')
    return session


def _syngas_mixed_session():
    """Session with a discrete variable alongside the constrained composition
    inputs, forcing acquisition through the mixed-variable optimizer path."""
    session = OptimizationSession()
    session.add_variable('H2', 'real', bounds=(0.0, 100.0))
    session.add_variable('CO', 'real', bounds=(0.0, 100.0))
    session.add_variable('CO2', 'real', bounds=(0.0, 100.0))
    session.add_variable('T', 'discrete', allowed_values=[200, 225, 250, 275, 300])

    np.random.seed(0)
    n = 40
    H2 = np.random.uniform(0, 100, n)
    CO = np.random.uniform(0, 100, n)
    CO2 = np.random.uniform(0, 100, n)
    T = np.random.choice([200, 225, 250, 275, 300], n)
    y = 0.5 * H2 - 0.2 * CO + 0.1 * CO2 - 0.05 * np.abs(T - 245) + np.random.normal(0, 1, n)
    df = pd.DataFrame({'H2': H2, 'CO': CO, 'CO2': CO2, 'T': T, 'yield': y})
    session.experiment_manager.target_columns = ['yield']
    session.experiment_manager.df = df
    session.train_model(backend='botorch')
    return session


# ---------------------------------------------------------------------------
# to_botorch_constraints: raw-space output (matches optimize_acqf bounds)
# ---------------------------------------------------------------------------

class TestToBotorchConstraintsRawSpace:
    """Constraints must be expressed in raw variable space, not normalized."""

    def test_equality_constraint_is_raw_space(self):
        space = SearchSpace()
        space.add_variable('H2', 'real', min=0.0, max=100.0)
        space.add_variable('CO', 'real', min=0.0, max=100.0)
        space.add_variable('CO2', 'real', min=0.0, max=100.0)
        space.add_constraint('equality', {'H2': 1.0, 'CO': 1.0, 'CO2': 1.0}, rhs=100.0)

        ineq, eq = space.to_botorch_constraints(['H2', 'CO', 'CO2'])

        assert ineq is None
        assert eq is not None and len(eq) == 1
        indices, coeffs, rhs = eq[0]
        assert sorted(indices.tolist()) == [0, 1, 2]
        # Raw-space coefficients are the identity coefficients (all 1.0), NOT
        # scaled by the variable range (which would give 100.0).
        assert torch.allclose(coeffs.double(), torch.tensor([1.0, 1.0, 1.0], dtype=torch.double))
        assert rhs == pytest.approx(100.0)

    def test_inequality_sign_convention_raw_space(self):
        # ALchemist inequality (coeff·x <= rhs) -> BoTorch (coeff·x >= rhs):
        # coefficients and rhs are negated, but NOT range-scaled.
        space = SearchSpace()
        space.add_variable('H2', 'real', min=0.0, max=100.0)
        space.add_variable('CO', 'real', min=0.0, max=100.0)
        space.add_constraint('inequality', {'H2': 1.0, 'CO': 1.0}, rhs=100.0)

        ineq, eq = space.to_botorch_constraints(['H2', 'CO'])

        assert eq is None
        assert ineq is not None and len(ineq) == 1
        indices, coeffs, rhs = ineq[0]
        assert torch.allclose(coeffs.double(), torch.tensor([-1.0, -1.0], dtype=torch.double))
        assert rhs == pytest.approx(-100.0)


# ---------------------------------------------------------------------------
# End-to-end: suggest_next honors constraints
# ---------------------------------------------------------------------------

class TestSuggestNextHonorsInputConstraints:

    def test_equality_constraint_satisfied(self):
        """H2 + CO + CO2 must sum to 100 within tolerance."""
        session = _syngas_session()
        session.add_input_constraint('equality', {'H2': 1.0, 'CO': 1.0, 'CO2': 1.0}, rhs=100.0)
        res = session.suggest_next(strategy='LogEI', goal='maximize')
        row = res.iloc[0]
        total = row['H2'] + row['CO'] + row['CO2']
        assert total == pytest.approx(100.0, abs=1.0)

    def test_inequality_constraint_satisfied(self):
        """H2 + CO + CO2 must be <= 100 within tolerance."""
        session = _syngas_session()
        session.add_input_constraint('inequality', {'H2': 1.0, 'CO': 1.0, 'CO2': 1.0}, rhs=100.0)
        res = session.suggest_next(strategy='LogEI', goal='maximize')
        row = res.iloc[0]
        total = row['H2'] + row['CO'] + row['CO2']
        assert total <= 100.0 + 1.0

    def test_multiple_constraints_satisfied(self):
        """Multiple simultaneous constraints all honored."""
        session = _syngas_session()
        # H2 + CO + CO2 == 100  and  H2 - CO <= 20
        session.add_input_constraint('equality', {'H2': 1.0, 'CO': 1.0, 'CO2': 1.0}, rhs=100.0)
        session.add_input_constraint('inequality', {'H2': 1.0, 'CO': -1.0}, rhs=20.0)
        res = session.suggest_next(strategy='LogEI', goal='maximize')
        row = res.iloc[0]
        assert (row['H2'] + row['CO'] + row['CO2']) == pytest.approx(100.0, abs=1.0)
        assert (row['H2'] - row['CO']) <= 20.0 + 1.0

    def test_subset_constraint_satisfied(self):
        """Constraint over a subset of variables is honored; other var free."""
        session = _syngas_session()
        session.add_input_constraint('inequality', {'H2': 1.0, 'CO': 1.0}, rhs=50.0)
        res = session.suggest_next(strategy='LogEI', goal='maximize')
        row = res.iloc[0]
        assert (row['H2'] + row['CO']) <= 50.0 + 1.0

    def test_raw_optimizer_candidate_is_feasible(self):
        """The candidate the optimizer returns (raw space) is itself feasible,
        not merely a post-processed value."""
        session = _syngas_session()
        session.add_input_constraint('equality', {'H2': 1.0, 'CO': 1.0, 'CO2': 1.0}, rhs=100.0)
        res = session.suggest_next(strategy='LogEI', goal='maximize')
        raw = session.acquisition.last_raw_candidate
        assert raw is not None
        # raw candidate columns follow model.original_feature_names order
        names = session.model.original_feature_names
        vals = {n: float(v) for n, v in zip(names, raw)}
        total = vals['H2'] + vals['CO'] + vals['CO2']
        assert total == pytest.approx(100.0, abs=1.0)


class TestMixedVariablePathHonorsConstraints:
    """A discrete variable routes acquisition through the mixed-variable
    optimizer (optimize_acqf_mixed_alternating). Constraints must be honored
    there too, not just on the purely-continuous optimize_acqf path."""

    def test_equality_constraint_satisfied_mixed_path(self):
        session = _syngas_mixed_session()
        session.add_input_constraint('equality', {'H2': 1.0, 'CO': 1.0, 'CO2': 1.0}, rhs=100.0)
        res = session.suggest_next(strategy='LogEI', goal='maximize')
        row = res.iloc[0]
        total = row['H2'] + row['CO'] + row['CO2']
        assert total == pytest.approx(100.0, abs=1.0)

    def test_inequality_constraint_satisfied_mixed_path(self):
        session = _syngas_mixed_session()
        session.add_input_constraint('inequality', {'H2': 1.0, 'CO': 1.0, 'CO2': 1.0}, rhs=100.0)
        res = session.suggest_next(strategy='LogEI', goal='maximize')
        row = res.iloc[0]
        total = row['H2'] + row['CO'] + row['CO2']
        assert total <= 100.0 + 1.0


class TestSklearnBackendRejectsInputConstraints:

    def test_sklearn_raises_on_registered_input_constraint(self):
        """The sklearn/skopt backend cannot express linear input constraints, so
        registering one and calling suggest_next must raise a clear error rather
        than silently ignoring the constraint."""
        session = OptimizationSession()
        session.add_variable('x1', 'real', bounds=(0.0, 1.0))
        session.add_variable('x2', 'real', bounds=(0.0, 1.0))
        session.add_input_constraint('inequality', {'x1': 1.0, 'x2': 1.0}, rhs=1.5)

        np.random.seed(1)
        n = 12
        df = pd.DataFrame({
            'x1': np.random.uniform(0, 1, n),
            'x2': np.random.uniform(0, 1, n),
            'yield': np.random.uniform(0, 10, n),
        })
        session.experiment_manager.target_columns = ['yield']
        session.experiment_manager.df = df
        session.train_model(backend='sklearn')

        with pytest.raises(ValueError, match='(?i)input constraint'):
            session.suggest_next(strategy='EI', goal='maximize')


class TestNoRegressionUnconstrained:

    def test_unconstrained_suggestion_deterministic_unchanged(self):
        """With no registered input constraints, suggestion is unaffected."""
        s1 = _syngas_session()
        r1 = s1.suggest_next(strategy='LogEI', goal='maximize')
        s2 = _syngas_session()
        r2 = s2.suggest_next(strategy='LogEI', goal='maximize')
        for col in ['H2', 'CO', 'CO2']:
            assert r1.iloc[0][col] == pytest.approx(r2.iloc[0][col], abs=1e-6)


class TestFindOptimumFeasible:
    """find_optimum must return a feasible optimum when input constraints exist."""

    def test_find_optimum_respects_equality(self):
        session = _syngas_session()
        session.add_input_constraint('equality', {'H2': 1.0, 'CO': 1.0, 'CO2': 1.0}, rhs=100.0)
        opt = session.find_optimum('maximize')
        x = opt['x_opt'].iloc[0]
        total = x['H2'] + x['CO'] + x['CO2']
        assert total == pytest.approx(100.0, abs=5.0)

    def test_find_optimum_respects_inequality(self):
        session = _syngas_session()
        session.add_input_constraint('inequality', {'H2': 1.0, 'CO': 1.0, 'CO2': 1.0}, rhs=100.0)
        opt = session.find_optimum('maximize')
        x = opt['x_opt'].iloc[0]
        assert (x['H2'] + x['CO'] + x['CO2']) <= 100.0 + 5.0

    def test_find_optimum_raises_when_no_feasible_grid_points(self):
        session = _syngas_session()
        session.add_input_constraint('equality', {'H2': 1.0, 'CO': 1.0, 'CO2': 1.0}, rhs=500.0)
        with pytest.raises(ValueError, match='(?i)feasible'):
            session.find_optimum('maximize')


class TestSklearnFindOptimumFeasible:
    """session.find_optimum is grid-based and backend-agnostic, so the sklearn
    backend also returns a constraint-feasible optimum (via the grid filter)."""

    def test_sklearn_find_optimum_respects_inequality(self):
        session = OptimizationSession()
        session.add_variable('x1', 'real', bounds=(0.0, 1.0))
        session.add_variable('x2', 'real', bounds=(0.0, 1.0))
        session.add_input_constraint('inequality', {'x1': 1.0, 'x2': 1.0}, rhs=1.0)
        np.random.seed(2)
        n = 15
        df = pd.DataFrame({
            'x1': np.random.uniform(0, 1, n),
            'x2': np.random.uniform(0, 1, n),
            'yield': np.random.uniform(0, 10, n),
        })
        session.experiment_manager.target_columns = ['yield']
        session.experiment_manager.df = df
        session.train_model(backend='sklearn')
        opt = session.find_optimum('maximize')
        x = opt['x_opt'].iloc[0]
        assert (x['x1'] + x['x2']) <= 1.0 + 1e-2
