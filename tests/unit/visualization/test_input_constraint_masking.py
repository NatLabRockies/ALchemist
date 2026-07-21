import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')

from alchemist_core import OptimizationSession


def _session():
    s = OptimizationSession()
    s.add_variable('H2', 'real', bounds=(0.0, 100.0))
    s.add_variable('CO', 'real', bounds=(0.0, 100.0))
    s.add_variable('CO2', 'real', bounds=(0.0, 100.0))
    np.random.seed(0)
    n = 20
    df = pd.DataFrame({
        'H2': np.random.uniform(0, 100, n),
        'CO': np.random.uniform(0, 100, n),
        'CO2': np.random.uniform(0, 100, n),
    })
    df['yield'] = 0.5 * df.H2 - 0.2 * df.CO + 0.1 * df.CO2
    s.experiment_manager.target_columns = ['yield']
    s.experiment_manager.df = df
    s.train_model(backend='botorch')
    return s


class TestGridMasking:
    def test_apply_feasibility_mask_sets_nan(self):
        """The shared masking helper NaNs infeasible cells of a Z grid."""
        s = _session()
        s.add_input_constraint('equality', {'H2': 1.0, 'CO': 1.0, 'CO2': 1.0}, rhs=100.0)
        grid_df = pd.DataFrame({
            'H2':  [50.0, 50.0],
            'CO':  [30.0, 30.0],
            'CO2': [20.0, 80.0],   # row0 sum=100 feasible, row1 sum=160 infeasible
        })
        Z = np.array([1.0, 2.0])
        Z_masked = s._apply_feasibility_mask(Z, grid_df)
        assert not np.isnan(Z_masked[0])
        assert np.isnan(Z_masked[1])

    def test_apply_feasibility_mask_preserves_shape(self):
        """Masking a 2D Z aligned with a flattened grid_df keeps the 2D shape."""
        s = _session()
        s.add_input_constraint('inequality', {'H2': 1.0, 'CO': 1.0, 'CO2': 1.0}, rhs=100.0)
        grid_df = pd.DataFrame({
            'H2':  [10.0, 90.0, 10.0, 90.0],
            'CO':  [10.0, 10.0, 90.0, 90.0],
            'CO2': [0.0, 0.0, 0.0, 0.0],
        })
        Z = np.arange(4, dtype=float).reshape(2, 2)
        Z_masked = s._apply_feasibility_mask(Z, grid_df)
        assert Z_masked.shape == (2, 2)
        # row with H2+CO=180 (index 3) infeasible -> NaN
        assert np.isnan(Z_masked.ravel()[3])

    def test_mask_noop_without_constraints(self):
        s = _session()
        grid_df = pd.DataFrame({'H2': [50.0], 'CO': [30.0], 'CO2': [20.0]})
        Z = np.array([1.0])
        Z_masked = s._apply_feasibility_mask(Z, grid_df)
        assert Z_masked.tolist() == [1.0]

    def test_plot_contour_masks_infeasible(self):
        """End-to-end: constrained contour is produced with a partial feasible
        region (inequality leaves a 2D feasible band, not measure-zero)."""
        s = _session()
        s.add_input_constraint('inequality', {'H2': 1.0, 'CO': 1.0}, rhs=60.0)
        fig = s.plot_contour('H2', 'CO')
        assert fig is not None

    def test_plot_contour_all_infeasible_does_not_crash(self):
        """An equality slice with a measure-zero feasible band must not crash;
        it falls back to the unmasked plot rather than an all-NaN grid."""
        s = _session()
        s.add_input_constraint('equality', {'H2': 1.0, 'CO': 1.0, 'CO2': 1.0}, rhs=100.0)
        fig = s.plot_contour('H2', 'CO')  # CO2 fixed at midpoint -> band nearly empty
        assert fig is not None

