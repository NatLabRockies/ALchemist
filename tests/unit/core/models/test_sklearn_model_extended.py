
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from alchemist_core.models.sklearn_model import SklearnModel
from alchemist_core.data.experiment_manager import ExperimentManager
from alchemist_core.data.search_space import SearchSpace
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

class TestSklearnModelExtended:
    
    @pytest.fixture
    def experiment_manager(self):
        em = MagicMock(spec=ExperimentManager)
        em.search_space = MagicMock(spec=SearchSpace)
        em.search_space.get_categorical_variables.return_value = []
        em.target_columns = ['Output']
        return em

    def test_generate_contour_data(self, experiment_manager):
        # Setup model
        model = SklearnModel(kernel_options={"kernel_type": "RBF"})
        model.model = MagicMock()
        model.model.predict.return_value = np.zeros(10000) # 100x100 grid
        model._is_trained = True
        model.feature_names = ["x1", "x2"]
        model.original_feature_names = ["x1", "x2"]
        
        # Test contour generation
        X, Y, Z = model.generate_contour_data(
            x_range=(0, 1), 
            y_range=(0, 1), 
            fixed_values={}, 
            x_idx=0, 
            y_idx=1
        )
        
        assert X.shape == (100, 100)
        assert Y.shape == (100, 100)
        assert Z.shape == (100, 100)
        
    def test_generate_contour_data_categorical_error(self, experiment_manager):
        model = SklearnModel(kernel_options={"kernel_type": "RBF"})
        model.model = MagicMock()  # Set model to simulate trained state
        model._is_trained = True
        model.feature_names = ["x1", "c1"]
        model.original_feature_names = ["x1", "c1"]
        model.categorical_variables = ["c1"]
        
        # Should raise error if trying to plot categorical variable on axis
        with pytest.raises(ValueError, match="Cannot create contour plot with categorical variables"):
            model.generate_contour_data(
                x_range=(0, 1), 
                y_range=(0, 1), 
                fixed_values={}, 
                x_idx=0, 
                y_idx=1  # c1 is at index 1
            )

    def test_compute_calibration_factors(self):
        model = SklearnModel(kernel_options={"kernel_type": "RBF"})
        
        # Setup cached CV results
        # Perfect calibration: z-scores have std=1
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.1])
        y_std = np.array([0.1, 0.1, 0.1]) # Errors are roughly 0.1
        
        model.cv_cached_results = {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_std': y_std
        }
        
        model._compute_calibration_factors()
        
        assert model.calibration_enabled is True
        assert 0.5 < model.calibration_factor < 1.5
        assert hasattr(model, 'cv_cached_results_calibrated')

    def test_compute_calibration_factors_invalid(self):
        model = SklearnModel(kernel_options={"kernel_type": "RBF"})
        
        # Invalid std (zero)
        model.cv_cached_results = {
            'y_true': np.array([1.0]),
            'y_pred': np.array([1.0]),
            'y_std': np.array([0.0])
        }
        
        model._compute_calibration_factors()
        assert model.calibration_enabled is False
        assert model.calibration_factor == 1.0

    def test_train_non_finite_bounds_retry(self, experiment_manager):
        # Mock data
        X = pd.DataFrame({"x1": [0.0, 1.0], "x2": [0.0, 1.0]})
        y = pd.Series([0.0, 1.0])
        noise = None
        experiment_manager.get_features_target_and_noise.return_value = (X, y, noise)
        
        model = SklearnModel(kernel_options={"kernel_type": "RBF"}, n_restarts_optimizer=5)
        
        # Mock GaussianProcessRegressor to fail first then succeed
        with patch('alchemist_core.models.sklearn_model.GaussianProcessRegressor') as MockGPR:
            instance = MockGPR.return_value
            
            # First fit raises ValueError about bounds
            def side_effect_fit(*args, **kwargs):
                if MockGPR.call_count == 1:
                    raise ValueError("fmin_l_bfgs_b requires that all bounds are finite")
                return None
                
            instance.fit.side_effect = side_effect_fit
            instance.kernel_ = ConstantKernel() * RBF()
            
            model.train(experiment_manager, cache_cv=False, calibrate_uncertainty=False)
            
            # Should have retried with n_restarts_optimizer=0
            assert MockGPR.call_count == 2
            call_args = MockGPR.call_args_list[1]
            assert call_args[1]['n_restarts_optimizer'] == 0

    def test_build_kernel_bad_length_scales(self):
        model = SklearnModel(kernel_options={"kernel_type": "RBF"})
        
        # Data with 0 variance (constant column)
        X = np.array([[1.0, 5.0], [1.0, 6.0]]) # Col 0 has std=0
        
        kernel = model._build_kernel(X)
        
        # Check that length scale bounds are safe (not 0)
        # The RBF kernel is the second component (after ConstantKernel)
        rbf_kernel = kernel.k2
        bounds = rbf_kernel.length_scale_bounds
        
        # First dimension had 0 std, should have been replaced by 1.0 -> bounds (1e-5, 1e5)
        # Or at least safe bounds
        assert bounds[0][1] > 1e-5

    def test_preprocess_subset_fallback(self):
        # Test fallback when scalers are missing
        model = SklearnModel(kernel_options={"kernel_type": "RBF"}, input_transform_type="standard")
        
        X_subset = pd.DataFrame({"x1": [1.0, 2.0]})
        
        # Should not crash even if _fold_input_scaler is missing and fit_scalers=False
        processed = model._preprocess_subset(X_subset, categorical_variables=[], fit_scalers=False)
        
        assert processed.shape == (2, 1)
        # Should be raw values since no scaler available
        assert np.allclose(processed, X_subset.values)


class TestSklearnCVCacheTestIndices:
    """Verify the sklearn CV cache exposes ``test_indices`` / ``fold_ids`` that
    map cached predictions back to original sample-row indices."""

    def _build_session(self, n_samples=20, seed=0):
        from alchemist_core import OptimizationSession

        session = OptimizationSession()
        session.add_variable('x1', 'real', bounds=(0.0, 1.0))
        session.add_variable('x2', 'real', bounds=(0.0, 1.0))

        rng = np.random.default_rng(seed)
        for _ in range(n_samples):
            x1 = float(rng.uniform(0, 1))
            x2 = float(rng.uniform(0, 1))
            y = x1 + 2 * x2 + float(rng.normal(0, 0.05))
            session.add_experiment({'x1': x1, 'x2': x2}, y)
        return session

    def test_cache_has_test_indices_and_fold_ids(self):
        session = self._build_session(n_samples=20, seed=11)
        results = session.train_model(backend='sklearn', kernel='Matern')
        assert results['success'] is True

        cache = session.model.cv_cached_results
        assert cache is not None
        for key in ('test_indices', 'fold_ids'):
            assert key in cache

        n = cache['y_true'].shape[0]
        test_indices = cache['test_indices']
        fold_ids = cache['fold_ids']

        assert test_indices.shape == (n,)
        assert test_indices.dtype == np.int64
        assert np.array_equal(np.sort(test_indices), np.arange(n))

        assert fold_ids.shape == (n,)
        assert fold_ids.dtype == np.int64
        assert fold_ids.min() >= 0
        assert fold_ids.max() < 5

        _, y_orig, _ = session.experiment_manager.get_features_target_and_noise()
        y_orig_arr = np.asarray(y_orig.values, dtype=np.float64)
        np.testing.assert_allclose(
            y_orig_arr[test_indices], cache['y_true'], rtol=0, atol=1e-12
        )

    def test_calibrated_cache_has_test_indices(self):
        session = self._build_session(n_samples=20, seed=12)
        session.train_model(backend='sklearn', kernel='Matern')

        cache = session.model.cv_cached_results
        cal = session.model.cv_cached_results_calibrated
        # Calibration may be disabled if z-scores are pathological; tolerate that
        # by only validating the contract when the calibrated copy was produced.
        if cal is None:
            pytest.skip("Calibration was disabled for this fixture; cv_cached_results_calibrated not built")
        for key in ('test_indices', 'fold_ids'):
            assert key in cal
            np.testing.assert_array_equal(cal[key], cache[key])

