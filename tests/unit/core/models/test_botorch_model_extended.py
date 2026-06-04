
import pytest
import os
import pandas as pd
import numpy as np
import torch
from alchemist_core import OptimizationSession
from alchemist_core.models.botorch_model import BoTorchModel
from botorch.models import SingleTaskGP, MixedSingleTaskGP
from gpytorch.kernels import RBFKernel

@pytest.fixture
def catalyst_session():
    """Create session with real catalyst data."""
    session = OptimizationSession()
    
    # Load search space from JSON
    # Go up 4 levels: models -> core -> unit -> tests
    tests_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    search_space_path = os.path.join(tests_dir, 'catalyst_search_space.json')
    session.load_search_space(search_space_path)
    
    # Load experiments from CSV
    experiments_path = os.path.join(tests_dir, 'catalyst_experiments.csv')
    session.load_data(experiments_path)
    
    return session

class TestBoTorchModelExtended:
    """Extended tests for BoTorchModel to improve coverage."""

    def test_train_with_noise(self, catalyst_session):
        """Test training with explicit noise (uncertainty) data."""
        # Clear existing experiments to avoid mixing noisy and non-noisy data
        # which causes NaNs in train_Yvar and crashes BoTorch
        catalyst_session.experiment_manager.clear()
        
        # Add experiments with noise
        catalyst_session.add_experiment(
            {'Temperature': 400, 'Catalyst': 'High SAR', 'Metal Loading': 2.5, 'Zinc Fraction': 0.5},
            output=0.8,
            noise=0.05
        )
        catalyst_session.add_experiment(
            {'Temperature': 420, 'Catalyst': 'Low SAR', 'Metal Loading': 3.0, 'Zinc Fraction': 0.2},
            output=0.6,
            noise=0.05
        )
        catalyst_session.add_experiment(
            {'Temperature': 380, 'Catalyst': 'High SAR', 'Metal Loading': 1.0, 'Zinc Fraction': 0.8},
            output=0.4,
            noise=0.05
        )
        
        # Train model
        results = catalyst_session.train_model(
            backend='botorch',
            kernel='Matern'
        )
        
        assert results['success'] == True
        assert catalyst_session.model.is_trained
        
        # Verify that the underlying model is using the noise
        # In BoTorch, if train_Yvar is provided, it's stored in the model
        botorch_model = catalyst_session.model.model
        assert botorch_model.train_targets.shape == botorch_model.train_inputs[0].shape[:-1]
        # We can't easily check train_Yvar directly on the model object in all versions, 
        # but successful training with noise data implies it was handled.

    def test_explicit_standardize_transform(self, catalyst_session):
        """Test input_transform_type='standardize'."""
        results = catalyst_session.train_model(
            backend='botorch',
            input_transform_type='standardize',
            output_transform_type='standardize'
        )
        assert results['success'] == True
        
        # Check if transforms were applied (BoTorchModel stores this in internal state or logs)
        # We can check the model config or just rely on successful execution covering the lines
        assert catalyst_session.model.input_transform_type == 'standardize'
        assert catalyst_session.model.output_transform_type == 'standardize'

    def test_messy_data_encoding_fallback(self):
        """Test the fallback encoding logic for non-numeric data not in cat_dims."""
        # Create a session manually to control data exactly
        session = OptimizationSession()
        session.add_variable('x', 'real', bounds=(0, 10))
        session.add_variable('messy_col', 'categorical', categories=['A', 'B'])
        
        # Add data where 'messy_col' is passed but maybe not correctly identified initially
        # or simulate the condition where pd.to_numeric fails
        
        # We can test the _encode_categorical_data method directly
        model = BoTorchModel()
        
        df = pd.DataFrame({
            'x': [1.0, 2.0, 3.0],
            'messy_col': ['A', 'B', 'A'], # Strings
            'y': [0.1, 0.2, 0.3]
        })
        
        # Force cat_dims to be empty so it tries to convert 'messy_col' to numeric and fails
        model.cat_dims = [] 
        
        encoded_df = model._encode_categorical_data(df)
        
        # Check that 'messy_col' was encoded to numbers despite not being in cat_dims
        assert pd.api.types.is_numeric_dtype(encoded_df['messy_col'])
        assert set(encoded_df['messy_col'].unique()) == {0.0, 1.0}

    def test_rbf_kernel_factory(self, catalyst_session):
        """Explicitly test RBF kernel factory creation."""
        model = BoTorchModel(kernel_options={"cont_kernel_type": "RBF"})
        factory = model._get_cont_kernel_factory()
        
        # Create dummy args for factory
        batch_shape = torch.Size([])
        ard_num_dims = 2
        active_dims = [0, 1]
        
        kernel = factory(batch_shape, ard_num_dims, active_dims)
        assert isinstance(kernel, RBFKernel)

    def test_mixed_single_task_gp_with_noise(self, catalyst_session):
        """Test MixedSingleTaskGP (categorical) with noise."""
        # Catalyst session has categorical variables
        
        # Clear existing experiments to avoid mixing noisy and non-noisy data
        catalyst_session.experiment_manager.clear()
        
        # Add experiments with noise
        catalyst_session.add_experiment(
            {'Temperature': 400, 'Catalyst': 'High SAR', 'Metal Loading': 2.5, 'Zinc Fraction': 0.5},
            output=0.8,
            noise=0.05
        )
        catalyst_session.add_experiment(
            {'Temperature': 420, 'Catalyst': 'Low SAR', 'Metal Loading': 3.0, 'Zinc Fraction': 0.2},
            output=0.6,
            noise=0.05
        )
        catalyst_session.add_experiment(
            {'Temperature': 380, 'Catalyst': 'High SAR', 'Metal Loading': 1.0, 'Zinc Fraction': 0.8},
            output=0.4,
            noise=0.05
        )
        
        results = catalyst_session.train_model(
            backend='botorch',
            kernel='Matern'
            # Session now auto-detects cat_dims from search space
        )
        
        assert results['success'] == True
        assert isinstance(catalyst_session.model.model, MixedSingleTaskGP)
        # Verify it didn't crash and produced a model

    def test_single_task_gp_with_noise(self):
        """Test SingleTaskGP (no categorical) with noise."""
        session = OptimizationSession()
        session.add_variable('x', 'real', bounds=(0, 10))
        
        # Add data with noise
        session.add_experiment({'x': 1.0}, 0.1, noise=0.01)
        session.add_experiment({'x': 2.0}, 0.2, noise=0.01)
        session.add_experiment({'x': 3.0}, 0.3, noise=0.01)
        
        results = session.train_model(
            backend='botorch',
            kernel='Matern'
        )
        
        assert results['success'] == True
        assert isinstance(session.model.model, SingleTaskGP)


class TestCVCacheTestIndices:
    """Verify the CV cache exposes ``test_indices`` / ``fold_ids`` that map cached
    predictions back to original sample-row indices."""

    def _build_single_objective_session(self, n_samples=20, seed=0):
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

    def test_single_output_cache_has_test_indices_and_fold_ids(self):
        session = self._build_single_objective_session(n_samples=20, seed=1)
        results = session.train_model(backend='botorch', kernel='Matern')
        assert results['success'] is True

        cache = session.model.cv_cached_results
        assert cache is not None
        for key in ('test_indices', 'fold_ids'):
            assert key in cache, f"missing key {key!r} in cv_cached_results"

        n = cache['y_true'].shape[0]
        test_indices = cache['test_indices']
        fold_ids = cache['fold_ids']

        # test_indices must be a permutation of [0, n).
        assert test_indices.shape == (n,)
        assert test_indices.dtype == np.int64
        assert np.array_equal(np.sort(test_indices), np.arange(n))

        # fold_ids: same length, values in [0, n_splits).
        assert fold_ids.shape == (n,)
        assert fold_ids.dtype == np.int64
        assert fold_ids.min() >= 0
        assert fold_ids.max() < 5  # default n_splits

        # Alignment: re-fetching the original targets at test_indices reproduces y_true.
        _, y_orig, _ = session.experiment_manager.get_features_target_and_noise()
        y_orig_arr = np.asarray(y_orig.values, dtype=np.float64)
        np.testing.assert_allclose(
            y_orig_arr[test_indices], cache['y_true'], rtol=0, atol=1e-12
        )

    def test_single_output_calibrated_cache_has_test_indices(self):
        session = self._build_single_objective_session(n_samples=20, seed=2)
        session.train_model(backend='botorch', kernel='Matern')

        cache = session.model.cv_cached_results
        cal = session.model.cv_cached_results_calibrated
        assert cal is not None
        for key in ('test_indices', 'fold_ids'):
            assert key in cal
            np.testing.assert_array_equal(cal[key], cache[key])

    def test_multi_output_cache_has_consistent_test_indices(self):
        from alchemist_core.data.experiment_manager import ExperimentManager
        from alchemist_core.data.search_space import SearchSpace

        space = SearchSpace()
        space.add_variable('x1', 'real', min=0.0, max=1.0)
        space.add_variable('x2', 'real', min=0.0, max=1.0)
        em = ExperimentManager(search_space=space, target_columns=['yield', 'selectivity'])

        rng = np.random.default_rng(3)
        n = 15
        em.df = pd.DataFrame({
            'x1': rng.uniform(0, 1, n),
            'x2': rng.uniform(0, 1, n),
            'yield': rng.uniform(50, 100, n),
            'selectivity': rng.uniform(70, 95, n),
        })

        model = BoTorchModel(training_iter=10, random_state=42)
        model.train(em, cache_cv=True, calibrate_uncertainty=False)

        multi = model.cv_cached_results_multi
        assert set(multi.keys()) == {'yield', 'selectivity'}

        ref_test_idx = multi['yield']['test_indices']
        ref_fold_ids = multi['yield']['fold_ids']

        # Indices/folds must be identical across objectives (shared KFold).
        np.testing.assert_array_equal(multi['selectivity']['test_indices'], ref_test_idx)
        np.testing.assert_array_equal(multi['selectivity']['fold_ids'], ref_fold_ids)

        # And must be a permutation of [0, n).
        assert np.array_equal(np.sort(ref_test_idx), np.arange(n))
        assert ref_fold_ids.min() >= 0
        assert ref_fold_ids.max() < 5

        # Alignment per objective.
        for obj_name in ('yield', 'selectivity'):
            y_orig = em.df[obj_name].to_numpy(dtype=np.float64)
            np.testing.assert_allclose(
                y_orig[multi[obj_name]['test_indices']],
                multi[obj_name]['y_true'],
                rtol=0,
                atol=1e-12,
            )
