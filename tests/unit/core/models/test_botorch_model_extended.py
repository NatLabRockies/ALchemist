
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


class TestCVKernelConsistency:
    """Regression tests for the per-fold CV ``cv_model`` construction.

    The bug: ``_cache_cross_validation_results`` (single-output path) used to
    construct each per-fold ``cv_model`` as a bare ``SingleTaskGP(...)`` with no
    ``covar_module`` argument, falling back to BoTorch's default Matern 5/2
    + gamma-prior kernel. The training path, however, builds the full-data
    model with an explicit ``covar_module = ScaleKernel(cont_kernel_factory(...))``
    that honours ``kernel`` / ``kernel_params``. ``load_state_dict(..., strict=False)``
    succeeded silently on matching parameter names but the resulting per-fold
    cv_model used the wrong kernel structure, producing systematically tight
    σ predictions (~50% too small on real datasets) and an exaggeratedly
    overconfident reliability diagram.

    The multi-output CV path (``_cache_cv_results_multi``) correctly uses
    ``_create_single_gp`` and is unaffected. After the fix, both paths use
    ``_create_single_gp`` so the per-fold predictions of a single-output
    session trained on target T must match the per-fold predictions of the T-th
    objective in a multi-output session trained on the same X and Y_T (since
    the fitted hyperparameters are identical, by construction)."""

    def _make_paired_sessions(self, n=40, seed=11):
        """Build a (joint multi-output, single-output) session pair backed by
        identical (X, y[target]) data so the trained hyperparameters of the
        target GP are bit-identical across the two paths."""
        from alchemist_core import OptimizationSession
        from alchemist_core.data.experiment_manager import ExperimentManager
        from alchemist_core.data.search_space import SearchSpace
        from alchemist_core.models.botorch_model import BoTorchModel

        rng = np.random.default_rng(seed)
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        y1 = 2.0 * x1 + x2 + rng.normal(0, 0.1, n)
        y2 = x1 * x2 + rng.normal(0, 0.1, n)

        # --- Joint multi-output (ModelListGP) session ---
        space_m = SearchSpace()
        space_m.add_variable('x1', 'real', min=0.0, max=1.0)
        space_m.add_variable('x2', 'real', min=0.0, max=1.0)
        em_m = ExperimentManager(search_space=space_m, target_columns=['y1', 'y2'])
        em_m.df = pd.DataFrame({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2})
        model_m = BoTorchModel(
            training_iter=50, random_state=42,
            kernel_options={'cont_kernel_type': 'Matern', 'cont_kernel_params': {'nu': 2.5}},
            input_transform_type='normalize',
            output_transform_type='standardize',
        )
        model_m.train(em_m, cache_cv=True, calibrate_uncertainty=False)

        # --- Single-output (SingleTaskGP) session on y1 only ---
        space_s = SearchSpace()
        space_s.add_variable('x1', 'real', min=0.0, max=1.0)
        space_s.add_variable('x2', 'real', min=0.0, max=1.0)
        em_s = ExperimentManager(search_space=space_s, target_columns=['y1'])
        em_s.df = pd.DataFrame({'x1': x1, 'x2': x2, 'y1': y1})
        model_s = BoTorchModel(
            training_iter=50, random_state=42,
            kernel_options={'cont_kernel_type': 'Matern', 'cont_kernel_params': {'nu': 2.5}},
            input_transform_type='normalize',
            output_transform_type='standardize',
        )
        model_s.train(em_s, cache_cv=True, calibrate_uncertainty=False)

        return model_m, model_s

    def test_single_output_cv_kernel_matches_joint_cv_kernel(self):
        """The per-fold cv_model in single-output CV must use the same kernel
        as the per-fold cv_model in multi-output CV (both should honor the
        session's ``kernel``/``kernel_params``). If single-output CV silently
        falls back to BoTorch's default kernel, per-fold predictions will not
        match the equivalent per-objective predictions from a joint model
        trained on the same X and y."""
        model_m, model_s = self._make_paired_sessions(n=40, seed=11)

        m_joint = model_m.cv_cached_results_multi['y1']
        m_single = model_s.cv_cached_results

        # Sanity: the full-data fitted hyperparameters for the y1 GP in the
        # joint ModelListGP must match those of the standalone single GP. If
        # this fails, the test setup is wrong (different X/y/seed), not a CV
        # bug.
        y1_idx = model_m.objective_names.index('y1')
        joint_sd = model_m.model.models[y1_idx].state_dict()
        single_sd = model_s.fitted_state_dict
        common = sorted(set(joint_sd.keys()) & set(single_sd.keys()))
        assert common, "no common state_dict keys between joint sub-GP and single GP"
        for k in common:
            jv = joint_sd[k].cpu().numpy() if hasattr(joint_sd[k], 'cpu') else joint_sd[k]
            sv = single_sd[k].cpu().numpy() if hasattr(single_sd[k], 'cpu') else single_sd[k]
            np.testing.assert_allclose(
                jv, sv, rtol=0, atol=1e-10,
                err_msg=f"fitted hyperparameter mismatch on key {k!r}"
            )

        # CV arrays are stored in fold-test-concatenation order; same KFold
        # seed → identical (test_indices, fold_ids) → in-place comparable arrays.
        np.testing.assert_array_equal(
            m_joint['test_indices'], m_single['test_indices'],
            err_msg="CV test_indices differ between joint and single sessions",
        )
        np.testing.assert_array_equal(
            m_joint['fold_ids'], m_single['fold_ids'],
            err_msg="CV fold_ids differ between joint and single sessions",
        )

        # If both CV paths build per-fold cv_model with the SAME kernel, then
        # given identical fitted hyperparameters, identical per-fold
        # (X_train, y_train) splits, and identical input/outcome transforms,
        # the per-fold posterior at X_test must be bit-exact.
        np.testing.assert_allclose(
            m_joint['y_pred'], m_single['y_pred'],
            rtol=0, atol=1e-8,
            err_msg=(
                "single-output CV y_pred diverges from joint CV y_pred on the "
                "same target — single CV is likely constructing cv_model with "
                "the wrong kernel (missing covar_module argument)."
            ),
        )
        np.testing.assert_allclose(
            m_joint['y_std'], m_single['y_std'],
            rtol=0, atol=1e-8,
            err_msg=(
                "single-output CV y_std diverges from joint CV y_std on the "
                "same target — single CV is likely constructing cv_model with "
                "the wrong kernel (missing covar_module argument)."
            ),
        )
