"""
Test model training functionality through OptimizationSession API.

This tests the model backends (sklearn, botorch) via the high-level Session interface,
covering different kernels, parameters, transforms, and training options.
Uses real catalyst experiment data for testing.
"""

import pytest
import os
import numpy as np
import pandas as pd
from alchemist_core import OptimizationSession


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


# Keep old fixtures as aliases for compatibility
@pytest.fixture
def simple_session(catalyst_session):
    """Alias for catalyst_session - provides real catalyst data."""
    return catalyst_session


@pytest.fixture
def mixed_session(catalyst_session):
    """Alias for catalyst_session - already has mixed variable types."""
    return catalyst_session


class TestSklearnModelTraining:
    """Test sklearn backend model training."""
    
    def test_train_matern_kernel(self, simple_session):
        """Test training with Matern kernel."""
        results = simple_session.train_model(
            backend='sklearn',
            kernel='Matern',
            kernel_params={'nu': 2.5}
        )
        
        assert results['success'] == True
        assert results['backend'] == 'sklearn'
        assert results['kernel'] == 'Matern'
        assert simple_session.model is not None
        assert simple_session.model.is_trained == True
    
    def test_train_rbf_kernel(self, simple_session):
        """Test training with RBF kernel."""
        results = simple_session.train_model(
            backend='sklearn',
            kernel='RBF'
        )
        
        assert results['success'] == True
        assert results['kernel'] == 'RBF'
        assert simple_session.model.is_trained == True
    
    def test_train_rational_quadratic(self, simple_session):
        """Test training with RationalQuadratic kernel."""
        results = simple_session.train_model(
            backend='sklearn',
            kernel='RationalQuadratic'
        )
        
        assert results['success'] == True
        assert results['kernel'] == 'RationalQuadratic'
    
    def test_cross_validation_metrics(self, simple_session):
        """Test that cross-validation metrics are returned."""
        results = simple_session.train_model(
            backend='sklearn',
            kernel='Matern'
        )
        
        assert 'metrics' in results
        metrics = results['metrics']
        
        # Check that key metrics are present
        assert 'cv_r2' in metrics or 'r2' in metrics
        assert 'cv_rmse' in metrics or 'rmse' in metrics
        assert 'cv_mae' in metrics or 'mae' in metrics
    
    def test_hyperparameters_returned(self, simple_session):
        """Test that hyperparameters are returned after training."""
        results = simple_session.train_model(
            backend='sklearn',
            kernel='Matern',
            kernel_params={'nu': 2.5}
        )
        
        assert 'hyperparameters' in results
        hyperparams = results['hyperparameters']
        
        # Should contain kernel information
        assert isinstance(hyperparams, dict)
        assert len(hyperparams) > 0
    
    def test_input_transform_standard(self, simple_session):
        """Test training with standard input scaling."""
        results = simple_session.train_model(
            backend='sklearn',
            kernel='Matern',
            input_transform_type='standard'
        )
        
        assert results['success'] == True
        assert hasattr(simple_session.model, 'input_scaler')
    
    def test_input_transform_minmax(self, simple_session):
        """Test training with minmax input scaling."""
        results = simple_session.train_model(
            backend='sklearn',
            kernel='Matern',
            input_transform_type='minmax'
        )
        
        assert results['success'] == True
        assert hasattr(simple_session.model, 'input_scaler')
    
    def test_output_transform(self, simple_session):
        """Test training with output standardization."""
        results = simple_session.train_model(
            backend='sklearn',
            kernel='Matern',
            output_transform_type='standard'
        )
        
        assert results['success'] == True
        assert hasattr(simple_session.model, 'output_scaler')
    
    def test_combined_transforms(self, simple_session):
        """Test training with both input and output transforms."""
        results = simple_session.train_model(
            backend='sklearn',
            kernel='Matern',
            input_transform_type='standard',
            output_transform_type='standard'
        )
        
        assert results['success'] == True
        assert hasattr(simple_session.model, 'input_scaler')
        assert hasattr(simple_session.model, 'output_scaler')
    
    def test_mixed_variable_types(self, mixed_session):
        """Test training with mixed variable types (real, integer, categorical)."""
        results = mixed_session.train_model(
            backend='sklearn',
            kernel='Matern'
        )
        
        assert results['success'] == True
        assert mixed_session.model.is_trained == True


class TestBoTorchModelTraining:
    """Test BoTorch backend model training."""
    
    def test_train_matern_kernel(self, simple_session):
        """Test training with Matern kernel."""
        results = simple_session.train_model(
            backend='botorch',
            kernel='Matern',
            kernel_params={'nu': 2.5}
        )
        
        assert results['success'] == True
        assert results['backend'] == 'botorch'
        assert results['kernel'] == 'Matern'
        assert simple_session.model is not None
        assert simple_session.model.is_trained == True
    
    def test_train_rbf_kernel(self, simple_session):
        """Test training with RBF kernel."""
        results = simple_session.train_model(
            backend='botorch',
            kernel='RBF'
        )
        
        assert results['success'] == True
        assert results['kernel'] == 'RBF'
    
    def test_default_transforms_applied(self, simple_session):
        """Test that BoTorch applies default input/output transforms."""
        results = simple_session.train_model(
            backend='botorch',
            kernel='Matern'
        )
        
        assert results['success'] == True
        # BoTorch should auto-apply normalize and standardize by default
        assert simple_session.model.is_trained == True
    
    def test_explicit_transforms(self, simple_session):
        """Test training with explicit transform specification."""
        results = simple_session.train_model(
            backend='botorch',
            kernel='Matern',
            input_transform_type='normalize',
            output_transform_type='standardize'
        )
        
        assert results['success'] == True
    
    def test_hyperparameters_returned(self, simple_session):
        """Test that hyperparameters are returned."""
        results = simple_session.train_model(
            backend='botorch',
            kernel='Matern',
            kernel_params={'nu': 2.5}
        )
        
        assert 'hyperparameters' in results
        hyperparams = results['hyperparameters']
        assert isinstance(hyperparams, dict)
    
    def test_mixed_variable_types(self, mixed_session):
        """Test training with mixed variable types."""
        results = mixed_session.train_model(
            backend='botorch',
            kernel='Matern'
        )

        assert results['success'] == True
        assert mixed_session.model.is_trained == True

    def test_train_ibnn_kernel(self, simple_session):
        """Test training with IBNN (Infinite-Width BNN) kernel."""
        results = simple_session.train_model(
            backend='botorch',
            kernel='IBNN',
            kernel_params={'ibnn_depth': 3}
        )

        assert results['success'] == True
        assert results['backend'] == 'botorch'
        assert results['kernel'] == 'IBNN'
        assert simple_session.model is not None
        assert simple_session.model.is_trained == True

    def test_ibnn_hyperparameters(self, simple_session):
        """Test that IBNN hyperparameters report kernel_type and depth."""
        simple_session.train_model(
            backend='botorch',
            kernel='IBNN',
            kernel_params={'ibnn_depth': 5}
        )

        hyperparams = simple_session.model.get_hyperparameters()
        assert hyperparams['kernel_type'] == 'IBNN'
        assert hyperparams['depth'] == 5
        # IBNN has no lengthscale
        assert 'nu' not in hyperparams

    def test_ibnn_predictions(self, simple_session):
        """Test that predictions work after training with IBNN kernel."""
        simple_session.train_model(
            backend='botorch',
            kernel='IBNN',
            kernel_params={'ibnn_depth': 3}
        )

        test_points = pd.DataFrame({
            'Temperature': [400.0, 425.0],
            'Catalyst': ['High SAR', 'Low SAR'],
            'Metal Loading': [2.0, 3.5],
            'Zinc Fraction': [0.5, 0.3]
        })

        mean, std = simple_session.model.predict(test_points, return_std=True)
        assert len(mean) == 2
        assert len(std) == 2
        assert all(s > 0 for s in std)


class TestModelComparison:
    """Test that sklearn and botorch backends produce reasonable results."""
    
    def test_both_backends_train_successfully(self, simple_session):
        """Test that both backends can train on same data."""
        # Train sklearn
        results_sklearn = simple_session.train_model(
            backend='sklearn',
            kernel='Matern'
        )
        assert results_sklearn['success'] == True
        
        # Create new session with same data for botorch
        session2 = OptimizationSession()
        
        # Load same search space and data
        tests_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        search_space_path = os.path.join(tests_dir, 'catalyst_search_space.json')
        session2.load_search_space(search_space_path)
        
        experiments_path = os.path.join(tests_dir, 'catalyst_experiments.csv')
        session2.load_data(experiments_path)
        
        # Train botorch
        results_botorch = session2.train_model(
            backend='botorch',
            kernel='Matern'
        )
        assert results_botorch['success'] == True
    
    def test_predictions_after_training(self, simple_session):
        """Test that predictions work after training."""
        simple_session.train_model(backend='sklearn', kernel='Matern')
        
        test_points = pd.DataFrame({
            'Temperature': [400.0, 425.0],
            'Catalyst': ['Low SAR', 'High SAR'],
            'Metal Loading': [2.5, 3.0],
            'Zinc Fraction': [0.5, 0.6]
        })
        
        pred_dict = simple_session.predict(test_points)
        assert isinstance(pred_dict, dict)
        target_name = list(pred_dict.keys())[0]
        predictions, uncertainties = pred_dict[target_name]
        
        assert len(predictions) == 2
        assert len(uncertainties) == 2
        assert all(uncertainties > 0)  # Uncertainties should be positive


class TestModelErrorHandling:
    """Test error handling in model training."""
    
    def test_train_without_data_fails(self):
        """Test that training without data raises error."""
        session = OptimizationSession()
        session.add_variable('x', 'real', bounds=(0, 1))
        
        with pytest.raises(ValueError, match="data|experiments"):
            session.train_model(backend='sklearn', kernel='Matern')
    
    def test_train_without_variables_fails(self):
        """Test that training without variables raises error."""
        session = OptimizationSession()
        
        with pytest.raises(ValueError):
            session.train_model(backend='sklearn', kernel='Matern')
    
    def test_invalid_backend_fails(self, simple_session):
        """Test that invalid backend raises error."""
        with pytest.raises(ValueError, match="backend|Unknown"):
            simple_session.train_model(backend='invalid_backend', kernel='Matern')
    
    def test_insufficient_data(self):
        """Test training with minimal data."""
        session = OptimizationSession()
        session.add_variable('x', 'real', bounds=(0, 1))
        
        # Add minimal experiments
        session.add_experiment({'x': 0.0}, 0.0)
        session.add_experiment({'x': 1.0}, 1.0)
        
        # Should work but may have warnings
        results = session.train_model(backend='sklearn', kernel='Matern')
        # At minimum, it should not crash
        assert results['success'] == True or 'error' in results


class TestModelRetraining:
    """Test model retraining scenarios."""
    
    def test_retrain_with_new_data(self, simple_session):
        """Test retraining after adding new data."""
        # Initial training
        results1 = simple_session.train_model(backend='sklearn', kernel='Matern')
        assert results1['success'] == True
        
        # Add more data
        np.random.seed(123)
        for i in range(5):
            temp = np.random.uniform(350, 450)
            catalyst = np.random.choice(['Low SAR', 'High SAR'])
            metal = np.random.uniform(0, 5)
            zinc = np.random.uniform(0, 1)
            y = np.random.uniform(0, 0.6)  # Simulated output
            simple_session.add_experiment({
                'Temperature': temp,
                'Catalyst': catalyst,
                'Metal Loading': metal,
                'Zinc Fraction': zinc
            }, y)
        
        # Retrain
        results2 = simple_session.train_model(backend='sklearn', kernel='Matern')
        assert results2['success'] == True
    
    def test_switch_backends(self, simple_session):
        """Test switching between backends."""
        # Train with sklearn
        simple_session.train_model(backend='sklearn', kernel='Matern')
        assert simple_session.model_backend == 'sklearn'
        
        # Switch to botorch
        simple_session.train_model(backend='botorch', kernel='Matern')
        assert simple_session.model_backend == 'botorch'
    
    def test_switch_kernels(self, simple_session):
        """Test training with different kernels."""
        # Train with Matern
        simple_session.train_model(backend='sklearn', kernel='Matern')
        
        # Retrain with RBF
        results = simple_session.train_model(backend='sklearn', kernel='RBF')
        assert results['kernel'] == 'RBF'


class TestModelStateQueries:
    """Test querying model state and information."""
    
    def test_model_summary_before_training(self):
        """Test model summary when no model is trained."""
        session = OptimizationSession()
        session.add_variable('x', 'real', bounds=(0, 1))
        
        summary = session.get_model_summary()
        assert summary is None
    
    def test_model_summary_after_training(self, simple_session):
        """Test model summary after training."""
        simple_session.train_model(backend='sklearn', kernel='Matern')
        
        summary = simple_session.get_model_summary()
        assert summary is not None
        assert isinstance(summary, dict)
    
    def test_hyperparameters_before_training(self):
        """Test that accessing model before training handles gracefully."""
        session = OptimizationSession()
        session.add_variable('x', 'real', bounds=(0, 1))
        
        # Model doesn't exist yet
        assert session.model is None
    
    def test_hyperparameters_after_training(self, simple_session):
        """Test accessing hyperparameters after training."""
        simple_session.train_model(backend='sklearn', kernel='Matern')
        
        # Can access model's hyperparameters directly
        hyperparams = simple_session.model.get_hyperparameters()
        assert hyperparams is not None
        assert isinstance(hyperparams, dict)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
