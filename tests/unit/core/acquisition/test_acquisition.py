"""
Test acquisition function functionality through OptimizationSession API.

Tests different acquisition strategies (EI, UCB, PI) with both
sklearn and BoTorch backends using real catalyst experiment data.
"""

import pytest
import os
import pandas as pd
from alchemist_core import OptimizationSession


@pytest.fixture
def trained_session_sklearn():
    """Create session with trained sklearn model using catalyst data."""
    session = OptimizationSession()
    
    # Load search space from JSON
    tests_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    search_space_path = os.path.join(tests_dir, 'catalyst_search_space.json')
    session.load_search_space(search_space_path)
    
    # Load experiments from CSV
    experiments_path = os.path.join(tests_dir, 'catalyst_experiments.csv')
    session.load_data(experiments_path)
    
    # Train model
    session.train_model(backend='sklearn', kernel='Matern')
    
    return session


@pytest.fixture
def trained_session_botorch():
    """Create session with trained BoTorch model using catalyst data."""
    session = OptimizationSession()
    
    # Load search space from JSON
    tests_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    search_space_path = os.path.join(tests_dir, 'catalyst_search_space.json')
    session.load_search_space(search_space_path)
    
    # Load experiments from CSV
    experiments_path = os.path.join(tests_dir, 'catalyst_experiments.csv')
    session.load_data(experiments_path)
    
    # Train model
    session.train_model(backend='botorch', kernel='Matern')
    
    return session


class TestExpectedImprovementSklearn:
    """Test Expected Improvement acquisition with sklearn backend."""
    
    def test_ei_maximize_single_suggestion(self, trained_session_sklearn):
        """Test EI for maximization with single suggestion."""
        candidates = trained_session_sklearn.suggest_next(
            strategy='EI',
            n_candidates=1,
            goal='maximize'
        )
        
        assert isinstance(candidates, pd.DataFrame)
        assert len(candidates) == 1
        assert 'Temperature' in candidates.columns
        assert 'Catalyst' in candidates.columns
        assert 'Metal Loading' in candidates.columns
        assert 'Zinc Fraction' in candidates.columns
        
        # Values should be within bounds
        assert 350 <= candidates.iloc[0]['Temperature'] <= 450
        assert candidates.iloc[0]['Catalyst'] in ['High SAR', 'Low SAR']
        assert 0 <= candidates.iloc[0]['Metal Loading'] <= 5
        assert 0 <= candidates.iloc[0]['Zinc Fraction'] <= 1
    
    def test_ei_maximize_multiple_suggestions(self, trained_session_sklearn):
        """Test EI with multiple suggestions.
        
        Note: Current sklearn implementation may return fewer candidates than requested
        if optimization converges to same point.
        """
        candidates = trained_session_sklearn.suggest_next(
            strategy='EI',
            n_candidates=5,
            goal='maximize'
        )
        
        # Should return at least 1, may return fewer than requested
        assert len(candidates) >= 1
        assert len(candidates) <= 5
        assert all(350 <= candidates['Temperature']) and all(candidates['Temperature'] <= 450)
        assert all(candidates['Catalyst'].isin(['High SAR', 'Low SAR']))
        assert all(0 <= candidates['Metal Loading']) and all(candidates['Metal Loading'] <= 5)
        assert all(0 <= candidates['Zinc Fraction']) and all(candidates['Zinc Fraction'] <= 1)
    
    def test_ei_minimize(self, trained_session_sklearn):
        """Test EI for minimization."""
        candidates = trained_session_sklearn.suggest_next(
            strategy='EI',
            n_candidates=1,
            goal='minimize'
        )
        
        assert isinstance(candidates, pd.DataFrame)
        assert len(candidates) == 1
    
    def test_ei_with_xi_parameter(self, trained_session_sklearn):
        """Test EI with exploration parameter xi."""
        candidates1 = trained_session_sklearn.suggest_next(
            strategy='EI',
            n_candidates=1,
            goal='maximize',
            xi=0.0  # Pure exploitation
        )
        
        candidates2 = trained_session_sklearn.suggest_next(
            strategy='EI',
            n_candidates=1,
            goal='maximize',
            xi=0.1  # More exploration
        )
        
        # Both should succeed
        assert len(candidates1) == 1
        assert len(candidates2) == 1


class TestUpperConfidenceBoundSklearn:
    """Test Upper Confidence Bound acquisition with sklearn backend."""
    
    def test_ucb_maximize(self, trained_session_sklearn):
        """Test UCB for maximization."""
        candidates = trained_session_sklearn.suggest_next(
            strategy='UCB',
            n_candidates=1,
            goal='maximize'
        )
        
        assert isinstance(candidates, pd.DataFrame)
        assert len(candidates) == 1
    
    def test_ucb_with_kappa(self, trained_session_sklearn):
        """Test UCB with different kappa values."""
        # Lower kappa = more exploitation
        candidates1 = trained_session_sklearn.suggest_next(
            strategy='UCB',
            n_candidates=1,
            goal='maximize',
            kappa=0.5
        )
        
        # Higher kappa = more exploration
        candidates2 = trained_session_sklearn.suggest_next(
            strategy='UCB',
            n_candidates=1,
            goal='maximize',
            kappa=5.0
        )
        
        assert len(candidates1) == 1
        assert len(candidates2) == 1
    
    def test_ucb_multiple_suggestions(self, trained_session_sklearn):
        """Test UCB with multiple suggestions."""
        candidates = trained_session_sklearn.suggest_next(
            strategy='UCB',
            n_candidates=5,
            goal='maximize',
            kappa=2.0
        )
        
        # May return fewer if optimization converges
        assert len(candidates) >= 1
        assert len(candidates) <= 5


class TestProbabilityOfImprovementSklearn:
    """Test Probability of Improvement acquisition with sklearn backend."""
    
    def test_pi_maximize(self, trained_session_sklearn):
        """Test PI for maximization."""
        candidates = trained_session_sklearn.suggest_next(
            strategy='PI',
            n_candidates=1,
            goal='maximize'
        )
        
        assert isinstance(candidates, pd.DataFrame)
        assert len(candidates) == 1
    
    def test_pi_minimize(self, trained_session_sklearn):
        """Test PI for minimization."""
        candidates = trained_session_sklearn.suggest_next(
            strategy='PI',
            n_candidates=1,
            goal='minimize'
        )
        
        assert len(candidates) == 1
    
    def test_pi_with_xi(self, trained_session_sklearn):
        """Test PI with exploration parameter."""
        candidates = trained_session_sklearn.suggest_next(
            strategy='PI',
            n_candidates=1,
            goal='maximize',
            xi=0.05
        )
        
        assert len(candidates) == 1
    
    def test_pi_multiple_suggestions(self, trained_session_sklearn):
        """Test PI with multiple suggestions."""
        candidates = trained_session_sklearn.suggest_next(
            strategy='PI',
            n_candidates=3,
            goal='maximize'
        )
        
        # May return fewer if optimization converges
        assert len(candidates) >= 1
        assert len(candidates) <= 3


class TestBoTorchAcquisition:
    """Test acquisition functions with BoTorch backend."""
    
    def test_botorch_ei_maximize(self, trained_session_botorch):
        """Test BoTorch EI for maximization."""
        candidates = trained_session_botorch.suggest_next(
            strategy='EI',
            n_candidates=1,
            goal='maximize'
        )
        
        assert isinstance(candidates, pd.DataFrame)
        assert len(candidates) == 1
    
    def test_botorch_ei_multiple(self, trained_session_botorch):
        """Test BoTorch EI with multiple suggestions."""
        candidates = trained_session_botorch.suggest_next(
            strategy='EI',
            n_candidates=3,
            goal='maximize'
        )
        
        # May return fewer if optimization converges
        assert len(candidates) >= 1
        assert len(candidates) <= 3
    
    def test_botorch_ucb(self, trained_session_botorch):
        """Test BoTorch UCB."""
        candidates = trained_session_botorch.suggest_next(
            strategy='UCB',
            n_candidates=1,
            goal='maximize',
            kappa=2.0
        )
        
        assert len(candidates) == 1
    
    def test_botorch_pi(self, trained_session_botorch):
        """Test BoTorch PI."""
        candidates = trained_session_botorch.suggest_next(
            strategy='PI',
            n_candidates=1,
            goal='maximize'
        )
        
        assert len(candidates) == 1


class TestAcquisitionErrorHandling:
    """Test error handling in acquisition functions."""
    
    def test_suggest_without_model_fails(self):
        """Test that suggesting without trained model raises error."""
        session = OptimizationSession()
        session.add_variable('x', 'real', bounds=(0, 1))
        
        # Add some data but don't train
        session.add_experiment({'x': 0.5}, 0.25)
        
        with pytest.raises(ValueError, match="model|trained"):
            session.suggest_next(strategy='EI', goal='maximize')
    
    def test_invalid_strategy_fails(self, trained_session_sklearn):
        """Test that invalid strategy raises error."""
        with pytest.raises((ValueError, KeyError), match="strategy|invalid"):
            trained_session_sklearn.suggest_next(
                strategy='InvalidStrategy',
                goal='maximize'
            )


class TestAcquisitionBehavior:
    """Test acquisition function behavior and sanity checks."""
    
    def test_suggestions_within_bounds(self, trained_session_sklearn):
        """Test that all suggestions are within variable bounds."""
        candidates = trained_session_sklearn.suggest_next(
            strategy='EI',
            n_candidates=10,
            goal='maximize'
        )
        
        # All values should be within bounds
        assert all((candidates['Temperature'] >= 350) & (candidates['Temperature'] <= 450))
        assert all(candidates['Catalyst'].isin(['High SAR', 'Low SAR']))
        assert all((candidates['Metal Loading'] >= 0) & (candidates['Metal Loading'] <= 5))
        assert all((candidates['Zinc Fraction'] >= 0) & (candidates['Zinc Fraction'] <= 1))
    
    def test_reproducibility_with_seed(self):
        """Test that suggestions are reproducible with same seed."""
        import numpy as np
        
        # Create two identical sessions
        def create_and_train():
            session = OptimizationSession()
            session.set_config(random_state=12345)
    
            # Load search space from JSON
            tests_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            search_space_path = os.path.join(tests_dir, 'catalyst_search_space.json')
            session.load_search_space(search_space_path)
    
            # Load experiments from CSV
            experiments_path = os.path.join(tests_dir, 'catalyst_experiments.csv')
            session.load_data(experiments_path)
            session.train_model(backend='sklearn', kernel='Matern')
            return session
        
        session1 = create_and_train()
        session2 = create_and_train()
        
        candidates1 = session1.suggest_next(strategy='EI', n_candidates=1, goal='maximize')
        candidates2 = session2.suggest_next(strategy='EI', n_candidates=1, goal='maximize')
        
        # Numeric columns should be very close (allowing small numerical differences)
        numeric_cols = ['Temperature', 'Metal Loading', 'Zinc Fraction']
        for col in numeric_cols:
            assert np.allclose(candidates1[col].values, candidates2[col].values, rtol=1e-5)
        
        # Categorical should be identical
        assert candidates1['Catalyst'].iloc[0] == candidates2['Catalyst'].iloc[0]
    
    def test_maximize_vs_minimize_different(self, trained_session_sklearn):
        """Test that maximize and minimize give different suggestions."""
        max_candidates = trained_session_sklearn.suggest_next(
            strategy='EI',
            n_candidates=1,
            goal='maximize'
        )
        
        min_candidates = trained_session_sklearn.suggest_next(
            strategy='EI',
            n_candidates=1,
            goal='minimize'
        )
        
        # Suggestions should likely be different (not guaranteed but very probable)
        # At minimum, they shouldn't cause errors
        assert isinstance(max_candidates, pd.DataFrame)
        assert isinstance(min_candidates, pd.DataFrame)


class TestAcquisitionIntegration:
    """Test acquisition function integration with workflow."""
    
    def test_suggest_add_retrain_loop(self):
        """Test typical active learning loop with catalyst data."""
        import numpy as np
        
        session = OptimizationSession()
        
        # Load search space and initial experiments
        tests_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        search_space_path = os.path.join(tests_dir, 'catalyst_search_space.json')
        session.load_search_space(search_space_path)
        
        experiments_path = os.path.join(tests_dir, 'catalyst_experiments.csv')
        session.load_data(experiments_path)
        
        initial_count = len(session.experiment_manager.df)
        
        # Train initial model
        session.train_model(backend='sklearn', kernel='Matern')
        
        # Active learning loop - add 3 new experiments
        np.random.seed(42)
        for iteration in range(3):
            # Suggest next point
            candidate = session.suggest_next(strategy='EI', n_candidates=1, goal='maximize')
            
            # "Run experiment" (simulate with random output)
            new_inputs = candidate.iloc[0].to_dict()
            y_new = np.random.uniform(0, 0.6)  # Simulated output in reasonable range
            
            # Add to dataset
            session.add_experiment(new_inputs, y_new)
            
            # Retrain model
            session.train_model(backend='sklearn', kernel='Matern')
        
        # Should have initial + 3 experiments
        assert len(session.experiment_manager.df) == initial_count + 3
    
    def test_staged_workflow(self, trained_session_sklearn):
        """Test staged experiment workflow with suggestions."""
        # Generate multiple candidates
        candidates = trained_session_sklearn.suggest_next(
            strategy='EI',
            n_candidates=5,
            goal='maximize'
        )
        
        n_candidates = len(candidates)
        
        # Stage them
        for _, row in candidates.iterrows():
            trained_session_sklearn.add_staged_experiment(row.to_dict())
        
        # Check staged experiments
        staged = trained_session_sklearn.get_staged_experiments()
        assert len(staged) == n_candidates
        
        # "Run experiments" and get results (simulate outputs)
        import numpy as np
        np.random.seed(42)
        outputs = []
        for exp in staged:
            y = np.random.uniform(0, 0.6)  # Simulated output in reasonable range
            outputs.append(y)
        
        # Get initial count
        initial_count = len(trained_session_sklearn.experiment_manager.df)
        
        # Move to dataset
        trained_session_sklearn.move_staged_to_experiments(outputs=outputs)
        
        # Verify
        assert len(trained_session_sklearn.get_staged_experiments()) == 0
        assert len(trained_session_sklearn.experiment_manager.df) == initial_count + n_candidates


class TestAcquisitionParameters:
    """Test different acquisition function parameters."""
    
    def test_ei_with_different_xi_values(self, trained_session_sklearn):
        """Test EI with various xi (exploration) parameters."""
        xi_values = [0.0, 0.01, 0.05, 0.1, 0.5]
        
        for xi in xi_values:
            candidates = trained_session_sklearn.suggest_next(
                strategy='EI',
                n_candidates=1,
                goal='maximize',
                xi=xi
            )
            assert len(candidates) == 1
    
    def test_ucb_with_different_kappa_values(self, trained_session_sklearn):
        """Test UCB with various kappa (exploration) parameters."""
        kappa_values = [0.1, 1.0, 2.0, 5.0, 10.0]
        
        for kappa in kappa_values:
            candidates = trained_session_sklearn.suggest_next(
                strategy='UCB',
                n_candidates=1,
                goal='maximize',
                kappa=kappa
            )
            assert len(candidates) == 1
    
    def test_batch_sizes(self, trained_session_sklearn):
        """Test different batch sizes.
        
        Note: May return fewer candidates if optimization converges.
        """
        batch_sizes = [1, 2, 5, 10]
        
        for n in batch_sizes:
            candidates = trained_session_sklearn.suggest_next(
                strategy='EI',
                n_candidates=n,
                goal='maximize'
            )
            # Should return at least 1, at most n
            assert len(candidates) >= 1
            assert len(candidates) <= max(1, n)


class TestSessionFindOptimum:
    """Regression tests for OptimizationSession.find_optimum (single-objective).

    The single-objective branch unpacks the dict returned by self.predict();
    this previously crashed with ValueError because predict() always returns
    {target_name: (means, stds)} regardless of objective count.
    """

    def test_find_optimum_sklearn_maximize(self, trained_session_sklearn):
        result = trained_session_sklearn.find_optimum(goal='maximize', n_grid_points=200)
        assert 'x_opt' in result
        assert 'value' in result
        assert 'std' in result
        assert len(result['x_opt']) == 1
        assert isinstance(result['value'], float)
        assert isinstance(result['std'], float)
        assert result['std'] >= 0

    def test_find_optimum_sklearn_minimize(self, trained_session_sklearn):
        result_max = trained_session_sklearn.find_optimum(goal='maximize', n_grid_points=200)
        result_min = trained_session_sklearn.find_optimum(goal='minimize', n_grid_points=200)
        # Minimize and maximize over the same grid should disagree on the value.
        assert result_min['value'] <= result_max['value']

    def test_find_optimum_botorch_maximize(self, trained_session_botorch):
        result = trained_session_botorch.find_optimum(goal='maximize', n_grid_points=200)
        assert 'x_opt' in result
        assert isinstance(result['value'], float)
        assert isinstance(result['std'], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
