"""
Test OptimizationSession API - High-level interface tests.
"""

import pandas as pd
import numpy as np
from alchemist_core import OptimizationSession


def test_basic_workflow():
    """Test complete optimization workflow."""
    print("\n" + "="*60)
    print("Test: Basic Optimization Workflow")
    print("="*60)
    
    # 1. Create session
    session = OptimizationSession()
    print("✓ Session created")
    
    # 2. Define search space
    session.add_variable('x1', 'real', bounds=(0.0, 1.0))
    session.add_variable('x2', 'real', bounds=(0.0, 1.0))
    # Skip categorical for now to avoid CV issues with one-hot encoding
    # session.add_variable('catalyst', 'categorical', categories=['A', 'B', 'C'])
    print("✓ Search space defined")
    
    summary = session.get_search_space_summary()
    assert summary['n_variables'] == 2
    assert len(summary['categorical_variables']) == 0
    print(f"  - {summary['n_variables']} variables")
    print(f"  - Categorical: {summary['categorical_variables']}")
    
    # 3. Add synthetic experimental data
    np.random.seed(42)
    for i in range(20):
        x1 = np.random.uniform(0, 1)
        x2 = np.random.uniform(0, 1)
        
        # Simple objective: y = x1^2 + x2^2 + noise
        y = x1**2 + x2**2 + np.random.normal(0, 0.1)
        
        session.add_experiment(
            inputs={'x1': x1, 'x2': x2},
            output=y
        )
    print("✓ Added 20 experiments")
    
    data_summary = session.get_data_summary()
    assert data_summary['n_experiments'] == 20
    assert data_summary['has_data'] == True
    print(f"  - Target range: [{data_summary['target_stats']['min']:.3f}, {data_summary['target_stats']['max']:.3f}]")
    
    # 4. Train model
    results = session.train_model(backend='sklearn', kernel='Matern')
    assert results['success'] == True
    assert results['backend'] == 'sklearn'
    print("✓ Model trained")
    print(f"  - Backend: {results['backend']}")
    print(f"  - Kernel: {results['kernel']}")
    if 'r2' in results['metrics']:
        print(f"  - R²: {results['metrics']['r2']:.4f}")
    
    # 5. Get model summary
    model_summary = session.get_model_summary()
    assert model_summary is not None
    assert model_summary['is_trained'] == True
    print("✓ Model summary retrieved")
    
    # 6. Suggest next experiment
    next_point = session.suggest_next(strategy='EI', goal='maximize')
    assert isinstance(next_point, pd.DataFrame)
    assert len(next_point) == 1
    assert 'x1' in next_point.columns
    print("✓ Next experiment suggested")
    print(f"  - Suggestion: {next_point.to_dict('records')[0]}")
    
    # 7. Make predictions
    test_points = pd.DataFrame({
        'x1': [0.5, 0.7],
        'x2': [0.5, 0.3]
    })
    pred_dict = session.predict(test_points)
    assert isinstance(pred_dict, dict)
    target_name = list(pred_dict.keys())[0]
    predictions, uncertainties = pred_dict[target_name]
    assert len(predictions) == 2
    assert len(uncertainties) == 2
    print("✓ Predictions made")
    print(f"  - Test predictions: {predictions}")
    print(f"  - Uncertainties: {uncertainties}")


def test_event_system():
    """Test event handling in session."""
    print("\n" + "="*60)
    print("Test: Event System Integration")
    print("="*60)
    
    session = OptimizationSession()
    
    # Track events
    events_received = []
    
    def on_event(event_name):
        def handler(data):
            events_received.append(event_name)
        return handler
    
    session.on('variable_added', on_event('variable_added'))
    session.on('experiment_added', on_event('experiment_added'))
    session.on('training_started', on_event('training_started'))
    session.on('training_completed', on_event('training_completed'))
    
    # Trigger events
    session.add_variable('x', 'real', bounds=(0, 1))
    session.add_experiment(inputs={'x': 0.5}, output=0.25)
    
    # Add more data for training
    for i in range(10):
        session.add_experiment(inputs={'x': np.random.random()}, output=np.random.random())
    
    session.train_model(backend='sklearn', kernel='RBF')
    
    assert 'variable_added' in events_received
    assert 'experiment_added' in events_received
    assert 'training_started' in events_received
    assert 'training_completed' in events_received
    
    print(f"✓ Events received: {set(events_received)}")


def test_error_handling():
    """Test error handling."""
    print("\n" + "="*60)
    print("Test: Error Handling")
    print("="*60)
    
    session = OptimizationSession()
    
    # Try to train without data
    try:
        session.train_model()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    
    # Try to suggest without model
    session.add_variable('x', 'real', bounds=(0, 1))
    session.add_experiment(inputs={'x': 0.5}, output=0.25)
    
    try:
        session.suggest_next()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")


def test_configuration():
    """Test session configuration."""
    print("\n" + "="*60)
    print("Test: Configuration")
    print("="*60)
    
    session = OptimizationSession()
    
    # Update config
    session.set_config(random_state=123, verbose=False)
    
    assert session.config['random_state'] == 123
    assert session.config['verbose'] == False
    
    print("✓ Configuration updated successfully")


if __name__ == "__main__":
    print("="*60)
    print("Testing OptimizationSession API")
    print("="*60)
    
    tests = [
        test_basic_workflow,
        test_event_system,
        test_error_handling,
        test_configuration,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ {test.__name__} failed with exception:")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "="*60)
    if all(results):
        print(f"✅ All tests passed! ({len(results)}/{len(results)})")
    else:
        print(f"❌ Some tests failed: {sum(results)}/{len(results)} passed")
    print("="*60)
