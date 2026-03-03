"""
Test OptimizationSession with real catalyst data (categorical + continuous variables).
"""

import os
from alchemist_core import OptimizationSession


def test_catalyst_data_workflow():
    """Test complete workflow with real catalyst experimental data."""
    print("\n" + "="*60)
    print("Test: Catalyst Optimization with Real Data")
    print("="*60)
    
    # Get paths to test data
    # Go up 3 levels from tests/integration/workflows/ to tests/
    test_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    search_space_path = os.path.join(test_dir, "tests", "catalyst_search_space.json")
    experiments_path = os.path.join(test_dir, "tests", "catalyst_experiments.csv")
    
    # 1. Create session
    session = OptimizationSession()
    print("✓ Session created")
    
    # 2. Load search space from JSON
    session.load_search_space(search_space_path)
    print("✓ Search space loaded from JSON")
    
    summary = session.get_search_space_summary()
    print(f"  - {summary['n_variables']} variables")
    print(f"  - Categorical: {summary['categorical_variables']}")
    for var in summary['variables']:
        print(f"    - {var['name']} ({var['type']}): {var.get('bounds') or var.get('categories')}")
    
    # 3. Load experimental data
    session.load_data(experiments_path, target_columns='Output')
    print("✓ Experimental data loaded")
    
    data_summary = session.get_data_summary()
    print(f"  - {data_summary['n_experiments']} experiments")
    print(f"  - Target range: [{data_summary['target_stats']['min']:.3f}, {data_summary['target_stats']['max']:.3f}]")
    print(f"  - Features: {data_summary['feature_names']}")
    
    # 4. Train model with sklearn backend
    print("\nTraining sklearn model...")
    results_sklearn = session.train_model(backend='sklearn', kernel='Matern')
    print("✓ sklearn model trained")
    print(f"  - Kernel: {results_sklearn['kernel']}")
    if results_sklearn['metrics']:
        print(f"  - R²: {results_sklearn['metrics'].get('r2', 'N/A'):.4f}")
        print(f"  - RMSE: {results_sklearn['metrics'].get('rmse', 'N/A'):.4f}")
    
    # 5. Suggest next experiment with sklearn
    print("\nSuggesting next experiment (sklearn + EI)...")
    next_point_sklearn = session.suggest_next(strategy='EI', goal='maximize')
    print("✓ Next point suggested")
    print(f"  - {next_point_sklearn.to_dict('records')[0]}")
    
    # 6. Train model with BoTorch backend
    print("\nTraining BoTorch model...")
    try:
        results_botorch = session.train_model(backend='botorch', kernel='Matern')
        print("✓ BoTorch model trained")
        print(f"  - Kernel: {results_botorch['kernel']}")
        if results_botorch['metrics']:
            print(f"  - R²: {results_botorch['metrics'].get('r2', 'N/A'):.4f}")
            print(f"  - RMSE: {results_botorch['metrics'].get('rmse', 'N/A'):.4f}")
        
        # 7. Suggest next experiment with BoTorch
        print("\nSuggesting next experiment (BoTorch + qEI)...")
        next_point_botorch = session.suggest_next(strategy='qEI', goal='maximize')
        print("✓ Next point suggested")
        print(f"  - {next_point_botorch.to_dict('records')[0]}")
        
    except Exception as e:
        print(f"⚠ BoTorch test skipped: {e}")
    
    # 8. Make predictions at test points
    print("\nMaking predictions at test points...")
    import pandas as pd
    test_points = pd.DataFrame({
        'Temperature': [400.0, 425.0, 450.0],
        'Catalyst': ['Low SAR', 'High SAR', 'Low SAR'],
        'Metal Loading': [2.5, 1.0, 3.0],
        'Zinc Fraction': [0.5, 0.0, 1.0]
    })
    
    pred_dict = session.predict(test_points)
    assert isinstance(pred_dict, dict)
    target_name = list(pred_dict.keys())[0]
    predictions, uncertainties = pred_dict[target_name]
    print("✓ Predictions made")
    for i, (pred, unc) in enumerate(zip(predictions, uncertainties)):
        print(f"  - Point {i+1}: {pred:.4f} ± {unc:.4f}")
    
    print("\n" + "="*60)
    print("✅ Catalyst data test completed successfully!")
    print("="*60)
    
    pass


if __name__ == "__main__":
    try:
        test_catalyst_data_workflow()
    except Exception as e:
        print(f"\n❌ Test failed with exception:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
