"""
Test OptimizationSession visualization methods.
"""

import pytest
import numpy as np
import pandas as pd
from alchemist_core import OptimizationSession

# Check if matplotlib is available
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Skip all tests if matplotlib not available
pytestmark = pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")


def create_trained_session():
    """Helper to create a session with trained model."""
    session = OptimizationSession()
    
    # Add variables
    session.add_variable('x1', 'real', bounds=(0.0, 1.0))
    session.add_variable('x2', 'real', bounds=(0.0, 1.0))
    
    # Add synthetic data
    np.random.seed(42)
    for i in range(20):
        x1 = np.random.uniform(0, 1)
        x2 = np.random.uniform(0, 1)
        y = x1**2 + x2**2 + np.random.normal(0, 0.1)
        session.add_experiment({'x1': x1, 'x2': x2}, output=y)
    
    # Train model
    session.train_model(backend='sklearn', kernel='Matern')
    
    return session


def test_matplotlib_import_check():
    """Test that matplotlib check works."""
    session = OptimizationSession()
    # Should not raise since we have matplotlib in this test
    session._check_matplotlib()


def test_model_trained_check():
    """Test that model trained check works."""
    session = OptimizationSession()
    
    with pytest.raises(ValueError, match="Model not trained"):
        session._check_model_trained()


def test_plot_parity_returns_figure():
    """Test that plot_parity returns matplotlib Figure."""
    session = create_trained_session()
    
    fig = session.plot_parity()
    
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    
    plt.close(fig)


def test_plot_parity_with_options():
    """Test plot_parity with various options."""
    session = create_trained_session()
    
    fig = session.plot_parity(
        sigma_multiplier=2.58,
        figsize=(10, 8),
        show_error_bars=True,
        show_metrics=True
    )
    
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_contour_returns_figure():
    """Test that plot_contour returns matplotlib Figure."""
    session = create_trained_session()
    
    fig = session.plot_contour('x1', 'x2')
    
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1  # At least one axis (may have colorbar)
    
    plt.close(fig)


def test_plot_contour_with_fixed_values():
    """Test contour plot with additional variables."""
    session = OptimizationSession()
    
    # Add 3 variables
    session.add_variable('temp', 'real', bounds=(300, 500))
    session.add_variable('pressure', 'real', bounds=(1, 10))
    session.add_variable('flow_rate', 'real', bounds=(10, 100))
    
    # Add data
    np.random.seed(42)
    for i in range(20):
        temp = np.random.uniform(300, 500)
        pressure = np.random.uniform(1, 10)
        flow = np.random.uniform(10, 100)
        y = 0.01*temp + 0.1*pressure + 0.001*flow + np.random.normal(0, 1)
        session.add_experiment({'temp': temp, 'pressure': pressure, 'flow_rate': flow}, output=y)
    
    session.train_model(backend='sklearn')
    
    fig = session.plot_contour(
        'temp', 'pressure',
        fixed_values={'flow_rate': 50},
        grid_resolution=30,
        show_experiments=True
    )
    
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_contour_invalid_variable():
    """Test that plot_contour raises error for invalid variable."""
    session = create_trained_session()
    
    with pytest.raises(ValueError, match="not in search space"):
        session.plot_contour('x1', 'nonexistent')


def test_plot_contour_requires_real_variables():
    """Test that plot_contour requires real variables."""
    session = OptimizationSession()
    session.add_variable('cat1', 'categorical', categories=['A', 'B', 'C'])
    session.add_variable('x1', 'real', bounds=(0, 1))
    
    # Add data and train
    for i in range(10):
        session.add_experiment({'cat1': 'A', 'x1': 0.5}, output=1.0)
    session.train_model(backend='sklearn')
    
    with pytest.raises(ValueError, match="must be 'real' type"):
        session.plot_contour('cat1', 'x1')


def test_plot_metrics_returns_figure():
    """Test that plot_metrics returns matplotlib Figure."""
    session = create_trained_session()
    
    fig = session.plot_metrics('rmse')
    
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_metrics_all_types():
    """Test all metric types."""
    session = create_trained_session()
    
    for metric in ['rmse', 'mae', 'r2', 'mape']:
        fig = session.plot_metrics(metric)
        assert isinstance(fig, Figure)
        plt.close(fig)


def test_plot_metrics_invalid_metric():
    """Test that invalid metric raises error."""
    session = create_trained_session()
    
    with pytest.raises(ValueError, match="Unknown metric"):
        session.plot_metrics('invalid')


def test_plot_without_model_raises_error():
    """Test that plotting without trained model raises error."""
    session = OptimizationSession()
    session.add_variable('x1', 'real', bounds=(0, 1))
    
    with pytest.raises(ValueError, match="Model not trained"):
        session.plot_parity()
    
    with pytest.raises(ValueError, match="Model not trained"):
        session.plot_contour('x1', 'x1')
    
    with pytest.raises(ValueError, match="Model not trained"):
        session.plot_metrics()


def test_visualization_methods_exist():
    """Test that all visualization methods are available."""
    session = OptimizationSession()
    
    assert hasattr(session, 'plot_parity')
    assert hasattr(session, 'plot_contour')
    assert hasattr(session, 'plot_metrics')
    assert callable(session.plot_parity)
    assert callable(session.plot_contour)
    assert callable(session.plot_metrics)


def test_plot_contour_with_categorical():
    """Test contour plot with categorical variable in search space."""
    session = OptimizationSession()
    
    session.add_variable('x1', 'real', bounds=(0, 1))
    session.add_variable('x2', 'real', bounds=(0, 1))
    session.add_variable('catalyst', 'categorical', categories=['A', 'B', 'C'])
    
    # Add data
    np.random.seed(42)
    for i in range(20):
        x1 = np.random.uniform(0, 1)
        x2 = np.random.uniform(0, 1)
        cat = np.random.choice(['A', 'B', 'C'])
        y = x1 + x2 + np.random.normal(0, 0.1)
        session.add_experiment({'x1': x1, 'x2': x2, 'catalyst': cat}, output=y)
    
    session.train_model(backend='sklearn')
    
    # Should use default (first category) for catalyst
    fig = session.plot_contour('x1', 'x2')
    assert isinstance(fig, Figure)
    plt.close(fig)
    
    
    # Should use specified value for catalyst
    fig = session.plot_contour('x1', 'x2', fixed_values={'catalyst': 'B'})
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_slice_basic():
    """Test basic 1D slice plot."""
    session = create_trained_session()
    
    fig = session.plot_slice('x1')
    assert isinstance(fig, Figure)
    
    # Should have a line plot
    ax = fig.axes[0]
    assert len(ax.lines) > 0
    
    plt.close(fig)


def test_plot_slice_with_fixed_values():
    """Test 1D slice plot with fixed values."""
    session = create_trained_session()
    
    fig = session.plot_slice('x1', fixed_values={'x2': 0.5})
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_slice_no_uncertainty():
    """Test 1D slice plot without uncertainty bands."""
    session = create_trained_session()
    
    fig = session.plot_slice('x1', show_uncertainty=False)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_slice_custom_sigma():
    """Test 1D slice plot with custom sigma values."""
    session = create_trained_session()
    
    # Test with custom sigma values
    fig = session.plot_slice('x1', show_uncertainty=[1.0, 2.0, 3.0])
    assert isinstance(fig, Figure)
    
    # Should have 3 uncertainty bands in legend
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert '±1.0σ' in labels
    assert '±2.0σ' in labels
    assert '±3.0σ' in labels
    
    plt.close(fig)


def test_plot_slice_with_categorical():
    """Test 1D slice plot with categorical variable fixed."""
    session = OptimizationSession()
    
    session.add_variable('x1', 'real', bounds=(0, 1))
    session.add_variable('x2', 'real', bounds=(0, 1))
    session.add_variable('catalyst', 'categorical', categories=['A', 'B', 'C'])
    
    # Add data
    np.random.seed(42)
    for i in range(20):
        x1 = np.random.uniform(0, 1)
        x2 = np.random.uniform(0, 1)
        cat = np.random.choice(['A', 'B', 'C'])
        y = x1 + x2 + np.random.normal(0, 0.1)
        session.add_experiment({'x1': x1, 'x2': x2, 'catalyst': cat}, output=y)
    
    session.train_model(backend='sklearn')
    
    fig = session.plot_slice('x1', fixed_values={'x2': 0.5, 'catalyst': 'B'})
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_slice_invalid_variable():
    """Test that plot_slice raises error for invalid variable."""
    session = create_trained_session()
    
    with pytest.raises(ValueError, match="not in search space"):
        session.plot_slice('nonexistent')


def test_plot_slice_categorical_variable():
    """Test that plot_slice raises error for categorical variable."""
    session = OptimizationSession()
    
    session.add_variable('x1', 'real', bounds=(0, 1))
    session.add_variable('catalyst', 'categorical', categories=['A', 'B'])
    
    # Add minimal data
    np.random.seed(42)
    for i in range(10):
        session.add_experiment(
            {'x1': np.random.uniform(0, 1), 'catalyst': np.random.choice(['A', 'B'])},
            output=np.random.uniform(0, 1)
        )
    
    session.train_model(backend='sklearn')
    
    with pytest.raises(ValueError, match="must be 'real' or 'integer' type"):
        session.plot_slice('catalyst')


def test_plot_parity_with_external_axes():
    """Test that plot_parity draws on caller-provided axes and returns the same figure."""
    session = create_trained_session()
    fig, ax = plt.subplots(figsize=(5, 4))
    result = session.plot_parity(ax=ax)
    assert result is fig  # Same figure, not a new one
    assert len(fig.axes) == 1
    plt.close(fig)


def test_plot_contour_with_external_axes():
    """Test that plot_contour draws on caller-provided axes and returns the same figure."""
    session = create_trained_session()
    fig, ax = plt.subplots(figsize=(5, 4))
    result = session.plot_contour('x1', 'x2', ax=ax)
    assert result is fig
    assert len(fig.axes) >= 2  # main axes + colorbar
    plt.close(fig)


def test_plot_metrics_with_external_axes():
    """Test that plot_metrics draws on caller-provided axes and returns the same figure."""
    session = create_trained_session()
    fig, ax = plt.subplots(figsize=(5, 4))
    result = session.plot_metrics(ax=ax)
    assert result is fig
    plt.close(fig)


def test_plot_qq_with_external_axes():
    """Test that plot_qq draws on caller-provided axes and returns the same figure."""
    session = create_trained_session()
    fig, ax = plt.subplots(figsize=(5, 4))
    result = session.plot_qq(ax=ax)
    assert result is fig
    plt.close(fig)


def test_plot_calibration_with_external_axes():
    """Test that plot_calibration draws on caller-provided axes and returns the same figure."""
    session = create_trained_session()
    fig, ax = plt.subplots(figsize=(5, 4))
    result = session.plot_calibration(ax=ax)
    assert result is fig
    plt.close(fig)


def test_plot_surface_returns_figure():
    """Test that plot_surface returns matplotlib Figure."""
    session = create_trained_session()

    fig = session.plot_surface('x1', 'x2')

    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_plot_surface_with_fixed_values():
    """Test surface plot with additional variables and fixed values."""
    session = OptimizationSession()

    session.add_variable('temp', 'real', bounds=(300, 500))
    session.add_variable('pressure', 'real', bounds=(1, 10))
    session.add_variable('flow_rate', 'real', bounds=(10, 100))

    np.random.seed(42)
    for i in range(20):
        temp = np.random.uniform(300, 500)
        pressure = np.random.uniform(1, 10)
        flow = np.random.uniform(10, 100)
        y = 0.01 * temp + 0.1 * pressure + 0.001 * flow + np.random.normal(0, 1)
        session.add_experiment({'temp': temp, 'pressure': pressure, 'flow_rate': flow}, output=y)

    session.train_model(backend='sklearn')

    fig = session.plot_surface(
        'temp', 'pressure',
        fixed_values={'flow_rate': 50},
        grid_resolution=20,
        show_experiments=True
    )

    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_surface_invalid_variable():
    """Test that plot_surface raises error for invalid variable."""
    session = create_trained_session()

    with pytest.raises(ValueError, match="not in search space"):
        session.plot_surface('x1', 'nonexistent')


def test_plot_surface_requires_real_variables():
    """Test that plot_surface requires real variables."""
    session = OptimizationSession()
    session.add_variable('cat1', 'categorical', categories=['A', 'B', 'C'])
    session.add_variable('x1', 'real', bounds=(0, 1))

    for i in range(10):
        session.add_experiment({'cat1': 'A', 'x1': 0.5}, output=1.0)
    session.train_model(backend='sklearn')

    with pytest.raises(ValueError, match="must be 'real' type"):
        session.plot_surface('cat1', 'x1')


def test_plot_uncertainty_surface_returns_figure():
    """Test that plot_uncertainty_surface returns matplotlib Figure."""
    session = create_trained_session()

    fig = session.plot_uncertainty_surface('x1', 'x2')

    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_plot_uncertainty_surface_invalid_variable():
    """Test that plot_uncertainty_surface raises error for invalid variable."""
    session = create_trained_session()

    with pytest.raises(ValueError, match="not in search space"):
        session.plot_uncertainty_surface('x1', 'nonexistent')


if __name__ == "__main__":
    # Run a simple test
    print("Testing session visualization methods...")
    session = create_trained_session()
    
    print("✓ Creating parity plot...")
    fig = session.plot_parity()
    plt.close(fig)
    
    print("✓ Creating contour plot...")
    fig = session.plot_contour('x1', 'x2')
    plt.close(fig)
    
    print("✓ Creating slice plot...")
    fig = session.plot_slice('x1')
    plt.close(fig)
    
    print("✓ Creating metrics plot...")
    fig = session.plot_metrics('rmse')
    plt.close(fig)
    
    print("\n✓ All tests passed!")

