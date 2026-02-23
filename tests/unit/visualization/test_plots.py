"""
Unit tests for alchemist_core.visualization module.

Tests pure plotting functions with numpy arrays.
"""

import pytest
import numpy as np
from alchemist_core.visualization.plots import (
    create_parity_plot,
    create_contour_plot,
    create_surface_plot,
    create_uncertainty_surface_plot,
    create_slice_plot,
    create_metrics_plot,
    create_qq_plot,
    create_calibration_plot,
)
from alchemist_core.visualization.helpers import (
    check_matplotlib,
    compute_z_scores,
    compute_calibration_metrics,
)


class TestHelpers:
    """Test helper functions."""
    
    def test_check_matplotlib(self):
        """Test matplotlib availability check."""
        # Should not raise if matplotlib is installed
        check_matplotlib()
    
    def test_compute_z_scores(self):
        """Test z-score computation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        y_std = np.array([0.1, 0.2, 0.15, 0.25, 0.1])
        
        z = compute_z_scores(y_true, y_pred, y_std)
        
        # Check shape
        assert z.shape == y_true.shape
        
        # Check calculation for first element
        expected_z0 = (1.0 - 1.1) / 0.1
        assert np.isclose(z[0], expected_z0)
        
        # Check no division by zero
        y_std_zero = np.array([0.0, 0.1, 0.1, 0.1, 0.1])
        z_safe = compute_z_scores(y_true, y_pred, y_std_zero)
        assert np.all(np.isfinite(z_safe))
    
    def test_compute_calibration_metrics(self):
        """Test calibration metrics computation."""
        # Create synthetic data with perfect calibration
        np.random.seed(42)
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.randn(100) * 0.1  # Small noise
        y_std = np.ones(100) * 0.1
        
        nominal, empirical = compute_calibration_metrics(y_true, y_pred, y_std)
        
        # Check shapes
        assert len(nominal) == len(empirical)
        assert len(nominal) > 0
        
        # Check range [0, 1]
        assert np.all(nominal >= 0) and np.all(nominal <= 1)
        assert np.all(empirical >= 0) and np.all(empirical <= 1)
        
        # For perfect calibration, empirical should be close to nominal
        # (may not be exact due to finite sample size)
        assert np.all(np.abs(empirical - nominal) < 0.3)  # Loose tolerance


class TestParityPlot:
    """Test parity plot creation."""
    
    def test_basic_parity_plot(self):
        """Test basic parity plot without error bars."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        
        fig, ax = create_parity_plot(y_true, y_pred, show_error_bars=False)
        
        assert fig is not None
        assert ax is not None
        assert ax.get_xlabel() == "Actual Values"
        assert ax.get_ylabel() == "Predicted Values"
    
    def test_parity_plot_with_error_bars(self):
        """Test parity plot with uncertainty."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        y_std = np.array([0.1, 0.2, 0.15, 0.25, 0.1])
        
        fig, ax = create_parity_plot(
            y_true, y_pred, y_std=y_std,
            show_error_bars=True,
            sigma_multiplier=1.96
        )
        
        assert fig is not None
        assert ax is not None
        
        # Check that title includes CI information
        title = ax.get_title()
        assert 'RMSE' in title
        assert '95% CI' in title
    
    def test_parity_plot_custom_title(self):
        """Test parity plot with custom title."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.1])
        
        custom_title = "My Custom Parity Plot"
        fig, ax = create_parity_plot(y_true, y_pred, title=custom_title)
        
        assert ax.get_title() == custom_title
    
    def test_parity_plot_with_existing_axes(self):
        """Test plotting on existing axes."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.1])
        
        returned_fig, returned_ax = create_parity_plot(y_true, y_pred, ax=ax)
        
        assert returned_fig is fig
        assert returned_ax is ax
        plt.close(fig)


class TestContourPlot:
    """Test contour plot creation."""
    
    def test_basic_contour_plot(self):
        """Test basic contour plot."""
        # Create meshgrid
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 20)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X) * np.cos(Y)  # Arbitrary function
        
        fig, ax, cbar = create_contour_plot(X, Y, Z, 'x_var', 'y_var')
        
        assert fig is not None
        assert ax is not None
        assert cbar is not None
        assert ax.get_xlabel() == "x_var"
        assert ax.get_ylabel() == "y_var"
    
    def test_contour_plot_with_experiments(self):
        """Test contour plot with experimental points overlay."""
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 20)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X) * np.cos(Y)
        
        exp_x = np.array([2.0, 5.0, 8.0])
        exp_y = np.array([3.0, 6.0, 4.0])
        
        fig, ax, cbar = create_contour_plot(X, Y, Z, 'x_var', 'y_var',
                                     exp_x=exp_x, exp_y=exp_y)
        
        assert fig is not None
        assert cbar is not None
        # Check legend exists
        legend = ax.get_legend()
        assert legend is not None
    
    def test_contour_plot_with_suggestions(self):
        """Test contour plot with suggestion points overlay."""
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 20)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X) * np.cos(Y)
        
        suggest_x = np.array([1.0, 9.0])
        suggest_y = np.array([1.0, 9.0])
        
        fig, ax, cbar = create_contour_plot(X, Y, Z, 'x_var', 'y_var',
                                     suggest_x=suggest_x, suggest_y=suggest_y)
        
        assert fig is not None
        assert cbar is not None
        legend = ax.get_legend()
        assert legend is not None


class TestSlicePlot:
    """Test 1D slice plot creation."""
    
    def test_basic_slice_plot(self):
        """Test basic slice plot without uncertainty."""
        x = np.linspace(0, 10, 50)
        y = np.sin(x)
        
        fig, ax = create_slice_plot(x, y, 'x_variable')
        
        assert fig is not None
        assert ax is not None
        assert ax.get_xlabel() == "x_variable"
        assert ax.get_ylabel() == "Predicted Output"
    
    def test_slice_plot_with_uncertainty(self):
        """Test slice plot with uncertainty bands."""
        x = np.linspace(0, 10, 50)
        y = np.sin(x)
        std = np.ones_like(x) * 0.1
        
        fig, ax = create_slice_plot(x, y, 'x_variable',
                                    std=std, sigma_bands=[1.0, 2.0])
        
        assert fig is not None
        legend = ax.get_legend()
        assert legend is not None
        
        # Check that legend has uncertainty bands
        labels = [t.get_text() for t in legend.get_texts()]
        assert any('σ' in label for label in labels)
    
    def test_slice_plot_with_experiments(self):
        """Test slice plot with experimental points."""
        x = np.linspace(0, 10, 50)
        y = np.sin(x)
        
        exp_x = np.array([2.0, 5.0, 8.0])
        exp_y = np.array([0.9, -0.96, 0.99])
        
        fig, ax = create_slice_plot(x, y, 'x_variable',
                                    exp_x=exp_x, exp_y=exp_y)
        
        assert fig is not None
        legend = ax.get_legend()
        assert legend is not None


class TestMetricsPlot:
    """Test metrics plot creation."""
    
    def test_rmse_plot(self):
        """Test RMSE learning curve."""
        sizes = np.array([5, 6, 7, 8, 9, 10])
        rmse = np.array([0.15, 0.12, 0.10, 0.08, 0.07, 0.06])
        
        fig, ax = create_metrics_plot(sizes, rmse, 'rmse')
        
        assert fig is not None
        assert ax is not None
        assert 'RMSE' in ax.get_ylabel()
        assert 'Number of Observations' in ax.get_xlabel()
    
    def test_r2_plot(self):
        """Test R² learning curve."""
        sizes = np.array([5, 6, 7, 8, 9, 10])
        r2 = np.array([0.5, 0.6, 0.7, 0.8, 0.85, 0.9])
        
        fig, ax = create_metrics_plot(sizes, r2, 'r2')
        
        assert fig is not None
        assert 'R²' in ax.get_ylabel()
    
    def test_mae_plot(self):
        """Test MAE learning curve."""
        sizes = np.array([5, 6, 7, 8])
        mae = np.array([0.12, 0.10, 0.08, 0.07])
        
        fig, ax = create_metrics_plot(sizes, mae, 'mae')
        
        assert fig is not None
        assert 'MAE' in ax.get_ylabel()


class TestQQPlot:
    """Test Q-Q plot creation."""
    
    def test_basic_qq_plot(self):
        """Test Q-Q plot with normally distributed z-scores."""
        np.random.seed(42)
        z_scores = np.random.randn(50)
        
        fig, ax = create_qq_plot(z_scores)
        
        assert fig is not None
        assert ax is not None
        assert 'Theoretical Quantiles' in ax.get_xlabel()
        assert 'Sample Quantiles' in ax.get_ylabel()
    
    def test_qq_plot_with_confidence_bands(self):
        """Test Q-Q plot with confidence bands for small samples."""
        z_scores = np.random.randn(20)
        
        fig, ax = create_qq_plot(z_scores, show_confidence_bands=True)
        
        assert fig is not None
        legend = ax.get_legend()
        assert legend is not None
    
    def test_qq_plot_without_confidence_bands(self):
        """Test Q-Q plot without confidence bands."""
        z_scores = np.random.randn(200)
        
        fig, ax = create_qq_plot(z_scores, show_confidence_bands=False)
        
        assert fig is not None


class TestCalibrationPlot:
    """Test calibration plot creation."""
    
    def test_basic_calibration_plot(self):
        """Test calibration curve."""
        nominal = np.array([0.5, 0.68, 0.95, 0.99])
        empirical = np.array([0.52, 0.70, 0.94, 0.98])
        
        fig, ax = create_calibration_plot(nominal, empirical)
        
        assert fig is not None
        assert ax is not None
        assert 'Nominal Coverage' in ax.get_xlabel()
        assert 'Empirical Coverage' in ax.get_ylabel()
        
        # Check axes limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        assert xlim[0] <= 0 and xlim[1] >= 1
        assert ylim[0] <= 0 and ylim[1] >= 1
    
    def test_perfect_calibration(self):
        """Test calibration plot with perfect calibration."""
        probs = np.linspace(0.1, 0.99, 10)
        
        fig, ax = create_calibration_plot(probs, probs)
        
        assert fig is not None
        legend = ax.get_legend()
        assert legend is not None


class TestSurfacePlot:
    """Test 3D surface plot creation."""

    def test_basic_surface_plot(self):
        """Test basic surface plot returns figure, axes, and colorbar."""
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 20)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X) * np.cos(Y)

        fig, ax, cbar = create_surface_plot(X, Y, Z, 'x_var', 'y_var')

        assert fig is not None
        assert ax is not None
        assert cbar is not None
        assert ax.get_xlabel() == 'x_var'
        assert ax.get_ylabel() == 'y_var'

    def test_surface_plot_with_experiments(self):
        """Test surface plot with experimental data overlay."""
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 20)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X) * np.cos(Y)

        exp_x = np.array([2.0, 5.0, 8.0])
        exp_y = np.array([3.0, 6.0, 4.0])
        exp_output = np.array([0.5, -0.3, 0.1])

        fig, ax, cbar = create_surface_plot(
            X, Y, Z, 'x_var', 'y_var',
            exp_x=exp_x, exp_y=exp_y, exp_output=exp_output
        )

        assert fig is not None
        assert cbar is not None
        legend = ax.get_legend()
        assert legend is not None

    def test_surface_plot_with_suggestions(self):
        """Test surface plot with suggestion points overlay."""
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 20)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X) * np.cos(Y)

        suggest_x = np.array([1.0, 9.0])
        suggest_y = np.array([1.0, 9.0])

        fig, ax, cbar = create_surface_plot(
            X, Y, Z, 'x_var', 'y_var',
            suggest_x=suggest_x, suggest_y=suggest_y
        )

        assert fig is not None
        assert cbar is not None
        legend = ax.get_legend()
        assert legend is not None

    def test_surface_plot_with_custom_axes(self):
        """Test surface plot with externally provided 3D axes."""
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = np.linspace(0, 10, 15)
        y = np.linspace(0, 10, 15)
        X, Y = np.meshgrid(x, y)
        Z = X + Y

        ret_fig, ret_ax, cbar = create_surface_plot(
            X, Y, Z, 'a', 'b', ax=ax
        )

        assert ret_fig is fig
        assert ret_ax is ax
        assert cbar is not None
        plt.close(fig)

    def test_surface_plot_custom_params(self):
        """Test surface plot with custom cmap, alpha, and title."""
        x = np.linspace(0, 5, 10)
        y = np.linspace(0, 5, 10)
        X, Y = np.meshgrid(x, y)
        Z = X * Y

        fig, ax, cbar = create_surface_plot(
            X, Y, Z, 'x', 'y',
            cmap='coolwarm', alpha=0.5,
            title='Custom Title', output_label='My Output'
        )

        assert ax.get_title() == 'Custom Title'
        assert fig is not None


class TestUncertaintySurfacePlot:
    """Test 3D uncertainty surface plot creation."""

    def test_basic_uncertainty_surface(self):
        """Test basic uncertainty surface plot."""
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 20)
        X, Y = np.meshgrid(x, y)
        unc = np.abs(np.sin(X) * np.cos(Y)) + 0.1

        fig, ax, cbar = create_uncertainty_surface_plot(X, Y, unc, 'x_var', 'y_var')

        assert fig is not None
        assert ax is not None
        assert cbar is not None
        assert ax.get_xlabel() == 'x_var'
        assert ax.get_ylabel() == 'y_var'

    def test_uncertainty_surface_with_experiments(self):
        """Test uncertainty surface with experimental overlay."""
        x = np.linspace(0, 10, 15)
        y = np.linspace(0, 10, 15)
        X, Y = np.meshgrid(x, y)
        unc = np.ones_like(X) * 0.5

        exp_x = np.array([3.0, 7.0])
        exp_y = np.array([4.0, 8.0])

        fig, ax, cbar = create_uncertainty_surface_plot(
            X, Y, unc, 'x', 'y', exp_x=exp_x, exp_y=exp_y
        )

        assert fig is not None
        legend = ax.get_legend()
        assert legend is not None

    def test_uncertainty_surface_with_suggestions(self):
        """Test uncertainty surface with suggestion overlay."""
        x = np.linspace(0, 10, 15)
        y = np.linspace(0, 10, 15)
        X, Y = np.meshgrid(x, y)
        unc = np.abs(np.sin(X)) + 0.01

        suggest_x = np.array([2.0])
        suggest_y = np.array([5.0])

        fig, ax, cbar = create_uncertainty_surface_plot(
            X, Y, unc, 'x', 'y',
            suggest_x=suggest_x, suggest_y=suggest_y
        )

        assert fig is not None
        legend = ax.get_legend()
        assert legend is not None

    def test_uncertainty_surface_custom_axes(self):
        """Test uncertainty surface with externally provided 3D axes."""
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = np.linspace(0, 5, 10)
        y = np.linspace(0, 5, 10)
        X, Y = np.meshgrid(x, y)
        unc = np.ones_like(X) * 0.2

        ret_fig, ret_ax, cbar = create_uncertainty_surface_plot(
            X, Y, unc, 'a', 'b', ax=ax
        )

        assert ret_fig is fig
        assert ret_ax is ax
        plt.close(fig)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
