"""
Pure plotting functions for ALchemist visualizations.

All functions are framework-agnostic and return matplotlib Figure/Axes objects.
They accept optional axes for embedding in existing figures.
"""

from typing import Optional, Dict, Tuple, Union, List, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def create_parity_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: Optional[np.ndarray] = None,
    sigma_multiplier: float = 1.96,
    show_error_bars: bool = True,
    show_metrics: bool = True,
    figsize: Tuple[float, float] = (8, 6),
    dpi: int = 100,
    title: Optional[str] = None,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """
    Create parity plot of actual vs predicted values.
    
    Pure plotting function with no session or model dependencies.
    
    Args:
        y_true: Actual experimental values
        y_pred: Model predicted values
        y_std: Prediction uncertainties (optional)
        sigma_multiplier: Error bar size multiplier (1.96 = 95% CI)
        show_error_bars: Display uncertainty error bars
        show_metrics: Include RMSE/MAE/R² in title
        figsize: Figure size (width, height) in inches
        dpi: Resolution in dots per inch
        title: Custom title (auto-generated if None and show_metrics=True)
        ax: Existing axes to plot on (creates new if None)
    
    Returns:
        Tuple of (Figure, Axes) objects
    
    Example:
        >>> fig, ax = create_parity_plot(y_true, y_pred, y_std)
        >>> fig.savefig('parity.png', dpi=300, bbox_inches='tight')
    """
    # If ax provided, use its figure; otherwise create new
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        should_tight_layout = True
    else:
        fig = ax.figure
        should_tight_layout = False
    
    # Calculate metrics if requested
    if show_metrics:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        try:
            r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
        except:
            r2 = np.nan
    
    # Plot data with optional error bars
    if show_error_bars and y_std is not None:
        yerr = sigma_multiplier * y_std
        ax.errorbar(y_true, y_pred, yerr=yerr, 
                   fmt='o', alpha=0.7, capsize=3, capthick=1,
                   elinewidth=1, markersize=5)
    else:
        ax.scatter(y_true, y_pred, alpha=0.7)
    
    # Add parity line (y=x)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Parity line')
    
    # Set labels
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    
    # Create title with metrics
    if title is None and show_metrics:
        ci_labels = {
            1.0: "68% CI",
            1.96: "95% CI",
            2.0: "95.4% CI",
            2.58: "99% CI",
            3.0: "99.7% CI"
        }
        ci_label = ci_labels.get(sigma_multiplier, f"{sigma_multiplier}σ")
        
        if show_error_bars and y_std is not None:
            title = (f"Cross-Validation Parity Plot\n"
                    f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}\n"
                    f"Error bars: ±{sigma_multiplier}σ ({ci_label})")
        else:
            title = (f"Cross-Validation Parity Plot\n"
                    f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    if title:
        ax.set_title(title)
    
    ax.legend()
    
    if should_tight_layout:
        fig.tight_layout()
    
    return fig, ax


def create_contour_plot(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    predictions_grid: np.ndarray,
    x_var: str,
    y_var: str,
    exp_x: Optional[np.ndarray] = None,
    exp_y: Optional[np.ndarray] = None,
    suggest_x: Optional[np.ndarray] = None,
    suggest_y: Optional[np.ndarray] = None,
    cmap: str = 'viridis',
    use_log_scale: bool = False,
    figsize: Tuple[float, float] = (8, 6),
    dpi: int = 100,
    title: str = "Contour Plot of Model Predictions",
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes, Any]:
    """
    Create 2D contour plot of model predictions.
    
    Args:
        x_grid: X-axis meshgrid values (2D array)
        y_grid: Y-axis meshgrid values (2D array)
        predictions_grid: Model predictions on grid (2D array)
        x_var: X variable name for axis label
        y_var: Y variable name for axis label
        exp_x: Experimental X values to overlay (optional)
        exp_y: Experimental Y values to overlay (optional)
        suggest_x: Suggested X values to overlay (optional)
        suggest_y: Suggested Y values to overlay (optional)
        cmap: Matplotlib colormap name
        use_log_scale: Use logarithmic color scale for values spanning orders of magnitude
        figsize: Figure size (width, height) in inches
        dpi: Resolution
        title: Plot title
        ax: Existing axes (creates new if None)
    
    Returns:
        Tuple of (Figure, Axes, Colorbar) - includes colorbar reference for management
    
    Example:
        >>> X, Y = np.meshgrid(x_range, y_range)
        >>> Z = model_predictions.reshape(X.shape)
        >>> fig, ax, cbar = create_contour_plot(X, Y, Z, 'temperature', 'pressure')
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        should_tight_layout = True
    else:
        fig = ax.figure
        should_tight_layout = False

    # Helper: create colorbar without permanently stealing space from the parent
    # axes.  When the caller owns the axes (ax was passed in), we use
    # make_axes_locatable so the colorbar axes is a child divider — removing it
    # later restores the parent to its original size.
    def _add_colorbar(mappable):
        if should_tight_layout:
            # New figure — normal colorbar is fine
            return fig.colorbar(mappable, ax=ax)
        else:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.08)
            return fig.colorbar(mappable, cax=cax)

    # Contour plot with optional log scaling
    if use_log_scale:
        from matplotlib.colors import LogNorm, SymLogNorm

        min_val = float(np.nanmin(predictions_grid))
        max_val = float(np.nanmax(predictions_grid))

        # For predominantly negative values (like LogEI), use negative to make positive
        if max_val <= 0:
            # All negative: flip sign to make positive for LogNorm, but keep track for colorbar
            plot_grid = -predictions_grid
            vmin_pos = -max_val  # Most negative original value (worst)
            vmax_pos = -min_val  # Closest to zero original value (best)

            # Create log-spaced levels in the positive space
            levels = np.logspace(np.log10(vmin_pos), np.log10(vmax_pos), 50)
            contour = ax.contourf(x_grid, y_grid, plot_grid, levels=levels, cmap=cmap, norm=LogNorm(vmin=vmin_pos, vmax=vmax_pos))

            # Create colorbar with negative value labels
            cbar = _add_colorbar(contour)

            # Create a custom formatter that shows negative values
            def fmt(x, pos):
                return f'{-x:.0e}'

            import matplotlib.ticker as ticker
            cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt))
        elif min_val >= 0:
            # All positive: use directly
            plot_grid = predictions_grid
            levels = np.logspace(np.log10(min_val + 1e-10), np.log10(max_val), 50)
            contour = ax.contourf(x_grid, y_grid, plot_grid, levels=levels, cmap=cmap, norm=LogNorm())
            cbar = _add_colorbar(contour)
        else:
            # Mixed signs: use SymLogNorm
            linthresh = max(abs(min_val), abs(max_val)) * 0.01
            contour = ax.contourf(x_grid, y_grid, predictions_grid, levels=50, cmap=cmap,
                                norm=SymLogNorm(linthresh=linthresh, vmin=min_val, vmax=max_val))
            cbar = _add_colorbar(contour)
    else:
        # Mask NaN values so contourf leaves those regions blank
        masked_grid = np.ma.masked_invalid(predictions_grid)

        if masked_grid.count() == 0:
            raise ValueError(
                "All predictions are NaN — the model returned no valid values for this grid. "
                "Check that the model trained successfully and that all input variables are correct."
            )

        min_val = float(masked_grid.min())
        max_val = float(masked_grid.max())

        # Generate levels that better handle extreme outliers
        # Use percentile-based approach to create more levels in the dense region
        valid_vals = masked_grid.compressed()
        percentiles = np.percentile(valid_vals, [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100])

        # Handle constant predictions (min == max) to avoid contourf failure
        if min_val == max_val:
            eps = max(abs(min_val) * 1e-6, 1e-10)
            min_val -= eps
            max_val += eps
            levels = np.linspace(min_val, max_val, 50)
        # If there's a large gap between percentiles, use adaptive levels
        elif (percentiles[-1] - percentiles[-2]) > 2 * (percentiles[-2] - percentiles[-3]):
            # Extreme outliers detected - create custom levels
            # More levels in the main data range, fewer in the outlier range
            main_levels = np.linspace(percentiles[1], percentiles[-2], 40)
            outlier_levels = np.linspace(percentiles[-2], max_val, 10)
            levels = np.concatenate([main_levels, outlier_levels])
        else:
            # Normal distribution - use uniform levels
            levels = np.linspace(min_val, max_val, 50)

        contour = ax.contourf(x_grid, y_grid, masked_grid, levels=levels, cmap=cmap,
                            vmin=min_val, vmax=max_val)
        cbar = _add_colorbar(contour)
    
    # Only set label if we haven't already created colorbar
    if not use_log_scale or (use_log_scale and not (max_val <= 0 and min_val < 0)):
        cbar.set_label('Predicted Output')
    else:
        cbar.set_label('Predicted Output')
    
    # Overlay experimental points
    if exp_x is not None and exp_y is not None and len(exp_x) > 0:
        ax.scatter(exp_x, exp_y, c='white', edgecolors='black', 
                  s=80, marker='o', label='Experiments', zorder=5)
    
    # Overlay suggestion points
    if suggest_x is not None and suggest_y is not None and len(suggest_x) > 0:
        ax.scatter(suggest_x, suggest_y, c='black',
                  s=120, marker='*', label='Suggestions', zorder=6)
    
    # Labels and title
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.set_title(title)
    
    # Legend if we have overlays
    if (exp_x is not None and len(exp_x) > 0) or (suggest_x is not None and len(suggest_x) > 0):
        ax.legend()
    
    if should_tight_layout:
        fig.tight_layout()
    
    return fig, ax, cbar


def create_slice_plot(
    x_values: np.ndarray,
    predictions: np.ndarray,
    x_var: str,
    std: Optional[np.ndarray] = None,
    sigma_bands: Optional[List[float]] = None,
    exp_x: Optional[np.ndarray] = None,
    exp_y: Optional[np.ndarray] = None,
    figsize: Tuple[float, float] = (8, 6),
    dpi: int = 100,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    prediction_label: str = 'Prediction',
    line_color: Optional[str] = None,
    line_width: Optional[float] = None
) -> Tuple[Figure, Axes]:
    """
    Create 1D slice plot with uncertainty bands.
    
    Shows model predictions along one variable while other variables are held fixed.
    Optionally displays uncertainty bands at multiple sigma levels.
    
    Args:
        x_values: X-axis values (1D array)
        predictions: Mean predictions (1D array)
        x_var: Variable name for X-axis label
        std: Standard deviations (1D array, optional)
        sigma_bands: List of sigma values for uncertainty bands (e.g., [1.0, 2.0])
        exp_x: Experimental X values to overlay (optional)
        exp_y: Experimental Y values to overlay (optional)
        figsize: Figure size (width, height) in inches
        dpi: Resolution
        title: Custom title (optional)
        ax: Existing axes (creates new if None)
        prediction_label: Label for the prediction line in legend (default: 'Prediction')
        line_color: Color for the prediction line (default: dark blue)
        line_width: Width of the prediction line (default: 2.6)
    
    Returns:
        Tuple of (Figure, Axes)
    
    Example:
        >>> x = np.linspace(20, 100, 100)
        >>> y_pred, y_std = model.predict(X_grid, return_std=True)
        >>> fig, ax = create_slice_plot(x, y_pred, 'temperature', 
        ...                             std=y_std, sigma_bands=[1.0, 2.0])
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        should_tight_layout = True
    else:
        fig = ax.figure
        should_tight_layout = False
    
    # Styling constants
    mean_color = "#0B3C5D"          # dark blue-teal, prints well
    exp_face   = "#E07A00"          # orange (colorblind-friendly vs blue)
    grid_alpha = 0.25
    
    # Plot uncertainty bands BEFORE mean line (for proper z-stacking)
    if std is not None and sigma_bands:
        # Sort largest to smallest for proper layering
        sigma_bands_sorted = sorted(sigma_bands, reverse=True)
        n = len(sigma_bands_sorted)
        
        # Sequential colormap: same hue, different lightness
        cmap = plt.get_cmap("Blues")
        
        for i, sigma in enumerate(sigma_bands_sorted):
            # i=0 is largest sigma (most transparent), i=n-1 is smallest (most opaque)
            t = i / max(1, n - 1)  # 0..1 ratio
            
            # Lighter tones for larger sigma, darker for smaller
            face = cmap(0.3 + 0.3 * t)  # 0.30 to 0.60 in Blues colormap
            edge = plt.matplotlib.colors.to_rgba(mean_color, 0.55)
            
            # Sigmoid-based alpha: smaller sigma → higher alpha (more opaque)
            alpha = 1.0 - 1.0 / (1.0 + np.exp(-sigma + 2.0))
            
            ax.fill_between(
                x_values,
                predictions - sigma * std,
                predictions + sigma * std,
                facecolor=plt.matplotlib.colors.to_rgba(face, alpha),
                edgecolor=edge,
                linewidth=0.9,
                label=f'±{sigma:.1f}σ',
                zorder=1
            )
    
    # Mean prediction on top
    ax.plot(
        x_values,
        predictions,
        color=line_color if line_color is not None else mean_color,
        linewidth=line_width if line_width is not None else 2.6,
        label=prediction_label,
        zorder=3
    )
    
    # Plot experimental points
    if exp_x is not None and exp_y is not None and len(exp_x) > 0:
        ax.scatter(
            exp_x,
            exp_y,
            s=70,
            facecolor=exp_face,
            edgecolor='black',
            linewidth=0.9,
            alpha=0.9,
            zorder=4,
            label=f'Experiments (n={len(exp_x)})'
        )
    
    # Labels and grid
    ax.set_xlabel(x_var)
    ax.set_ylabel('Predicted Output')
    ax.grid(True, alpha=grid_alpha)
    ax.set_axisbelow(True)
    
    if title:
        ax.set_title(title)
    
    # Create legend with smart ordering: Prediction, bands (small->large), Experiments
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        from alchemist_core.visualization.helpers import sort_legend_items
        sorted_indices = sort_legend_items(labels)
        sorted_handles = [handles[i] for i in sorted_indices]
        sorted_labels = [labels[i] for i in sorted_indices]
        ax.legend(sorted_handles, sorted_labels)
    
    if should_tight_layout:
        fig.tight_layout()
    
    return fig, ax


def create_voxel_plot(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    predictions_grid: np.ndarray,
    x_var: str,
    y_var: str,
    z_var: str,
    exp_x: Optional[np.ndarray] = None,
    exp_y: Optional[np.ndarray] = None,
    exp_z: Optional[np.ndarray] = None,
    suggest_x: Optional[np.ndarray] = None,
    suggest_y: Optional[np.ndarray] = None,
    suggest_z: Optional[np.ndarray] = None,
    cmap: str = 'viridis',
    alpha: float = 0.5,
    use_log_scale: bool = False,
    figsize: Tuple[float, float] = (10, 8),
    dpi: int = 100,
    title: str = "3D Voxel Plot of Model Predictions",
    ax: Optional[Any] = None  # 3D axes
) -> Tuple[Figure, Any]:
    """
    Create 3D voxel plot of model predictions over a variable space.
    
    Visualizes the model's predicted response surface by varying three variables
    while holding others constant. Uses volumetric rendering to show the 3D
    prediction landscape.
    
    Args:
        x_grid: X-axis meshgrid values (3D array)
        y_grid: Y-axis meshgrid values (3D array)
        z_grid: Z-axis meshgrid values (3D array)
        predictions_grid: Model predictions on grid (3D array)
        x_var: X variable name for axis label
        y_var: Y variable name for axis label
        z_var: Z variable name for axis label
        exp_x: Experimental X values to overlay (optional)
        exp_y: Experimental Y values to overlay (optional)
        exp_z: Experimental Z values to overlay (optional)
        suggest_x: Suggested X values to overlay (optional)
        suggest_y: Suggested Y values to overlay (optional)
        suggest_z: Suggested Z values to overlay (optional)
        cmap: Matplotlib colormap name
        alpha: Transparency level (0=transparent, 1=opaque)
        use_log_scale: Use logarithmic color scale
        figsize: Figure size (width, height) in inches
        dpi: Resolution
        title: Plot title
        ax: Existing 3D axes (creates new if None)
    
    Returns:
        Tuple of (Figure, Axes3D) objects
    
    Example:
        >>> # Create 3D grid
        >>> x = np.linspace(0, 10, 20)
        >>> y = np.linspace(0, 10, 20)
        >>> z = np.linspace(0, 10, 20)
        >>> X, Y, Z = np.meshgrid(x, y, z)
        >>> predictions = model.predict(grid)
        >>> fig, ax = create_voxel_plot(X, Y, Z, predictions, 'temp', 'pressure', 'flow')
    
    Note:
        - Requires 3D arrays for x_grid, y_grid, z_grid, predictions_grid
        - Use alpha to control transparency (lower values show interior structure)
        - Computationally expensive for high-resolution grids
    """
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.colors import Normalize, LogNorm
    
    # Create figure and 3D axes if not provided
    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        should_tight_layout = True
    else:
        fig = ax.figure
        should_tight_layout = False
    
    # Normalize predictions for colormapping
    min_val = predictions_grid.min()
    max_val = predictions_grid.max()
    
    if use_log_scale and min_val > 0:
        norm = LogNorm(vmin=min_val, vmax=max_val)
    else:
        norm = Normalize(vmin=min_val, vmax=max_val)
    
    # Get colormap
    cm = plt.get_cmap(cmap)
    
    # Create voxel colors based on predictions
    # Flatten arrays for easier manipulation
    colors = cm(norm(predictions_grid))
    colors[..., -1] = alpha  # Set alpha channel
    
    # Create voxel plot using scatter3D with marker size based on grid resolution
    # Flatten the 3D grids
    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()
    z_flat = z_grid.ravel()
    pred_flat = predictions_grid.ravel()
    
    # Calculate marker size based on grid spacing
    # Use smaller markers for denser grids
    n_points = len(x_flat)
    marker_size = max(10, 1000 / (n_points ** (1/3)))
    
    # Plot as 3D scatter with colors
    scatter = ax.scatter(
        x_flat, y_flat, z_flat,
        c=pred_flat,
        cmap=cmap,
        norm=norm,
        alpha=alpha,
        s=marker_size,
        marker='o',
        edgecolors='none'
    )
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Predicted Output', rotation=270, labelpad=20)
    
    # Overlay experimental points if provided
    if exp_x is not None and exp_y is not None and exp_z is not None and len(exp_x) > 0:
        ax.scatter(
            exp_x, exp_y, exp_z,
            c='white', 
            edgecolors='black',
            s=100, 
            marker='o', 
            label='Experiments',
            linewidths=2,
            depthshade=True
        )
    
    # Overlay suggestion points if provided
    if suggest_x is not None and suggest_y is not None and suggest_z is not None and len(suggest_x) > 0:
        ax.scatter(
            suggest_x, suggest_y, suggest_z,
            c='black',
            s=150,
            marker='*',
            label='Suggestions',
            linewidths=2,
            depthshade=True
        )
    
    # Set labels and title
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.set_zlabel(z_var)
    ax.set_title(title)
    
    # Add legend if we have overlays
    if (exp_x is not None and len(exp_x) > 0) or (suggest_x is not None and len(suggest_x) > 0):
        ax.legend(loc='upper left')
    
    if should_tight_layout:
        fig.tight_layout()
    
    return fig, ax


def create_metrics_plot(
    training_sizes: np.ndarray,
    metric_values: np.ndarray,
    metric_name: str,
    figsize: Tuple[float, float] = (8, 6),
    dpi: int = 100,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """
    Create learning curve plot showing metric vs training size.
    
    Displays how model performance improves as more experimental data is added.
    
    Args:
        training_sizes: X-axis values (number of observations)
        metric_values: Y-axis values (metric at each training size)
        metric_name: Metric name ('rmse', 'mae', 'r2', 'mape')
        figsize: Figure size (width, height) in inches
        dpi: Resolution
        ax: Existing axes (creates new if None)
    
    Returns:
        Tuple of (Figure, Axes)
    
    Example:
        >>> sizes = np.array([5, 6, 7, 8, 9, 10])
        >>> rmse = np.array([0.15, 0.12, 0.10, 0.08, 0.07, 0.06])
        >>> fig, ax = create_metrics_plot(sizes, rmse, 'rmse')
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        should_tight_layout = True
    else:
        fig = ax.figure
        should_tight_layout = False
    
    # Plot as line with markers
    ax.plot(training_sizes, metric_values, marker='o', linewidth=2,
           markersize=6, color='#2E86AB')
    
    # Labels
    metric_labels = {
        'rmse': 'RMSE',
        'mae': 'MAE',
        'r2': 'R²',
        'mape': 'MAPE (%)'
    }
    label = metric_labels.get(metric_name.lower(), metric_name.upper())
    
    ax.set_xlabel("Number of Observations")
    ax.set_ylabel(label)
    ax.set_title(f"{label} vs Number of Observations")
    ax.grid(True, alpha=0.3)
    
    if should_tight_layout:
        fig.tight_layout()
    
    return fig, ax


def create_qq_plot(
    z_scores: np.ndarray,
    figsize: Tuple[float, float] = (8, 6),
    dpi: int = 100,
    show_confidence_bands: bool = True,
    title: str = "Q-Q Plot: Standardized Residuals",
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """
    Create Q-Q plot of standardized residuals.
    
    Compares the distribution of standardized residuals to a standard normal
    distribution. Points following the diagonal line indicate well-calibrated
    uncertainty estimates.
    
    Args:
        z_scores: Standardized residuals (z-scores)
        figsize: Figure size (width, height) in inches
        dpi: Resolution
        show_confidence_bands: Add approximate 95% CI bands for small samples
        title: Plot title
        ax: Existing axes (creates new if None)
    
    Returns:
        Tuple of (Figure, Axes)
    
    Example:
        >>> z = (y_true - y_pred) / y_std
        >>> fig, ax = create_qq_plot(z)
    """
    from scipy import stats
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        should_tight_layout = True
    else:
        fig = ax.figure
        should_tight_layout = False
    
    # Sort z-scores
    z_sorted = np.sort(z_scores)
    
    # Theoretical quantiles from standard normal distribution
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(z_scores)))
    
    # Scatter plot
    ax.scatter(theoretical_quantiles, z_sorted, alpha=0.7, s=30,
              edgecolors='k', linewidth=0.5)
    
    # Perfect calibration line (y=x)
    min_val = min(theoretical_quantiles.min(), z_sorted.min())
    max_val = max(theoretical_quantiles.max(), z_sorted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--',
           linewidth=2, label='Perfect calibration')
    
    # Confidence bands for small samples
    if show_confidence_bands and len(z_scores) < 100:
        se = 1.96 / np.sqrt(len(z_scores))
        ax.fill_between([min_val, max_val],
                       [min_val - se, max_val - se],
                       [min_val + se, max_val + se],
                       alpha=0.2, color='red', label='Approximate 95% CI')
    
    # Labels and legend
    ax.set_xlabel("Theoretical Quantiles (Standard Normal)")
    ax.set_ylabel("Sample Quantiles (Standardized Residuals)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if should_tight_layout:
        fig.tight_layout()
    
    return fig, ax


def create_calibration_plot(
    nominal_probs: np.ndarray,
    empirical_coverage: np.ndarray,
    figsize: Tuple[float, float] = (8, 6),
    dpi: int = 100,
    title: str = "Calibration Curve",
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """
    Create calibration curve (reliability diagram).
    
    Shows whether predicted confidence intervals have the correct coverage.
    Points on the diagonal line indicate well-calibrated uncertainty.
    Points above the line indicate overconfident predictions (intervals too narrow).
    Points below the line indicate underconfident predictions (intervals too wide).
    
    Args:
        nominal_probs: Expected coverage probabilities (X-axis)
        empirical_coverage: Observed coverage fractions (Y-axis)
        figsize: Figure size (width, height) in inches
        dpi: Resolution
        title: Plot title
        ax: Existing axes (creates new if None)
    
    Returns:
        Tuple of (Figure, Axes)
    
    Example:
        >>> nominal = np.array([0.68, 0.95, 0.99])
        >>> empirical = np.array([0.72, 0.94, 0.98])
        >>> fig, ax = create_calibration_plot(nominal, empirical)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        should_tight_layout = True
    else:
        fig = ax.figure
        should_tight_layout = False
    
    # Empirical coverage line
    ax.plot(nominal_probs, empirical_coverage, 'o-', linewidth=2,
           markersize=6, label='Empirical coverage', color='steelblue')
    
    # Perfect calibration line (y=x)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect calibration')
    
    # Labels and legend
    ax.set_xlabel("Nominal Coverage Probability")
    ax.set_ylabel("Empirical Coverage")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    if should_tight_layout:
        fig.tight_layout()
    
    return fig, ax


def create_regret_plot(
    iterations: np.ndarray,
    observed_values: np.ndarray,
    show_cumulative: bool = False,
    goal: str = 'maximize',
    predicted_means: Optional[np.ndarray] = None,
    predicted_stds: Optional[np.ndarray] = None,
    sigma_bands: Optional[List[float]] = None,
    figsize: Tuple[float, float] = (8, 6),
    dpi: int = 100,
    title: Optional[str] = None,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """
    Create regret curve (best observed value vs iteration).
    
    Shows the cumulative best result achieved over the course of optimization.
    A flattening curve indicates convergence (no further improvements found).
    
    This is also known as "simple regret" or "incumbent trajectory" in the
    Bayesian optimization literature.
    
    Args:
        iterations: Iteration numbers (typically 0, 1, 2, ... or experiment indices)
        observed_values: Actual experimental outputs at each iteration
        goal: 'maximize' or 'minimize' - determines how "best" is computed
        predicted_means: Optional array of max(posterior mean) at each iteration
        predicted_stds: Optional array of std at max(posterior mean) at each iteration
        sigma_bands: List of sigma values for uncertainty bands (e.g., [1.0, 2.0])
        figsize: Figure size (width, height) in inches
        dpi: Resolution
        title: Custom title (auto-generated if None)
        ax: Existing axes (creates new if None)
    
    Returns:
        Tuple of (Figure, Axes)
    
    Example:
        >>> iterations = np.arange(len(outputs))
        >>> fig, ax = create_regret_plot(iterations, outputs, goal='maximize')
    
    Notes:
        For maximization, plots cumulative maximum (best so far).
        For minimization, plots cumulative minimum (best so far).
        Curve should increase/decrease monotonically and flatten at convergence.
        If predicted_means/stds provided, also shows model's predicted best with uncertainty.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        should_tight_layout = True
    else:
        fig = ax.figure
        should_tight_layout = False
    
    # Styling constants (matching slice plots)
    mean_color = "#0B3C5D"          # dark blue-teal, prints well
    exp_face   = "#E07A00"          # orange (colorblind-friendly vs blue)
    grid_alpha = 0.25
    
    # Compute cumulative best
    if goal.lower() == 'maximize':
        cumulative_best = np.maximum.accumulate(observed_values)
    else:
        cumulative_best = np.minimum.accumulate(observed_values)

    ylabel = 'Objective Function Value'
    
    # Plot cumulative best as line
    if show_cumulative:
        ax.plot(iterations, cumulative_best, linewidth=2.5, 
               color='#E07A00', label='Best observed', zorder=3)
    
    # Overlay actual observations as scatter points (using same orange as slice plots)
    ax.scatter(iterations, observed_values, s=70, alpha=0.9,
              facecolor=exp_face, edgecolors='black', linewidth=0.9,
              label='All observations', zorder=2)
    
    # Plot predicted best (max posterior mean) with uncertainty if provided
    if predicted_means is not None:
        if predicted_stds is not None and sigma_bands:
            # Sort largest to smallest for proper layering
            sigma_bands_sorted = sorted(sigma_bands, reverse=True)
            n = len(sigma_bands_sorted)
            
            # Sequential colormap: same hue, different lightness (matching slice plots)
            cmap = plt.get_cmap("Blues")
            
            for i, sigma in enumerate(sigma_bands_sorted):
                # i=0 is largest sigma (most transparent), i=n-1 is smallest (most opaque)
                t = i / max(1, n - 1)  # 0..1 ratio
                
                # Lighter tones for larger sigma, darker for smaller
                face = cmap(0.3 + 0.3 * t)  # 0.30 to 0.60 in Blues colormap
                edge = plt.matplotlib.colors.to_rgba(mean_color, 0.55)
                
                # Sigmoid-based alpha: smaller sigma → higher alpha (more opaque)
                alpha = 1.0 - 1.0 / (1.0 + np.exp(-sigma + 2.0))
                
                ax.fill_between(
                    iterations,
                    predicted_means - sigma * predicted_stds,
                    predicted_means + sigma * predicted_stds,
                    facecolor=plt.matplotlib.colors.to_rgba(face, alpha),
                    edgecolor=edge,
                    linewidth=0.9,
                    label=f'±{sigma:.1f}σ',
                    zorder=1
                )
                
        ax.plot(iterations, predicted_means, linewidth=2.6, 
               color=mean_color, linestyle='-', 
               label='Max posterior mean', zorder=3)
    
    # Labels and title
    ax.set_xlabel("Experiment Number")
    ax.set_ylabel(ylabel)
    
    if title is None:
        title = f"Optimization Progress ({'maximization' if goal.lower() == 'maximize' else 'minimization'})"
    ax.set_title(title)
    
    ax.grid(True, alpha=grid_alpha)
    ax.set_axisbelow(True)
    ax.legend()
    
    if should_tight_layout:
        fig.tight_layout()
    
    return fig, ax


def create_probability_of_improvement_plot(
    iterations: np.ndarray,
    max_pi_values: np.ndarray,
    figsize: Tuple[float, float] = (8, 6),
    dpi: int = 100,
    title: Optional[str] = None,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """
    Create probability of improvement convergence curve.
    
    Shows the maximum probability of improvement (PI) available in the search
    space at each iteration. As optimization progresses and good regions are
    explored, max(PI) should decrease, indicating convergence.
    
    This is computed retroactively by:
    1. Training GP incrementally with observations 0:i
    2. Computing PI across the search space
    3. Taking the maximum PI value
    
    Args:
        iterations: Iteration numbers where PI was evaluated
        max_pi_values: Maximum PI value in search space at each iteration
        figsize: Figure size (width, height) in inches
        dpi: Resolution
        title: Custom title (auto-generated if None)
        ax: Existing axes (creates new if None)
    
    Returns:
        Tuple of (Figure, Axes)
    
    Example:
        >>> # Computed retroactively from session
        >>> fig = session.plot_probability_of_improvement()
    
    Notes:
        - Values range from 0 to 1 (probabilities)
        - Decreasing trend indicates optimization converging
        - Values near 0 suggest little room for improvement
        - Useful for determining stopping criteria
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        should_tight_layout = True
    else:
        fig = ax.figure
        should_tight_layout = False
    
    # Plot max PI values with markers
    ax.plot(iterations, max_pi_values, linewidth=2.5,
           marker='o', markersize=6, color='#5B9BD5', 
           label='Max PI in search space')
    
    # Add horizontal reference lines
    ax.axhline(y=0.5, color='gray', linestyle='--', 
              linewidth=1, alpha=0.4, label='PI = 0.5')
    ax.axhline(y=0.1, color='orange', linestyle='--',
              linewidth=1, alpha=0.4, label='PI = 0.1')
    
    # Labels and title
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Maximum Probability of Improvement")
    
    if title is None:
        title = "Probability of Improvement Convergence"
    ax.set_title(title)
    
    ax.set_ylim([0, 1.05])  # PI is a probability
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    if should_tight_layout:
        fig.tight_layout()
    
    return fig, ax


# ==============================================================================
# VISUALIZATION ARCHITECTURE NOTES
# ==============================================================================
#
# This module implements a systematic framework for visualizing Gaussian Process
# models in the context of Bayesian optimization. The visualization space can be
# organized along two axes:
#
# 1. WHAT TO VISUALIZE (3 fundamental quantities):
#    ------------------------------------------------
#    a) Posterior Mean
#       - Model's best estimate of the objective function
#       - ✅ IMPLEMENTED: 1D slice plots, 2D contour plots, 3D voxel plots
#    
#    b) Posterior Uncertainty
#       - Model's confidence/uncertainty in predictions
#       - ✅ IMPLEMENTED: 1D slice plots (as bands around mean)
#       - ❌ NOT YET: 2D contour plots of uncertainty, 3D voxel plots
#    
#    c) Acquisition Function
#       - Decision-making criteria under uncertainty (EI, PI, UCB, etc.)
#       - Shows where the optimization algorithm will sample next
#       - Same dimensionality as posterior mean/uncertainty
#       - ✅ IMPLEMENTED: 1D slice plots, 2D contour plots (via session API)
#       - ❌ NOT YET: 3D voxel plots
#
# 2. HOW TO VISUALIZE (dimensionality of visualization):
#    -----------------------------------------------------
#    a) 1D Slice - Fix all but 1 variable (✅ FULLY IMPLEMENTED)
#       - create_slice_plot(): Shows posterior mean + uncertainty bands
#       - Experimental points can be overlaid
#       - Custom sigma bands: [1.0, 2.0, 3.0] for ±1σ, ±2σ, ±3σ
#       - Used by: session.plot_slice(), session.plot_acquisition_slice()
#    
#    b) 2D Contour - Fix all but 2 variables (✅ MOSTLY IMPLEMENTED)
#       - create_contour_plot(): Shows posterior mean as colored contours
#       - Experimental points and suggestions can be overlaid
#       - Used by: session.plot_contour(), session.plot_acquisition_contour()
#       - ❌ NOT YET: uncertainty as separate subplot or transparency overlay
#    
#    c) 3D Voxel - Fix all but 3 variables (✅ NEWLY IMPLEMENTED - Jan 2026)
#       - create_voxel_plot(): 3D scatter visualization for response surfaces
#       - Uses matplotlib 3D scatter with color mapping and transparency
#       - Adjustable alpha parameter for seeing interior structure
#       - Experimental points and suggestions can be overlaid
#       - Used by: session.plot_voxel()
#       - Requires 3+ continuous (real/integer) variables
#       - ❌ NOT YET: uncertainty visualization, acquisition function plots
#       - Note: Computationally expensive (O(N³) evaluations)
#
# IMPLEMENTATION STATUS MATRIX (Jan 22, 2026):
# -----------------------------------------
#                     1D Slice    2D Contour    3D Voxel
# Posterior Mean         ✅           ✅           ✅
# Posterior Uncertainty  ✅           ✅           ✅
# Acquisition Function   ✅           ✅           ✅
#
# COMPLETE! All 9 visualization combinations implemented.
#
# API CONSISTENCY NOTES:
# ----------------------
# Current naming conventions:
# - plot_slice: show_uncertainty (Union[bool, List[float]])
# - plot_parity: show_error_bars (bool)
# - All: show_experiments (bool)
# - plot_slice: n_points (int) - 1D sampling
# - plot_contour: grid_resolution (int) - 2D grid (N×N)
# - plot_voxel: grid_resolution (int) - 3D grid (N×N×N, default: 15)
# - plot_voxel: alpha (float) - transparency (0-1, default: 0.5)
#
# These differences are intentional (slice shows bands, parity shows error bars;
# 1D/2D/3D sampling densities), but maintain consistency within dimensionality.
#
# ==============================================================================


def create_uncertainty_contour_plot(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    uncertainty_grid: np.ndarray,
    x_var: str,
    y_var: str,
    exp_x: Optional[np.ndarray] = None,
    exp_y: Optional[np.ndarray] = None,
    suggest_x: Optional[np.ndarray] = None,
    suggest_y: Optional[np.ndarray] = None,
    cmap: str = 'Reds',
    figsize: Tuple[float, float] = (8, 6),
    dpi: int = 100,
    title: str = "Posterior Uncertainty (Standard Deviation)",
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes, Any]:
    """
    Create 2D contour plot of posterior uncertainty (standard deviation).
    
    Visualizes where the model is most uncertain about predictions, showing
    regions that may benefit from additional sampling. Higher values indicate
    greater uncertainty.
    
    Args:
        x_grid: X-axis meshgrid values (2D array)
        y_grid: Y-axis meshgrid values (2D array)
        uncertainty_grid: Posterior standard deviations on grid (2D array)
        x_var: X variable name for axis label
        y_var: Y variable name for axis label
        exp_x: Experimental X values to overlay (optional)
        exp_y: Experimental Y values to overlay (optional)
        suggest_x: Suggested X values to overlay (optional)
        suggest_y: Suggested Y values to overlay (optional)
        cmap: Matplotlib colormap name (default: 'Reds' - darker = more uncertain)
        figsize: Figure size (width, height) in inches
        dpi: Resolution
        title: Plot title
        ax: Existing axes (creates new if None)
    
    Returns:
        Tuple of (Figure, Axes, Colorbar)
    
    Example:
        >>> X, Y = np.meshgrid(x_range, y_range)
        >>> _, std = model.predict(grid, return_std=True)
        >>> uncertainty = std.reshape(X.shape)
        >>> fig, ax, cbar = create_uncertainty_contour_plot(X, Y, uncertainty, 'temp', 'pressure')
    
    Note:
        - Useful for identifying under-explored regions
        - Typically used with 'Reds' or 'YlOrRd' colormaps
        - High uncertainty near data gaps is expected
        - Can guide where to sample next for exploration
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        should_tight_layout = True
    else:
        fig = ax.figure
        should_tight_layout = False
    
    # Contour plot of uncertainty
    min_val = uncertainty_grid.min()
    max_val = uncertainty_grid.max()
    
    levels = np.linspace(min_val, max_val, 50)
    contour = ax.contourf(x_grid, y_grid, uncertainty_grid, levels=levels, 
                         cmap=cmap, vmin=min_val, vmax=max_val)
    
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Posterior Standard Deviation', rotation=270, labelpad=20)
    
    # Overlay experimental points
    if exp_x is not None and exp_y is not None and len(exp_x) > 0:
        ax.scatter(exp_x, exp_y, c='white', edgecolors='black', 
                  s=80, marker='o', label='Experiments', zorder=5)
    
    # Overlay suggestion points
    if suggest_x is not None and suggest_y is not None and len(suggest_x) > 0:
        ax.scatter(suggest_x, suggest_y, c='black',
                  s=120, marker='*', label='Suggestions', zorder=6)
    
    # Labels and title
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.set_title(title)
    
    # Legend if we have overlays
    if (exp_x is not None and len(exp_x) > 0) or (suggest_x is not None and len(suggest_x) > 0):
        ax.legend()
    
    if should_tight_layout:
        fig.tight_layout()
    
    return fig, ax, cbar


def create_uncertainty_voxel_plot(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    uncertainty_grid: np.ndarray,
    x_var: str,
    y_var: str,
    z_var: str,
    exp_x: Optional[np.ndarray] = None,
    exp_y: Optional[np.ndarray] = None,
    exp_z: Optional[np.ndarray] = None,
    suggest_x: Optional[np.ndarray] = None,
    suggest_y: Optional[np.ndarray] = None,
    suggest_z: Optional[np.ndarray] = None,
    cmap: str = 'Reds',
    alpha: float = 0.5,
    figsize: Tuple[float, float] = (10, 8),
    dpi: int = 100,
    title: str = "3D Posterior Uncertainty",
    ax: Optional[Any] = None
) -> Tuple[Figure, Any]:
    """
    Create 3D voxel plot of posterior uncertainty over variable space.
    
    Visualizes where the model is most uncertain in 3D, helping identify
    under-explored regions that may benefit from additional sampling.
    
    Args:
        x_grid: X-axis meshgrid values (3D array)
        y_grid: Y-axis meshgrid values (3D array)
        z_grid: Z-axis meshgrid values (3D array)
        uncertainty_grid: Posterior standard deviations on grid (3D array)
        x_var: X variable name for axis label
        y_var: Y variable name for axis label
        z_var: Z variable name for axis label
        exp_x: Experimental X values to overlay (optional)
        exp_y: Experimental Y values to overlay (optional)
        exp_z: Experimental Z values to overlay (optional)
        suggest_x: Suggested X values to overlay (optional)
        suggest_y: Suggested Y values to overlay (optional)
        suggest_z: Suggested Z values to overlay (optional)
        cmap: Matplotlib colormap name (default: 'Reds')
        alpha: Transparency level (0=transparent, 1=opaque)
        figsize: Figure size (width, height) in inches
        dpi: Resolution
        title: Plot title
        ax: Existing 3D axes (creates new if None)
    
    Returns:
        Tuple of (Figure, Axes3D) objects
    
    Example:
        >>> X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        >>> _, std = model.predict(grid, return_std=True)
        >>> uncertainty = std.reshape(X.shape)
        >>> fig, ax = create_uncertainty_voxel_plot(X, Y, Z, uncertainty, 'temp', 'press', 'flow')
    
    Note:
        - Higher values = greater uncertainty
        - Useful for planning exploration strategies
        - Shows data-sparse regions in 3D
        - Computationally expensive (O(N³) evaluations)
    """
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.colors import Normalize
    
    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        should_tight_layout = True
    else:
        fig = ax.figure
        should_tight_layout = False
    
    # Normalize uncertainty for colormapping
    min_val = uncertainty_grid.min()
    max_val = uncertainty_grid.max()
    norm = Normalize(vmin=min_val, vmax=max_val)
    
    # Get colormap
    cm = plt.get_cmap(cmap)
    
    # Flatten arrays
    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()
    z_flat = z_grid.ravel()
    uncertainty_flat = uncertainty_grid.ravel()
    
    # Calculate marker size based on grid spacing
    n_points = len(x_flat)
    marker_size = max(10, 1000 / (n_points ** (1/3)))
    
    # Plot as 3D scatter with colors
    scatter = ax.scatter(
        x_flat, y_flat, z_flat,
        c=uncertainty_flat,
        cmap=cmap,
        norm=norm,
        alpha=alpha,
        s=marker_size,
        marker='o',
        edgecolors='none'
    )
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Posterior Standard Deviation', rotation=270, labelpad=20)
    
    # Overlay experimental points
    if exp_x is not None and exp_y is not None and exp_z is not None and len(exp_x) > 0:
        ax.scatter(
            exp_x, exp_y, exp_z,
            c='white', 
            edgecolors='black',
            s=100, 
            marker='o', 
            label='Experiments',
            linewidths=2,
            depthshade=True
        )
    
    # Overlay suggestion points
    if suggest_x is not None and suggest_y is not None and suggest_z is not None and len(suggest_x) > 0:
        ax.scatter(
            suggest_x, suggest_y, suggest_z,
            c='blue',
            edgecolors='black',
            s=150,
            marker='*',
            label='Suggestions',
            linewidths=2,
            depthshade=True
        )
    
    # Set labels and title
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.set_zlabel(z_var)
    ax.set_title(title)
    
    # Add legend if we have overlays
    if (exp_x is not None and len(exp_x) > 0) or (suggest_x is not None and len(suggest_x) > 0):
        ax.legend(loc='upper left')
    
    if should_tight_layout:
        fig.tight_layout()
    
    return fig, ax


def create_acquisition_voxel_plot(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    acquisition_grid: np.ndarray,
    x_var: str,
    y_var: str,
    z_var: str,
    exp_x: Optional[np.ndarray] = None,
    exp_y: Optional[np.ndarray] = None,
    exp_z: Optional[np.ndarray] = None,
    suggest_x: Optional[np.ndarray] = None,
    suggest_y: Optional[np.ndarray] = None,
    suggest_z: Optional[np.ndarray] = None,
    cmap: str = 'hot',
    alpha: float = 0.5,
    use_log_scale: bool = False,
    figsize: Tuple[float, float] = (10, 8),
    dpi: int = 100,
    title: str = "3D Acquisition Function",
    ax: Optional[Any] = None
) -> Tuple[Figure, Any]:
    """
    Create 3D voxel plot of acquisition function over variable space.
    
    Visualizes the acquisition function in 3D, showing "hot spots" where
    the optimization algorithm believes the next experiment should be conducted.
    Higher values indicate more promising regions.
    
    Args:
        x_grid: X-axis meshgrid values (3D array)
        y_grid: Y-axis meshgrid values (3D array)
        z_grid: Z-axis meshgrid values (3D array)
        acquisition_grid: Acquisition function values on grid (3D array)
        x_var: X variable name for axis label
        y_var: Y variable name for axis label
        z_var: Z variable name for axis label
        exp_x: Experimental X values to overlay (optional)
        exp_y: Experimental Y values to overlay (optional)
        exp_z: Experimental Z values to overlay (optional)
        suggest_x: Suggested X values to overlay (optional)
        suggest_y: Suggested Y values to overlay (optional)
        suggest_z: Suggested Z values to overlay (optional)
        cmap: Matplotlib colormap name (default: 'hot')
        alpha: Transparency level (0=transparent, 1=opaque)
        use_log_scale: Use logarithmic color scale
        figsize: Figure size (width, height) in inches
        dpi: Resolution
        title: Plot title
        ax: Existing 3D axes (creates new if None)
    
    Returns:
        Tuple of (Figure, Axes3D) objects
    
    Example:
        >>> X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        >>> acq_values = evaluate_acquisition(model, grid, acq_func='ei')
        >>> acq_grid = acq_values.reshape(X.shape)
        >>> fig, ax = create_acquisition_voxel_plot(X, Y, Z, acq_grid, 'temp', 'press', 'flow')
    
    Note:
        - Higher values = more promising for next experiment
        - Use with EI, PI, UCB, or other acquisition functions
        - Helps visualize exploration-exploitation tradeoff
        - Suggestions should align with high-value regions
        - Computationally expensive (O(N³) evaluations)
    """
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.colors import Normalize, LogNorm
    
    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        should_tight_layout = True
    else:
        fig = ax.figure
        should_tight_layout = False
    
    # Normalize acquisition values for colormapping
    min_val = acquisition_grid.min()
    max_val = acquisition_grid.max()
    
    if use_log_scale and min_val > 0:
        norm = LogNorm(vmin=min_val, vmax=max_val)
    else:
        norm = Normalize(vmin=min_val, vmax=max_val)
    
    # Get colormap
    cm = plt.get_cmap(cmap)
    
    # Flatten arrays
    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()
    z_flat = z_grid.ravel()
    acq_flat = acquisition_grid.ravel()
    
    # Calculate marker size based on grid spacing
    n_points = len(x_flat)
    marker_size = max(10, 1000 / (n_points ** (1/3)))
    
    # Plot as 3D scatter with colors
    scatter = ax.scatter(
        x_flat, y_flat, z_flat,
        c=acq_flat,
        cmap=cmap,
        norm=norm,
        alpha=alpha,
        s=marker_size,
        marker='o',
        edgecolors='none'
    )
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Acquisition Function Value', rotation=270, labelpad=20)

    # Overlay experimental points
    if exp_x is not None and exp_y is not None and exp_z is not None and len(exp_x) > 0:
        ax.scatter(
            exp_x, exp_y, exp_z,
            c='cyan',
            edgecolors='black',
            s=100,
            marker='o',
            label='Experiments',
            linewidths=2,
            depthshade=True
        )

    # Overlay suggestion points (should be in high-acquisition regions)
    if suggest_x is not None and suggest_y is not None and suggest_z is not None and len(suggest_x) > 0:
        ax.scatter(
            suggest_x, suggest_y, suggest_z,
            c='black',
            s=150,
            marker='*',
            label='Suggestions',
            linewidths=2,
            depthshade=True
        )

    # Set labels and title
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.set_zlabel(z_var)
    ax.set_title(title)

    # Add legend if we have overlays
    if (exp_x is not None and len(exp_x) > 0) or (suggest_x is not None and len(suggest_x) > 0):
        ax.legend(loc='upper left')

    if should_tight_layout:
        fig.tight_layout()

    return fig, ax


def create_pareto_plot(
    Y: np.ndarray,
    pareto_mask: np.ndarray,
    objective_names: List[str],
    directions: Optional[List[str]] = None,
    ref_point: Optional[List[float]] = None,
    show_hypervolume: bool = True,
    suggested_points: Optional[np.ndarray] = None,
    constraint_boundaries: Optional[Dict[str, float]] = None,
    figsize=(8, 6), dpi=100, title=None, ax=None
) -> Tuple[Figure, Axes]:
    """Create a Pareto frontier plot for 2-objective optimization.

    Args:
        Y: (n_samples, n_objectives) array of observed objective values
        pareto_mask: Boolean mask identifying Pareto-optimal points
        objective_names: Names for each objective axis
        directions: 'maximize' or 'minimize' per objective (default: all maximize)
        ref_point: Reference point for hypervolume shading
        show_hypervolume: Whether to shade the dominated hypervolume region
        suggested_points: (n, 2) array of newly suggested points to overlay
        constraint_boundaries: {objective_name: value} for dashed constraint lines
        figsize: Figure size
        dpi: Figure DPI
        title: Optional title
        ax: Optional existing Axes

    Returns:
        (Figure, Axes)
    """
    should_create_fig = ax is None
    if should_create_fig:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.get_figure()

    if directions is None:
        directions = ['maximize'] * Y.shape[1]

    # Split Pareto vs dominated
    pareto_Y = Y[pareto_mask]
    dominated_Y = Y[~pareto_mask]

    # Plot dominated points
    if len(dominated_Y) > 0:
        ax.scatter(dominated_Y[:, 0], dominated_Y[:, 1],
                   c='#aaaaaa', alpha=0.5, s=40, label='Dominated', zorder=2)

    # Plot Pareto points
    if len(pareto_Y) > 0:
        ax.scatter(pareto_Y[:, 0], pareto_Y[:, 1],
                   c='#2196F3', edgecolors='black', s=80, linewidth=0.8,
                   label='Pareto optimal', zorder=3)

        # Draw stepped Pareto front line
        sort_idx = np.argsort(pareto_Y[:, 0])
        sorted_pareto = pareto_Y[sort_idx]

        step_x = []
        step_y = []
        for i in range(len(sorted_pareto)):
            if i > 0:
                step_x.append(sorted_pareto[i, 0])
                step_y.append(sorted_pareto[i - 1, 1])
            step_x.append(sorted_pareto[i, 0])
            step_y.append(sorted_pareto[i, 1])

        ax.plot(step_x, step_y, 'b-', alpha=0.6, linewidth=1.5, zorder=2)

        # Hypervolume shading
        if show_hypervolume and ref_point is not None:
            fill_x = [ref_point[0]] + step_x + [sorted_pareto[-1, 0], ref_point[0]]
            fill_y = [ref_point[1]] + step_y + [ref_point[1], ref_point[1]]
            ax.fill(fill_x, fill_y, alpha=0.1, color='blue', label='Hypervolume')
            ax.scatter([ref_point[0]], [ref_point[1]], marker='x', c='red', s=100,
                       linewidths=2, label='Ref point', zorder=4)

    # Overlay suggested points
    if suggested_points is not None and len(suggested_points) > 0:
        ax.scatter(suggested_points[:, 0], suggested_points[:, 1],
                   c='#FF9800', marker='*', s=150, edgecolors='black', linewidth=0.5,
                   label='Suggested', zorder=5)

    # Constraint boundaries as dashed lines
    if constraint_boundaries:
        for obj_name, value in constraint_boundaries.items():
            if obj_name == objective_names[0]:
                ax.axvline(x=value, color='red', linestyle='--', alpha=0.7,
                          label=f'{obj_name} bound')
            elif obj_name == objective_names[1]:
                ax.axhline(y=value, color='red', linestyle='--', alpha=0.7,
                          label=f'{obj_name} bound')

    ax.set_xlabel(objective_names[0])
    ax.set_ylabel(objective_names[1])
    ax.set_title(title or 'Pareto Frontier')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    if should_create_fig:
        fig.tight_layout()

    return fig, ax


def create_hypervolume_convergence_plot(
    iterations: np.ndarray,
    observed_hv: np.ndarray,
    show_cumulative: bool = False,
    predicted_hv: Optional[np.ndarray] = None,
    predicted_hv_std: Optional[np.ndarray] = None,
    sigma_bands: Optional[List[float]] = None,
    ref_point: Optional[List[float]] = None,
    figsize: Tuple[float, float] = (8, 6),
    dpi: int = 100,
    title: Optional[str] = None,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """Create hypervolume convergence plot for multi-objective optimization.

    Mirrors create_regret_plot() in style and behaviour but uses hypervolume as
    the progress metric.

    - Scatter points show the observed Pareto-front hypervolume at each iteration.
    - An optional cumulative-best line is shown when *show_cumulative* is True.
    - When *predicted_hv* (and optionally *predicted_hv_std* + *sigma_bands*)
      are provided, a model-predicted hypervolume line with uncertainty bands
      is overlaid — analogous to the "max posterior mean" line in the
      single-objective regret plot.

    Args:
        iterations: 1-based iteration indices.
        observed_hv: Hypervolume of the observed Pareto front at each iteration.
        show_cumulative: Show cumulative best HV line (default False).
        predicted_hv: Model-predicted hypervolume at each iteration (optional).
        predicted_hv_std: Std of predicted hypervolume (optional).
        sigma_bands: Sigma multipliers for uncertainty bands (e.g. [1.0, 2.0]).
        ref_point: Reference point used for hypervolume (annotated if given).
        figsize: Figure size.
        dpi: Figure DPI.
        title: Custom title.
        ax: Pre-existing axes.

    Returns:
        (Figure, Axes)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        should_tight_layout = True
    else:
        fig = ax.figure
        should_tight_layout = False

    mean_color = "#0B3C5D"
    exp_face = "#E07A00"

    # Optional cumulative best line (off by default, matching single-obj behaviour)
    if show_cumulative:
        cum_max = np.maximum.accumulate(observed_hv)
        ax.plot(iterations, cum_max, linewidth=2.5, color=exp_face,
                label='Cumulative best HV', zorder=3)

    # Scatter observed HV (per-experiment contribution / delta HV)
    ax.scatter(iterations, observed_hv, s=70, alpha=0.9,
               facecolor=exp_face, edgecolors='black', linewidth=0.9,
               label='HV contribution', zorder=2)

    # Predicted HV with uncertainty bands
    if predicted_hv is not None:
        valid = ~np.isnan(predicted_hv)
        if valid.any():
            iters_v = iterations[valid]
            hv_v = predicted_hv[valid]

            if predicted_hv_std is not None and sigma_bands:
                std_v = predicted_hv_std[valid]
                sigma_bands_sorted = sorted(sigma_bands, reverse=True)
                n = len(sigma_bands_sorted)
                cmap = plt.get_cmap("Blues")

                for idx, sigma in enumerate(sigma_bands_sorted):
                    t = idx / max(1, n - 1)
                    face = cmap(0.3 + 0.3 * t)
                    alpha_band = 1.0 - 1.0 / (1.0 + np.exp(-sigma + 2.0))
                    ax.fill_between(
                        iters_v,
                        hv_v - sigma * std_v,
                        hv_v + sigma * std_v,
                        alpha=alpha_band,
                        facecolor=face,
                        edgecolor=plt.matplotlib.colors.to_rgba(mean_color, 0.55),
                        linewidth=0.5,
                        label=f'±{sigma:.1f}σ',
                        zorder=1,
                    )

            ax.plot(iters_v, hv_v, linewidth=2.6, color=mean_color,
                    linestyle='-',
                    label='Max posterior HV', zorder=4)

    ax.set_xlabel('Experiment Number')
    ax.set_ylabel('Hypervolume')

    if title is None:
        title = 'Hypervolume Convergence'
        if ref_point is not None:
            title += f'\n(ref point: {ref_point})'

    ax.set_title(title)
    ax.set_axisbelow(True)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.25)

    if should_tight_layout:
        fig.tight_layout()

    return fig, ax


def create_surface_plot(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    predictions_grid: np.ndarray,
    x_var: str,
    y_var: str,
    output_label: str = "Predicted Output",
    exp_x: Optional[np.ndarray] = None,
    exp_y: Optional[np.ndarray] = None,
    exp_output: Optional[np.ndarray] = None,
    suggest_x: Optional[np.ndarray] = None,
    suggest_y: Optional[np.ndarray] = None,
    cmap: str = 'viridis',
    alpha: float = 0.9,
    figsize: Tuple[float, float] = (10, 8),
    dpi: int = 100,
    title: str = "3D Surface Plot of Model Predictions",
    ax: Optional[Any] = None
) -> Tuple[Figure, Any, Any]:
    """
    Create 3D surface plot of model predictions.

    Like the 2D contour plot, varies two variables (X, Y) while holding
    others fixed, but renders the predicted output on the Z axis as a
    3D surface colored by the prediction values.

    Args:
        x_grid: X-axis meshgrid values (2D array)
        y_grid: Y-axis meshgrid values (2D array)
        predictions_grid: Model predictions on grid (2D array, same shape as x_grid)
        x_var: X variable name for axis label
        y_var: Y variable name for axis label
        output_label: Z-axis label (default: "Predicted Output")
        exp_x: Experimental X values to overlay (optional)
        exp_y: Experimental Y values to overlay (optional)
        exp_output: Experimental output values to overlay (optional)
        suggest_x: Suggested X values to overlay (optional)
        suggest_y: Suggested Y values to overlay (optional)
        cmap: Matplotlib colormap name
        alpha: Surface transparency (0=transparent, 1=opaque)
        figsize: Figure size (width, height) in inches
        dpi: Resolution
        title: Plot title
        ax: Existing 3D axes (creates new if None)

    Returns:
        Tuple of (Figure, Axes3D, Colorbar)

    Example:
        >>> X, Y = np.meshgrid(x_range, y_range)
        >>> Z = model_predictions.reshape(X.shape)
        >>> fig, ax, cbar = create_surface_plot(X, Y, Z, 'temperature', 'pressure')
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib.colors import Normalize

    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        should_tight_layout = True
    else:
        fig = ax.figure
        should_tight_layout = False

    norm = Normalize(vmin=float(np.nanmin(predictions_grid)),
                     vmax=float(np.nanmax(predictions_grid)))

    surf = ax.plot_surface(
        x_grid, y_grid, predictions_grid,
        cmap=cmap, norm=norm, alpha=alpha,
        edgecolor='none', antialiased=True
    )

    cbar = fig.colorbar(surf, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label(output_label, rotation=270, labelpad=20)

    # Overlay experimental points
    if (exp_x is not None and exp_y is not None
            and exp_output is not None and len(exp_x) > 0):
        ax.scatter(
            exp_x, exp_y, exp_output,
            c='white', edgecolors='black', s=80, marker='o',
            label='Experiments', depthshade=True, zorder=5
        )

    # Overlay suggestion points at the max prediction height for visibility
    if suggest_x is not None and suggest_y is not None and len(suggest_x) > 0:
        suggest_z = np.full_like(suggest_x, float(np.nanmax(predictions_grid)))
        ax.scatter(
            suggest_x, suggest_y, suggest_z,
            c='black', s=120, marker='*',
            label='Suggestions', depthshade=True, zorder=6
        )

    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.set_zlabel(output_label)
    ax.set_title(title)

    if ((exp_x is not None and len(exp_x) > 0)
            or (suggest_x is not None and len(suggest_x) > 0)):
        ax.legend(loc='upper left')

    if should_tight_layout:
        fig.tight_layout()

    return fig, ax, cbar


def create_uncertainty_surface_plot(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    uncertainty_grid: np.ndarray,
    x_var: str,
    y_var: str,
    exp_x: Optional[np.ndarray] = None,
    exp_y: Optional[np.ndarray] = None,
    suggest_x: Optional[np.ndarray] = None,
    suggest_y: Optional[np.ndarray] = None,
    cmap: str = 'Reds',
    alpha: float = 0.9,
    figsize: Tuple[float, float] = (10, 8),
    dpi: int = 100,
    title: str = "3D Uncertainty Surface (Standard Deviation)",
    ax: Optional[Any] = None
) -> Tuple[Figure, Any, Any]:
    """
    Create 3D surface plot of posterior uncertainty.

    Like the 2D uncertainty contour plot, but renders the prediction
    standard deviation on the Z axis as a 3D surface.

    Args:
        x_grid: X-axis meshgrid values (2D array)
        y_grid: Y-axis meshgrid values (2D array)
        uncertainty_grid: Posterior standard deviations on grid (2D array)
        x_var: X variable name for axis label
        y_var: Y variable name for axis label
        exp_x: Experimental X values to overlay (optional)
        exp_y: Experimental Y values to overlay (optional)
        suggest_x: Suggested X values to overlay (optional)
        suggest_y: Suggested Y values to overlay (optional)
        cmap: Matplotlib colormap name (default: 'Reds')
        alpha: Surface transparency (0=transparent, 1=opaque)
        figsize: Figure size (width, height) in inches
        dpi: Resolution
        title: Plot title
        ax: Existing 3D axes (creates new if None)

    Returns:
        Tuple of (Figure, Axes3D, Colorbar)

    Example:
        >>> X, Y = np.meshgrid(x_range, y_range)
        >>> _, std = model.predict(grid, return_std=True)
        >>> unc = std.reshape(X.shape)
        >>> fig, ax, cbar = create_uncertainty_surface_plot(X, Y, unc, 'temp', 'pressure')
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib.colors import Normalize

    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        should_tight_layout = True
    else:
        fig = ax.figure
        should_tight_layout = False

    norm = Normalize(vmin=float(np.nanmin(uncertainty_grid)),
                     vmax=float(np.nanmax(uncertainty_grid)))

    surf = ax.plot_surface(
        x_grid, y_grid, uncertainty_grid,
        cmap=cmap, norm=norm, alpha=alpha,
        edgecolor='none', antialiased=True
    )

    cbar = fig.colorbar(surf, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Posterior Standard Deviation', rotation=270, labelpad=20)

    # Overlay experimental points at z=0 for reference
    if exp_x is not None and exp_y is not None and len(exp_x) > 0:
        exp_z = np.zeros_like(exp_x)
        ax.scatter(
            exp_x, exp_y, exp_z,
            c='white', edgecolors='black', s=80, marker='o',
            label='Experiments', depthshade=True, zorder=5
        )

    # Overlay suggestion points
    if suggest_x is not None and suggest_y is not None and len(suggest_x) > 0:
        suggest_z = np.full_like(suggest_x, float(np.nanmax(uncertainty_grid)))
        ax.scatter(
            suggest_x, suggest_y, suggest_z,
            c='black', s=120, marker='*',
            label='Suggestions', depthshade=True, zorder=6
        )

    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.set_zlabel('Uncertainty (σ)')
    ax.set_title(title)

    if ((exp_x is not None and len(exp_x) > 0)
            or (suggest_x is not None and len(suggest_x) > 0)):
        ax.legend(loc='upper left')

    if should_tight_layout:
        fig.tight_layout()

    return fig, ax, cbar
