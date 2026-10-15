# Visualization Module Documentation

**Module**: `alchemist_core.visualization`  
**Created**: January 8, 2026  
**Updated**: January 22, 2026  
**Status**: Complete (3x3 Grid Implemented)

---

## Overview

The visualization module provides a complete framework for visualizing Gaussian Process models in Bayesian optimization. It implements a systematic 3×3 grid of visualization types:

- **What to visualize**: Posterior Mean, Posterior Uncertainty, Acquisition Function
- **Dimensionality**: 1D Slice, 2D Contour, 3D Voxel

All functions are pure plotting utilities with no session or UI dependencies, returning matplotlib `Figure` and `Axes` objects for maximum flexibility.

## Complete 3×3 Visualization Grid

| Quantity | 1D Slice | 2D Contour | 3D Voxel |
|----------|----------|------------|----------|
| **Posterior Mean** | `create_slice_plot()` | `create_contour_plot()` | `create_voxel_plot()` |
| **Posterior Uncertainty** | `create_slice_plot()` (with σ bands) | `create_uncertainty_contour_plot()` | `create_uncertainty_voxel_plot()` |
| **Acquisition Function** | `create_slice_plot()` (reused) | `create_contour_plot()` (reused) | `create_acquisition_voxel_plot()` |

### Session API Methods

The `OptimizationSession` class provides high-level methods that wrap these plotting functions:

| Quantity | 1D | 2D | 3D |
|----------|----|----|-----|
| **Mean** | `plot_slice()` | `plot_contour()` | `plot_voxel()` |
| **Uncertainty** | `plot_slice(show_uncertainty=True)` | `plot_uncertainty_contour()` | `plot_uncertainty_voxel()` |
| **Acquisition** | `plot_acquisition_slice()` | `plot_acquisition_contour()` | `plot_acquisition_voxel()` |

## Architecture

```
alchemist_core/visualization/
├── __init__.py          # Module exports
├── plots.py             # Pure plotting functions (600+ lines)
├── helpers.py           # Utility functions
└── README.md            # This file
```

### Design Principles

1. **Pure Functions**: No session or UI dependencies
2. **Framework Agnostic**: Works with notebooks, scripts, desktop UI, web
3. **Flexible Embedding**: Optional `ax` parameter for existing figures
4. **Consistent API**: Similar signatures across all plot types
5. **Type Hints**: Full typing for better IDE support

---

## API Reference

### Plotting Functions

All plotting functions follow this pattern:
- Accept numpy arrays and parameters as input
- Return `(Figure, Axes)` tuple
- Accept optional `ax` parameter for embedding
- Apply `tight_layout()` only when creating new figure

#### `create_parity_plot()`

Create parity plot of actual vs predicted values.

```python
from alchemist_core.visualization import create_parity_plot

fig, ax = create_parity_plot(
    y_true=np.array([...]),
    y_pred=np.array([...]),
    y_std=np.array([...]),        # Optional
    sigma_multiplier=1.96,         # 95% CI
    show_error_bars=True,
    show_metrics=True,
    figsize=(8, 6),
    dpi=100,
    title=None,                    # Auto-generated
    ax=None                        # Create new if None
)
```

**Features**:
- Error bars with configurable sigma multiplier
- Automatic RMSE/MAE/R² calculation
- Parity line (y=x) reference
- Confidence interval labels (68%, 95%, 99%)

#### `create_contour_plot()`

Create 2D contour plot of model predictions.

```python
from alchemist_core.visualization import create_contour_plot

X, Y = np.meshgrid(x_range, y_range)
Z = predictions.reshape(X.shape)

fig, ax = create_contour_plot(
    x_grid=X,
    y_grid=Y,
    predictions_grid=Z,
    x_var='temperature',
    y_var='pressure',
    exp_x=exp_data_x,              # Optional overlay
    exp_y=exp_data_y,
    suggest_x=next_points_x,       # Optional suggestions
    suggest_y=next_points_y,
    cmap='viridis',
    figsize=(8, 6),
    dpi=100,
    title='Contour Plot',
    ax=None
)
```

**Features**:
- 20-level contour filling
- Colorbar with label
- Experimental points overlay (white circles)
- Suggestion points overlay (red stars)
- Automatic legend when overlays present

#### `create_slice_plot()`

Create 1D slice plot with uncertainty bands.

```python
from alchemist_core.visualization import create_slice_plot

fig, ax = create_slice_plot(
    x_values=np.linspace(0, 100, 100),
    predictions=y_mean,
    x_var='temperature',
    std=y_std,                     # Optional
    sigma_bands=[1.0, 2.0],        # Multiple bands
    exp_x=exp_data_x,              # Optional experiments
    exp_y=exp_data_y,
    figsize=(8, 6),
    dpi=100,
    title=None,
    ax=None
)
```

**Features**:
- Multiple uncertainty bands with proper z-stacking
- Sequential colormap (Blues) for band opacity
- Sigmoid-based alpha scaling
- Smart legend ordering (Prediction → bands → Experiments)
- Professional styling (colorblind-friendly colors)

#### `create_voxel_plot()`

Create 3D voxel plot of model predictions.

```python
from alchemist_core.visualization import create_voxel_plot

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
predictions = model.predict(grid)

fig, ax = create_voxel_plot(
    x_grid=X,
    y_grid=Y,
    z_grid=Z,
    predictions_grid=predictions.reshape(X.shape),
    x_var='temperature',
    y_var='pressure',
    z_var='flow_rate',
    exp_x=exp_x,                   # Optional overlays
    exp_y=exp_y,
    exp_z=exp_z,
    suggest_x=next_x,
    suggest_y=next_y,
    suggest_z=next_z,
    cmap='viridis',
    alpha=0.5,                     # Transparency
    use_log_scale=False,
    figsize=(10, 8),
    dpi=100,
    title='3D Voxel Plot',
    ax=None                        # Can pass existing 3D axes
)
```

**Features**:
- 3D scatter visualization with color mapping
- Adjustable transparency (alpha) to see interior
- Experimental and suggestion point overlays
- Colorbar with proper 3D positioning
- Automatic marker sizing based on grid resolution

#### `create_uncertainty_contour_plot()`

Create 2D contour plot of posterior uncertainty (standard deviation).

```python
from alchemist_core.visualization import create_uncertainty_contour_plot

X, Y = np.meshgrid(x_range, y_range)
_, std = model.predict(grid, return_std=True)
uncertainty = std.reshape(X.shape)

fig, ax, cbar = create_uncertainty_contour_plot(
    x_grid=X,
    y_grid=Y,
    uncertainty_grid=uncertainty,
    x_var='temperature',
    y_var='pressure',
    exp_x=exp_x,
    exp_y=exp_y,
    suggest_x=sugg_x,
    suggest_y=sugg_y,
    cmap='Reds',                   # Darker = more uncertain
    figsize=(8, 6),
    dpi=100,
    title='Posterior Uncertainty',
    ax=None
)
```

**Features**:
- Shows where model is most uncertain
- Default 'Reds' colormap (darker = higher uncertainty)
- Useful for identifying under-explored regions
- Experimental points show where data exists
- Can guide exploration strategies

**Use Cases**:
- Planning where to sample next (high uncertainty regions)
- Validating model coverage across variable space
- Identifying data-sparse regions

#### `create_uncertainty_voxel_plot()`

Create 3D voxel plot of posterior uncertainty.

```python
from alchemist_core.visualization import create_uncertainty_voxel_plot

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
_, std = model.predict(grid, return_std=True)
uncertainty = std.reshape(X.shape)

fig, ax = create_uncertainty_voxel_plot(
    x_grid=X,
    y_grid=Y,
    z_grid=Z,
    uncertainty_grid=uncertainty,
    x_var='temperature',
    y_var='pressure',
    z_var='flow_rate',
    exp_x=exp_x,
    exp_y=exp_y,
    exp_z=exp_z,
    suggest_x=sugg_x,
    suggest_y=sugg_y,
    suggest_z=sugg_z,
    cmap='Reds',
    alpha=0.5,
    figsize=(10, 8),
    dpi=100,
    title='3D Posterior Uncertainty',
    ax=None
)
```

**Features**:
- 3D visualization of model uncertainty
- Shows data-sparse volumes in 3D space
- Adjustable transparency for interior visibility
- Helps plan 3D exploration strategies

**Performance Note**: O(N³) evaluations - use `grid_resolution=10-15` for reasonable speed.

#### `create_acquisition_voxel_plot()`

Create 3D voxel plot of acquisition function values.

```python
from alchemist_core.visualization import create_acquisition_voxel_plot

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
acq_values = evaluate_acquisition(model, grid, acq_func='ei')
acq_grid = acq_values.reshape(X.shape)

fig, ax = create_acquisition_voxel_plot(
    x_grid=X,
    y_grid=Y,
    z_grid=Z,
    acquisition_grid=acq_grid,
    x_var='temperature',
    y_var='pressure',
    z_var='flow_rate',
    exp_x=exp_x,
    exp_y=exp_y,
    exp_z=exp_z,
    suggest_x=sugg_x,              # Should align with hot spots
    suggest_y=sugg_y,
    suggest_z=sugg_z,
    cmap='hot',
    alpha=0.5,
    use_log_scale=False,           # Auto-enabled for logei/logpi
    figsize=(10, 8),
    dpi=100,
    title='3D Acquisition Function (EI)',
    ax=None
)
```

**Features**:
- Visualizes where to sample next in 3D
- "Hot spots" indicate promising regions
- Use with EI, PI, UCB, or other acquisition functions
- Suggestions overlay to show algorithm decisions
- Helps understand exploration-exploitation tradeoff in 3D

**Colormap Recommendations**:
- 'hot': Good for acquisition (bright = high value)
- 'plasma': Alternative with good perceptual uniformity
- 'viridis': Also works, but 'hot' is more intuitive

#### `create_metrics_plot()`

Create learning curve showing metric vs training size.

```python
from alchemist_core.visualization import create_metrics_plot

fig, ax = create_metrics_plot(
    training_sizes=np.array([5, 6, 7, ...]),
    metric_values=np.array([0.15, 0.12, 0.10, ...]),
    metric_name='rmse',            # 'rmse', 'mae', 'r2', 'mape'
    figsize=(8, 6),
    dpi=100,
    ax=None
)
```

**Features**:
- Line plot with markers
- Automatic axis labels for each metric type
- Grid for readability

#### `create_qq_plot()`

Create Q-Q plot of standardized residuals.

```python
from alchemist_core.visualization import create_qq_plot

fig, ax = create_qq_plot(
    z_scores=np.array([...]),      # Standardized residuals
    figsize=(8, 6),
    dpi=100,
    show_confidence_bands=True,    # For samples < 100
    title='Q-Q Plot',
    ax=None
)
```

**Features**:
- Comparison to standard normal distribution
- Perfect calibration reference line
- Approximate 95% CI bands for small samples
- Automatic quantile computation

#### `create_calibration_plot()`

Create calibration curve (reliability diagram).

```python
from alchemist_core.visualization import create_calibration_plot

fig, ax = create_calibration_plot(
    nominal_probs=np.array([0.68, 0.95, 0.99]),
    empirical_coverage=np.array([0.70, 0.94, 0.98]),
    figsize=(8, 6),
    dpi=100,
    title='Calibration Curve',
    ax=None
)
```

**Features**:
- Nominal vs empirical coverage
- Perfect calibration reference line
- Fixed [0, 1] axis limits

---

### Helper Functions

#### `check_matplotlib()`

Check if matplotlib is available, raise ImportError if not.

```python
from alchemist_core.visualization import check_matplotlib

check_matplotlib()  # Raises ImportError if not installed
```

#### `compute_z_scores()`

Compute standardized residuals (z-scores).

```python
from alchemist_core.visualization import compute_z_scores

z = compute_z_scores(
    y_true=np.array([...]),
    y_pred=np.array([...]),
    y_std=np.array([...])
)
# z = (y_true - y_pred) / y_std
```

**Note**: Adds small epsilon (1e-10) to avoid division by zero.

#### `compute_calibration_metrics()`

Compute nominal vs empirical coverage for calibration curves.

```python
from alchemist_core.visualization import compute_calibration_metrics

nominal, empirical = compute_calibration_metrics(
    y_true=np.array([...]),
    y_pred=np.array([...]),
    y_std=np.array([...]),
    prob_levels=None               # Default: np.arange(0.10, 1.00, 0.05)
)
```

**Returns**: `(nominal_probs, empirical_coverage)` tuple

---

## Usage Examples

### Basic Usage

```python
import numpy as np
from alchemist_core.visualization import create_parity_plot

# Synthetic data
y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

# Create plot
fig, ax = create_parity_plot(y_true, y_pred)

# Save or display
fig.savefig('parity.png', dpi=300, bbox_inches='tight')
# OR in notebook: fig  (auto-displays)
```

### Embedding in Existing Figure

```python
import matplotlib.pyplot as plt
from alchemist_core.visualization import create_parity_plot, create_slice_plot

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot on existing axes
create_parity_plot(y_true, y_pred, ax=ax1)
create_slice_plot(x_values, predictions, 'temperature', ax=ax2)

fig.tight_layout()
plt.show()
```

### Complete Workflow

```python
from alchemist_core.visualization import (
    create_parity_plot,
    create_contour_plot,
    compute_z_scores,
    create_qq_plot
)

# 1. Get CV results (from model)
y_true = cv_results['y_true']
y_pred = cv_results['y_pred']
y_std = cv_results['y_std']

# 2. Create parity plot
fig1, _ = create_parity_plot(y_true, y_pred, y_std)
fig1.savefig('parity.png')

# 3. Check calibration with Q-Q plot
z = compute_z_scores(y_true, y_pred, y_std)
fig2, _ = create_qq_plot(z)
fig2.savefig('qq.png')

# 4. Create contour plot
X, Y = np.meshgrid(x_range, y_range)
Z = model.predict(grid_points).reshape(X.shape)
fig3, _ = create_contour_plot(X, Y, Z, 'x', 'y')
fig3.savefig('contour.png')
```

---

## Integration Points

### Session API Integration

The Session API delegates to these functions:

```python
# alchemist_core/session.py

def plot_parity(self, ...):
    from alchemist_core.visualization.plots import create_parity_plot
    
    # Get data from model
    cv_results = self._check_cv_results(use_calibrated)
    
    # Delegate to pure function
    fig, ax = create_parity_plot(
        y_true=cv_results['y_true'],
        y_pred=cv_results['y_pred'],
        y_std=cv_results['y_std'],
        ...
    )
    return fig
```

### UI Integration

The Desktop UI can call Session API or use functions directly:

```python
# ui/visualizations.py

def plot_parity(self):
    # Option 1: Call Session API (recommended)
    fig = self.session.plot_parity(...)
    
    # Option 2: Call pure function directly (if needed)
    from alchemist_core.visualization.plots import create_parity_plot
    fig, ax = create_parity_plot(...)
    
    # Embed in canvas
    self._embed_figure(fig)
```

### REST API Integration

The REST API can return plot data or rendered images:

```python
# api/routers/visualizations.py

@router.get("/parity")
async def get_parity_plot(session_id: str):
    session = get_session(session_id)
    
    # Get figure from session
    fig = session.plot_parity()
    
    # Option 1: Return as PNG image
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    return StreamingResponse(buf, media_type='image/png')
    
    # Option 2: Return data for client-side rendering
    # Extract data from figure and return JSON
```

---

## Testing

Unit tests cover all plotting functions:

```bash
pytest tests/unit/visualization/test_plots.py -v
```

**Test Coverage**:
- ✅ All 6 plotting functions
- ✅ Helper functions
- ✅ Error handling
- ✅ Edge cases (empty arrays, single points)
- ✅ Embedding in existing axes
- ✅ Custom styling options

**Results**: 21/21 tests passing

---

## Future Enhancements

### Potential Additions

1. **Feature Importance Plot**: Bar chart of feature importance
2. **Pareto Front Plot**: For multi-objective optimization
3. **Acquisition Function Plot**: Show acquisition landscape
4. **Residual Plot**: Residuals vs predicted values
5. **Correlation Matrix**: Heatmap of variable correlations

### Styling Improvements

1. **StyleConfig Class**: Centralized styling configuration
2. **Theme Support**: Light/dark themes
3. **Publication Mode**: IEEE/Nature/Science style presets
4. **Interactive Plots**: Plotly backend support

### Performance

1. **Lazy Imports**: Defer matplotlib import until needed
2. **Caching**: Cache rendered figures for repeated requests
3. **Vectorization**: Optimize meshgrid operations

---

## Migration Guide

For developers migrating from old code:

### Before (UI code)
```python
# ui/visualizations.py - 100 lines of plotting code
def plot_parity(self):
    # Manual CV computation
    # Direct matplotlib calls
    # Embedded in self.ax
```

### After (with module)
```python
# ui/visualizations.py - 10 lines
def plot_parity(self):
    fig = self.session.plot_parity(...)
    self._embed_figure(fig)
```

**Benefits**:
- 90% code reduction
- Consistent behavior
- Easier to maintain
- Testable in isolation

---

## Dependencies

**Required**:
- `numpy`
- `matplotlib`
- `scipy` (for stats functions)
- `scikit-learn` (for metrics in parity plot)

**Optional**:
- None (all dependencies already in ALchemist)

---

## Performance Characteristics

| Function | Typical Time | Notes |
|----------|-------------|-------|
| `create_parity_plot` | ~50ms | Fast, O(n) scatter |
| `create_contour_plot` | ~200ms | Depends on grid resolution |
| `create_slice_plot` | ~100ms | Multiple fill_between calls |
| `create_metrics_plot` | ~30ms | Simple line plot |
| `create_qq_plot` | ~40ms | Sorting dominates |
| `create_calibration_plot` | ~30ms | Simple line plot |

*Benchmarked on M1 Mac with typical data sizes*

---

## Version History

- **v0.1.0** (Jan 8, 2026): Initial implementation
  - 6 plotting functions
  - 3 helper functions
  - 21 unit tests
  - Complete documentation

---

## Contact

Questions or issues? See main ALchemist documentation or contact:
- Email: ccoatney@nlr.gov
- GitHub: https://github.com/NatLabRockies/ALchemist
