# 3D Surface & Uncertainty Plots

Surface plots provide an intuitive three-dimensional view of the surrogate model's predictions and uncertainty estimates across two input dimensions. They complement the standard 2D contour plots, offering a more visually expressive way to communicate model landscapes.

---

## 3D Prediction Surface

The **prediction surface** renders the surrogate model's mean prediction as a continuous 3D surface over a two-variable grid, with experimental observations and (optionally) acquisition suggestions overlaid as scatter points.

**When to use:**
- Communicating optimization results to stakeholders who find 3D plots more intuitive
- Identifying curvature and interaction structure between two key variables
- Presentations and publications

The surface is colored by predicted output value (using a diverging colormap), and experiment points are shown as markers above or below the surface at their actual output values.

---

## 3D Uncertainty Surface

The **uncertainty surface** shows the surrogate model's predictive standard deviation (epistemic uncertainty) across the variable space — the same information shown in the 2D uncertainty contour, but rendered as a 3D surface.

**When to use:**
- Identifying regions of high uncertainty that are candidates for active learning
- Verifying that the model is well-calibrated (low uncertainty near data, high away from data)
- Communicating where the model is most confident vs. most uncertain

---

## Web UI

After training a model, open the **Visualizations** panel and select:

- **3D Surface Plot** — predicted mean surface
- **3D Uncertainty Surface** — predictive uncertainty surface

Use the variable dropdowns to select which two variables appear on the X and Y axes. All other variables are held at their midpoints.

---

## Python API

```python
from alchemist_core import OptimizationSession
from alchemist_core.visualization.plots import create_surface_plot, create_uncertainty_surface_plot
import numpy as np

session = OptimizationSession()
# ... load variables, data, train model ...

# Get grid data from session
grid_data = session.compute_contour_data(x_var="Temperature", y_var="Pressure")

# 3D prediction surface
fig, ax, cbar = create_surface_plot(
    x_grid=grid_data["x_grid"],
    y_grid=grid_data["y_grid"],
    predictions_grid=grid_data["predictions_grid"],
    x_var="Temperature",
    y_var="Pressure",
    output_label="Predicted Yield",
    exp_x=grid_data.get("exp_x"),
    exp_y=grid_data.get("exp_y"),
    exp_output=grid_data.get("exp_output"),
)
fig.savefig("surface_plot.png", dpi=150)

# 3D uncertainty surface
fig2, ax2, cbar2 = create_uncertainty_surface_plot(
    x_grid=grid_data["x_grid"],
    y_grid=grid_data["y_grid"],
    uncertainty_grid=grid_data["uncertainty_grid"],
    x_var="Temperature",
    y_var="Pressure",
)
fig2.savefig("uncertainty_surface.png", dpi=150)
```

---

## Interpretation

### Prediction Surface
- **Peaks** (high values) indicate predicted optimum regions — potential targets for the next experiment
- **Valleys** may represent unfavorable conditions or constraints
- **Curvature** reveals quadratic structure — steep surfaces suggest strong variable effects
- **Experimental points** shown above/below the surface reveal how well the model fits the data

### Uncertainty Surface
- **High peaks** near unexplored regions → model is uncertain, good candidates for exploration
- **Low, flat regions** near your data → model is confident here
- **If uncertainty is high everywhere:** You may need more experiments before the model is reliable

---

## Related Plots

- [Contour Plot](contour_plot.md) — 2D top-down view of model predictions
- [Calibration Curve](calibration_curve.md) — validates model uncertainty calibration
