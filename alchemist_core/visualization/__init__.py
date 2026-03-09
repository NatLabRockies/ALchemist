"""
Visualization module for ALchemist.

Pure plotting functions with no session or UI dependencies.
All functions return matplotlib Figure/Axes objects for maximum flexibility.
"""

from alchemist_core.visualization.plots import (
    create_parity_plot,
    create_contour_plot,
    create_surface_plot,
    create_slice_plot,
    create_voxel_plot,
    create_metrics_plot,
    create_qq_plot,
    create_calibration_plot,
    create_regret_plot,
    create_probability_of_improvement_plot,
    create_uncertainty_contour_plot,
    create_uncertainty_surface_plot,
    create_uncertainty_voxel_plot,
    create_acquisition_voxel_plot,
    create_pareto_plot,
    create_hypervolume_convergence_plot,
)

from alchemist_core.visualization.helpers import (
    check_matplotlib,
    compute_z_scores,
    compute_calibration_metrics,
    generate_subplot_labels,
    resolve_subplot_labels,
    annotate_subplot_label,
)

__all__ = [
    'create_parity_plot',
    'create_contour_plot',
    'create_surface_plot',
    'create_slice_plot',
    'create_voxel_plot',
    'create_metrics_plot',
    'create_qq_plot',
    'create_calibration_plot',
    'create_regret_plot',
    'create_probability_of_improvement_plot',
    'create_uncertainty_contour_plot',
    'create_uncertainty_surface_plot',
    'create_uncertainty_voxel_plot',
    'create_acquisition_voxel_plot',
    'create_pareto_plot',
    'create_hypervolume_convergence_plot',
    'check_matplotlib',
    'compute_z_scores',
    'compute_calibration_metrics',
    'generate_subplot_labels',
    'resolve_subplot_labels',
    'annotate_subplot_label',
]
