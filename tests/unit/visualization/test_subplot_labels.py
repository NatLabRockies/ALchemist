"""
Unit tests for subplot annotation label feature.

Tests generate_subplot_labels, resolve_subplot_labels, annotate_subplot_label,
and the subplot_label parameter on low-level create_*_plot functions.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from alchemist_core.visualization.helpers import (
    generate_subplot_labels,
    resolve_subplot_labels,
    annotate_subplot_label,
)
from alchemist_core.visualization.plots import (
    create_parity_plot,
    create_slice_plot,
    create_contour_plot,
    create_metrics_plot,
    create_qq_plot,
    create_calibration_plot,
    create_regret_plot,
)


# ---------------------------------------------------------------------------
# generate_subplot_labels
# ---------------------------------------------------------------------------

class TestGenerateSubplotLabels:
    def test_default_format(self):
        assert generate_subplot_labels(3) == ["(a)", "(b)", "(c)"]

    def test_uppercase_format(self):
        assert generate_subplot_labels(2, fmt="{ALPHA})") == ["A)", "B)"]

    def test_numeric_format(self):
        assert generate_subplot_labels(4, fmt="({num})") == ["(1)", "(2)", "(3)", "(4)"]

    def test_single(self):
        assert generate_subplot_labels(1) == ["(a)"]

    def test_zero(self):
        assert generate_subplot_labels(0) == []


# ---------------------------------------------------------------------------
# resolve_subplot_labels
# ---------------------------------------------------------------------------

class TestResolveSubplotLabels:
    def test_none_returns_none(self):
        assert resolve_subplot_labels(None, 3) is None

    def test_false_returns_none(self):
        assert resolve_subplot_labels(False, 3) is None

    def test_true_auto_generates(self):
        result = resolve_subplot_labels(True, 3)
        assert result == ["(a)", "(b)", "(c)"]

    def test_true_custom_fmt(self):
        result = resolve_subplot_labels(True, 2, fmt="{ALPHA}.")
        assert result == ["A.", "B."]

    def test_string_wraps(self):
        result = resolve_subplot_labels("(x)", 1)
        assert result == ["(x)"]

    def test_list_passthrough(self):
        labels = ["fig 1", "fig 2"]
        assert resolve_subplot_labels(labels, 2) == labels

    def test_list_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="3 items.*2 axes"):
            resolve_subplot_labels(["a", "b", "c"], 2)

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="bool, str, or list"):
            resolve_subplot_labels(42, 1)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# annotate_subplot_label
# ---------------------------------------------------------------------------

class TestAnnotateSubplotLabel:
    def test_adds_text_2d(self):
        fig, ax = plt.subplots()
        annotate_subplot_label(ax, "(a)")
        texts = [t for t in ax.texts if t.get_text() == "(a)"]
        assert len(texts) == 1
        assert texts[0].get_fontweight() == "bold"
        plt.close(fig)

    def test_adds_text_3d(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        annotate_subplot_label(ax, "(b)")
        # Axes3D stores text2D calls in .texts as well
        texts = [t for t in ax.texts if t.get_text() == "(b)"]
        assert len(texts) == 1
        plt.close(fig)

    def test_upper_right_location(self):
        fig, ax = plt.subplots()
        annotate_subplot_label(ax, "(z)", loc="upper right")
        texts = [t for t in ax.texts if t.get_text() == "(z)"]
        assert len(texts) == 1
        plt.close(fig)

    def test_custom_fontsize(self):
        fig, ax = plt.subplots()
        annotate_subplot_label(ax, "(c)", fontsize=16)
        text_obj = [t for t in ax.texts if t.get_text() == "(c)"][0]
        assert text_obj.get_fontsize() == 16
        plt.close(fig)


# ---------------------------------------------------------------------------
# subplot_label on low-level plot functions
# ---------------------------------------------------------------------------

class TestPlotFunctionsSubplotLabel:
    """Verify the subplot_label parameter threads through to each plot."""

    def test_parity_plot_label(self):
        y_true = np.array([1, 2, 3, 4, 5], dtype=float)
        y_pred = np.array([1.1, 2.0, 2.9, 4.2, 5.0])
        fig, ax = create_parity_plot(y_true, y_pred, subplot_label="(a)")
        assert any(t.get_text() == "(a)" for t in ax.texts)
        plt.close(fig)

    def test_parity_plot_no_label(self):
        y_true = np.array([1, 2, 3], dtype=float)
        y_pred = np.array([1, 2, 3], dtype=float)
        fig, ax = create_parity_plot(y_true, y_pred)
        assert not any(t.get_text().startswith("(") for t in ax.texts)
        plt.close(fig)

    def test_slice_plot_label(self):
        x = np.linspace(0, 1, 50)
        y = np.sin(x)
        fig, ax = create_slice_plot(
            x_values=x, predictions=y, x_var="x",
            subplot_label="(b)"
        )
        assert any(t.get_text() == "(b)" for t in ax.texts)
        plt.close(fig)

    def test_contour_plot_label(self):
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        X, Y = np.meshgrid(x, y)
        Z = X + Y
        fig, ax, _ = create_contour_plot(
            x_grid=X, y_grid=Y, predictions_grid=Z,
            x_var="x", y_var="y", subplot_label="(c)"
        )
        assert any(t.get_text() == "(c)" for t in ax.texts)
        plt.close(fig)

    def test_metrics_plot_label(self):
        fig, ax = create_metrics_plot(
            training_sizes=np.arange(5),
            metric_values=np.random.rand(5),
            metric_name="RMSE",
            subplot_label="(d)"
        )
        assert any(t.get_text() == "(d)" for t in ax.texts)
        plt.close(fig)

    def test_qq_plot_label(self):
        z_scores = np.random.randn(100)
        fig, ax = create_qq_plot(z_scores=z_scores, subplot_label="(e)")
        assert any(t.get_text() == "(e)" for t in ax.texts)
        plt.close(fig)

    def test_calibration_plot_label(self):
        nominal = np.array([0.1, 0.2, 0.5, 0.8, 0.9])
        empirical = np.array([0.12, 0.18, 0.52, 0.79, 0.88])
        fig, ax = create_calibration_plot(
            nominal_probs=nominal,
            empirical_coverage=empirical,
            subplot_label="(f)"
        )
        assert any(t.get_text() == "(f)" for t in ax.texts)
        plt.close(fig)

    def test_regret_plot_label(self):
        fig, ax = create_regret_plot(
            iterations=np.arange(10),
            observed_values=np.random.rand(10),
            subplot_label="(g)"
        )
        assert any(t.get_text() == "(g)" for t in ax.texts)
        plt.close(fig)
