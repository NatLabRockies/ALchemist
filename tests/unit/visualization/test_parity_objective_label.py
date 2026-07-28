import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
from alchemist_core.visualization.plots import create_parity_plot


def test_parity_axis_label_uses_provided_objective_label():
    y = np.array([1.0, 2.0, 3.0])
    fig, ax = create_parity_plot(y, y, show_metrics=False, objective_label="carbonyl (a.u.)")
    assert "carbonyl (a.u.)" in ax.get_xlabel()
    assert "carbonyl (a.u.)" in ax.get_ylabel()


def test_parity_axis_label_defaults_when_no_label():
    y = np.array([1.0, 2.0, 3.0])
    fig, ax = create_parity_plot(y, y, show_metrics=False)
    assert ax.get_xlabel() == "Actual Values"
    assert ax.get_ylabel() == "Predicted Values"
