"""The Q-Q plot's x positions and band (T5), and the one helper every drawer uses."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest
from scipy import stats

from alchemist_core.visualization.helpers import qq_confidence_band, qq_theoretical_quantiles
from alchemist_core.visualization.plots import create_qq_plot

REPO = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("n", [5, 10, 25, 107])
def test_positions_are_blom_and_symmetric(n):
    q = qq_theoretical_quantiles(n)
    expected = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
    np.testing.assert_allclose(q, expected)
    np.testing.assert_allclose(q, -q[::-1], atol=1e-12)


def test_extremes_depend_on_n():
    # linspace(0.01, 0.99, n) puts them at +/-2.33 for every n.
    assert qq_theoretical_quantiles(10)[-1] == pytest.approx(1.547, abs=0.01)
    assert qq_theoretical_quantiles(100)[-1] == pytest.approx(2.508, abs=0.01)


def test_positions_estimate_expected_normal_order_statistics():
    rng = np.random.default_rng(0)
    n = 20
    mean_sorted = np.sort(rng.standard_normal((20000, n)), axis=1).mean(axis=0)
    assert np.abs(qq_theoretical_quantiles(n) - mean_sorted).max() < 0.03


def test_band_is_wider_in_the_tails_and_contains_the_positions():
    n = 30
    lo, hi = qq_confidence_band(n)
    q = qq_theoretical_quantiles(n)
    assert (lo < q).all() and (q < hi).all()
    width = hi - lo
    assert width[0] > 1.5 * width[n // 2]
    assert width[-1] > 1.5 * width[n // 2]


def test_band_has_its_nominal_pointwise_coverage():
    rng = np.random.default_rng(1)
    n = 25
    lo, hi = qq_confidence_band(n, level=0.95)
    z = np.sort(rng.standard_normal((40000, n)), axis=1)
    coverage = ((z >= lo) & (z <= hi)).mean(axis=0)
    np.testing.assert_allclose(coverage, 0.95, atol=0.01)


def test_plot_draws_those_positions_and_a_varying_band():
    z = np.random.default_rng(2).standard_normal(20)
    fig, ax = create_qq_plot(z)
    points = ax.collections[0].get_offsets()
    np.testing.assert_allclose(np.asarray(points)[:, 0], qq_theoretical_quantiles(20))
    np.testing.assert_allclose(np.asarray(points)[:, 1], np.sort(z))

    band = ax.collections[1].get_paths()[0].vertices
    # fill_between's outline: the lower edge runs forward, so read its lowest y per x.
    lo, _ = qq_confidence_band(20)
    assert band[:, 1].min() == pytest.approx(lo.min(), abs=1e-6)
    matplotlib.pyplot.close(fig)


@pytest.mark.parametrize(
    "path",
    ["alchemist_core/visualization/plots.py", "api/routers/visualizations.py", "ui/visualizations.py"],
)
def test_no_drawer_keeps_its_own_copy(path):
    source = (REPO / path).read_text(encoding="utf-8")
    assert "linspace(0.01, 0.99" not in source
    assert "1.96 / np.sqrt" not in source


def test_desktop_parity_fallback_has_no_leaky_cv_of_its_own():
    source = (REPO / "ui/visualizations.py").read_text(encoding="utf-8")
    assert "load_state_dict" not in source
    assert "optimized_kernel" not in source
    assert "_cross_validate(" in source
