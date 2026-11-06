"""F1: the "Calibrate Uncertainty" control had no effect, because calibration runs on every train.

Predicted σ is always scaled by the cross-validation factor (display only;
acquisition uses the raw model). The control is gone, and a caller who passes
``calibration_enabled=False`` is told it does nothing.
"""
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alchemist_core.session import OptimizationSession

REPO = Path(__file__).resolve().parents[3]


def _session():
    s = OptimizationSession()
    s.add_variable("x1", "real", min=0.0, max=1.0)
    s.add_variable("x2", "real", min=0.0, max=1.0)
    rng = np.random.default_rng(0)
    for x1, x2 in rng.random((12, 2)):
        s.add_experiment({"x1": float(x1), "x2": float(x2)}, output=float(np.sin(3 * x1) + x2 + rng.normal(0, 0.05)))
    return s


@pytest.mark.parametrize("backend", ["sklearn", "botorch"])
def test_the_flag_changes_nothing_a_user_can_see(backend):
    grid = pd.DataFrame({"x1": [0.2, 0.7], "x2": [0.4, 0.9]})
    stds = []
    for kwargs in ({}, {"calibration_enabled": False}, {"calibration_enabled": True}):
        s = _session()
        s.train_model(backend=backend, kernel="Matern", **kwargs)
        assert s.model.calibration_enabled is True
        stds.append(s.model.predict(grid, return_std=True)[1])
    np.testing.assert_allclose(stds[1], stds[0], rtol=1e-6)
    np.testing.assert_allclose(stds[2], stds[0], rtol=1e-6)


def test_false_is_refused_out_loud(caplog):
    s = _session()
    with caplog.at_level(logging.WARNING, logger="alchemist_core.session"):
        s.train_model(backend="sklearn", kernel="RBF", calibration_enabled=False)
    assert "calibration_enabled=False has no effect" in caplog.text


@pytest.mark.parametrize("kwargs", [{}, {"calibration_enabled": True}])
def test_no_warning_when_the_flag_agrees_with_what_happens(kwargs, caplog):
    s = _session()
    with caplog.at_level(logging.WARNING, logger="alchemist_core.session"):
        s.train_model(backend="sklearn", kernel="RBF", **kwargs)
    assert "calibration_enabled" not in caplog.text


def test_saved_false_config_replays_with_calibration_still_on():
    """Every web session saved before this fix holds calibration_enabled=False."""
    s = _session()
    s.train_model(backend="botorch", kernel="Matern", calibration_enabled=False)
    config = s.get_last_training_config()
    assert config["calibration_enabled"] is False
    r = _session()
    r.train_model(**config)
    assert r.model.calibration_enabled is True


def test_desktop_panel_has_no_calibrate_checkbox():
    source = (REPO / "ui" / "gpr_panel.py").read_text(encoding="utf-8")
    assert "calibrate_uncertainty" not in source
    assert "Calibrate Uncertainty" not in source


def test_desktop_advice_does_not_name_the_removed_checkbox():
    source = (REPO / "ui" / "notifications.py").read_text(encoding="utf-8")
    assert not re.search(r"Calibrate Uncertainty' checked", source)
