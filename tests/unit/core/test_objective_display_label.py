from alchemist_core.session import OptimizationSession


def _session():
    s = OptimizationSession()
    s.add_variable("x", "real", bounds=(0.0, 10.0))
    return s


def test_display_label_falls_back_to_raw_name():
    s = _session()
    assert s.objective_display_label("Output") == "Output"


def test_display_label_with_label_and_unit():
    s = _session()
    s.set_objective_metadata({"Output": {"label": "carbonyl", "unit": "a.u."}})
    assert s.objective_display_label("Output") == "carbonyl (a.u.)"


def test_display_label_with_label_no_unit():
    s = _session()
    s.set_objective_metadata({"Output": {"label": "carbonyl", "unit": None}})
    assert s.objective_display_label("Output") == "carbonyl"


def test_display_label_empty_label_falls_back():
    s = _session()
    s.set_objective_metadata({"Output": {"label": None, "unit": "a.u."}})
    assert s.objective_display_label("Output") == "Output"


def test_plot_parity_uses_objective_label(monkeypatch):
    import numpy as np
    import pytest
    pytest.importorskip("matplotlib")
    s = OptimizationSession()
    s.add_variable("x", "real", bounds=(0.0, 10.0))
    # seed data + train so plot_parity has cv_results
    for i in range(6):
        s.add_experiment({"x": float(i)}, output=float(i) * 0.1)
    s.set_objective_metadata({"Output": {"label": "carbonyl", "unit": "a.u."}})
    s.train_model(backend="sklearn", kernel="rbf")
    fig = s.plot_parity(show_metrics=False)
    ax = fig.axes[0]
    assert "carbonyl (a.u.)" in ax.get_xlabel()
