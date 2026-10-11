"""
get_data_summary must stay JSON-safe for tiny datasets.

Regression for the web app's first Add Point: with one data row the sample
standard deviation is NaN, Starlette refuses to serialize NaN (400 "Out of
range float values are not JSON compliant"), and the UI read that 400 as
"Session not found" and cleared the session.
"""

import json
import math

import pytest

from alchemist_core import OptimizationSession


def _session(outputs):
    session = OptimizationSession()
    session.add_variable("x1", "real", bounds=(0.0, 1.0))
    session.add_variable("x2", "real", bounds=(0.0, 1.0))
    for i, y in enumerate(outputs):
        session.add_experiment(inputs={"x1": 0.1 * (i + 1), "x2": 0.05 * (i + 1)}, output=y)
    return session


def _assert_json_safe(summary):
    # allow_nan=False is what Starlette's JSONResponse uses.
    json.dumps(summary, allow_nan=False)


def test_zero_rows_summary():
    summary = _session([]).get_data_summary()
    assert summary == {"n_experiments": 0, "has_data": False}
    _assert_json_safe(summary)


def test_one_row_summary_has_no_std():
    summary = _session([0.7]).get_data_summary()
    _assert_json_safe(summary)
    assert summary["n_experiments"] == 1
    assert summary["has_data"] is True
    stats = summary["target_stats"]
    assert stats["min"] == pytest.approx(0.7)
    assert stats["max"] == pytest.approx(0.7)
    assert stats["mean"] == pytest.approx(0.7)
    # Sample std (ddof=1) is undefined for one observation: None, not NaN.
    assert stats["std"] is None


def test_two_rows_summary_has_finite_std():
    summary = _session([0.5, 0.7]).get_data_summary()
    _assert_json_safe(summary)
    stats = summary["target_stats"]
    assert summary["n_experiments"] == 2
    assert stats["min"] == pytest.approx(0.5)
    assert stats["max"] == pytest.approx(0.7)
    assert stats["mean"] == pytest.approx(0.6)
    assert stats["std"] == pytest.approx(math.sqrt(0.02))


def test_two_identical_rows_summary_std_zero():
    summary = _session([0.5, 0.5]).get_data_summary()
    _assert_json_safe(summary)
    assert summary["target_stats"]["std"] == pytest.approx(0.0)
