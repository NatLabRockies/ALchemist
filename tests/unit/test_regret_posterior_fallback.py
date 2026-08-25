"""
Regression tests for posterior-prediction fallback in regret plots.

Background (see session.py _compute_posterior_predictions):
When plot_regret refits a fresh GP on each data prefix df[0:i], small /
ill-conditioned prefixes can raise botorch ModelFittingError under the
default ``normalize``/``standardize`` transforms. Previously the failure
handler left that iteration's predicted mean/std as NaN, producing a visible
gap in the regret plot. The fallback ladder must recover a genuine GP
prediction instead of leaving NaN.

The catalyst fixture (tests/catalyst_experiments.csv) has a categorical
variable, so the covar_module is an AdditiveKernel and the reuse-hyperparameter
fast-path is skipped -> every iteration re-optimizes. With botorch 0.17 /
gpytorch 1.15 the 11- and 12-experiment prefixes fail to fit under
normalize/standardize but fit successfully with transforms disabled.
"""

import json
import os

import numpy as np
import pytest

from alchemist_core.session import OptimizationSession

# Skip cleanly if botorch backend isn't available in the environment.
botorch = pytest.importorskip("botorch")

TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATALYST_CSV = os.path.join(TESTS_DIR, "catalyst_experiments.csv")
CATALYST_SS = os.path.join(TESTS_DIR, "catalyst_search_space.json")


def _build_catalyst_session(n_rows=22):
    """Build a trained single-objective session on the catalyst fixture.

    Uses the first ``n_rows`` experiments. n_rows=22 spans the prefixes
    (11, 12) that trigger ModelFittingError under normalize/standardize.
    """
    import pandas as pd

    df = pd.read_csv(CATALYST_CSV)
    with open(CATALYST_SS) as fh:
        ss = json.load(fh)
    target = "Output"

    session = OptimizationSession()
    for var in ss:
        if var["type"] == "Real":
            session.add_variable(
                var["name"], "real", bounds=(var["min"], var["max"])
            )
        else:
            session.add_variable(
                var["name"], "categorical", categories=var["values"]
            )

    for idx in range(min(n_rows, len(df))):
        row = df.iloc[idx]
        inputs = {var["name"]: row[var["name"]] for var in ss}
        session.add_experiment(inputs, output=row[target])

    session.train_model(backend="botorch", kernel="Matern")
    return session


def test_no_interior_nans_in_posterior_predictions():
    """Interior iterations must never be NaN, even when a subset fit fails.

    Leading NaNs (i < start_iteration) are expected; interior NaNs would
    produce a gap in the regret plot and are the bug under test.
    """
    session = _build_catalyst_session(n_rows=22)
    start_iteration = 5

    means, stds = session._compute_posterior_predictions(
        goal="maximize",
        backend="botorch",
        kernel="Matern",
        n_grid_points=200,
        start_iteration=start_iteration,
        reuse_hyperparameters=True,
        use_calibrated_uncertainty=False,
    )

    # Interior region is start_iteration-1 .. end (0-indexed).
    interior_means = means[start_iteration - 1:]
    interior_stds = stds[start_iteration - 1:]

    assert not np.any(np.isnan(interior_means)), (
        f"Interior NaNs in predicted_means at indices "
        f"{np.where(np.isnan(interior_means))[0] + (start_iteration - 1)}"
    )
    assert not np.any(np.isnan(interior_stds)), (
        f"Interior NaNs in predicted_stds at indices "
        f"{np.where(np.isnan(interior_stds))[0] + (start_iteration - 1)}"
    )


def test_recovered_predictions_are_finite_and_in_range():
    """Recovered values must be genuine, finite, sane objective-scale numbers."""
    session = _build_catalyst_session(n_rows=22)
    start_iteration = 5

    means, stds = session._compute_posterior_predictions(
        goal="maximize",
        backend="botorch",
        kernel="Matern",
        n_grid_points=200,
        start_iteration=start_iteration,
        reuse_hyperparameters=True,
        use_calibrated_uncertainty=False,
    )

    interior_means = means[start_iteration - 1:]
    interior_stds = stds[start_iteration - 1:]

    assert np.all(np.isfinite(interior_means))
    assert np.all(np.isfinite(interior_stds))
    # Objective (Output) is a conversion/selectivity fraction; predicted max
    # mean should stay in a sane band even for the transform-free fallback fit.
    assert np.all(interior_means > -1.0)
    assert np.all(interior_means < 2.0)
    assert np.all(interior_stds >= 0.0)
