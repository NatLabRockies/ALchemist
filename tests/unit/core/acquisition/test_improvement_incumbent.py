"""Regression tests for the improvement-family acquisition incumbent (best_f).

Bug: best_f was set to max(Y_orig) — the raw observed maximum of a NOISY
target. With a noise-fitting GP + Standardize outcome transform, the posterior
mean cannot reach that noise-inflated value anywhere, so Expected/Probability of
Improvement have negative improvement everywhere and collapse to a monotone
function of sigma (pure exploration): they return low-mean, max-uncertainty
points instead of high-value ones. UCB (which never uses best_f) is unaffected.

Fix: best_f is the best posterior mean at the training points (the noise-free
incumbent), the standard BoTorch convention for noisy observations.
"""

import logging
import warnings

import numpy as np
import pandas as pd
import pytest

from alchemist_core import OptimizationSession

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)


COLS = ['H2', 'CO', 'CO2', 'T', 'P', 'SV']


def _noisy_session():
    """All-continuous session with a noisy target (surfaces the bug)."""
    s = OptimizationSession()
    s.add_variable('H2', 'real', bounds=(0.0, 100.0))
    s.add_variable('CO', 'real', bounds=(0.0, 100.0))
    s.add_variable('CO2', 'real', bounds=(0.0, 100.0))
    s.add_variable('T', 'real', bounds=(200.0, 300.0))
    s.add_variable('P', 'real', bounds=(1000.0, 2000.0))
    s.add_variable('SV', 'real', bounds=(0.1, 1.0))
    np.random.seed(5)
    n = 90
    H2 = np.random.uniform(20, 80, n)
    CO = np.random.uniform(10, 60, n)
    CO2 = np.clip(100 - H2 - CO, 0, 100)
    T = np.random.uniform(200, 300, n)
    P = np.random.uniform(1000, 2000, n)
    SV = np.random.uniform(0.1, 1.0, n)
    y = 45 - 0.3 * np.abs(H2 - 63) - 0.15 * CO2 - 0.03 * np.abs(T - 245) + np.random.normal(0, 1.5, n)
    df = pd.DataFrame({'H2': H2, 'CO': CO, 'CO2': CO2, 'T': T, 'P': P, 'SV': SV, 'yield': y})
    s.experiment_manager.target_columns = ['yield']
    s.experiment_manager.df = df
    s.train_model(backend='botorch', kernel='Matern')
    return s


def _mean_sigma(session, sugg):
    r = session.predict(pd.DataFrame([sugg])[COLS])
    mean, std = [v[0] for v in list(r.values())[0]]
    return float(mean), float(std)


class TestImprovementFamilyIncumbent:
    """EI/LogEI/PI must exploit toward high-mean points on noisy data, not
    collapse to the max-uncertainty region like a pure-exploration acquisition."""

    @pytest.mark.parametrize('strategy', ['LogEI', 'EI', 'PI'])
    def test_improvement_family_not_degenerate(self, strategy):
        session = _noisy_session()
        ucb = session.suggest_next(strategy='UCB', goal='maximize').iloc[0].to_dict()
        ucb_mean, _ = _mean_sigma(session, ucb)

        sugg = session.suggest_next(strategy=strategy, goal='maximize').iloc[0].to_dict()
        mean, sigma = _mean_sigma(session, sugg)

        # The improvement-family suggestion should have a predicted mean in the
        # same ballpark as UCB's (a high-value point), not a degenerate low-mean
        # max-uncertainty point. Allow a margin but reject the collapse (which
        # produced mean ~28 vs UCB ~43).
        assert mean >= ucb_mean - 8.0, (
            f"{strategy} returned a degenerate low-mean suggestion "
            f"(mean={mean:.1f}) vs UCB (mean={ucb_mean:.1f})"
        )
