"""Cross-validation must score models that never saw the held-out rows (T1, T2, T3, T4, T22)."""
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from botorch.exceptions.errors import ModelFittingError
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from sklearn.model_selection import KFold

import alchemist_core.models.botorch_model as bm
import alchemist_core.models.sklearn_model as sm
from alchemist_core.data.experiment_manager import ExperimentManager
from alchemist_core.data.search_space import SearchSpace
from alchemist_core.models.botorch_model import BoTorchModel
from alchemist_core.models.sklearn_model import SklearnModel


def _manager(n=25, targets=("Output",), noise=None, seed=0):
    """A noisy 2-D response, so hyperparameters differ visibly between folds."""
    space = SearchSpace()
    space.add_variable("x1", "real", min=0.0, max=1.0)
    space.add_variable("x2", "real", min=0.0, max=1.0)
    em = ExperimentManager(search_space=space, target_columns=list(targets))
    rng = np.random.default_rng(seed)
    x1, x2 = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    df = pd.DataFrame({"x1": x1, "x2": x2})
    for i, t in enumerate(targets):
        df[t] = np.sin(6 * x1 + i) + x2 ** 2 + 0.25 * rng.normal(size=n)
    if noise is not None:
        df["Noise"] = noise
    em.df = df
    return em


def _botorch(**kw):
    return BoTorchModel(input_transform_type="normalize", output_transform_type="standardize", **kw)


def _sklearn(**kw):
    return SklearnModel(
        kernel_options={"kernel_type": "Matern", "matern_nu": 2.5},
        n_restarts_optimizer=1,
        input_transform_type="standard",
        output_transform_type="standard",
        **kw,
    )


@pytest.fixture
def fit_spy():
    """Records every GP that BoTorchModel fits, in order."""
    fitted = []

    def spy(mll, *a, **kw):
        fitted.append(mll.model)
        return fit_gpytorch_mll(mll, *a, **kw)

    with patch.object(bm, "fit_gpytorch_mll", side_effect=spy):
        yield fitted


def _refit_fold_by_hand(em, train_idx, test_idx):
    """The honest answer for one fold, built without going through BoTorchModel."""
    X, y, _ = em.get_features_target_and_noise()
    X_t = torch.tensor(X.values, dtype=torch.float64)
    y_t = torch.tensor(y.values, dtype=torch.float64).unsqueeze(-1)
    torch.manual_seed(42)
    gp = SingleTaskGP(
        X_t[train_idx],
        y_t[train_idx],
        covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=2, active_dims=[0, 1])),
        input_transform=Normalize(d=2),
        outcome_transform=Standardize(m=1),
    )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    gp.eval()
    with torch.no_grad():
        return gp.posterior(X_t[test_idx]).mean.squeeze(-1).numpy()


class TestBoTorchCvRefits:
    """T1."""

    def test_single_objective_fits_each_fold_from_its_own_rows(self, fit_spy):
        em = _manager()
        model = _botorch()
        model.train(em)

        assert len(fit_spy) == 6, "one full-data fit, then one per fold"
        for fold_gp in fit_spy[1:]:
            assert fold_gp.train_targets.shape[-1] == 20

    def test_fold_predictions_match_an_independent_refit(self):
        em = _manager()
        model = _botorch()
        model.train(em)
        c = model.cv_cached_results

        X, _, _ = em.get_features_target_and_noise()
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for fold_id, (train_idx, test_idx) in enumerate(kf.split(X)):
            mine = c["y_pred"][c["fold_ids"] == fold_id]
            np.testing.assert_allclose(
                mine, _refit_fold_by_hand(em, train_idx, test_idx), rtol=1e-2, atol=1e-2
            )

    def test_fold_hyperparameters_are_not_the_full_data_ones(self, fit_spy):
        model = _botorch()
        model.train(_manager())
        assert len(fit_spy) == 6
        full = fit_spy[0].covar_module.base_kernel.lengthscale.detach()
        for fold_gp in fit_spy[1:]:
            fold = fold_gp.covar_module.base_kernel.lengthscale.detach()
            assert not torch.allclose(full, fold)

    def test_multi_objective_fits_each_fold_of_each_objective(self, fit_spy):
        model = _botorch()
        model.train(_manager(targets=("a", "b")))
        assert len(fit_spy) == 2 + 2 * 5
        assert set(model.cv_cached_results_multi) == {"a", "b"}

    def test_folds_train_on_their_share_of_a_noise_column(self, fit_spy):
        noise = np.linspace(0.01, 0.2, 25)
        model = _botorch()
        model.train(_manager(noise=noise))
        assert len(fit_spy) == 6
        for fold_gp in fit_spy[1:]:
            assert isinstance(fold_gp.likelihood, FixedNoiseGaussianLikelihood)
            assert fold_gp.likelihood.noise.numel() == 20

    def test_a_fold_that_cannot_fit_is_dropped_loudly_never_reused(self, fit_spy, caplog):
        calls = {"n": 0}

        def flaky(mll, *a, **kw):
            calls["n"] += 1
            if calls["n"] == 3:  # full fit, fold 0, then fold 1 fails
                raise ModelFittingError("no")
            return fit_gpytorch_mll(mll, *a, **kw)

        logger = bm.logger
        logger.addHandler(caplog.handler)
        try:
            with patch.object(bm, "fit_gpytorch_mll", side_effect=flaky):
                model = _botorch()
                model.train(_manager())
        finally:
            logger.removeHandler(caplog.handler)

        assert 1 not in set(model.cv_cached_results["fold_ids"])
        assert len(model.cv_cached_results["y_pred"]) == 20
        assert "fold 1" in caplog.text

    def test_no_cached_results_when_every_fold_fails(self):
        calls = {"n": 0}

        def only_full(mll, *a, **kw):
            calls["n"] += 1
            if calls["n"] > 1:
                raise ModelFittingError("no")
            return fit_gpytorch_mll(mll, *a, **kw)

        with patch.object(bm, "fit_gpytorch_mll", side_effect=only_full):
            model = _botorch()
            model.train(_manager())
        assert model.is_trained
        assert model.cv_cached_results is None

    def test_retraining_gives_the_same_scores(self):
        a, b = _botorch(), _botorch()
        a.train(_manager())
        b.train(_manager())
        np.testing.assert_array_equal(a.cv_cached_results["y_pred"], b.cv_cached_results["y_pred"])






class _GPSpy:
    """Replaces the GP class in sklearn_model and keeps every instance."""

    def __init__(self):
        self.real = sm.GaussianProcessRegressor
        self.made = []

    def __call__(self, *a, **kw):
        gp = self.real(*a, **kw)
        self.made.append(gp)
        return gp
