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

    def test_a_fold_that_cannot_fit_is_dropped_loudly_never_reused(self):
        calls = {"n": 0}

        def flaky(mll, *a, **kw):
            calls["n"] += 1
            if calls["n"] == 3:  # full fit, fold 0, then fold 1 fails
                raise ModelFittingError("no")
            return fit_gpytorch_mll(mll, *a, **kw)

        with patch.object(bm, "fit_gpytorch_mll", side_effect=flaky), patch.object(bm.logger, "warning") as warn:
            model = _botorch()
            model.train(_manager())

        assert 1 not in set(model.cv_cached_results["fold_ids"])
        assert len(model.cv_cached_results["y_pred"]) == 20
        assert any("fold 1" in str(c.args[0]) for c in warn.call_args_list)

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


class TestBoTorchIterationCap:
    """T22: BoTorch drops ``options=``; the cap has to reach SciPy."""

    def test_default_fit_runs_to_convergence(self):
        """A 50-iteration default stopped real 6-D fits short and moved their proposals."""
        seen = []

        def spy(mll, *a, **kw):
            seen.append(kw)
            return fit_gpytorch_mll(mll, *a, **kw)

        with patch.object(bm, "fit_gpytorch_mll", side_effect=spy):
            model = _botorch()
            model.train(_manager())
        assert model.training_iter is None
        assert seen
        assert all("optimizer_kwargs" not in kw and "options" not in kw for kw in seen)

    def test_cap_changes_the_fit(self):
        one, many = _botorch(training_iter=1), _botorch(training_iter=200)
        one.train(_manager(), cache_cv=False)
        many.train(_manager(), cache_cv=False)
        ls_one = one.model.covar_module.base_kernel.lengthscale.detach()
        ls_many = many.model.covar_module.base_kernel.lengthscale.detach()
        assert not torch.allclose(ls_one, ls_many, rtol=1e-3)

    @pytest.mark.parametrize("targets", [("Output",), ("a", "b")])
    def test_every_fit_passes_the_cap_where_botorch_reads_it(self, targets):
        seen = []

        def spy(mll, *a, **kw):
            seen.append(kw)
            return fit_gpytorch_mll(mll, *a, **kw)

        with patch.object(bm, "fit_gpytorch_mll", side_effect=spy):
            _botorch(training_iter=7).train(_manager(targets=targets))
        assert seen
        for kw in seen:
            assert "options" not in kw
            assert kw["optimizer_kwargs"] == {"options": {"maxiter": 7}}

    def test_evaluate_uses_the_same_cap(self):
        em = _manager(n=8)
        model = _botorch(training_iter=7)
        model.train(em, cache_cv=False)
        seen = []

        def spy(mll, *a, **kw):
            seen.append(kw)
            return fit_gpytorch_mll(mll, *a, **kw)

        with patch.object(bm, "fit_gpytorch_mll", side_effect=spy):
            model.evaluate(em, cv_splits=3)
        assert seen
        assert all(kw.get("optimizer_kwargs") == {"options": {"maxiter": 7}} for kw in seen)


class TestBoTorchEvaluateKernel:
    """T3."""

    @pytest.mark.parametrize(
        "options, kernel_type",
        [
            ({"cont_kernel_type": "Matern", "matern_nu": 1.5}, MaternKernel),
            ({"cont_kernel_type": "RBF"}, RBFKernel),
        ],
    )
    def test_folds_use_the_chosen_kernel(self, fit_spy, options, kernel_type):
        em = _manager(n=8)
        model = _botorch(kernel_options=options)
        model.train(em, cache_cv=False)
        fit_spy.clear()
        model.evaluate(em, cv_splits=3)
        assert fit_spy
        for gp in fit_spy:
            assert isinstance(gp.covar_module, ScaleKernel)
            assert isinstance(gp.covar_module.base_kernel, kernel_type)
            if kernel_type is MaternKernel:
                assert gp.covar_module.base_kernel.nu == 1.5

    def test_folds_use_the_noise_column(self, fit_spy):
        em = _manager(n=8, noise=np.linspace(0.01, 0.2, 8))
        model = _botorch()
        model.train(em, cache_cv=False)
        fit_spy.clear()
        model.evaluate(em, cv_splits=3)
        assert fit_spy
        assert all(isinstance(gp.likelihood, FixedNoiseGaussianLikelihood) for gp in fit_spy)


class _GPSpy:
    """Replaces the GP class in sklearn_model and keeps every instance."""

    def __init__(self):
        self.real = sm.GaussianProcessRegressor
        self.made = []

    def __call__(self, *a, **kw):
        gp = self.real(*a, **kw)
        self.made.append(gp)
        return gp


class TestSklearnCvFitsFolds:
    """T2."""

    def test_cached_cv_optimises_each_fold_kernel(self):
        spy = _GPSpy()
        with patch.object(sm, "GaussianProcessRegressor", spy):
            _sklearn().train(_manager())
        folds = spy.made[1:]
        assert len(folds) == 5
        for gp in folds:
            assert gp.optimizer is not None
            assert not np.allclose(gp.kernel_.theta, gp.kernel.theta)

    @pytest.mark.parametrize("debug", [False, True])
    def test_evaluate_optimises_each_fold_kernel(self, debug):
        em = _manager(n=9)
        model = _sklearn()
        model.train(em, cache_cv=False)
        spy = _GPSpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with patch.object(sm, "GaussianProcessRegressor", spy):
                model.evaluate(em, cv_splits=3, debug=debug)
        assert spy.made
        for gp in spy.made:
            assert gp.optimizer is not None
            assert not np.allclose(gp.kernel_.theta, gp.kernel.theta)

    def test_fold_keeps_the_configured_noise_in_scaled_units(self):
        noise = np.linspace(0.01, 0.2, 25)
        spy = _GPSpy()
        with patch.object(sm, "GaussianProcessRegressor", spy):
            model = _sklearn()
            em = _manager(noise=noise)
            model.train(em)
        X, y, _ = em.get_features_target_and_noise()
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for gp, (train_idx, _) in zip(spy.made[1:], kf.split(X)):
            expected = noise[train_idx] / np.var(y.values[train_idx])
            np.testing.assert_allclose(gp.alpha, expected, rtol=1e-9)

    def test_cv_std_is_in_output_units_for_every_scaler(self):
        """The fold's own scaler sets the units; MinMax's ``scale_`` is an inverse range."""
        em = _manager()
        y = em.df["Output"]
        em.df["Output"] = 1000 * y
        for scaling in ("standard", "minmax", "robust"):
            model = SklearnModel(
                kernel_options={"kernel_type": "Matern", "matern_nu": 2.5},
                n_restarts_optimizer=1,
                input_transform_type="standard",
                output_transform_type=scaling,
            )
            model.train(em)
            sd = np.median(model.cv_cached_results["y_std"])
            assert 10 < sd < 5000, f"{scaling}: σ {sd} is not in the output's units"

    def test_a_failed_fold_is_dropped_loudly(self):
        real = SklearnModel._fit_gp
        calls = {"n": 0}

        def flaky(self, *a, **kw):
            calls["n"] += 1
            if calls["n"] == 2:
                raise np.linalg.LinAlgError("singular")
            return real(self, *a, **kw)

        with patch.object(SklearnModel, "_fit_gp", flaky), patch.object(sm.logger, "warning") as warn:
            model = _sklearn()
            model.train(_manager())
        assert 0 not in set(model.cv_cached_results["fold_ids"])
        assert any("fold 0" in str(c.args[0]) for c in warn.call_args_list)


class TestCrossValidateIsShared:
    """The desktop's legacy parity path calls this instead of keeping its own leaky copy."""

    def test_botorch(self):
        em = _manager()
        model = _botorch()
        model.train(em)
        X, y, _ = em.get_features_target_and_noise()
        again = model._cross_validate(X, y)
        np.testing.assert_allclose(again["y_pred"], model.cv_cached_results["y_pred"], rtol=1e-6)

    def test_sklearn(self):
        em = _manager()
        model = _sklearn()
        model.train(em)
        X, y, _ = em.get_features_target_and_noise()
        again = model._cross_validate(X, y)
        np.testing.assert_allclose(again["y_pred"], model.cv_cached_results["y_pred"], rtol=1e-6)

    def test_too_few_rows(self):
        em = _manager()
        model = _botorch()
        model.train(em)
        X, y, _ = em.get_features_target_and_noise()
        assert model._cross_validate(X.iloc[:4], y.iloc[:4]) is None


class TestCvSigmaIncludesNoise:
    """T4: a held-out measurement carries noise, so its z-score needs the noise term."""

    def _latent_var(self, gp, X_test):
        gp.eval()
        with torch.no_grad():
            return gp.posterior(X_test).variance.squeeze(-1).numpy()

    def _folds(self, em):
        X, _, _ = em.get_features_target_and_noise()
        return torch.tensor(X.values, dtype=torch.float64), list(
            KFold(n_splits=5, shuffle=True, random_state=42).split(X)
        )

    def test_learned_noise(self, fit_spy):
        em = _manager()
        model = _botorch()
        model.train(em, calibrate_uncertainty=False)
        assert len(fit_spy) == 6
        c = model.cv_cached_results
        X_t, folds = self._folds(em)
        y_var = float(np.var(em.df["Output"].values[folds[0][0]], ddof=1))
        for fold_id, (gp, (_, test_idx)) in enumerate(zip(fit_spy[1:], folds)):
            sd = c["y_std"][c["fold_ids"] == fold_id]
            latent = self._latent_var(gp, X_t[test_idx])
            assert (sd ** 2 > latent).all()
            # The added term is the fold's noise, in output units (order of the data's variance).
            added = sd ** 2 - latent
            assert np.allclose(added, added[0])
            assert 1e-4 * y_var < added[0] < y_var

    def test_known_noise_is_each_held_out_rows_own(self, fit_spy):
        noise = np.linspace(0.01, 0.2, 25)
        em = _manager(noise=noise)
        model = _botorch()
        model.train(em, calibrate_uncertainty=False)
        assert len(fit_spy) == 6
        c = model.cv_cached_results
        X_t, folds = self._folds(em)
        for fold_id, (gp, (_, test_idx)) in enumerate(zip(fit_spy[1:], folds)):
            sd = c["y_std"][c["fold_ids"] == fold_id]
            np.testing.assert_allclose(sd ** 2, self._latent_var(gp, X_t[test_idx]) + noise[test_idx], rtol=1e-6)

    def test_multi_objective(self, fit_spy):
        em = _manager(targets=("a", "b"))
        model = _botorch()
        model.train(em)
        c = model.cv_cached_results_multi["a"]
        X_t, folds = self._folds(em)
        first_fold_gp = fit_spy[2]
        sd = c["y_std"][c["fold_ids"] == 0]
        assert (sd ** 2 > self._latent_var(first_fold_gp, X_t[folds[0][1]])).all()

    def test_sklearn_adds_the_noise_column(self):
        noise = np.linspace(0.01, 0.2, 25)
        em = _manager(noise=noise)
        model = _sklearn()
        model.train(em, calibrate_uncertainty=False)
        c = model.cv_cached_results
        assert (c["y_std"] ** 2 > noise[c["test_indices"]]).all()

    def test_acquisition_facing_sigma_stays_latent(self):
        """Scope guard: which σ acquisition sees is a separate decision."""
        em = _manager()
        model = _botorch()
        model.train(em)
        X, _, _ = em.get_features_target_and_noise()
        _, sd = model.predict_with_std(X.iloc[:3])
        with torch.no_grad():
            post = model.model.posterior(torch.tensor(X.iloc[:3].values, dtype=torch.float64))
        np.testing.assert_allclose(sd, post.variance.squeeze(-1).sqrt().numpy(), rtol=1e-9)
