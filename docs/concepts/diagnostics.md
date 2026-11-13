# Reading the diagnostics

A Gaussian process reports a mean and an uncertainty at every point, and the diagnostics test both against data the model did not train on. This page explains the parity plot, Q-Q plot, calibration curve and error metrics, what each can and cannot tell you, and how ALchemist computes them.

## Cross-validated predictions

A prediction at a training point tests nothing, because the model has seen the answer. The diagnostics fit on all but one fold, predict the held-out fold, and repeat, so every observation gets one out-of-sample prediction $\hat\mu_i$ with uncertainty $\hat\sigma_i$ ([Rasmussen and Williams 2006, §5.4.2](references.md#rasmussen-2006)). The folds share training data, so errors are correlated and every threshold below is approximate ([Bastos and O'Hagan 2009](references.md#bastos-2009)). Optimization data concentrate where the policy found them promising ([Garnett 2023, ch. 7](references.md#garnett-2023)), so error describes the sampled region.

## Parity plot

The x axis is the observed value and the y axis the cross-validated prediction, with a dashed y = x line and ±1.96 $\hat\sigma_i$ error bars by default. A point above the line is an over-prediction; below, an under-prediction.

## Standardized residuals

The Q-Q plot and calibration curve both work from

$$z_i = \frac{y_i - \hat\mu_i}{\hat\sigma_i}.$$

If the model is right, the $z_i$ behave like standard-normal draws, with mean 0 and standard deviation 1 ([Bastos and O'Hagan 2009](references.md#bastos-2009)). A positive Mean(z) means the observations sit above the predictions: under-prediction. For independent normal draws the sampling errors of the mean and standard deviation are about $1/\sqrt n$ and $1/\sqrt{2n}$: 0.18 and 0.13 at $n = 30$.

The posterior describes the latent function $\phi$; a noisy measurement $y$ has the extra variance $\sigma_n^2$ ([Garnett 2023, eqs. 2.19–2.20](references.md#garnett-2023)):

$$p(\phi \mid x, \mathcal D) = \mathcal N(\mu, \sigma^2), \qquad p(y \mid x, \mathcal D) = \mathcal N(\mu, \sigma^2 + \sigma_n^2).$$

Dividing noisy residuals by the latent $\sigma$ makes a correct model look over-confident.

## Q-Q plot

Sort the $z_i$ and plot them against standard-normal quantiles $\Phi^{-1}(p_i)$ ([Wilk and Gnanadesikan 1968](references.md#wilk-1968)). If $z \approx \mathcal N(m, s^2)$ the points fall on

$$z_{(i)} \approx m + s\,\Phi^{-1}(p_i),$$

so the intercept is the bias and the slope is the spread. Std(z) > 1 makes the cloud steeper than the diagonal (below it on the left, above it on the right) instead of shifting it, and Std(z) < 1 makes it shallower. Bias moves the whole cloud up or down; points leaving the line only at the ends indicate a few large errors.

## Calibration curve

For each nominal level $p$, form the interval $\hat\mu_i \pm \Phi^{-1}\!\big(\tfrac{1+p}{2}\big)\hat\sigma_i$ and measure the fraction $c(p)$ of observations inside. A calibrated model has $c(p) \approx p$ ([Gneiting, Balabdaoui and Raftery 2007](references.md#gneiting-2007a)); $c$ against $p$ is the reliability diagram ([DeGroot and Fienberg 1983](references.md#degroot-1983)). Coverage below nominal (a curve under the diagonal) means intervals too narrow: over-confident. Above nominal means too wide: under-confident.

For sampling noise, the exact binomial interval for the true coverage given $k$ of $n$ points inside ([Clopper and Pearson 1934](references.md#clopper-1934)) is

$$\big[\, B_{\alpha/2}(k,\ n-k+1),\ B_{1-\alpha/2}(k+1,\ n-k) \,\big],$$

where $B_q(a, b)$ is the $q$-quantile of Beta$(a, b)$. For 17 of 20 inside a nominal 95% interval (0.85) it runs from 0.62 to 0.97 and contains 0.95. ALchemist does not draw these bands, and they assume independent points.

## Error metrics

With $e_i = y_i - \hat\mu_i$, RMSE $= \sqrt{\overline{e^2}}$, MAE $= \overline{|e|}$ and $R^2 = 1 - \sum e_i^2 / \sum (y_i - \bar y)^2$ follow `sklearn.metrics`; ALchemist computes MAPE itself, as $100\,\overline{|e_i| / (|y_i| + 10^{-9})}$, a percentage. MAE ≤ RMSE always, with equality only when every $|e_i|$ is equal. $R^2$ depends on the spread of the sampled $y$ and is negative when the predictions do worse than the mean of $y$, which is common in small-$n$ cross-validation. MAPE explodes when any $y$ is near zero. On noisy data the expected squared error is $\sigma_n^2$ plus the model's error about the latent function ([Garnett 2023, eq. 2.20](references.md#garnett-2023)), so RMSE should not be expected to fall below $\sigma_n$, though a small sample can by chance. No universal cut-off separates good from bad; compare RMSE with your noise and the response range.

## From symptom to action

Causes are the usual suspects and remedies are things to try. Hyperparameters fitted to few points are noisy ([Garnett 2023, §4.3](references.md#garnett-2023)), and the kernel sets their effect on $\sigma$ ([Garnett 2023, §3.3–3.4](references.md#garnett-2023)).

| What you see | Likely cause | What to try |
|---|---|---|
| Std(z) above 1: Q-Q steeper than the diagonal, curve below it, "Over-conf" | $\hat\sigma$ too small: noise underestimated, lengthscales too long or outputscale too small, response rougher than the kernel allows | Check the Noise column holds variances; try a Matern kernel with lower $\nu$; add points where $\lvert z\rvert$ is large |
| Std(z) below 1: Q-Q shallower, curve above the diagonal, "Under-conf" | $\hat\sigma$ too large: noise overestimated or entered in the wrong units, outputscale too large, very short lengthscales | Check the Noise column's units; read the fitted hyperparameters. Often less harmful, but a wide $\sigma$ inflates exploration in UCB and EI |
| Mean(z) far from 0: Q-Q cloud shifted, parity points mostly on one side | Systematic bias (positive Mean(z) is under-prediction): a trend the kernel cannot carry, or edge points that must be extrapolated | Check the edge points in the parity plot; add observations there |
| Q-Q ends peel away, or the curve crosses the diagonal (above at low levels, below at high) | Heavy tails: a few large errors from a bad measurement or an abrupt change; or $\hat\sigma$ good in some regions and poor in others, which one global scale cannot fix | Find the largest $\lvert z\rvert$ on the parity plot and re-check those measurements; see whether errors cluster in one region |
| $R^2$ falls or RMSE rises as data accumulate, z diagnostics fine | The sampled region narrows, shrinking the spread of $y$; or the noise floor | Compare RMSE with $\sigma_n$; this is not overfitting by itself |

!!! info "How ALchemist does it"
    One cached set of cross-validated predictions per trained model feeds the parity, Q-Q and calibration plots. It is computed in [`alchemist_core.models`](../reference/python/models.md) and drawn by [`alchemist_core.visualization.plots`](../reference/python/visualization.md).

    - Folds: `_cache_cross_validation_results` runs shuffled 5-fold CV (seed 42 by default) on every train, skipped below 5 observations.
    - BoTorch: each fold builds a GP with the session's kernel and Noise column and fits its hyperparameters on that fold's rows only; $\hat\sigma$ is the predictive one, with the held-out row's Noise variance or else the fold's learned noise. A fold that cannot be fitted is left out with a logged warning.
    - scikit-learn: each fold builds a fresh kernel that starts from the initial hyperparameters (constant 1, lengthscale = standard deviation of each scaled input, for RBF and Matérn; RationalQuadratic starts from scikit-learn's defaults) and optimizes them on that fold's rows, with at most 2 restarts where the final model has 30. $\hat\sigma$ adds the held-out row's Noise variance; without a Noise column the GP has no noise term to add, so noisy data can read as over-confident.
    - Q-Q plot: $p_i = (i - 0.375)/(n + 0.25)$ (Blom), so the extreme quantiles grow with $n$. The band, drawn below 100 points, is each sorted value's exact pointwise 95% interval under a standard normal, from the Beta distribution of order statistics; it widens in the tails, and with many points some will fall outside by chance.
    - Metrics plot: `evaluate` repeats the CV on the first $n$ rows in data order, $n \ge 5$. BoTorch refits every fold and pools predictions; scikit-learn averages per-fold metrics.
    - σ-scaling: `_compute_calibration_factors` sets $s = \mathrm{Std}(z)$ with `ddof=1`, and `predict(return_std=True)` multiplies the latent $\sigma$ by $s$, although $s$ was measured on the predictive one. The scaled CV copy feeds "Use Calibrated Results". This changes displayed $\sigma$ only: BoTorch acquisition uses the raw GP and scikit-optimize the unscaled estimator. Because $s$ comes from the same residuals, the calibrated Q-Q has Std(z) = 1 by construction.
    - Status bands: per level, with $d = c - p$, $\lvert d\rvert < 0.05$ is "Good", 0.05 to 0.10 is "~ Acceptable", $d > 0.10$ is "Under-conf" and $d < -0.10$ is "Over-conf", both with a warning. These are ALchemist's cut-offs, not a test: at $n = 20$ the standard error of $c$ is up to 0.11.

!!! note
    Calibration runs on every single-objective train that has cross-validation, and there is no setting to turn it off. The API field `calibration_enabled` is accepted but has no effect; passing `False` logs a warning.

## Further reading

- [Garnett 2023](references.md#garnett-2023): GP posterior, noise, hyperparameters (ch. 2–4).
- [Rasmussen and Williams 2006](references.md#rasmussen-2006): cross-validation for GPs.
- [Bastos and O'Hagan 2009](references.md#bastos-2009): GP diagnostics.
- [Gneiting, Balabdaoui and Raftery 2007](references.md#gneiting-2007a): calibration.
- [Kuleshov et al. 2018](references.md#kuleshov-2018): isotonic recalibration.

In the app: [Check the model](../guide/check-model.md).
