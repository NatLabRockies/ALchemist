# Reading the diagnostics

A Gaussian process reports a mean and an uncertainty at every point, and the diagnostics test both against data the model did not train on. This page explains the parity plot, Q-Q plot, calibration curve and error metrics, what each can and cannot tell you, and how ALchemist computes them.

## Cross-validated predictions

A prediction at a training point tests nothing, because the model has seen the answer. The diagnostics split the observations into folds, fit on all but one fold, predict the held-out fold, and repeat, so every observation gets one out-of-sample prediction $\hat\mu_i$ with uncertainty $\hat\sigma_i$ ([Rasmussen and Williams 2006, §5.4.2](references.md#rasmussen-2006)). The folds share training data, so the errors are correlated and every threshold below is approximate ([Bastos and O'Hagan 2009](references.md#bastos-2009)). Optimization data concentrate where the policy found them promising ([Garnett 2023, ch. 7](references.md#garnett-2023)), so the error describes the region you sampled.

## Parity plot

The x axis is the observed value and the y axis the cross-validated prediction (`plots.py:96-108`), with a dashed y = x line and ±1.96 $\hat\sigma_i$ error bars by default. A point above the line is an over-prediction (predicted above actual); a point below is an under-prediction.

## Standardized residuals

The Q-Q plot and calibration curve both work from

$$z_i = \frac{y_i - \hat\mu_i}{\hat\sigma_i}.$$

If the model is right, the $z_i$ behave like standard-normal draws, with mean 0 and standard deviation 1 ([Bastos and O'Hagan 2009](references.md#bastos-2009)). A positive Mean(z) means the observations sit above the predictions: under-prediction. For independent normal draws the sampling error of the mean is about $1/\sqrt n$ and of the standard deviation about $1/\sqrt{2n}$: 0.18 and 0.13 at $n = 30$.

Which $\sigma$ goes in the denominator matters. The GP posterior describes the latent function $\phi$; a noisy measurement $y$ has the extra variance $\sigma_n^2$ ([Garnett 2023, eqs. 2.19–2.20](references.md#garnett-2023)):

$$p(\phi \mid x, \mathcal D) = \mathcal N(\mu, \sigma^2), \qquad p(y \mid x, \mathcal D) = \mathcal N(\mu, \sigma^2 + \sigma_n^2).$$

Dividing noisy residuals by the latent $\sigma$ makes a correct model look over-confident.

## Q-Q plot

Sort the $z_i$ and plot them against standard-normal quantiles $\Phi^{-1}(p_i)$ ([Wilk and Gnanadesikan 1968](references.md#wilk-1968)). If $z \approx \mathcal N(m, s^2)$ the points fall on

$$z_{(i)} \approx m + s\,\Phi^{-1}(p_i),$$

so the intercept is the bias and the slope is the spread. Std(z) > 1 makes the cloud steeper than the diagonal (below it on the left, above it on the right) instead of shifting it, and Std(z) < 1 makes it shallower. Bias moves the whole cloud up or down; points leaving the line only at the ends indicate a few large errors.

## Calibration curve

For each nominal level $p$, form the interval $\hat\mu_i \pm \Phi^{-1}\!\big(\tfrac{1+p}{2}\big)\hat\sigma_i$ and measure the fraction $c(p)$ of observations inside. A calibrated model has $c(p) \approx p$ ([Gneiting, Balabdaoui and Raftery 2007](references.md#gneiting-2007a)); $c$ against $p$ is the reliability diagram ([DeGroot and Fienberg 1983](references.md#degroot-1983)). Coverage below nominal, a curve under the diagonal, means the intervals are too narrow: over-confident. Coverage above nominal means they are too wide: under-confident.

For sampling noise, the exact binomial interval for the true coverage given $k$ of $n$ points inside ([Clopper and Pearson 1934](references.md#clopper-1934)) is

$$\big[\, B_{\alpha/2}(k,\ n-k+1),\ B_{1-\alpha/2}(k+1,\ n-k) \,\big],$$

where $B_q(a, b)$ is the $q$-quantile of Beta$(a, b)$. For 17 of 20 points inside a nominal 95% interval (0.85) it runs from 0.62 to 0.97 and contains 0.95, so 20 points cannot tell those apart. ALchemist does not draw these bands, and they assume independent points.

## Error metrics

With $e_i = y_i - \hat\mu_i$, the `sklearn.metrics` definitions are RMSE $= \sqrt{\overline{e^2}}$, MAE $= \overline{|e|}$, $R^2 = 1 - \sum e_i^2 / \sum (y_i - \bar y)^2$ and MAPE $= 100\,\overline{|e_i| / (|y_i| + 10^{-9})}$. MAE ≤ RMSE always, with equality only when every $|e_i|$ is equal. $R^2$ depends on the spread of the sampled $y$. MAPE explodes when any $y$ is near zero. On noisy data RMSE cannot fall below $\sigma_n$, because $y$ varies by $\sigma_n$ about the latent function ([Garnett 2023, eq. 2.20](references.md#garnett-2023)). No universal cut-off separates good from bad; compare RMSE with your noise and the response range.

## From symptom to action

Causes are the usual suspects, not a diagnosis; remedies are things to try. Hyperparameters fitted to few points are noisy ([Garnett 2023, §4.3](references.md#garnett-2023)), and the kernel sets their effect on $\sigma$ ([Garnett 2023, §3.3–3.4](references.md#garnett-2023)).

| What you see | Likely cause | What to try |
|---|---|---|
| Std(z) above 1: Q-Q steeper than the diagonal, curve below it, "Over-conf" | $\hat\sigma$ too small: noise left out of the latent $\sigma$ (T4) or underestimated, lengthscales too long or outputscale too small, response rougher than the kernel allows | Check the Noise column holds variances; try a Matern kernel with lower $\nu$; add points where $\lvert z\rvert$ is large |
| Std(z) below 1: Q-Q shallower, curve above the diagonal, "Under-conf" | $\hat\sigma$ too large: noise overestimated or entered in the wrong units, outputscale too large, very short lengthscales | Check the Noise column's units; read the fitted hyperparameters. Often less harmful, but a wide $\sigma$ inflates exploration in UCB and EI |
| Mean(z) far from 0: Q-Q cloud shifted, parity points mostly on one side | Systematic bias (positive Mean(z) is under-prediction): a trend the kernel cannot carry, or edge points that must be extrapolated | Check the edge points in the parity plot; add observations there |
| Q-Q straight in the middle, ends peel away | Heavy tails: a few large errors from a bad measurement or an abrupt change | Find the largest $\lvert z\rvert$ on the parity plot; re-check those measurements |
| Curve crosses the diagonal (above at low levels, below at high) | The same heavy tails, or $\hat\sigma$ good in some regions and poor in others, which one global scale cannot fix | As above; see whether errors cluster in one region |
| $R^2$ falls or RMSE rises as data accumulate, z diagnostics fine | The sampled region narrows, shrinking the spread of $y$; or the noise floor | Compare RMSE with $\sigma_n$; this is not overfitting by itself |

!!! info "How ALchemist does it"
    One cached set of cross-validated predictions per trained model feeds every diagnostic, computed in [`alchemist_core.models`](../reference/python/models.md) and drawn by [`alchemist_core.visualization.plots`](../reference/python/visualization.md).

    - Folds: `_cache_cross_validation_results` runs shuffled 5-fold CV (seed 42 by default) on every train, skipped below 5 observations (`botorch_model.py:1118-1141`, `sklearn_model.py:585-597`).
    - BoTorch: each fold builds a GP with the session's kernel, then loads the full-data fitted state; $\hat\sigma$ is the latent posterior $\sigma$ (`botorch_model.py:1177-1192`).
    - scikit-learn: each fold builds a fresh kernel and fits with `optimizer=None`, so hyperparameters are never re-optimised; $\hat\sigma$ comes from a GP with no noise term (`sklearn_model.py:615-631`).
    - Metrics plot: `evaluate` repeats the CV on the first $n$ rows in data order, $n \ge 5$. BoTorch refits every fold and pools predictions (`botorch_model.py:662-772`); scikit-learn averages per-fold metrics (`sklearn_model.py:569`).
        - σ-scaling: `_compute_calibration_factors` sets $s = \mathrm{Std}(z)$ with `ddof=1` (`botorch_model.py:1316`, `sklearn_model.py:696`). `predict(return_std=True)` multiplies $\sigma$ by $s$ (`botorch_model.py:534-536`, `sklearn_model.py:417-419`), and the scaled CV copy feeds the "Use Calibrated Results" toggle. This changes displayed $\sigma$ only: BoTorch acquisition uses the raw GP (`botorch_acquisition.py:225-305`) and scikit-optimize the unscaled estimator (`skopt_acquisition.py:89-92`). Because $s$ comes from the same residuals, the calibrated Q-Q has Std(z) = 1 by construction.
    - Status bands: per level, with $d = c - p$, $\lvert d\rvert < 0.05$ is "Good", 0.05 to 0.10 is "~ Acceptable", $d > 0.10$ is "Under-conf" and $d < -0.10$ is "Over-conf", both with a warning (`CalibrationCurve.tsx:54-63`, `ui/visualizations.py:1041-1048`). They are ALchemist's cut-offs, not a test: at $n = 20$ the standard error of $c$ is already up to 0.11.

!!! warning "Known issue"
    T1: BoTorch cross-validation reloads the full-data fitted state in every fold, so parity, Q-Q and calibration use hyperparameters that saw the held-out points and the metrics are optimistic (RMSE 0.135 against 0.178 refitted, in a 25-point synthetic check). T2: scikit-learn cross-validation and its metrics plot use kernels that were never fitted, so Std(z) describes the starting values. T3: without categorical variables, the BoTorch metrics plot fits BoTorch's default kernel, not yours. T4: the cross-validation $\sigma$ leaves out the noise term, so noisy data read as over-confident and the calibration factor absorbs the noise. All are tracked for a 0.4.x release.

!!! warning "Known issue"
    T5: the Q-Q x axis uses `linspace(0.01, 0.99, n)`, not plotting positions such as $(i - 0.5)/n$, so the extreme theoretical quantiles sit at ±2.33 for every $n$ (±1.64 at $n = 10$ and ±2.58 at $n = 100$ with the usual positions); read the tails loosely. The shaded band is a constant ±1.96/$\sqrt n$ for $n < 100$, the 95% half-width for Mean(z), not a band for sorted values (`plots.py:805-821`, `QQPlot.tsx:52-56`). Tracked for a 0.4.x release.

!!! warning "Known issue"
    F1: the "Calibrate uncertainty" checkbox on the train panel has no effect. Calibration runs on every train whenever cross-validation exists (`botorch_model.py:358`, `sklearn_model.py:386`); the flag only makes scikit-learn repeat the step (`session.py:1591-1594`). Tracked for a 0.4.x release.

## Further reading

- [Garnett 2023](references.md#garnett-2023): the GP posterior, noise and hyperparameters (ch. 2–4).
- [Kuleshov et al. 2018](references.md#kuleshov-2018): isotonic recalibration, a richer alternative to one scale factor.
- [Rasmussen and Williams 2006](references.md#rasmussen-2006): cross-validation for GPs.
- [Bastos and O'Hagan 2009](references.md#bastos-2009): diagnostics for GP emulators.
- [Gneiting, Balabdaoui and Raftery 2007](references.md#gneiting-2007a): calibration and sharpness.

In the app: [Check the model](../guide/check-model.md).
