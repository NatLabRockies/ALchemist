# Gaussian processes and kernels

A Gaussian process (GP) is the surrogate model behind ALchemist's predictions and its uncertainty bands. Knowing what its kernel and noise settings assume tells you how to read its suggestions and plots.

## From prior to posterior

A GP is a distribution over functions. The prior $f \sim \mathcal{GP}(\mu, K)$ gives the average level $\mu$ and a kernel $K(x, x')$ saying how strongly values at two inputs move together. Observing $\mathbf{y}$ at inputs $X$ gives a posterior that is again a GP ([Garnett 2023, §2.2](references.md#garnett-2023); [Rasmussen and Williams 2006, ch. 2](references.md#rasmussen-2006)).

The latent function $f$ and the observation $y$ differ by measurement error: $y = f(x) + \varepsilon$, with $\varepsilon \sim \mathcal{N}(0, \mathbf{N})$ ([Garnett 2023, eq. 2.15](references.md#garnett-2023)). Common choices are one shared variance, $\sigma_n^2 \mathbf{I}$ (homoskedastic, eq. 2.16), and a variance per run, $\operatorname{diag}\sigma_n^2(x)$ (heteroskedastic, eq. 2.17). With $\mathbf{K} = K(X, X)$, the posterior is

$$
\mu_{\mathcal{D}}(x) = \mu(x) + K(x, X)\,(\mathbf{K} + \mathbf{N})^{-1}(\mathbf{y} - \boldsymbol{\mu}), \qquad
K_{\mathcal{D}}(x, x') = K(x, x') - K(x, X)\,(\mathbf{K} + \mathbf{N})^{-1} K(X, x')
$$

([Garnett 2023, eq. 2.19](references.md#garnett-2023); [Rasmussen and Williams 2006, eqs. 2.23-2.24](references.md#rasmussen-2006) for $\mu = 0$, $\mathbf{N} = \sigma_n^2\mathbf{I}$). With $\mathbf{N} = 0$ the posterior passes exactly through the data ([Garnett 2023, eq. 2.10](references.md#garnett-2023)).

The posterior variance $\sigma^2(x) = K_{\mathcal{D}}(x, x)$ describes the latent $f$. A new measurement is scattered around $f$ by the noise as well:

$$
p(y \mid x, \mathcal{D}) = \mathcal{N}\big(y;\ \mu_{\mathcal{D}}(x),\ \sigma^2(x) + \sigma_n^2\big)
$$

([Garnett 2023, eq. 2.20](references.md#garnett-2023)). A band from $\sigma$ alone shows where $f$ may lie. A band covering new measurements must include $\sigma_n^2$.

## Kernels

ALchemist uses an output scale times a unit-scale shape, $K(x, x') = \sigma_f^2\, k(d)$ ([Garnett 2023, eq. 3.20](references.md#garnett-2023)). The distance $d$ divides each coordinate by its own length scale,

$$
d = \sqrt{\sum_i \left(\frac{x_i - x_i'}{\ell_i}\right)^{2}}
$$

([Garnett 2023, eq. 3.24](references.md#garnett-2023)). A separate $\ell_i$ per input is automatic relevance determination (ARD, [Garnett 2023, eq. 3.25](references.md#garnett-2023); [Rasmussen and Williams 2006, §5.1](references.md#rasmussen-2006)). An input the response barely depends on tends to get a long $\ell_i$, but with few runs length scales are weakly determined, so do not read $1/\ell_i$ as importance.

| Kernel | $k(d)$ | Smoothness |
|---|---|---|
| RBF | $\exp(-d^2/2)$ ([eq. 3.12](references.md#garnett-2023)) | infinitely differentiable |
| Matérn, $\nu = 1/2$ | $\exp(-d)$ ([eq. 3.11](references.md#garnett-2023)) | continuous, not differentiable |
| Matérn, $\nu = 3/2$ | $(1 + \sqrt{3}\,d)\,e^{-\sqrt{3}\,d}$ ([eq. 3.13](references.md#garnett-2023)) | once differentiable |
| Matérn, $\nu = 5/2$ | $(1 + \sqrt{5}\,d + \tfrac{5}{3}d^2)\,e^{-\sqrt{5}\,d}$ ([eq. 3.14](references.md#garnett-2023)) | twice differentiable |
| Rational quadratic | $\left(1 + \frac{d^2}{2\alpha}\right)^{-\alpha}$ ([R&W eq. 4.19](references.md#rasmussen-2006)) | infinitely differentiable (a scale mixture of RBFs) |

Matérn sample paths are $\lceil\nu\rceil - 1$ times differentiable; $\nu \to \infty$ gives the RBF ([Garnett 2023, §3.3](references.md#garnett-2023)). Garnett calls $\nu = 5/2$ a prudent default for Bayesian optimization ([Snoek et al. 2012](references.md#snoek-2012)) and the RBF's infinite smoothness often unrealistic ([Stein 1999](references.md#stein-1999)).

### Fitting

Length scales, output scale and noise are hyperparameters $\theta$, chosen by maximizing the marginal likelihood, optionally weighted by a prior ([Garnett 2023, eq. 4.9](references.md#garnett-2023)):

$$
\log p(\mathbf{y} \mid X, \theta) = -\tfrac{1}{2}\Big[(\mathbf{y} - \boldsymbol{\mu})^{\top}(\mathbf{K} + \mathbf{N})^{-1}(\mathbf{y} - \boldsymbol{\mu}) + \log|\mathbf{K} + \mathbf{N}| + n\log 2\pi\Big]
$$

([Garnett 2023, eq. 4.8](references.md#garnett-2023)). The first term rewards fit and the second penalizes complexity. A single best $\theta$ ignores uncertainty in $\theta$, which matters most with few runs ([Garnett 2023, §4.4 and §9.3](references.md#garnett-2023)).

### Infinite-width neural-network kernel (IBNN)

The IBNN kernel is the GP that a deep, infinitely wide ReLU network defines ([Lee et al. 2018](references.md#lee-2018)). It starts from $K^0(x, x') = \sigma_b^2 + \sigma_w^2\, x\!\cdot\!x'/d_{\text{in}}$ (eq. 6 there) and applies one recursion per layer (eq. 4 there), where $\sigma_w^2$ and $\sigma_b^2$ are weight and bias variances. It has no length scales and is not stationary: it depends on $x \cdot x'$, so shifting the inputs changes it. [Li et al. (2023)](references.md#li-2023) found it promising as a surrogate, especially in higher dimensions.

## Categorical variables

A kernel needs a similarity between categories. A common construction tests whether two categories match and combines that with a continuous kernel by sums and products, which are valid kernels ([Garnett 2023, eq. 3.29](references.md#garnett-2023)). For continuous inputs $x$ and a categorical input $c$, the terms are $k_{\text{cont}}(x, x') + k_{\text{cat}}(c, c')$ and $k_{\text{cont}}(x, x')\,k_{\text{cat}}(c, c')$, with $k_{\text{cat}} = 1$ for matching categories and smaller otherwise. Another route encodes categories as numeric columns for an ordinary kernel; Gryffin builds expert knowledge about the categories into the model ([Häse et al. 2021](references.md#hase-2021)).

!!! info "How ALchemist does it"
    Two classes implement the surrogate: `BoTorchModel` in `alchemist_core/models/botorch_model.py` and `SklearnModel` in `alchemist_core/models/sklearn_model.py` ([Models reference](../reference/python/models.md)). `OptimizationSession.train_model` in `alchemist_core/session.py` builds them from the `kernel`, `kernel_params`, `input_transform_type` and `output_transform_type` arguments.

    | | BoTorch | scikit-learn |
    |---|---|---|
    | Kernels | RBF, Matérn with $\nu \in \{0.5, 1.5, 2.5\}$ (default 2.5), IBNN with `ibnn_depth` (default 3) | RBF, Matérn (default $\nu = 1.5$; the web list adds `inf`), RationalQuadratic |
    | Length scales | one per input for RBF and Matérn; none for IBNN | one per input for RBF and Matérn; one shared for RationalQuadratic |
    | Output scale and mean | `ScaleKernel`, constant mean | `ConstantKernel` times the kernel, zero mean |
    | Noise, no Noise column | one learned variance, log-normal prior | none: `alpha` stays 1e-10, so the fit interpolates |
    | Noise column | per-run variances `train_Yvar`, fixed | per-run values passed as `alpha` |
    | Categorical variables | `MixedSingleTaskGP` | one-hot columns (first level dropped), ordinary kernel |
    | Hyperparameter fit | `fit_gpytorch_mll` with BoTorch's default L-BFGS-B settings (bug T22) | L-BFGS-B, 30 restarts |
    | Input transforms | `normalize` (to [0, 1]), `standardize`, `none` | `minmax`, `standard`, `robust`, `none` |
    | Output transforms | `standardize`, `none` | `standard`, `none` |

    Reported uncertainties are latent: `predict` returns $\sigma$ without the $\sigma_n^2$ term. Both backends then multiply it by a cross-validation calibration factor computed on every train; the factor partly absorbs the noise (bug T4), so check [Diagnostics](diagnostics.md) before adding $\sigma_n^2$ to a calibrated $\sigma$. Acquisition uses the uncalibrated value. The web toggle does not turn calibration off (bug F1).

    For categoricals, BoTorch codes each level as an integer by its position in the variable's declared list, so a level with no runs yet can still be suggested. When at least one input is continuous, it fits the sum-plus-product construction above. The sum and product terms each have their own continuous and categorical kernels, so each has its own length scales. The categorical kernel decays with the mean mismatch across categorical inputs, with one $\ell$ per input.

    Defaults depend on how you train. Python `train_model` uses scikit-learn, Matérn and no transforms; with `backend="botorch"` and no transform arguments it applies `normalize` and `standardize`. Over REST, omitted transforms get the same backend defaults. The web Model panel starts on scikit-learn with RBF and no transforms; choosing BoTorch selects Matérn 2.5 with `normalize` and `standardize` and always sends the selected transforms, so `none` chosen there trains unscaled.

!!! warning "Known issue"
    Bug T6: for BoTorch, the input transform `standardize` is built as `Normalize`, which rescales to [0, 1], not to zero mean and unit variance. Use `normalize` for inputs; `standardize` is correct for outputs. Tracked for a 0.4.x release.

!!! warning "Known issue"
    Bug T17: in scikit-learn only the `standard` output transform works. `minmax` and `robust`, which the web list offers, silently leave the output unscaled. Tracked for a 0.4.x release.

!!! warning "Known issue"
    Bug T18: with a Noise column and `standard` output scaling, scikit-learn passes `alpha` unscaled while `y` is scaled, so the noise is mis-weighted. Leave output scaling off, or enter the noise already divided by the variance of `y`. Bug T19: the code treats the Noise column as a variance, though the API descriptions say "measurement uncertainty"; if you have a standard deviation, enter its square. Tracked for a 0.4.x release.

!!! warning "Known issue"
    Bug F2: the scikit-learn Matérn $\nu$ sent from the web, or as `kernel_params={"nu": ...}` from Python, is ignored. The fitted kernel always has $\nu = 1.5$. From Python, `kernel_params={"matern_nu": ...}` is honored. BoTorch maps $\nu$ correctly. Tracked for a 0.4.x release.

!!! warning "Known issue"
    Bug T22: ALchemist passes its BoTorch iteration limit (`training_iter`, 50) to `fit_gpytorch_mll` in a form that BoTorch ignores (checked on 0.18.1), so the cap never applies. Every BoTorch fit, including the cross-validation refits, runs L-BFGS-B with BoTorch's default settings. Tracked for a 0.4.x release.

## Further reading

- [Garnett (2023)](references.md#garnett-2023), Chapters 2 to 4.
- [Rasmussen and Williams (2006)](references.md#rasmussen-2006), Chapters 2, 4, 5.
- [Stein (1999)](references.md#stein-1999), Matérn versus RBF.
- [Lee et al. (2018)](references.md#lee-2018) and [Li et al. (2023)](references.md#li-2023), IBNN.
- [Häse et al. (2021)](references.md#hase-2021), categoricals in chemistry.
