# Gaussian processes and kernels

A Gaussian process (GP) is the surrogate model behind ALchemist's predictions and its uncertainty bands. Knowing what its kernel and noise settings assume tells you how to read every suggestion and plot.

## From prior to posterior

A GP is a distribution over functions. The prior $f \sim \mathcal{GP}(\mu, K)$ says what functions are plausible before any data arrive: $\mu$ is the average level and the kernel $K(x, x')$ says how strongly the values at two inputs move together. Observing runs $\mathbf{y}$ at inputs $X$ updates the prior to a posterior that is again a GP ([Garnett 2023, §2.2](references.md#garnett-2023); [Rasmussen and Williams 2006, ch. 2](references.md#rasmussen-2006)).

The latent function $f$ and the observation $y$ differ by measurement error: $y = f(x) + \varepsilon$, with $\varepsilon \sim \mathcal{N}(0, \mathbf{N})$ ([Garnett 2023, eq. 2.15](references.md#garnett-2023)). Two common choices for $\mathbf{N}$ are a single shared noise variance, $\sigma_n^2 \mathbf{I}$ (homoskedastic, eq. 2.16), and a separate variance per run, $\operatorname{diag}\sigma_n^2(x)$ (heteroskedastic, eq. 2.17). With $\mathbf{K} = K(X, X)$, the posterior is

$$
\mu_{\mathcal{D}}(x) = \mu(x) + K(x, X)\,(\mathbf{K} + \mathbf{N})^{-1}(\mathbf{y} - \boldsymbol{\mu}), \qquad
K_{\mathcal{D}}(x, x') = K(x, x') - K(x, X)\,(\mathbf{K} + \mathbf{N})^{-1} K(X, x')
$$

([Garnett 2023, eq. 2.19](references.md#garnett-2023); with $\mu = 0$ and $\mathbf{N} = \sigma_n^2\mathbf{I}$ these are [Rasmussen and Williams 2006, eqs. 2.23 and 2.24](references.md#rasmussen-2006)). With $\mathbf{N} = 0$ the posterior passes exactly through the data ([Garnett 2023, eq. 2.10](references.md#garnett-2023)).

The posterior variance $\sigma^2(x) = K_{\mathcal{D}}(x, x)$ describes the latent $f$. A new measurement is scattered around $f$ by the noise as well:

$$
p(y \mid x, \mathcal{D}) = \mathcal{N}\big(y;\ \mu_{\mathcal{D}}(x),\ \sigma^2(x) + \sigma_n^2\big)
$$

([Garnett 2023, eq. 2.20](references.md#garnett-2023)). A band drawn from $\sigma$ alone shows where $f$ may lie. A band that covers new measurements must include $\sigma_n^2$.

## Kernels

ALchemist writes a kernel as an output scale times a unit-scale shape, $K(x, x') = \sigma_f^2\, k(d)$ ([Garnett 2023, eq. 3.20](references.md#garnett-2023)). The distance $d$ divides each coordinate by its own length scale,

$$
d = \sqrt{\sum_i \left(\frac{x_i - x_i'}{\ell_i}\right)^{2}}
$$

([Garnett 2023, eq. 3.24](references.md#garnett-2023)). A separate $\ell_i$ per input is automatic relevance determination (ARD, [Garnett 2023, eq. 3.25](references.md#garnett-2023); [Rasmussen and Williams 2006, §5.1](references.md#rasmussen-2006)). If the response barely depends on input $i$, its fitted $\ell_i$ tends to be long, but with few runs length scales are only weakly determined, so do not read $1/\ell_i$ as an importance score.

| Kernel | $k(d)$ | Smoothness |
|---|---|---|
| RBF | $\exp(-d^2/2)$ ([eq. 3.12](references.md#garnett-2023)) | infinitely differentiable |
| Matérn, $\nu = 1/2$ | $\exp(-d)$ ([eq. 3.11](references.md#garnett-2023)) | continuous, not differentiable |
| Matérn, $\nu = 3/2$ | $(1 + \sqrt{3}\,d)\,e^{-\sqrt{3}\,d}$ ([eq. 3.13](references.md#garnett-2023)) | once differentiable |
| Matérn, $\nu = 5/2$ | $(1 + \sqrt{5}\,d + \tfrac{5}{3}d^2)\,e^{-\sqrt{5}\,d}$ ([eq. 3.14](references.md#garnett-2023)) | twice differentiable |
| Rational quadratic | $(1 + d^2/2\alpha)^{-\alpha}$ ([R&W eq. 4.19](references.md#rasmussen-2006)) | infinitely differentiable (a scale mixture of RBFs) |

Matérn sample paths are $\lceil\nu\rceil - 1$ times differentiable, and $\nu \to \infty$ gives the RBF ([Garnett 2023, §3.3](references.md#garnett-2023)). Garnett notes that $\nu = 5/2$ is a prudent default for Bayesian optimization ([Snoek et al. 2012](references.md#snoek-2012)) and that the RBF's infinite smoothness is often unrealistic ([Stein 1999](references.md#stein-1999)).

**Fitting.** The kernel's length scales, the output scale and the noise are hyperparameters $\theta$. They are chosen by maximizing the marginal likelihood, optionally weighted by a prior on $\theta$ ([Garnett 2023, eq. 4.9](references.md#garnett-2023)):

$$
\log p(\mathbf{y} \mid X, \theta) = -\tfrac{1}{2}\Big[(\mathbf{y} - \boldsymbol{\mu})^{\top}(\mathbf{K} + \mathbf{N})^{-1}(\mathbf{y} - \boldsymbol{\mu}) + \log|\mathbf{K} + \mathbf{N}| + n\log 2\pi\Big]
$$

([Garnett 2023, eq. 4.8](references.md#garnett-2023)). The first term rewards fitting the data and the second penalizes complexity. A single best $\theta$ ignores uncertainty in $\theta$ itself, which matters most with few runs ([Garnett 2023, §4.4 and §9.3](references.md#garnett-2023)).

**Infinite-width neural-network kernel (IBNN).** The IBNN kernel is the GP that a deep, infinitely wide ReLU network defines ([Lee et al. 2018](references.md#lee-2018)). It starts from $K^0(x, x') = \sigma_b^2 + \sigma_w^2\, x\!\cdot\!x'/d_{\text{in}}$ (eq. 6 there) and applies one recursion per layer (eq. 4 there), where $\sigma_w^2$ and $\sigma_b^2$ are weight and bias variances. It has no length scales and depends on $x \cdot x'$, so it is not stationary and changes if you shift the inputs. [Li et al. (2023)](references.md#li-2023) found infinite-width networks promising as surrogates, especially in higher dimensions, with no single surrogate best on every problem.

## Categorical variables

A kernel needs a similarity between categories. A common construction compares whether two categories match and combines that with a continuous kernel by sums and products, which are valid because sums and products of kernels are kernels ([Garnett 2023, eq. 3.29](references.md#garnett-2023)). Another route is to encode categories as numeric columns and use an ordinary kernel. Gryffin takes a different route and builds expert knowledge about the categories into the model ([Häse et al. 2021](references.md#hase-2021)).

!!! info "How ALchemist does it"
    Two classes implement the surrogate: `BoTorchModel` in `alchemist_core/models/botorch_model.py` and `SklearnModel` in `alchemist_core/models/sklearn_model.py` ([Models reference](../reference/python/models.md)). `OptimizationSession.train_model` in `alchemist_core/session.py` builds them from the `kernel`, `kernel_params`, `input_transform_type` and `output_transform_type` arguments.

    | | BoTorch | scikit-learn |
    |---|---|---|
    | Kernels | RBF, Matérn with $\nu \in \{0.5, 1.5, 2.5\}$ (default 2.5), IBNN with `ibnn_depth` (default 3) | RBF, Matérn (default $\nu = 1.5$; the web list adds `inf`), RationalQuadratic |
    | Length scales | one per input for RBF and Matérn; none for IBNN | one per input for RBF and Matérn; one shared for RationalQuadratic |
    | Output scale and mean | `ScaleKernel`, constant mean | `ConstantKernel` times the kernel, zero mean |
    | Noise, no Noise column | one noise variance, learned, with a log-normal prior | none: `alpha` stays at the library's 1e-10, so the fit interpolates |
    | Noise column | per-run variances `train_Yvar`, fixed | per-run values passed as `alpha`, unscaled |
    | Categorical variables | `MixedSingleTaskGP` | one-hot columns (first level dropped), ordinary kernel |
    | Hyperparameter fit | marginal likelihood with `fit_gpytorch_mll`, 50 iterations by default | marginal likelihood, L-BFGS-B, 30 restarts |
    | Input transforms | `normalize` (to [0, 1]), `standardize`, `none` | `minmax`, `standard`, `robust`, `none` |
    | Output transforms | `standardize`, `none` | `standard`, `none` |

    Reported uncertainties are latent: `predict` returns $\sigma$ without the $\sigma_n^2$ term in both backends. The Noise column holds a variance per run, not a standard deviation. In scikit-learn only the `standard` output transform takes effect: `minmax` and `robust`, which the web list also offers, currently leave the output unscaled. For categoricals, BoTorch codes each level as an integer in order of first appearance in the training data (`_encode_categorical_data`) and fits, when at least one input is continuous, $K = \sigma_1^2\,[\,k_{\text{cont}} + \sigma_2^2\,k_{\text{cat}}\,] + \sigma_3^2\,k_{\text{cont}}\,k_{\text{cat}}$. Each $k_{\text{cont}}$ is the chosen continuous kernel with its own length scales. The categorical part is $k_{\text{cat}} = \exp\!\big(-\tfrac{1}{m}\sum_j \delta_j/\ell_j\big)$, where $\delta_j$ is 1 if the $j$-th of the $m$ categorical inputs differs between the two points and 0 if it matches.

    Defaults depend on how you train. Python `train_model` uses scikit-learn, a Matérn kernel and no transforms; with `backend="botorch"` and no transform arguments it applies `normalize` to inputs and `standardize` to outputs. The web Model panel starts on scikit-learn with an RBF kernel (BoTorch: Matérn 2.5) and sends no transform unless you select one.

!!! warning "Known issue"
    Bug W2: the web Model panel's default of no transforms reaches the BoTorch model as "no transform", and on data far from unit scale the fit can fail, with length scales stuck at their starting value and $R^2$ of 0. Enable the advanced options and choose `normalize` for inputs and `standardize` for outputs. Tracked for a 0.4.x release.

!!! warning "Known issue"
    Bug T6: for BoTorch, the input transform named `standardize` is built as `Normalize`, which rescales to [0, 1] rather than to zero mean and unit variance. Use `normalize` for inputs and keep `standardize` for outputs, where it does standardize. Tracked for a 0.4.x release.

!!! warning "Known issue"
    Bug F2: the scikit-learn Matérn $\nu$ sent from the web, or as `kernel_params={"nu": ...}` from Python, is ignored. The fitted kernel always has $\nu = 1.5$. From Python, `kernel_params={"matern_nu": ...}` is honored. BoTorch maps $\nu$ correctly. Tracked for a 0.4.x release.

!!! warning "Known issue"
    Bug A11: BoTorch indexes categorical levels from the training data, not from the levels you declared, so a level with no recorded runs can never be suggested. Include every level in the initial design. Tracked for a 0.4.x release.

## Further reading

- [Garnett (2023)](references.md#garnett-2023), Chapters 2 to 4.
- [Rasmussen and Williams (2006)](references.md#rasmussen-2006), Chapters 2, 4 and 5.
- [Stein (1999)](references.md#stein-1999), the case for Matérn over RBF.
- [Lee et al. (2018)](references.md#lee-2018) and [Li et al. (2023)](references.md#li-2023), the IBNN kernel.
- [Häse et al. (2021)](references.md#hase-2021), categorical variables in chemistry.
