# Acquisition functions

An acquisition function scores every candidate experiment from the Gaussian-process posterior, and ALchemist suggests the highest-scoring one. The choice sets how hard the loop pushes toward the best value so far versus regions the model knows little about.

## Notation

At an input $x$ the posterior (see [Gaussian processes](gaussian-processes.md)) gives a mean $\mu(x)$ and standard deviation $\sigma(x)$. The incumbent $f^*$ is the value a new point must beat, $\Phi$ and $\phi$ are the standard normal cdf and pdf, and $z = (\mu - f^*)/\sigma$. Formulas maximize.

## Improvement-based functions

**Probability of improvement** (PI, [Kushner 1964](references.md#kushner-1964)) is the probability that a new value exceeds a threshold $\tau = f^* + \varepsilon$. For a Gaussian process ([Garnett 2023, eq. 7.5-7.6 and 8.21](references.md#garnett-2023)):

$$\alpha_{\mathrm{PI}}(x) = \Pr\big(f(x) > \tau\big) = \Phi\!\left(\frac{\mu(x) - \tau}{\sigma(x)}\right)$$

PI ignores the size of the gain, so it is more risk-averse than EI ([Garnett 2023, §7.5](references.md#garnett-2023)). The margin $\varepsilon$ (called $\xi$ in scikit-optimize) pushes it toward exploration.

**Expected improvement** (EI, [Jones et al. 1998](references.md#jones-1998)) weights the size of the gain ([Garnett 2023, eq. 7.3 and 8.9](references.md#garnett-2023)):

$$\alpha_{\mathrm{EI}}(x) = (\mu - f^*)\,\Phi(z) + \sigma\,\phi(z)$$

For minimization, $f^*$ is the lowest value and $\alpha_{\mathrm{EI}} = (f^* - \mu)\,\Phi(z') + \sigma\,\phi(z')$ with $z' = (f^* - \mu)/\sigma$ ([Jones et al. 1998](references.md#jones-1998)).

With noisy observations, the best observed value overstates what the model can reach. [Garnett (2023, eq. 8.17-8.18)](references.md#garnett-2023) compares the plug-ins $f^* \approx \max y$ and the less-biased $f^* \approx \max_i \mu(x_i)$ over the observed inputs; [Picheny et al. (2013)](references.md#picheny-2013) benchmark such choices. Noisy EI integrates over the latent values instead ([Letham et al. 2019](references.md#letham-2019)).

A flat acquisition surface stalls gradient-based optimizers ([Garnett 2023, §9.2](references.md#garnett-2023)), and EI and its gradient underflow to zero where improvement is nearly impossible ([Ament et al. 2023](references.md#ament-2023)). **LogEI** computes $\log \alpha_{\mathrm{EI}}$ stably and leaves the maximizer unchanged. It does not model noise.

## Upper confidence bound

$$\alpha_{\mathrm{UCB}}(x) = \mu(x) + \beta\,\sigma(x)$$

where $\beta = \Phi^{-1}(\pi)$ for confidence level $\pi$ ([Garnett 2023, eq. 8.25, §7.8](references.md#garnett-2023)). Many libraries use $\sqrt{\beta}\,\sigma$ ([Garnett 2023, footnote 21, §8.4](references.md#garnett-2023)), so quote the convention. The guarantees of [Srinivas et al. (2010)](references.md#srinivas-2010) need $\beta$ to grow with the iteration count; a fixed $\beta$ is a heuristic.

!!! info "How ALchemist does it"
    `BoTorchAcquisition` ([Python reference](../reference/python/acquisition.md)), called by `OptimizationSession.suggest_next` ([reference](../reference/python/session.md), [guide](../guide/suggest.md)).

    - `EI` builds BoTorch's `LogExpectedImprovement`, not plain EI (`_create_acquisition_function`). `LogEI` builds the same object, so the two choose the same point, unless an outcome constraint is registered (below).
    - The incumbent is the best posterior mean at the training inputs, highest for maximize and lowest for minimize (`_compute_incumbent_best_f`), the second plug-in above (fallback: best observed value). With outcome constraints registered, only training inputs whose posterior mean satisfies every constraint are candidates for the incumbent. PI and EI take no margin ($\varepsilon = 0$).
    - `UCB` is BoTorch's `UpperConfidenceBound`, which uses $\sqrt{\beta}\,\sigma$, with default $\beta = 0.5$ (multiplier $\sqrt{0.5} \approx 0.71$, constant); `qUCB` matches it at $q = 1$. From Python pass `beta=`. The REST API has no $\beta$ field and the web app's qUCB $\beta$ control is sent as `kappa` and ignored, so both use $\beta = 0.5$ (T13, below).
    - `EI`, `LogEI`, `PI`, `LogPI` and `UCB` pass the goal on (`maximize=`). With an outcome constraint registered, `EI` and `LogEI` both build BoTorch's `qExpectedImprovement` at $q = 1$ with the constraint, since the analytic functions have no constraint support; the goal is kept through a negating objective.

## Batches by Monte Carlo

The value of $q$ points at once is an integral over their joint posterior, estimated by sampling $y_i = \boldsymbol{\mu} + \Lambda z_i$, where $\boldsymbol{\mu}$ and $S$ are the joint posterior mean and covariance of the $q$ values, $\Lambda\Lambda^\top = S$ and $z_i$ is standard normal ([Garnett 2023, eq. 11.13-11.15, §11.3](references.md#garnett-2023)):

$$\alpha(X) \approx \frac{1}{n}\sum_{i=1}^{n} \Delta(X, y_i)$$

For qEI, $\Delta$ is the best improvement over $f^*$ among the $q$ samples ([Ginsbourger et al. 2010](references.md#ginsbourger-2010)). Samples that are smooth in $z_i$ give gradients ([Wilson et al. 2018](references.md#wilson-2018), [Balandat et al. 2020](references.md#balandat-2020)).

!!! info "How ALchemist does it"
    - `suggest_next` passes `n_suggestions` on as the batch size $q$. Only `q`-prefixed functions (`qEI`, `qUCB`, `qNIPV`, `qEHVI`, `qNEHVI`) score a joint batch. `EI`, `LogEI`, `PI`, `LogPI` and `UCB` score one point, so asking them for more than one raises a `ValueError` that names `qEI`, `qUCB` and `qIPV` (the alias of `qNIPV`; HTTP 400 over REST). The scikit-learn backend suggests one point at a time and refuses any batch the same way.
    - The default is 128 seeded Sobol samples (`mc_samples`).
    - No pending points are set ([Garnett 2023, §11.4](references.md#garnett-2023)), so queued suggestions do not influence the next batch.

    - BoTorch's Monte Carlo functions always maximize their objective, so for `goal="minimize"` ALchemist gives `qEI` and `qUCB` an objective that negates the model output (and negates the incumbent with it). Test: on the bowl $f = (x_1 - 0.3)^2 + (x_2 - 0.7)^2$ with 12 random points, every strategy minimizing lands beside the minimum $(0.3, 0.7)$ and maximizing lands at the corner $(1, 0)$.

!!! warning "Known issue"
    T13: through the REST API and web app, UCB and qUCB always use $\beta = 0.5$: the API has no $\beta$ field, and the web app's qUCB control sends `kappa`, which the BoTorch path drops. Set `beta` from Python. Tracked for a 0.4.x release.

## Exploration only: integrated posterior variance

qNIPV scores a candidate by how much it lowers the posterior variance averaged over the search space, $-\tfrac{1}{|M|}\sum_{m \in M}\mathrm{Var}[f(m) \mid \mathcal{D} \cup \{x\}]$ over sample points $M$. A Gaussian-process variance does not depend on the observed $y$ ([Garnett 2023, eq. 2.19](references.md#garnett-2023)), so it is a space-filling criterion in the information-gain family ([Garnett 2023, §6.3, §7.6](references.md#garnett-2023)).

!!! info "How ALchemist does it"
    `qNIPV` (alias `qIPV`) draws 500 uniform random points inside the search-space bounds as $M$ (`n_mc_points`). It ignores the goal, accepts `n_suggestions` above 1, and errors if context variables are registered.

## Multi-objective: EHVI

With several objectives the target is the Pareto front. Progress is the volume dominated by the observations above a reference point $r$ ([Zitzler and Thiele 1999](references.md#zitzler-1999)); expected hypervolume improvement ([Emmerich et al. 2006](references.md#emmerich-2006)) is its expected gain:

$$\alpha_{\mathrm{EHVI}}(x) = \mathbb{E}\big[\mathrm{HV}(\mathcal{P} \cup \{y\};\, r) - \mathrm{HV}(\mathcal{P};\, r)\big]$$

The reference point must be clearly worse than anything attainable ([Garnett 2023, §11.7](references.md#garnett-2023)); hypervolume depends on $r$ and objective scale. qEHVI ([Daulton et al. 2020](references.md#daulton-2020)) handles batches; qNEHVI ([Daulton et al. 2021](references.md#daulton-2021)) also integrates over uncertainty in observed values, for noisy objectives.

!!! info "How ALchemist does it"
    `qEHVI` and `qNEHVI` are in the core (`_create_mobo_acquisition_function`) and the desktop app. Objectives are converted to maximization form; qEHVI partitions the observed values directly. Pass `ref_point` in the original units. The default lies, per objective, beyond the worst observed value by 10 percent of its magnitude, with a minimum margin of $10^{-6}$ (`_compute_default_ref_point`), recomputed on every call, so it moves as data arrive. The web app has no multi-objective acquisition or Pareto view yet; see [Multi-objective](../guide/more/multi-objective.md).

## Scikit-learn backend and GP-Hedge

The sklearn backend (`SkoptAcquisition`) uses scikit-optimize for `EI`, `PI`, `UCB` and `gp_hedge`. GP-Hedge ([Brochu et al. 2011](references.md#brochu-2011)) picks among EI, PI and UCB by a softmax over gains accumulated across iterations.

!!! info "How ALchemist does it"
    - $\xi = 0.01$ in the units of the response, so its effect depends on output scale. `UCB` is skopt's lower confidence bound on the sign-flipped response, giving $\mu + \kappa\sigma$ with $\kappa = 1.96$, the 97.5 percent normal quantile ([Garnett 2023, eq. 8.25](references.md#garnett-2023)); Python and the web app use 1.96, while a REST call without `kappa` defaults to 2.0 (`AcquisitionRequest`).
    - The incumbent is the best observed value, not the best posterior mean.
    - Every `suggest_next` call builds a fresh `SkoptAcquisition`, whose constructor builds a new skopt `Optimizer` that refits a clone of the estimator from its initial kernel on normalized inputs. That model is not the one shown in the diagnostics.

!!! warning "Known issue"
    T8: skopt's `Optimizer` defaults to `n_initial_points = 10`, which ALchemist does not change, so with fewer than 10 observations the sklearn backend returns a random point. `gp_hedge` starts from zero gains on every call, so its softmax is uniform and it picks at random among the EI, PI and LCB proposals. Tracked for a 0.4.x release.

## Further reading

- [Garnett (2023)](references.md#garnett-2023), chapters 7, 8, 9 and 11.
- [Ament et al. (2023)](references.md#ament-2023), LogEI.
- [Wilson et al. (2018)](references.md#wilson-2018), Monte-Carlo acquisition.
- [Daulton et al. (2020)](references.md#daulton-2020), multi-objective acquisition.
