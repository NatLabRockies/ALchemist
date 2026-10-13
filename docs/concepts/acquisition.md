# Acquisition functions

An acquisition function scores every candidate experiment using the Gaussian-process posterior, and ALchemist suggests the candidate with the highest score. The choice sets how hard the loop pushes toward the best value so far and how much it spends on regions the model knows little about.

## Notation

At an input $x$ the posterior (see [Gaussian processes](gaussian-processes.md)) gives a mean $\mu(x)$ and standard deviation $\sigma(x)$. The incumbent $f^*$ is the value a new point must beat, $\Phi$ and $\phi$ are the standard normal cdf and pdf, and $z = (\mu - f^*)/\sigma$. Formulas are for maximization.

## Improvement-based functions

**Probability of improvement** (PI, [Kushner 1964](references.md#kushner-1964)) is the probability that a new value exceeds a threshold $\tau = f^* + \varepsilon$. For a Gaussian process ([Garnett 2023, eq. 7.5-7.6 and 8.21](references.md#garnett-2023)):

$$\alpha_{\mathrm{PI}}(x) = \Pr\big(f(x) > \tau\big) = \Phi\!\left(\frac{\mu(x) - \tau}{\sigma(x)}\right)$$

PI ignores the size of the gain, making it more risk-averse than EI ([Garnett 2023, §7.5](references.md#garnett-2023)). The margin $\varepsilon$ (called $\xi$ in scikit-optimize) pushes it toward exploration.

**Expected improvement** (EI, [Jones et al. 1998](references.md#jones-1998)) weights the size of the gain ([Garnett 2023, eq. 7.3 and 8.9](references.md#garnett-2023)):

$$\alpha_{\mathrm{EI}}(x) = (\mu - f^*)\,\Phi(z) + \sigma\,\phi(z)$$

For minimization, $f^*$ is the lowest value and $\alpha_{\mathrm{EI}} = (f^* - \mu)\,\Phi(z') + \sigma\,\phi(z')$ with $z' = (f^* - \mu)/\sigma$ ([Jones et al. 1998](references.md#jones-1998)).

**Noisy incumbents.** Noise makes the best observed value overstate what the model can reach. [Garnett (2023, eq. 8.17-8.18)](references.md#garnett-2023) compares the plug-ins $f^* \approx \max y$ and the less-biased $f^* \approx \max_i \mu(x_i)$ over the observed inputs. Noisy EI integrates over the latent values instead ([Letham et al. 2019](references.md#letham-2019); [Picheny et al. 2013](references.md#picheny-2013)).

**LogEI.** EI and its gradient underflow to zero where improvement is nearly impossible, leaving optimizers on a flat surface ([Garnett 2023, §9.2](references.md#garnett-2023)). LogEI computes $\log \alpha_{\mathrm{EI}}$ stably; the maximizer is unchanged ([Ament et al. 2023](references.md#ament-2023)). It is a numerical fix, not a noise model.

## Upper confidence bound

$$\alpha_{\mathrm{UCB}}(x) = \mu(x) + \beta\,\sigma(x)$$

where $\beta = \Phi^{-1}(\pi)$ for confidence level $\pi$ ([Garnett 2023, eq. 8.25, §7.8](references.md#garnett-2023)). Many libraries use $\sqrt{\beta}\,\sigma$ ([Garnett 2023, footnote 21, §8.4](references.md#garnett-2023)); quote the convention with the number. The guarantees of [Srinivas et al. (2010)](references.md#srinivas-2010) need $\beta$ to grow with the iteration count; a fixed $\beta$ is a heuristic.

!!! info "How ALchemist does it"
    `BoTorchAcquisition` in `alchemist_core/acquisition/botorch_acquisition.py` ([Python reference](../reference/python/acquisition.md)), called by `OptimizationSession.suggest_next` ([reference](../reference/python/session.md), [guide](../guide/suggest.md)).

    - `EI` builds BoTorch's `LogExpectedImprovement`, not plain EI (`_create_acquisition_function`, lines 218-227). `LogEI` builds the same object (line 241) unless an outcome constraint is registered, so the two choose the same point.
    - The incumbent is the best posterior mean at the training inputs, highest for maximize and lowest for minimize (`_compute_incumbent_best_f`, lines 145-175), the second plug-in above, with the best observed value as fallback. PI and EI take no margin ($\varepsilon = 0$).
    - `UCB` is BoTorch's `UpperConfidenceBound`, which uses $\sqrt{\beta}\,\sigma$, with default $\beta = 0.5$ (line 121): a multiplier of $\sqrt{0.5} \approx 0.71$ on $\sigma$, constant over iterations; `qUCB` (BoTorch's `qUpperConfidenceBound`) matches it at $q = 1$, with the same default (line 124). From Python pass `beta=`; the REST API and web app have no $\beta$ field (`AcquisitionRequest` in `api/models/requests.py`) and use 0.5.
    - `EI`, `LogEI`, `PI`, `LogPI` and `UCB` pass the goal on (`maximize=`), so minimize works.

## Batches by Monte Carlo

The value of $q$ points at once is an integral over their joint posterior, estimated by sampling $y_i = \boldsymbol{\mu} + \Lambda z_i$ with $\Lambda\Lambda^\top = S$ and $z_i$ standard normal ([Garnett 2023, eq. 11.13-11.15, §11.3](references.md#garnett-2023)):

$$\beta(X) \approx \frac{1}{n}\sum_{i=1}^{n} \Delta(X, y_i)$$

For qEI, $\Delta$ is the best improvement over $f^*$ among the $q$ samples ([Ginsbourger et al. 2010](references.md#ginsbourger-2010)). Writing samples as smooth functions of $z_i$ gives gradients ([Wilson et al. 2018](references.md#wilson-2018), [Balandat et al. 2020](references.md#balandat-2020)).

!!! info "How ALchemist does it"
    - `n_suggestions` becomes the batch size $q$ (`session.py`, line 1940). Names starting with `q` (`qEI`, `qUCB`, `qNIPV`, `qEHVI`, `qNEHVI`) optimize the $q$ points jointly (line 478).
    - The default is 128 seeded Sobol samples (`mc_samples`, line 127).
    - The code never sets pending points ([Garnett 2023, §11.4](references.md#garnett-2023)), so suggestions waiting in the queue do not influence the next batch.

!!! warning "Known issue"
    A1: on the BoTorch backend, `EI`, `PI` or `UCB` with `n_suggestions` above 1 raises an `AssertionError` (they need $q = 1$), and the REST API returns a bare 500. On the sklearn backend one row comes back. Use `qEI` or `qUCB` for batches. Tracked for a 0.4.x release.

!!! warning "Known issue"
    T7: `qEI` and `qUCB` are built without a `maximize` argument (lines 275-290) and always maximize, so `goal="minimize"` is not honoured. Test: on the bowl $f = (x_1 - 0.3)^2 + (x_2 - 0.7)^2$ with 12 random points, `EI` and `UCB` with `goal="minimize"` suggested points beside the minimum (true values 0.002, 0.028); `qEI` and `qUCB` suggested the corner $(1, 0)$, the worst point (0.98), and `qUCB` matched its `goal="maximize"` output. Until then, negate the response and maximize instead. Tracked for a 0.4.x release.

## Exploration only: integrated posterior variance

qNIPV scores a candidate by how much it lowers the posterior variance averaged over the search space, $-\tfrac{1}{|M|}\sum_{m \in M}\mathrm{Var}[f(m) \mid \mathcal{D} \cup \{x\}]$ over sample points $M$. A Gaussian-process variance does not depend on the observed $y$ ([Garnett 2023, eq. 2.19](references.md#garnett-2023)), so it is a space-filling criterion in the information-gain family ([Garnett 2023, §6.3, §7.6](references.md#garnett-2023)), related to uncertainty sampling ([§10.5](references.md#garnett-2023)).

!!! info "How ALchemist does it"
    `qNIPV` (alias `qIPV`) draws 500 uniform random points inside the search-space bounds as $M$ (`n_mc_points`, lines 295-300). It ignores the goal, accepts `n_suggestions` above 1, and raises an error if context variables are registered (lines 422-426).

## Multi-objective: EHVI

With several objectives the target is the Pareto front. Progress is the volume dominated by the observations above a reference point $r$ ([Zitzler and Thiele 1999](references.md#zitzler-1999)); expected hypervolume improvement ([Emmerich et al. 2006](references.md#emmerich-2006)) is its expected gain:

$$\alpha_{\mathrm{EHVI}}(x) = \mathbb{E}\big[\mathrm{HV}(\mathcal{P} \cup \{y\};\, r) - \mathrm{HV}(\mathcal{P};\, r)\big]$$

The reference point must be clearly worse than anything attainable ([Garnett 2023, §11.7](references.md#garnett-2023)), and hypervolume depends on $r$ and on each objective's scale. qEHVI ([Daulton et al. 2020](references.md#daulton-2020)) extends EHVI to batches; qNEHVI ([Daulton et al. 2021](references.md#daulton-2021)) also integrates over uncertainty in the observed values, for noisy objectives. A common heuristic is a fixed $r$ slightly worse than the worst acceptable value of each objective.

!!! info "How ALchemist does it"
    `qEHVI` and `qNEHVI` are in the core (`_create_mobo_acquisition_function`, lines 326-394) and the desktop app (`ui/acquisition_panel.py`). Objectives are converted to maximization form; qEHVI partitions the observed values directly (line 368); qNEHVI uses the training inputs as its baseline (lines 380-389). Pass `ref_point` in the original units. The default is, per objective, the worst observed value minus 10 percent of its magnitude, with a minimum margin of $10^{-6}$ (`_compute_default_ref_point`, lines 311-324), recomputed from the data on every call, so it moves as data arrive. The REST API and web app have no multi-objective acquisition or Pareto view yet; see [Multi-objective](../guide/more/multi-objective.md).

## Scikit-learn backend and GP-Hedge

The sklearn backend (`SkoptAcquisition`, `alchemist_core/acquisition/skopt_acquisition.py`) uses scikit-optimize for `EI`, `PI`, `UCB` and `gp_hedge`. GP-Hedge ([Brochu et al. 2011](references.md#brochu-2011)) picks among EI, PI and UCB by a softmax over gains accumulated across iterations.

!!! info "How ALchemist does it"
    - $\xi = 0.01$ in the units of the response, so its effect depends on the output scale (lines 73-74). `UCB` is skopt's lower confidence bound on the sign-flipped response, giving $\mu + \kappa\sigma$ with $\kappa = 1.96$, the 97.5 percent normal quantile (lines 62, 75-76; [Garnett 2023, eq. 8.25](references.md#garnett-2023)).
    - The incumbent is the best observed value, not the best posterior mean.
    - Every `suggest_next` call builds a fresh skopt `Optimizer` (`session.py`, line 1919). It refits a clone of the estimator from its initial kernel on skopt's normalized inputs, so it is not the model shown in the diagnostics.

!!! warning "Known issue"
    T8: skopt's `Optimizer` defaults to `n_initial_points = 10`, which ALchemist does not change, so with fewer than 10 observations the sklearn backend returns a random point. `gp_hedge` starts from zero gains on every call, so its softmax is uniform and it picks at random among the EI, PI and LCB proposals; the web app's "based on their past performance" does not hold. Tracked for a 0.4.x release.

## Further reading

- [Garnett (2023)](references.md#garnett-2023), chapters 7, 8, 9 and 11.
- [Ament et al. (2023)](references.md#ament-2023), LogEI.
- [Srinivas et al. (2010)](references.md#srinivas-2010), UCB regret bounds.
- [Wilson et al. (2018)](references.md#wilson-2018) and [Balandat et al. (2020)](references.md#balandat-2020), Monte-Carlo acquisition.
- [Daulton et al. (2020)](references.md#daulton-2020), multi-objective acquisition.
