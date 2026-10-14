# Bayesian optimization

Bayesian optimization (BO) finds good experimental conditions by fitting a probabilistic model to the results so far and using that model to choose the next run. It is the loop ALchemist runs: steps 3 to 7 of the guide are one pass around it.

## The loop

Sequential optimization starts from a dataset $\mathcal{D}$ of runs, which may be empty. It chooses a point with a policy, observes the result there, adds the pair to the dataset, and repeats until a termination condition is met ([Garnett 2023, Alg. 1.1](references.md#garnett-2023)). BO is a way of building the policy. It fits a surrogate model to $\mathcal{D}$, scores every candidate point with an acquisition function $\alpha$, and observes at the highest score:

$$
x_{t+1} \in \arg\max_{x \in \mathcal{X}} \; \alpha(x;\, \mathcal{D}_t)
$$

This is the generic form of an optimization policy ([Garnett 2023, eq. 7.1](references.md#garnett-2023)). The surrogate in ALchemist is a Gaussian process, described in [Gaussian processes and kernels](gaussian-processes.md); the acquisition functions are described in [Acquisition](acquisition.md).

BO suits problems where each evaluation is expensive, results are noisy, and there is no closed form or gradient to exploit. Garnett calls this black-box optimization and counts it among BO's main strengths ([Garnett 2023, §1.1](references.md#garnett-2023)). A common heuristic puts the practical range at a handful to a few dozen variables, although [Hvarfner et al. (2024)](references.md#hvarfner-2024) report that a plain GP with suitable priors does well in high dimensions. If you can afford hundreds of runs, or the process has a trusted physical model, a classical design from [Design of experiments](design-of-experiments.md) may serve you better. For surveys of the method see [Shahriari et al. (2016)](references.md#shahriari-2016) and [Frazier (2018)](references.md#frazier-2018).

## Why uncertainty drives exploration

Conditioned on the data, a GP gives a normal distribution for the objective at every point, with a posterior mean $\mu(x)$ and a posterior standard deviation $\sigma(x)$ ([Garnett 2023, §2.2](references.md#garnett-2023)). Near an observation $\sigma$ is small and the model is confident. Far from every observation $\sigma$ is large, and the true value could be much better or much worse than $\mu$ suggests.

An acquisition function turns the pair $(\mu, \sigma)$ into one score. A point scores well if its mean is high (exploitation) or its uncertainty is large (exploration). The next run goes where the two are best balanced, so the model's uncertainty is what sends the search into unvisited regions. The same dependence limits the method: if the model is badly wrong or has too little data, its uncertainty is wrong too, and the policy inherits the error ([Garnett 2023, §9.3](references.md#garnett-2023)).

## Starting: the initial design

When there is almost no data, the belief behind the policy is poor, and so is the policy. Garnett therefore advises beginning with a small set of runs chosen without reference to any model, for example random sampling, a low-discrepancy sequence or a Latin hypercube ([Garnett 2023, §9.3](references.md#garnett-2023), which cites [Jones et al. 1998](references.md#jones-1998) for this). The book gives no count. A common heuristic, usually attributed to [Jones et al. (1998)](references.md#jones-1998), is about ten runs per variable. Treat it as a starting point, not a rule.

## Stopping

Many BO runs stop when a budget of runs is spent. Garnett also treats termination as a decision: stop when no further observation is expected to be worth its cost ([Garnett 2023, §5.4](references.md#garnett-2023)). In practice this becomes a threshold on the acquisition function, stopping when

$$
\max_{x \in \mathcal{X}} \big( \alpha(x;\, \mathcal{D}) - c(x) \big) < 0
$$

where $c(x)$ is the cost or threshold at $x$ ([Garnett 2023, eq. 9.9](references.md#garnett-2023)). Garnett also describes purely data-driven rules, such as stopping when no progress has been made over a set number of runs ([Garnett 2023, §9.3](references.md#garnett-2023)). A statistic such as the model's $R^2$ on the runs so far measures how well the surrogate fits where you have already looked. It says nothing about whether better conditions remain, so it is not a stopping rule.

## A chemistry example

[Shields et al. (2021)](references.md#shields-2021) used BO to optimize chemical reactions over mixed continuous and categorical choices such as reagents, and compared it with expert chemists. Their setting, with few runs and a mix of numeric and categorical variables, resembles the one ALchemist targets.

!!! info "How ALchemist does it"
    The guide steps are the loop, implemented by `OptimizationSession` in `alchemist_core/session.py` ([Session reference](../reference/python/session.md)).

    | Loop stage | Guide step | Session method |
    |---|---|---|
    | Initial design | [3. Initial design](../guide/initial-design.md) | `generate_initial_design`, default method `"lhs"` |
    | Observe and add to $\mathcal{D}$ | [4. Run and record](../guide/run-and-record.md) | `add_experiment` |
    | Fit the surrogate | [5. Train a model](../guide/train.md) | `train_model` |
    | Check the surrogate | [6. Check the model](../guide/check-model.md) | the diagnostics, see [Diagnostics](diagnostics.md) |
    | Maximize the acquisition | [7. Suggest next](../guide/suggest.md) | `suggest_next`, default strategy `"EI"` |

    After step 7 you run the suggested experiment and return to step 4. For the space-filling methods `generate_initial_design` defaults `n_points` to 10 whatever the number of variables; pass your own count.

    ALchemist has no automatic stopping rule: nothing in `Session` ends the loop, so you decide when to stop. For the decision-based view above, `plot_probability_of_improvement` (in the [Session reference](../reference/python/session.md)) draws the largest probability of improvement found on a grid over the search space after each run. That is the quantity in the stopping rule with $\alpha$ set to probability of improvement. It is computed afterwards, reusing the final model's hyperparameters by default, and ALchemist applies no threshold to it.

## Further reading

- [Garnett (2023)](references.md#garnett-2023), Chapter 1 (the loop), Chapter 2 (the GP posterior), Chapters 7 and 8 (policies) and §9.3 (starting and stopping).
- [Shahriari et al. (2016)](references.md#shahriari-2016), a concise survey.
- [Frazier (2018)](references.md#frazier-2018), a short tutorial.
- [Jones et al. (1998)](references.md#jones-1998), the paper behind expected improvement and the initial-design heuristic.
- [Shields et al. (2021)](references.md#shields-2021), BO in chemical synthesis.
