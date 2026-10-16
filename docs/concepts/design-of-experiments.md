# Design of experiments

A design of experiments chooses which conditions to run before you see any results, so the data answer your question as cheaply as possible. In ALchemist it supplies the initial points for the optimization loop. Factors below are coded to $[-1, +1]$: $x_{\text{coded}} = (x - x_{\text{mid}})/x_{\text{half-range}}$.

## Space-filling designs

Latin hypercube sampling (LHS) is stratified random sampling: it cuts each factor's range into $n$ equal strata and puts exactly one point in each stratum ([McKay et al. 1979](references.md#mckay-1979)). Maximin LHS picks the Latin hypercube that maximizes the smallest pairwise distance, $\max_D \min_{i \ne j} \lVert x_i - x_j \rVert$ ([Morris and Mitchell 1995](references.md#morris-1995)). Sobol and Halton sequences, unlike LHS, are deterministic low-discrepancy sequences ([Sobol 1967](references.md#sobol-1967); [Halton 1960](references.md#halton-1960)). See [Santner et al. 2018, ch. 5](references.md#santner-2018).

A small space-filling design is a model-independent start for Bayesian optimization ([Garnett 2023, §9.3](references.md#garnett-2023)).

!!! info "How ALchemist does it"
    `generate_initial_design` in `alchemist_core/utils/doe.py` supports `random`, `lhs` (default), `sobol`, `halton` and `hammersly`; `n_points` defaults to 10. `lhs` (scikit-optimize) keeps the best of its random Latin hypercubes under `lhs_criterion` (`maximin` default, `correlation`, `ratio`); `sobol` is randomly shifted. `halton` is scikit-optimize's `Halton` sampler, one van der Corput sequence per dimension on its own prime base, and `hammersly` its `Hammersly` sampler, a Halton set with one dimension replaced by a regular grid; both are fixed sequences, so a seed reproduces them but does not change them. With input constraints they oversample ($4n$ up to $4096n$ draws) and keep feasible points. scikit-optimize warns that Sobol sequences lose their balance when $n$ is not a power of two. See the [Python reference](../reference/python/design.md) and the [initial design guide](../guide/initial-design.md).

## Classical designs

| Design | Runs | Supports |
|---|---|---|
| Full factorial ([Montgomery 2017, ch. 6](references.md#montgomery-2017)) | $\prod_{i=1}^{k} L_i$ for $L_i$ levels of factor $i$ | all interactions |
| Fractional factorial ([Box and Hunter 1961](references.md#box-hunter-1961)) | $2^{k-p}$ | main effects, some interactions |
| Central composite, CCD ([Box and Wilson 1951](references.md#box-wilson-1951)) | $2^k + 2k + n_c$ | quadratic |
| Box-Behnken ([Box and Behnken 1960](references.md#box-behnken-1960)) | $2k(k-1) + n_c$ | quadratic |
| Plackett-Burman ([Plackett and Burman 1946](references.md#plackett-1946)) | $4\lceil (k+1)/4 \rceil$ | main effects |
| Generalized subset, GSD ([Surowiec et al. 2017](references.md#surowiec-2017)) | about $\prod_i L_i / r$ for reduction $r$ | main effects, mixed levels |

$k$ is the number of factors and $n_c$ the number of center points.

**Resolution** says what a fraction confuses ([Montgomery 2017, ch. 8](references.md#montgomery-2017)). In resolution III, main effects are clear of each other but aliased with two-factor interactions (2FIs). In resolution IV, main effects are clear of 2FIs, but 2FIs are aliased with each other. In resolution V, main effects and 2FIs are clear of each other. Resolution IV can only show that some 2FI in an alias chain is active; resolving individual 2FIs needs V.

Central composite designs add $2k$ axial points at distance $\alpha$ from the center. The rotatable $\alpha = (2^k)^{1/4}$ makes prediction variance depend only on distance from the center ([Box and Hunter 1957](references.md#box-hunter-1957)). With $\alpha = 1$ the design is face-centered, and the orthogonal $\alpha$ depends on the center-point count ([Myers et al. 2016, §8.4.1–8.4.5](references.md#myers-2016)). The Box-Behnken formula matches the classical tables for $k = 3$ to $5$; larger $k$ uses incomplete block designs ([Box and Behnken 1960](references.md#box-behnken-1960); [Myers et al. 2016, §8.4.7](references.md#myers-2016)).

!!! info "How ALchemist does it"
    The classical builders in `alchemist_core/utils/doe.py` (via pyDOE) take `n_center` (default 1), `generators`, `ccd_alpha` (`orthogonal`, `rotatable`), `ccd_face` (`circumscribed`, `inscribed`, `faced`), `n_levels` and `gsd_reduction` (default 2). Run counts at `n_center = 1`:

    | $k$ | CCD | Box-Behnken | Plackett-Burman | Fractional (default) | 2-level full factorial |
    |---|---|---|---|---|---|
    | 3 | 16 | 13 | 5 | 5 | 9 |
    | 4 | 26 | 25 | 9 | 9 | 17 |
    | 5 | 44 | 41 | 9 | 9 | 33 |

    The CCD count is $2^k + 2k + 2n_c$, not $2^k + 2k + n_c$: center points enter both the factorial and axial blocks, and the core is always the full $2^k$. Box-Behnken follows $2k(k-1) + n_c$ and needs $k \ge 3$. The default generators are $2^{3-1}$ and $2^{4-1}$ (resolution III and IV) and $2^{5-2}$, $2^{6-3}$, $2^{7-4}$ (all III); other $k$ get a full factorial. A resolution V fraction such as $2^{5-1}$ needs your own `generators`.

!!! warning "Known issue"
    T9: a textbook circumscribed CCD puts its axial points outside the factorial cube, which cannot fit the variable bounds. ALchemist divides the coded design by $\alpha$ so the axial points land on the bounds, leaving the factorial points at $\pm 1/\alpha$. That is the inscribed design, so the default `ccd_face="circumscribed"` returns the same points as `"inscribed"`, and since 0.4.0 logs a warning saying so. Use `"faced"` for every point on the bounds or the centre.

!!! warning "Known issue"
    T11: the estimability gate below treats `fractional_factorial` as an "interaction" design. All default generators are below resolution V (III for $k = 3, 5, 6, 7$; IV for $k = 4$), so none supports that model, and a constraint that drops any point raises `DesignNotEstimableError`. Tracked for a 0.4.x release.

## Optimal designs

An optimal design picks $n$ runs from candidate points to be best for a model you name, $y = f(x)^\top \beta + \varepsilon$, with design matrix $X$ ($n \times p$). Each criterion summarizes $X^\top X$ ([Atkinson et al. 2007, ch. 10](references.md#atkinson-2007); [Goos and Jones 2011, §2.3.8 and §4.3.5](references.md#goos-2011)):

$$
\text{D: } \max\, \lvert X^\top X \rvert \qquad
\text{A: } \min\, \operatorname{tr}\!\big[(X^\top X)^{-1}\big] \qquad
\text{I: } \min\, \tfrac{1}{\lvert\mathcal{X}\rvert}\!\int_{\mathcal{X}} f(x)^\top (X^\top X)^{-1} f(x)\, dx
$$

D minimizes the confidence-ellipsoid volume for $\beta$, A the average parameter variance, I the average prediction variance over $\mathcal{X}$. D-efficiency, with factors coded to $[-1, +1]$, is

$$
\text{D-eff} = 100 \cdot \big\lvert X^\top X / n \big\rvert^{1/p}
$$

([Goos and Jones 2011, §4.3.3](references.md#goos-2011)). A two-level orthogonal design reaches 100 for a first-order model; quadratic models on a cube cannot.

**Exchange algorithms** swap design points for candidates while the criterion improves. Fedorov's tries all swaps ([Fedorov 1972](references.md#fedorov-1972); [Atkinson et al. 2007, ch. 12](references.md#atkinson-2007)); DETMAX adds and deletes points in deterministic excursions beyond $n$ ([Mitchell 1974](references.md#mitchell-1974)); coordinate exchange moves one coordinate at a time and needs no candidate list ([Meyer and Nachtsheim 1995](references.md#meyer-1995)). Sequential methods build or augment a design point by point ([Wynn 1970](references.md#wynn-1970); [Dykstra 1971](references.md#dykstra-1971)).

!!! info "How ALchemist does it"
    `run_optimal_design` in `alchemist_core/utils/optimal_design.py` takes `criterion` (`D`, `A`, `I`), `algorithm`, and a model as `model_type` (`linear`, `interaction`, `quadratic`) or an `effects` list such as `Temperature*Pressure`. It selects from a candidate grid and needs `n_points` of at least $p$; through `generate_initial_design` the grid has `n_levels` points per variable (5 unless you pass 3 or more). It starts from a greedy maximin-distance selection. With $M = X^\top X / n$, the scores are $\lvert M \rvert$ (D), $-\operatorname{tr}(M^{-1})$ (A) and $-\operatorname{tr}(M_{\text{mom}} M^{-1})$ (I), where $M_{\text{mom}}$ averages over the candidate grid, so the I integral is a sum over candidates. `sequential` makes one improving pass over the points; the other four repeat best-single-swap passes until nothing improves or `max_iter` (200) is reached.

!!! warning "Known issue"
    T10: `detmax` and `modified_fedorov` run the Fedorov code. All three share one branch, with no DETMAX excursions, so the three names give identical points and only `info["algorithm"]` differs; since 0.4.0 the two aliases log a warning saying so. `simple_exchange` evaluates the same swaps, so only `sequential` differs.

## Constraints and optimal designs

Linear input constraints cut a polytope $\{x : Ax \le b,\ \ell \le x \le u\}$ out of the box. A classical design loses the points outside it, and its structure with them. Optimal designs choose from the feasible region ([Goos and Jones 2011, §5.3.3](references.md#goos-2011)), and D-optimal designs tend to place runs at its extremes ([Atkinson et al. 2007, ch. 11](references.md#atkinson-2007)). See the [constraints guide](../guide/constraints.md).

!!! info "How ALchemist does it"
    For classical designs, `generate_initial_design` drops constraint-violating points and then applies the estimability gate. If every point is dropped it raises `InfeasibleRegionError`; if only some are, `_inestimable_terms` compares the rank of the implied model's design matrix, built from the survivors, to its column count. The implied model is quadratic for `ccd` and `box_behnken`, interaction for `fractional_factorial`, linear for `plackett_burman` and `gsd`, and for `full_factorial` quadratic when `n_levels` is at least 3, otherwise interaction. A rank-deficient remnant raises `DesignNotEstimableError` naming the inestimable terms; `allow_infeasible=True` returns it with a warning.

    For optimal designs, `run_optimal_design` builds an augmented candidate set with `augment_with_boundary` in `alchemist_core/utils/constrained_region.py`. It keeps the feasible grid points, adds feasible projections of infeasible points onto the constraint boundaries, and adds the polytope's vertices (up to five numeric variables, otherwise `vertex_enumeration_skipped` is set). Counts appear in `info["feasibility"]`. See the [optimal design guide](../guide/more/optimal-design.md).

## Split-plot designs

A completely randomized design assumes you can reset every factor on every run. Some factors cannot be reset that cheaply: a furnace takes an hour to settle, a catalyst bed is repacked once a day. The experimenter then holds such a factor fixed for a group of runs and changes the easy factors inside the group ([Jones and Nachtsheim 2009](references.md#jones-nachtsheim-2009)). The group is a whole plot, the hard-to-change factors are whole-plot factors, and the runs inside it are subplots. Randomization is restricted: whole plots are put in random order and runs are randomized within each whole plot, but the two are not shuffled together.

The restriction changes the analysis: runs in one whole plot share the error of that plot, so they are correlated. A standard model adds a random whole-plot term, $y = X\beta + Z\gamma + \varepsilon$ with $\gamma \sim N(0, \sigma_\gamma^2 I)$ and $\varepsilon \sim N(0, \sigma_\varepsilon^2 I)$, where $Z$ assigns each run to its plot ([Goos and Vandebroek 2001](references.md#goos-2001); [Jones and Nachtsheim 2009](references.md#jones-nachtsheim-2009)). Whole-plot factors are then tested against the whole-plot error and easy factors against the run-to-run error, so a split-plot experiment analyzed as if it were completely randomized gives wrong inference ([Jones and Nachtsheim 2009](references.md#jones-nachtsheim-2009); [Kowalski et al. 2007](references.md#kowalski-2007)). Optimal split-plot designs choose their runs given the variance ratio $\sigma_\gamma^2 / \sigma_\varepsilon^2$ ([Goos and Vandebroek 2001](references.md#goos-2001); [Goos and Vandebroek 2003](references.md#goos-2003); [Goos and Jones 2011, ch. 10](references.md#goos-2011)).

ALchemist gives a controllable variable one of three change levels: `easy` (the default), `hard` and `very_hard`. One restricted level makes a split-plot design; two make a split-split-plot design, with whole plots (`very_hard`), subplots (`hard`) and runs (`easy`).

!!! info "How ALchemist does it"
    `generate_split_plot_design` in `alchemist_core/utils/split_plot.py` (also `session.generate_split_plot_design`) nests the existing design methods. You pass `designs={level: {"method": ..., ...}}` with one entry per level present; each entry takes any `generate_initial_design` method except `optimal`, with that method's arguments. The outermost level gets a design of its own, the next level is generated under each of its rows with the outer values fixed, and `easy` is innermost. Constraints that tie a level to an outer level are reduced per plot; a plot with no feasible design raises `InfeasibleRegionError`. Classical methods default to `n_center=0` inside a split-plot design.

    | Option | Default | Effect |
    |---|---|---|
    | `replicates` | 1 | `{"hard": 2}` repeats that level's design; each repeat is a new plot with its own ID. |
    | `easy_mode`, `middle_mode` | `"fresh"` | `"fresh"` draws a new design under each outer plot; `"crossed"` reuses one under every outer plot. `middle_mode` is for `hard` when a `very_hard` level exists. |
    | `reference` | none | `{"position": "ends" or "start", "values": {...}}` adds reference runs of the easy variables to every innermost plot (`"ends"`, the default when `reference` is given, adds one first and one last; `"start"` adds one first); unnamed variables sit at the middle of their range (the first level for a categorical). |
    | `order` | `"random"` | Per level, `"random"` shuffles plots (for `easy`, the runs inside a plot); `"as_generated"` keeps the generated order. |
    | `random_seed` | none | Makes the design reproducible. |

    Each row has every controllable variable plus `WholePlot`, `Subplot` (three levels only), `RunOrder` (1 to `n_runs` across the design) and `Reason`. The call returns `(points, info)`; `info` holds `levels`, `n_plots`, `runs_per_plot`, `n_runs` and `warnings`. It warns when every whole-plot setting appears only once: with no replicated whole plots there is no pure-error estimate of the whole-plot variance. The same check runs on the `hard` settings inside each `very_hard` plot. Designs over 5,000 runs are refused. See the [Python reference](../reference/python/session.md) and the [split-plot guide](../guide/more/split-plot.md).

!!! warning "What ALchemist does not do"
    ALchemist generates the design; it does not analyze it as a split-plot experiment. `WholePlot`, `Subplot` and `RunOrder` are stored with the data but are never model inputs, so the Gaussian process treats every run as independent, with no whole-plot error term. The `changes` level does not affect acquisition, and optimal split-plot designs are not built. For inference on whole-plot factors, export the data and fit a mixed model elsewhere.

## Further reading

- [Montgomery 2017](references.md#montgomery-2017), chapters 6, 8, 11, and [Myers et al. 2016](references.md#myers-2016), chapter 8: factorials, fractions, response surfaces, central composite and Box-Behnken designs.
- [Goos and Jones 2011](references.md#goos-2011), chapters 2, 4, 5 and 10: D and I optimality, categorical factors, constrained regions, a split-plot design.
- [Atkinson et al. 2007](references.md#atkinson-2007), chapters 10 to 12: criteria, D-optimum designs, algorithms.
- [Santner et al. 2018](references.md#santner-2018), chapter 5: space-filling designs.
- [Garnett 2023, §12.1](references.md#garnett-2023): optimal design as the precursor of Bayesian optimization.
- [Jones and Nachtsheim 2009](references.md#jones-nachtsheim-2009) and [Kowalski et al. 2007](references.md#kowalski-2007): what split-plot designs are and how to run and analyze them.
