# Design of experiments

A design of experiments chooses which conditions to run before you see any results, so the data answer your question as cheaply as possible. In ALchemist it supplies the initial points for the optimization loop. Factors below are coded to $[-1, +1]$: $x_{\text{coded}} = (x - x_{\text{mid}})/x_{\text{half-range}}$.

## Space-filling designs

Latin hypercube sampling (LHS) is stratified random sampling: it cuts each factor's range into $n$ equal strata and puts exactly one point in each stratum ([McKay et al. 1979](references.md#mckay-1979)). Maximin LHS picks the Latin hypercube that maximizes the smallest pairwise distance, $\max_D \min_{i \ne j} \lVert x_i - x_j \rVert$ ([Morris and Mitchell 1995](references.md#morris-1995)). Sobol and Halton sequences, unlike LHS, are deterministic low-discrepancy sequences ([Sobol 1967](references.md#sobol-1967); [Halton 1960](references.md#halton-1960)). See [Santner et al. 2018, ch. 5](references.md#santner-2018).

A small space-filling design is a model-independent start for Bayesian optimization ([Garnett 2023, §9.3](references.md#garnett-2023)).

!!! info "How ALchemist does it"
    `generate_initial_design` in `alchemist_core/utils/doe.py` supports `random`, `lhs` (default), `sobol`, `halton` and `hammersly`; `n_points` defaults to 10. `lhs` (scikit-optimize) keeps the best of its random Latin hypercubes under `lhs_criterion` (`maximin` default, `correlation`, `ratio`); `sobol` is randomly shifted. With input constraints they oversample ($4n$ up to $4096n$ draws) and keep feasible points. scikit-optimize warns that Sobol sequences lose their balance when $n$ is not a power of two. See the [Python reference](../reference/python/design.md) and the [initial design guide](../guide/initial-design.md).

!!! warning "Known issue"
    T12: `halton` is not a Halton sequence. `method="halton"` runs scikit-optimize's `Hammersly` sampler, the same one as `hammersly`: a Halton set with one dimension replaced by a regular grid. Tracked for a 0.4.x release.

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
    T9: a textbook circumscribed CCD puts its axial points outside the factorial cube. ALchemist divides the coded design by $\alpha$ so the axial points land on the bounds, leaving the factorial points at $\pm 1/\alpha$. That is the inscribed design, so the default `ccd_face="circumscribed"` returns it; the two gave identical points at $k = 3$. Tracked for a 0.4.x release.

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
    T10: `detmax` and `modified_fedorov` run the Fedorov code. All three share one branch, with no DETMAX excursions. `simple_exchange` evaluates the same swaps, so only `sequential` differs. Tracked for a 0.4.x release.

## Constraints and optimal designs

Linear input constraints cut a polytope $\{x : Ax \le b,\ \ell \le x \le u\}$ out of the box. A classical design loses the points outside it, and its structure with them. Optimal designs choose from the feasible region ([Goos and Jones 2011, §5.3.3](references.md#goos-2011)), and D-optimal designs tend to place runs at its extremes ([Atkinson et al. 2007, ch. 11](references.md#atkinson-2007)). See the [constraints guide](../guide/constraints.md).

!!! info "How ALchemist does it"
    For classical designs, `generate_initial_design` drops constraint-violating points and then applies the estimability gate. If every point is dropped it raises `InfeasibleRegionError`; if only some are, `_inestimable_terms` compares the rank of the implied model's design matrix, built from the survivors, to its column count. The implied model is quadratic for `ccd` and `box_behnken`, interaction for `fractional_factorial`, linear for `plackett_burman` and `gsd`, and for `full_factorial` quadratic when `n_levels` is at least 3, otherwise interaction. A rank-deficient remnant raises `DesignNotEstimableError` naming the inestimable terms; `allow_infeasible=True` returns it with a warning.

    For optimal designs, `run_optimal_design` builds an augmented candidate set with `augment_with_boundary` in `alchemist_core/utils/constrained_region.py`. It keeps the feasible grid points, adds feasible projections of infeasible points onto the constraint boundaries, and adds the polytope's vertices (up to five numeric variables, otherwise `vertex_enumeration_skipped` is set). Counts appear in `info["feasibility"]`. See the [optimal design guide](../guide/more/optimal-design.md).

## Split-plot designs

!!! note "Being rewritten for ALchemist 0.4.0"
    This section will cover split-plot designs for hard-to-change variables: whole plots, subplots and restricted randomization.

## Further reading

- [Montgomery 2017](references.md#montgomery-2017), chapters 6, 8, 11: factorials, fractions, response surfaces.
- [Myers et al. 2016](references.md#myers-2016), chapter 8: central composite and Box-Behnken designs.
- [Goos and Jones 2011](references.md#goos-2011), chapters 2, 4 and 5: D and I optimality, categorical factors, constrained regions.
- [Atkinson et al. 2007](references.md#atkinson-2007), chapters 10 to 12: criteria, D-optimum designs, algorithms.
- [Santner et al. 2018](references.md#santner-2018), chapter 5: space-filling designs.
- [Garnett 2023, §12.1](references.md#garnett-2023): optimal design as the precursor of Bayesian optimization.
