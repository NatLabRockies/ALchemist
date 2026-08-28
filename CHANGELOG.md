# Changelog

All notable changes to ALchemist are documented here.

---

## [Unreleased]

Work surfaced while deploying ALchemist on an inductive RCC campaign (real
non-`Output` objective column, batch qEI acquisitions, hardware-adjusted
conditions), and a pass making linear input constraints work everywhere they
were already advertised.

### Breaking Changes

All four come from this work. The first three are direct consequences of the
constrained-DoE feature; the fourth is a side effect of hardening the shared
variable-registration path the feature depends on. **Unconstrained behavior is
unchanged at every seed**, locked by golden tests over every DoE method.

- A constrained **classical** design (`ccd`, `box_behnken`, `full_factorial`,
  `fractional_factorial`, `plackett_burman`, `gsd`) that loses structural
  points now raises `DesignNotEstimableError` when the surviving points can no
  longer estimate the design's implied model. It previously returned the
  degraded remnant with a log warning — a rank-deficient design that still
  looked like the design you asked for. The message names the terms that
  became inestimable. Pass `allow_infeasible=True` for the old behavior.
  Designs that lose only harmless points (a replicated center point) still
  return normally. Over REST this is a `400` with
  `"error_type": "DesignNotEstimableError"`.
- A constrained **optimal** design returns **different points for the same
  seed**, because it now selects from a candidate set that is filtered *and*
  augmented on the feasible boundary rather than from a plain filtered
  lattice. This is the fix, not a regression.
- `SearchSpace.add_constraint` / `session.add_input_constraint` now **reject
  non-numeric variables** (`categorical`, `context`) at registration, along
  with non-finite `rhs` and coefficients. Such constraints were already
  non-functional — they failed later, inside feasibility filtering.
- **Variable bounds are now validated, so documents that used to load are now
  refused.** `SearchSpace.add_variable` — and therefore `POST /variables`,
  `PUT /variables/{name}` and `/variables/load` — reject a non-finite `min` or
  `max` (`inf`, `NaN`) and a `real` span that is not float64-representable
  even though each endpoint is. None of these guards existed before; all three
  values were previously accepted and produced a variable that was unusable
  downstream (a `NaN` bound broke every subsequent export; an overflowing span
  sampled to a single distinct value). A stored search-space file carrying one
  of these values still opens in the desktop loader but now fails over REST.

### New Features
- **Suggested-vs-actual provenance.** Every experiment now records what the model
  suggested versus what was actually run, durable in the session file and audit
  log. Each dataset row carries a hidden `ProvenanceId` joining it to a full
  provenance record (suggested inputs, actual inputs, per-variable delta,
  strategy, acquisition params, output, noise, timestamp). Closes the loophole
  where actuals could deviate arbitrarily from a suggestion while still being
  labeled with an acquisition strategy. The web "Add Point" flow now routes
  through the work-queue `complete()` lifecycle; new endpoints
  `GET /experiments/provenance` and `/{id}` expose the records. `ProvenanceId`
  is excluded from the model input matrix at every site.
- **Read-only suggested + editable "actual" conditions in the Add Point dialog.**
  Suggested values are shown read-only (rounded per variable — integers to whole
  numbers, reals to 3 decimals, raw precision in a tooltip) alongside editable
  "actual" boxes pre-filled with the rounded suggestion. The recorded data point
  uses the actual (hardware-adjusted) values.
- **Acquisition suggestions now carry their iteration number** end to end, so the
  Add Point dialog shows the real iteration instead of `N/A`; a whole batch is
  recorded under one iteration.
- **Linear input constraints are now settable over REST.** `POST`/`GET`
  `/sessions/{id}/constraints` and `DELETE .../{name}`. Constraint names are
  unique and are the delete identity, so one call removes exactly one
  constraint. Previously constraints could only be set from Python, so no REST
  or web consumer could use the feature at all. New documentation page:
  *Constraining the Variable Space*.
- **Design responses carry a `feasibility` block.** `/initial-design` and
  `/optimal-design` report which constraints applied, how many structural
  points a classical design dropped, the optimal-design candidate accounting
  (candidates total, feasible, boundary and vertex points added), and an
  `estimability` verdict. `null` when no constraints are registered. This
  provenance previously existed only in a server log line.
- **`/variables/load` accepts the `{variables, constraints}` document** that
  `SearchSpace.save_to_json` writes, alongside the bare variable list. The dict
  form is validated atomically — a bad variable or constraint anywhere in the
  file rejects the whole document and leaves the session untouched — and
  loaded constraints pass the same validation as `POST /constraints`.
- **`/variables/export?include_constraints=true`** returns that same document,
  so `load → export → load` round-trips constraints. The default export stays a
  bare variable array, which is an existing cross-surface contract.

### Bug Fixes
- **Session load dropped all data when the objective column was not named
  `Output`.** A long-standing bug (present since ≤ v0.3.3): the loader hardcoded
  the target column, so any session whose objective was named something else
  (e.g. `Methane umol/g/Wh`) silently restored 0 experiments (rows were swallowed
  by a per-row `try/except`). The loader now resolves the target from a persisted
  `target_columns` key or infers it from the non-variable/non-metadata columns;
  `save_session` now records `target_columns` for self-describing files. Fully
  backward compatible across versions.
- **Batch acquisition points appeared identical in the Add Point dialog.** The
  dialog seeded form state from props at mount only, and stepping through a batch
  swapped the prop without remounting, so every point showed the first point's
  values. Fixed by keying the dialog per suggestion so it remounts with fresh
  state. (The underlying acquisition and staging were always correct — this was
  a display bug.)
- **Staged suggestions persisted across sessions ("ghost points").** The staged-
  suggestions restore effect only ever *added* to state and never cleared, so
  starting a new session left stale suggestions in the UI. It now always resolves
  to a definitive value (clearing when the session has none), with a race guard.
- **"Retrain model" was silently ignored when recording results via the work
  queue.** The queue-complete endpoint had no auto-train; it now honors the
  retrain control like the direct add-experiment path.
- **Regret / hypervolume plots showed gaps where a subset GP failed to fit.**
  The posterior overlay refits a fresh GP on each data prefix; small or
  ill-conditioned prefixes (notably categorical/`AdditiveKernel` models, where
  the hyperparameter-reuse fast path is skipped and every iteration
  re-optimizes) could raise `ModelFittingError`, leaving that iteration's
  prediction as NaN. Both the single-objective and MOBO paths now walk a
  fallback ladder — requested transforms, then transforms disabled, then
  escalating Cholesky jitter (1e-3, 1e-2) — and every value plotted remains a
  genuine prediction from an actually-fitted GP, never interpolated or
  fabricated. If the whole ladder fails, NaN is still the last resort.
- **Linear input constraints were honored in only two of four places.** They
  were respected by acquisition and by space-filling DoE, but **classical**
  designs generated their points ignoring constraints and post-hoc-dropped the
  infeasible ones with a log warning, leaving a design rank-deficient for the
  model it claimed to fit; and **optimal** designs had no constraint awareness
  at all, so the exchange algorithm spent its whole budget selecting points it
  would not be allowed to keep. Optimal designs now select from a candidate set
  that is filtered *and* augmented with points on the feasible region's
  boundary — a filtered lattice has none, and an optimal design wants precisely
  those extremes — so a constrained D/A/I-optimal design is now genuinely
  optimal over its region. An equality constraint over `real` variables is now
  satisfiable for the first time — optimal design is the only method that
  places points on the zero-volume slice such a constraint defines. Note the
  model must not contain the terms the equality makes collinear: with
  `x1 + x2 == rhs`, the intercept and the two main effects are exactly
  dependent, so pass an `effects` list that drops one of the tied variables
  rather than a `model_type` shortcut. The rank-deficiency error names this
  case and its remedy.
- **Non-model variables were spread straight through a constraint.** Variables
  absent from every model term were overwritten with a shuffled range *after*
  selection, undoing all feasibility work for those columns. They are now drawn
  from each row's feasible interval.
- **A `context` variable silently dropped a real variable from every design.**
  Design generation zipped the full variable list against samples drawn from
  dimensions that exclude `context`, so a context variable anywhere but last
  produced points missing an optimization variable, with its values relabeled
  onto the context column — no error, straight into the experiment table, the
  model and the acquisition function. The classical and optimal paths were
  affected more severely than space-filling. The desktop design table hit the
  same skew.
- **Editing or deleting a variable could silently corrupt the search space.**
  `PUT`/`DELETE` on a variable rebuilt dimensions by an index derived from a
  list that includes `context` entries, applied against one that does not. In
  range, `PUT` returned 200 and destroyed a different variable's dimension —
  invisible through the API's own read path, since `GET /variables` reads the
  other list. Both routes now go through one validated core path that replaces
  the dimension in place, so ordering is preserved — which is also how `PUT`
  picks up the new bound guards listed under Breaking Changes. Before this
  work no route validated bounds at all, and `PUT` built its dimension inline,
  so it would have bypassed the guards even once they existed.
- **Space-filling designs returned values that were not JSON-serializable.** An
  `integer` variable came back as `np.int64`, which `json.dumps` refuses, so the
  endpoint failed outright; `discrete` variables had been leaking `np.float64`
  silently for as long. Every variable type now returns a native scalar.
- **A provably infeasible constrained design ground for minutes and blocked the
  whole server.** The reject-and-resample loop escalated its batch size to
  4096× before giving up — measured at 480 s to a correct 400 — and the design
  endpoints ran that work directly on the ASGI event loop, so one such request
  starved every other request in the process, not just its own. Impossible
  regions are now proven empty up front and refused immediately with
  `InfeasibleRegionError` (`400`), and design generation runs in a threadpool
  like the other heavy endpoints.
- **`random_seed` was applied through process-global state.** Two concurrent
  seeded design requests interleaved and neither got the design its seed names.
  The seed is now honored per call.
- **Constraints were lost on server restart.** `SearchSpace.save_to_json`
  persisted them, but the *session* serializer did not, so every API session —
  and every crash-recovery backup — silently dropped its constraints on reload.
- **Constraint errors were indistinguishable from any other bad request.** Both
  are `ValueError` subclasses and so already returned `400`, but under
  `"error_type": "ValueError"`, alongside malformed bounds and unknown methods.
  They now report `DesignNotEstimableError` and `InfeasibleRegionError`, which
  carry different remedies.

### Maintenance / Internal
- De-duplicated three copies of the model-input metadata-exclusion logic into a
  single `ExperimentManager.metadata_columns()` helper; routed the BoTorch
  evaluation and Pareto feature-selection paths through it as well.
- `BoTorchModel` now carries a per-instance `cholesky_jitter` attribute
  (defaulting to the module value). Callers that need to escalate jitter for one
  ill-conditioned fit set it on their own instance instead of rebinding the
  `_CHOLESKY_JITTER` module global, which would leak across concurrent API
  requests and every other GPyTorch consumer in the process.
- Production web build (`tsc -b`) no longer fails on `*.test.ts(x)` in fresh
  checkouts (test files excluded from `tsconfig.app.json`); vitest still
  type-checks them.
- New `alchemist_core.utils.constrained_region` module: the feasible-region
  primitives (candidate filtering, boundary and vertex augmentation, emptiness
  and measure-zero proofs) and `InfeasibleRegionError`.
  `DesignNotEstimableError` lives in `alchemist_core.utils.doe`. Both subclass
  `ValueError`, so existing `except ValueError` handlers keep working.

---

## [0.3.4] — 2026-06-02

### Maintenance
- **Removed the unused `ax-platform` dependency.** It was no longer referenced by any model or acquisition backend (it had been dropped as a backend option but never pruned). It was also the only package forcing `ipywidgets` and the JupyterLab widgets extension into the install, whose very long asset paths triggered install failures on Windows (the 260-character `MAX_PATH` limit). Removing it fixes that failure class and trims `ipywidgets`, `plotly`, `sympy`, and `pyre-extensions` from the dependency tree. The BoTorch backend is unaffected.

### Documentation
- Removed internal planning/spec notes that had been inadvertently published to the docs site, and added `exclude_docs` / `.gitignore` guards to keep dev notes off the public docs going forward.

---

## [0.3.3] — 2026-03-20

### New Features
- **Full DoE suite** — classical RSM (CCD, Box-Behnken, Full/Fractional Factorial), screening (Plackett-Burman, GSD), and optimal designs (D/A/I-optimal with five exchange algorithms)
- **AI-assisted effect selection** — LLM-powered model term suggestions via OpenAI or local Ollama; optional Edison Scientific literature search integration
- **Discrete variable type** — numerical variables restricted to a specific set of allowed values (e.g., SAR ratios); full BO and DoE integration
- **IBNN kernel for BoTorch** — Infinite-Width Bayesian Neural Network kernel alongside Matern and RBF
- **Multi-objective variable role management** — explicit variable/target/drop column assignment on CSV upload; multi-target support in web UI, desktop, and Python API
- **3D surface and uncertainty surface plots** via `create_surface_plot()` and `create_uncertainty_surface_plot()`

### Improvements
- `pyDOE` pinned to 0.9.5 for reproducibility
- D-efficiency computation corrected for degenerate information matrices
- LLM panel: Beta badge added to toggle button; fixes for hallucinated citations and Ollama base URL normalization

### Documentation
- New pages: Classical & Screening Designs, Optimal Design, AI-Assisted Effect Selection, Multi-Objective Optimization, Staged Experiments (Python API), 3D Surface Plots, DoE theory background
- Updated: Variable Space setup (Discrete type), BoTorch Backend (IBNN kernel), Home feature list

---

## [0.3.2] — 2026-02-05

### New Features
- **Visualization module** (`alchemist_core/visualization/`) — pure plotting functions usable in notebooks, scripts, and the web API without UI dependencies
- **Staged experiments API** — `/api/v1/sessions/{id}/experiments/staged` endpoints for autonomous reactor-in-the-loop workflows
- **WebSocket-based session events** — lock status, experiment additions, and model training pushed to the frontend in real time (replaces polling)
- **Session reconnection fix** — URL `?session=` parameter now takes priority over stale recovery backups
- **Join Session UI** — paste a session ID on the landing page to connect to a running session
- Target column selection on CSV upload; flexible multi-objective preparation

### Improvements
- `BoTorchModel` updated to float64 tensors for numerical stability
- Corrected PI computation and sklearn acquisition functions for maximization
- `load_session()` now supports both static and instance usage patterns

### Infrastructure
- Repository migrated from NREL to NatLabRockies GitHub organization

---

## [0.3.1] — 2025-12-12

### New Features
- **WebSocket-based session locking** — replaces 5-second HTTP polling with instant (< 100 ms) lock status events; auto-reconnect with 5-second retry
- **Session lock REST endpoints** — UUID token-based lock/unlock with force-unlock recovery
- **Comprehensive unit test suite** — OptimizationSession, SklearnModel, AuditLog, EventEmitter, and multi-client scenarios

### Bug Fixes
- Multi-client iteration tracking: ExperimentManager now calculates iteration as `max(existing) + 1` automatically
- BoTorch/sklearn backend switching: automatic transform type mapping to prevent scaling errors
- Sklearn GP numerical stability: clamp predicted std to ≥ 1e-6; validate calibration factors before applying

### Documentation
- Auto-generated Python API reference using mkdocstrings
- Reorganized navigation; corrected endpoint paths, parameter names, and variable type names throughout

---

## [0.3.0] — 2025-11-24

### New Features
- **Production-ready packaging** — pre-built web UI bundled in the Python wheel; `pip install alchemist-nrel` includes both desktop and web apps
- **Entry points** — `alchemist` (desktop GUI) and `alchemist-web` (React + FastAPI) installed as CLI commands
- **Docker support** — production Dockerfile with multi-stage build, Docker Compose configuration, health checks, and volume mounting
- **Custom build hooks** — automatic React UI compilation during `python -m build`

### Improvements
- Flexible CORS via `ALLOWED_ORIGINS` environment variable
- Static file serving checks `api/static/` (production) before `alchemist-web/dist/` (development)

---

[0.3.3]: https://github.com/NatLabRockies/ALchemist/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/NatLabRockies/ALchemist/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/NatLabRockies/ALchemist/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/NatLabRockies/ALchemist/releases/tag/v0.3.0
