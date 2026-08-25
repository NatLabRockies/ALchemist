# Changelog

All notable changes to ALchemist are documented here.

---

## [Unreleased]

Work surfaced while deploying ALchemist on an inductive RCC campaign (real
non-`Output` objective column, batch qEI acquisitions, hardware-adjusted
conditions).

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

### Maintenance / Internal
- De-duplicated three copies of the model-input metadata-exclusion logic into a
  single `ExperimentManager.metadata_columns()` helper; routed the BoTorch
  evaluation and Pareto feature-selection paths through it as well.
- Production web build (`tsc -b`) no longer fails on `*.test.ts(x)` in fresh
  checkouts (test files excluded from `tsconfig.app.json`); vitest still
  type-checks them.

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
