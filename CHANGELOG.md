# Changelog

All notable changes to ALchemist are documented here.

---

## [0.3.3] — TBD

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
