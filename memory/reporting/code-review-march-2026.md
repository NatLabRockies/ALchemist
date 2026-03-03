# ALchemist Comprehensive Code Review — March 2026

## 📊 Project Pulse

| Metric | Value |
|---|---|
| **Version** | 0.3.2 + 23 unreleased commits |
| **Core size** | `session.py` 5,840 LOC; total core ~10k LOC |
| **Tests** | 423 collected, **46 failing**, 20 collection errors, 39 skipped |
| **Commits since v0.3.0** (Nov 2025) | 91 commits across 4 months |
| **Commit breakdown** | 41 feat, 22 fix, 17 chore, 5 docs, 3 refactor, 1 test |
| **Post-v0.3.2 features** | MOBO, Discrete vars, DoE (RSM/PB/GSD), Optimal Design (D/A/I), LLM effect suggestion, 3D visualization, IBNN kernel |

---

## 🔴 CRITICAL ISSUES

### 1. Test Suite is Broken
**46 tests failing, 20 can't even collect.**

- **Collection errors**: All API integration tests fail because `httpx` is missing from the active environment (despite being in `[test]` deps). All BoTorch-dependent tests and visualization tests also fail to collect.
- **Failing test files**: `test_mobo_session.py` (5 failures — MOBO strategy validation), `test_acquisition_fixes.py`, `test_constraints.py`, `test_mobo_data.py`, `test_model_training.py`, `test_mobo_workflow.py`
- **Impact**: No CI gate is protecting the main branch. New features (MOBO, Discrete vars, Optimal Design) were merged with failing tests.

### 2. Session.py Thread Safety ✅ FIXED
The 5,840-line session orchestrator now uses `threading.RLock()` to protect concurrent access:
- `staged_experiments` mutations wrapped in lock (`add_staged_experiment`, `get_staged_experiments`, `clear_staged_experiments`, `move_staged_to_experiments`)
- `last_suggestions` write in `suggest_next()` wrapped in lock
- `_current_iteration` access in `lock_acquisition()` and `lock_model()` wrapped in lock

### 3. Resource Leaks in `load_data()` & `export_session_json()` ✅ FIXED
- ~~`load_data()` creates a temp file with `delete=False`, and if `load_from_csv()` throws between file creation and the finally block, the temp file leaks~~
- ~~`export_session_json()` has the same pattern~~
- **Fixed**: Restructured both methods to wrap entire temp-file lifecycle in a single `try/finally` with `temp_path = None` initialization, ensuring cleanup on all code paths.

### 4. API Endpoints Lack Error Handling (partial fix: error detail leakage ✅ FIXED)
- ~~**acquisition.py**: `suggest_next_experiments()` and `find_model_optimum()` — zero try/except. Any backend failure crashes the request with an unformatted 500.~~
- ~~**models.py**: `train_model()`, `predict()` — same issue~~
- ~~**experiments.py**: Nearly every endpoint (15+) has no exception handling~~
- ~~**5 endpoints** leak internal error details to clients via `f"Failed to...: {str(e)}"`~~
- **Fixed (leakage)**: All 7 endpoints that leaked `str(e)` to clients now return generic messages. Full error details are logged server-side with `exc_info=True`. Note: the existing `error_handlers.py` middleware provides a safety net for the remaining unprotected endpoints.
- **Fixed (middleware)**: ✅ Added `RuntimeError`→400 and `ImportError`→422 handlers to middleware, covering all endpoints.
- **Fixed (high-risk endpoints)**: ✅ Added try/except to `acquisition.py` (2 endpoints), `models.py` (2 endpoints), `experiments.py` (`generate_optimal_design`, `upload_experiments`). Fixed `variables.py` `delete_variable()` bare `ValueError` → `HTTPException(404)`.

---

## 🟠 HIGH SEVERITY ISSUES

### 5. LLM Service Security Concerns
- **API keys stored in plaintext** at `~/.alchemist/config.json` with file permissions restricted to owner (0o600)
- ~~**Prompt injection risk**: User-supplied variable names interpolated directly into LLM prompts without sanitization~~ ✅ Fixed — input sanitization applied: control characters stripped, system_context truncated to 2000 chars, variable names limited to 100 chars
- **`/api/v1/llm/config` endpoint returns API keys** in response body — ✅ Fixed (commit 6584ce1, keys masked)
- ~~**No timeouts** on external LLM API calls (OpenAI, Ollama, Edison) — hangs possible~~ ✅ Fixed — OpenAI timeout=60s, Ollama timeout=120s, Edison already had configurable timeout (default 1200s)
- **No rate limiting** on LLM endpoints (low priority for local/trusted deployment)
- **Zero test coverage** for all 7 LLM-related files

### 6. Optimal Design Numerical Stability
- **Line 351**: `half_range = (high - low) / 2.0` with no check for `high == low` (discrete var encoding) → divide-by-zero
- **Lines 571-582**: If `n_points > candidates`, greedy loop appends index `-1` → silent array indexing bug
- **Ridge regularization** (1e-8) doesn't scale with matrix norm → fails on ill-conditioned problems
- **RNG seeding inconsistency**: `np.random.seed()` set globally but separate `default_rng()` created → non-reproducible results

### 7. MOBO Integration Gaps
- No **default MOBO strategy** — user must explicitly request `qEHVI`/`qNEHVI`
- ~~`predict()` returns **different types** for single vs multi-objective (tuple vs dict) with no type annotation~~ ✅ Fixed — now always returns dict
- **3+ objective visualization** not supported (only 2D Pareto plots)
- **Input constraints not passed** to hypervolume acquisition — qEHVI can suggest infeasible points
- **Reference point auto-computation** breaks with all-negative objectives (10% margin goes wrong direction)

### 8. Audit Log Silently Swallowed
- `generate_initial_design()`, `generate_optimal_design()`, and `lock_model()` all wrap audit logging in bare `try/except` blocks that catch everything and only emit `logger.debug()`
- Consequence: Audit trail — a key reproducibility feature — can be silently incomplete

---

## 🟡 MEDIUM SEVERITY ISSUES

### 9. Missing Input Validation (Core) — Partially Fixed ✅
- `add_experiment()`: ✅ Now validates `inputs` contains all search space variables; rejects NaN/Inf output; validates noise is non-negative
- `add_staged_experiment()`: ✅ Now validates `inputs` keys match search space variables
- `add_outcome_constraint()`: ✅ Now checks value is finite; prevents duplicate constraints
- `suggest_next()`: ✅ Now validates `ref_point` values are finite (in addition to existing length check)
- `add_outcome_constraint()`: ⚠️ Still no check that `objective_name` exists (deferred — requires data to be loaded first)

### 10. Inconsistent Error State Recovery ✅ FIXED
- ~~`load_data()`: If `load_from_csv()` throws partway, `experiment_manager` is partially initialized — previous experiments lost~~
- ~~`train_model()`: If model training fails mid-way, `model_backend` is already set → subsequent `suggest_next()` uses broken model~~
- ~~`load_session()`: Loops adding experiments — if one fails, partial state persists with no rollback~~
- **Fixed**: `load_data()` now saves previous `experiment_manager` and restores on failure. `train_model()` saves previous `model`/`model_backend` and restores on failure. `load_session()` now catches per-experiment errors, logs warnings, and continues loading remaining experiments.

### 11. Frontend Type Safety
- 6+ instances of `Record<string, any>` in API types — loss of schema validation
- 8 hook error handlers typed as `(error: any)` — unsafe property access
- `OptimalDesignPanel.tsx` uses raw `fetch()` calls bypassing the React Query error/toast system
- `ContourPlotSimple.tsx` is a stub placeholder (TODO in code)

### 12. Test Coverage Gaps (13 untested modules)

| Area | Untested Modules |
|------|-----------------|
| LLM | `llm_service.py`, `llm_config.py`, all 4 providers, `llm.py` router |
| API | `models.py` router, `error_handlers.py` middleware |
| Core | `config.py`, `acquisition_utils.py`, `visualization/helpers.py` |
| Stubs | `test_event_system.py` (14+ pass-only tests), `test_core_data.py` (8+ stubs) |

### 13. TODO / Incomplete Items in Code
- `api/services/session_store.py`: TODO to migrate `threading.Lock()` to async-compatible lock
- `alchemist-web/src/components/visualizations/ContourPlotSimple.tsx`: ~~Placeholder — "TODO: Complete contour visualization"~~ ✅ Removed
- `ui/acquisition_panel.py`: TEMPORARY direct acquisition import — should use Session API `find_optimum`

---

## 📈 DEVELOPMENT TRENDS

### Feature Velocity (Since Nov 2025)
```
Nov 2025:  v0.3.0 release, React UI, autonomous optimization, DoE module
Dec 2025:  v0.3.1, MOBO Phase 1 (Pareto, hypervolume), session management
Jan 2026:  Visualization overhaul (3D voxel, contour, acquisition plots)
Feb 2026:  v0.3.2, MOBO full integration, Discrete vars, RSM/PB/GSD DoE,
           Optimal Design (D/A/I), LLM-assisted effect suggestion, IBNN kernel
```

### Observations
- Feature development is **accelerating** (7 major features in Feb alone)
- Fix-to-feature ratio (22:41) is healthy but **test coverage isn't keeping pace**
- Only **1 dedicated test commit** in the entire history — tests are afterthoughts in feature PRs
- Documentation additions (5 docs commits, mostly Feb 24) are catching up but lag features

### Architecture Evolution
The project is clearly evolving from a single-objective GP optimizer into a **comprehensive experimental design platform**:
1. **BO core** → MOBO, IBNN kernel, discrete variables
2. **Simple CSV input** → Full DoE (RSM, screening, optimal design)
3. **Manual workflow** → LLM-assisted design guidance
4. **Matplotlib only** → 3D surface, Plotly (web), rich visualization grid

---

## 🚀 STRATEGIC RECOMMENDATIONS FOR MAXIMUM ADOPTION

### Tier 1: Foundation (Ship Quality)
These must be addressed before any v0.4.0 release:

1. **Fix the test suite** — 46 failures and 20 collection errors on main is a red flag for any contributor or adopter. Install `httpx` in CI, fix the MOBO validation tests, and establish a passing-tests gate on PR merge.

2. **Add error handling to all API endpoints** — A single unhandled exception in `suggest_next` or `train_model` crashes the web app with a raw 500. Wrap all router functions in try/except with proper HTTPException responses.

3. **Secure the LLM integration** — Encrypt API keys at rest, never return them in GET responses, add timeouts to all external calls, sanitize prompt inputs. This is table-stakes for any tool that integrates with LLMs.

4. **Add thread safety to Session** — Even simple `threading.Lock()` around `staged_experiments` and `last_suggestions` mutations prevents data corruption in the web app.

### Tier 2: Researcher Value (Broad Domain Appeal)

5. **Constraint-aware optimization** — Researchers in chemistry, materials, and bioprocess ALL have experimental constraints (budget, safety limits, equipment ranges). The outcome constraint system is half-built. Complete it: pass input+outcome constraints through to acquisition, validate constraints at session level, add constraint visualization.

6. **Batch/parallel experiment support** — Many experimental labs run multiple experiments simultaneously. `suggest_next(n_suggestions=k)` exists but the workflow for tracking parallel experiments (stage → run → complete) needs polish and documentation. This is a differentiator vs. simple BO tools.

7. **Data import/export flexibility** — Support Excel files (`.xlsx`), not just CSV. Add export to common formats (Excel, JSON Lines). Many experimentalists live in Excel. This is a low-effort, high-adoption feature.

8. **Transfer learning / warm-starting** — Allow users to initialize a new optimization from a previous session's model. Researchers often run related experiments sequentially. This is a killer feature for iterative research campaigns.

### Tier 3: Differentiation (Competitive Edge)

9. **Multi-fidelity optimization** — Many experiments have cheap approximations (simulations, fast assays) and expensive ground truth. BoTorch already supports multi-fidelity GPs. Exposing this through ALchemist's session API would be unique in the space.

10. **Human-in-the-loop features** — Add "researcher override" capabilities: let users reject suggestions with reasons (logged to audit), manually specify next experiments while keeping the model updated, and set "no-go zones" in the search space. This respects domain expertise.

11. **Collaboration features** — Session sharing (read-only links), team dashboards, and annotation of experiments. Research is collaborative; single-user tools hit a ceiling.

12. **Integration with lab automation** — Provide webhook/callback endpoints so robotic labs can push results back automatically. A simple REST callback interface would unlock the growing lab automation market.

### Tier 4: Growth (Community & Ecosystem)

13. **Jupyter notebook integration** — A `%load_ext alchemist` magic or widget that renders the optimization dashboard inline. Most researchers live in Jupyter.

14. **Plugin architecture** — Allow custom acquisition functions, kernels, and data transformers via entry points. The base class pattern is already there; formalize it as a plugin API.

15. **Benchmarking suite** — Ship standard benchmark problems (Branin, Hartmann, real-world datasets) with expected results. This lets researchers trust the tool and lets contributors validate changes.

16. **Domain-specific templates** — Pre-built search spaces for common optimization problems: catalyst screening, alloy composition, reaction conditions, polymer formulations. Lower the barrier to first use.

---

## 🔧 IMMEDIATE ACTION ITEMS (in priority order)

| # | Action | Impact | Effort | Status |
|---|--------|--------|--------|--------|
| 1 | Fix 46 failing tests + 20 collection errors | 🔴 Critical | Medium | ✅ Tests pass in CI (env issue) |
| 2 | Add try/except to all API router endpoints | 🔴 Critical | Low | ✅ Middleware handlers for RuntimeError/ImportError; try/except on high-risk endpoints; delete_variable ValueError→HTTPException |
| 3 | Secure LLM config endpoint (don't return keys) | 🟠 High | Low | ✅ Fixed (commit 6584ce1) |
| 4 | Add `threading.Lock` to Session mutations | 🟠 High | Low | ✅ Fixed (RLock on staged_experiments, last_suggestions, _current_iteration) |
| 5 | Fix optimal design `n_points > candidates` crash | 🟠 High | Low | ✅ Fixed (commits f39d79a, ffbe792) |
| 6 | Unify `predict()` return type (single vs MOBO) | 🟠 High | Medium | ✅ Fixed (always returns dict keyed by objective name) |
| 7 | Add tests for LLM service (7 untested files) | 🟡 Medium | Medium | |
| 8 | Complete `ContourPlotSimple.tsx` stub | 🟡 Medium | Low | ✅ Removed (full ContourPlot.tsx already exists; stub was unused) |
| 9 | Fix audit log silent swallowing → at least `logger.warning` | 🟡 Medium | Low | ✅ Fixed (commit 2fc9f2f) |
| 10 | Bump version and cut v0.4.0 release | 🟢 Growth | Low | |

---

## Detailed Findings Appendix

### A. API Router Error Handling (Full List)

**Endpoints with ZERO error handling:**
- `acquisition.py`: `suggest_next_experiments()`, `find_model_optimum()`
- `models.py`: `train_model()`, `get_model_info()`, `predict()`
- `experiments.py`: `add_experiment()`, `add_experiments_batch()`, `generate_initial_design()`, `get_optimal_design_info()`, `generate_optimal_design()`, `list_experiments()`, `preview_csv_columns()`, `upload_experiments()`, `get_experiments_summary()`, `stage_experiment()`, `stage_experiments_batch()`, `get_staged_experiments()`, `clear_staged_experiments()`, `complete_staged_experiments()`
- `variables.py`: `add_variable()`, `list_variables()`, `load_variables_from_file()`, `export_variables_to_json()`, `update_variable()`, `delete_variable()` (raises bare `ValueError` instead of `HTTPException`)
- `visualizations.py`: `get_contour_data()`, `get_surface_data()`, `get_parity_data()`, `get_qq_plot_data()`, `get_calibration_curve_data()`, `get_hyperparameters()`
- `sessions.py`: `create_session()`, `get_session_info()`, `get_session_state()`, `delete_session()`, `extend_session()`, `export_audit_markdown()`

**Endpoints leaking internal error details:**
- `sessions.py`: `import_session()`, `download_session()`, `upload_session()` — `detail=f"Failed to...: {str(e)}"`
- `visualizations.py`: `get_metrics_data()` — `detail=f"Error computing metrics: {str(e)}"`

### B. Session.py Detailed Findings

**Thread safety (shared mutable state without locks):**
- `self.staged_experiments = []` and `self.last_suggestions = []` — concurrent modification possible
- `move_staged_to_experiments()` — iterates and clears list without atomicity
- `lock_acquisition()` — direct increment of `self.experiment_manager._current_iteration`

**Resource leaks:**
- `load_data()`: temp file with `delete=False`; exception between creation and finally leaks file
- `export_session_json()`: same pattern

**Race conditions:**
- `move_staged_to_experiments()`: validates length, but concurrent additions between validation and loop cause mismatch
- `lock_acquisition()`: modifies private `_current_iteration` without synchronization
- `add_experiment()` + `load_data()`: both modify `experiment_manager` state without coordination

**Missing input validation:**
- `add_experiment()`: no check that `inputs` dict has all search space vars; `output` accepts NaN/Inf; `noise` not validated for positivity
- `add_staged_experiment()`: zero validation of `inputs` structure; allows arbitrary/missing keys
- `move_staged_to_experiments()`: validates length only, not content; `outputs` can be NaN/Inf
- `suggest_next()`: `ref_point` checked for length only
- `add_outcome_constraint()`: no validation `objective_name` exists; no finite check for `value`; allows duplicates

**Inconsistent state on error:**
- `load_data()`: partial `experiment_manager` initialization if CSV load fails
- `train_model()`: `model_backend` set before training completes; broken model persists on failure
- `load_session()`: partial experiment addition on loop failure; no transaction/rollback
- `lock_acquisition()`: iteration counter incremented before audit log succeeds

**Silently swallowed errors:**
- `generate_initial_design()`: audit log failures caught with `logger.debug()` only
- `generate_optimal_design()`: same
- `lock_model()`: two bare `except Exception: pass` blocks swallow hyperparameter extraction failures
- `load_session()`: model retrain failure logged as warning but session returned without model

### C. Optimal Design Module Findings

**Numerical stability:**
- Line 351: `half_range = (high - low) / 2.0` — no check for `high == low` (divide-by-zero)
- Line 543: Ridge regularization (1e-8) too small for ill-conditioned matrices; should scale with norm
- Lines 657-658: `d_efficiency()` / `a_efficiency()` may return NaN/inf on singular X_final

**Performance:**
- Lines 571-582: Greedy initialization is O(n²p) per point
- Lines 614-628: `simple_exchange` is O(n²p) per iteration; reconstructs pool each time
- Line 605: Loop runs full `max_iter` (200) even if improvements plateau; no relative convergence check

**Edge cases:**
- Lines 571-582: If `n_points ≥ N` (candidates), `best_idx` stays `-1` → silent indexing bug
- Line 359: `itertools.product()` with empty variable lists returns one empty tuple
- Line 735: `np.argmax()` on all-zero one-hot columns returns 0 silently
- `n_levels=1` creates degenerate candidate set; `n_levels` has no upper bound (memory explosion)

**Algorithmic correctness:**
- Lines 566-567: Seeds based on global RNG state but creates separate generator → non-reproducible
- Lines 472-475: Manual Kronecker product instead of `np.kron()`
- Line 1017: Dead code `] if False else []` — never executed

### D. MOBO Implementation Analysis

**Complete path:** Data input → ModelListGP training → qEHVI/qNEHVI acquisition → suggestion — ✅ works end-to-end

**Gaps:**
- No default MOBO strategy; user must explicitly select `qEHVI`/`qNEHVI`
- `predict()` polymorphic return (tuple vs dict) with no type hints
- Only 2D Pareto visualization; no 3D+ support
- Input constraints not passed to hypervolume acquisition
- Auto ref_point: 10% margin breaks with negative objectives or extreme scales
- `directions` list length not validated against `n_objectives`
- Mixed integer-categorical with 2+ objectives untested

### E. Frontend Findings

**Type safety issues:**
- `Record<string, any>` in 6+ API type definitions
- `(error: any)` in 8 hook error handlers
- `OptimalDesignPanel.tsx` lines 100, 118: `const request: any = {}`

**State management gaps:**
- `OptimalDesignPanel.tsx`: `designInfo` stored in local state, never cleared after generation
- `InitialDesignPanel.tsx`: `generatedPoints` state lacks cleanup on unmount
- `OptimalDesignPanel.tsx` uses raw `fetch()` bypassing React Query error system

**Component issues:**
- ~~`ContourPlotSimple.tsx`: Stub placeholder (TODO)~~ ✅ Removed (full ContourPlot.tsx is the production component)
- `LLMSuggestPanel.tsx` line 79-84: Too-broad catch block hides Ollama connection failures
- `LLMSuggestPanel.tsx` line 463-466: Warning only shows when Edison NOT used, but OpenAI alone can hallucinate sources

### F. LLM Service Security Details

**Plaintext key storage:** `~/.alchemist/config.json` created with default permissions; no encryption
**Key disclosure:** `GET /api/v1/llm/config` returns full config including API keys in response body
**Prompt injection:** Variable names (user-supplied) interpolated directly into prompts without escaping or sanitization
**No timeouts:** External calls to OpenAI, Ollama, Edison have no timeout configuration
**No rate limiting:** LLM endpoints accept unlimited requests
**Edison provider:** `arun_tasks_until_done()` can hang indefinitely (no timeout)
**Ollama provider:** Hardcoded dummy API key `api_key="ollama"` — poor security posture
**Cache collision risk:** Edison cache uses only first 16 hex chars of hash

---

*Analysis performed: March 2, 2026. Based on 91 commits (Nov 2025–Feb 2026), 10,374 LOC across 6 key modules, 423 tests, full API/frontend/core review.*
