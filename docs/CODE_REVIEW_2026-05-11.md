# ALchemist Code Review — Pre-ChemCatBio Meeting

**Date:** 2026-05-11
**Scope:** `alchemist_core/`, `api/`, `alchemist-web/`, README + docs + examples
**Focus:** correctness, code quality, demo-readiness
**Test baseline:** 849 passed in `alchemist-env` (3m 27s) at review time; **852 passed (1m 53s) after fixes below**

This document lists findings in priority order. Each item has a file:line reference and a recommended fix. The "Demo readiness checklist" at the end groups the quick wins worth landing before the talk.

---

## Status of fixes landed this session

| ID | Status | Notes |
|----|--------|-------|
| **P0-1** | ✅ Fixed | `session.py:1644` — `means, stds = self.predict(grid)[target_name]`; 3 regression tests added in `tests/unit/core/acquisition/test_acquisition.py` (sklearn maximize/minimize + botorch maximize). |
| **P0-3** | ✅ Fixed | `botorch_model.py` — removed module-level `gpytorch.settings.cholesky_jitter(1e-2)` global mutation. Added `_CHOLESKY_JITTER = 1e-4` and wrapped each `fit_gpytorch_mll` call (3 sites: lines 333, 437, 736) in `with gpytorch.settings.cholesky_jitter(_CHOLESKY_JITTER):`. |
| **P2-1** | ✅ Fixed | `alchemist-web/src/App.tsx:337-340` — debug `console.log` block removed. |

Remaining items below are unaddressed.

---

## P0 — Blockers / correctness bugs

### P0-1. `find_optimum()` is broken for single-objective sessions ✅ FIXED
**File:** `alchemist_core/session.py:1644`

`self.predict` returns `Dict[str, Tuple[ndarray, ndarray]]` for **both** single- and multi-objective cases (see line 1701: `return {target_name: (preds, stds)}`). Unpacking a one-key dict into two names raised `ValueError: not enough values to unpack (expected 2, got 1)`.

`find_optimum` is part of the documented public API (`docs/api/acquisition.md:19,35`), so any user following the docs hit this immediately.

**Fix applied:**
```python
target_name = self.objective_names[0]
means, stds = self.predict(grid)[target_name]
```

Regression tests added in `tests/unit/core/acquisition/test_acquisition.py::TestSessionFindOptimum` covering sklearn maximize, sklearn minimize (asserts min ≤ max), and botorch maximize.

### P0-2. Linear input constraints are sign-flipped end-to-end
**Files:** `alchemist_core/data/search_space.py:344-371` (user-facing) and `:376-422` (BoTorch conversion)

`add_constraint` docstring promises `inequality ⇒ sum(coeff·x) <= rhs`. `to_botorch_constraints` forwards `(indices, coeffs, rhs)` to BoTorch unchanged. BoTorch's `optimize_acqf` documents `inequality_constraints` as `sum(coeff_i · X[idx_i]) >= rhs` — the opposite half-space. Result: any user-defined linear constraint silently optimizes in the infeasible region.

The docstring at line 380 ("`sum - rhs <= 0`") is also incorrect about BoTorch's convention.

**Fix:** in `to_botorch_constraints`, negate both coefficients and rhs:

```python
constraint_tuple = (
    torch.tensor(indices, dtype=torch.long),
    torch.tensor([-c_ for c_ in coeffs], dtype=torch.double),
    -c['rhs'],
)
```

Add a regression test: define `x0 + x1 <= 1`, call `suggest_next`, assert returned points satisfy it.

**Demo impact:** low — no example or doc uses `add_constraint`. Worth fixing because the bug is fully silent.

### P0-3. Global `cholesky_jitter(1e-2)` inflates GP uncertainty ✅ FIXED
**File:** `alchemist_core/models/botorch_model.py:22` (was)

BoTorch's default is `1e-6`. `1e-2` was enormous — large enough to materially widen posterior bands on small/clean toy demos and to mask kernel-fit problems. It was also set at module load, polluting the global GPyTorch state for every consumer that imports `alchemist_core`.

**Fix applied:** removed the module-level call. Introduced `_CHOLESKY_JITTER = 1e-4` constant and wrapped each `fit_gpytorch_mll` call in `with gpytorch.settings.cholesky_jitter(_CHOLESKY_JITTER):` (single-output train, MOBO ModelListGP per-objective train, CV per-fold train). 1e-4 retains a safety margin over BoTorch's 1e-6 default for ill-conditioned kernels without globally distorting uncertainty bands. Full test suite passes (852/852).

---

## P1 — Important quality / footguns

### P1-1. Duplicate `set_config` definition silently shadows the first
**File:** `alchemist_core/session.py:1726` and `:2253`
Two methods on the same class with the same name; Python keeps the second. Either one is dead code or behavior has drifted. Diff them, pick one, delete the other.

### P1-2. `qEHVI` silently drops outcome constraints
**File:** `alchemist_core/acquisition/botorch_acquisition.py:306-333`
The constrained branch only fires for `qnehvi`. For `qehvi`, `self.outcome_constraints` is built and discarded. Constrained MOBO demos pointed at `qEHVI` will silently ignore the constraints.

**Fix:** raise (or log a warning) when `outcome_constraints` are present and `qehvi` is selected, directing the user to `qNEHVI`.

### P1-3. `goal=['minimize']` silently maximizes
**File:** `alchemist_core/session.py:1462, 1478`

```python
maximize=(goal.lower() == 'maximize') if isinstance(goal, str) else True,
```
Passing a list defaults `maximize=True`. Easy presenter typo (especially when toggling between single- and multi-objective snippets).

**Fix:** for single-objective, `raise TypeError` on list-typed `goal`, or honor `goal[0]`.

### P1-4. `load_session` `retrain_on_load=False` ignored in classmethod form
**File:** `alchemist_core/session.py` `load_session` (~line 2080)
The dual-mode function (`Session.load_session(path)` vs `session.load_session(path)`) consumes `path` into `self`, leaving `filepath=None`, then defaults retraining to True. The `retrain_on_load=False` kwarg is silently dropped.

**Fix:** make `load_session` a proper `@classmethod`; or split into `load_session(cls, ...)` and `reload_into(self, ...)`.

### P1-5. `to_botorch_bounds` returns a dict but the caller's tensor-shortcut is dead code
**File:** `alchemist_core/data/search_space.py:193` returns `Dict[str, np.ndarray]`. `alchemist_core/acquisition/botorch_acquisition.py:695-698` checks for a tensor — never true — so the fallback rebuilds bounds via `get_tunable_variable_names()` ordering. With context/derived variables, this ordering is not guaranteed to match `model.original_feature_names`. Silent misalignment between bounds and the GP's feature axis.

**Fix:** either make `to_botorch_bounds` return a tensor matching `feature_names`, or add an explicit `assert tunable_names == [n for n in model.original_feature_names if n not in context]` plus a regression test.

### P1-6. `DerivedFeatureTransform.transform` is per-row Python in the hot path
**File:** `alchemist_core/models/transforms.py`
`apply(lambda row: func(row.to_dict()))` is invoked thousands of times per `optimize_acqf` call (raw samples × restarts × q). Becomes the dominant cost for any problem using derived variables. Declared `input_cols` are also not enforced — undeclared columns are silently usable from the func.

**Fix:** vectorize over the input matrix; pass only declared `input_cols` and raise on missing dependencies.

### P1-7. Audit log captures `search_space_definition` only on first lock
**File:** `alchemist_core/session.py:1762, 1806`
`if self.audit_log.search_space_definition is None: ...` — if variables are added/changed between locks, only the first snapshot is recorded. The audit-log promise of reproducibility is partially broken.

**Fix:** deep-copy the current variables into each `AuditEntry` (or into a list keyed by iteration).

### P1-8. `BoTorchAcquisition.select_next` swallows all exceptions in the alternating path
**File:** `alchemist_core/acquisition/botorch_acquisition.py:458-491`
`except Exception: pass # fall through` hides the real error and the fallback path can subsequently `AttributeError` on `batch_acq_values.numel()` — making misconfig look like a low-level crash.

**Fix:** `except Exception as e: logger.warning("alternating optimizer failed: %r — falling back", e)`.

### P1-9. Sync ML work runs on the event loop
**File:** `api/routers/models.py:35-44`
`session.train_model(...)` is called directly inside `async def`. GP fits block the loop and freeze every concurrent WS/HTTP. The right pattern is already used at `api/routers/visualizations.py:383-388` (`run_in_threadpool`). Audit `acquisition.py`, `experiments.py`, `models.py` for the same.

### P1-10. `SessionStore.threading.Lock` taken inside async handlers
**File:** `api/services/session_store.py:18-26` (with a TODO already in place)
Same blast radius as P1-9. Plan referenced in `memory/SESSION_LOCKING_ASYNC_PLAN.md` — migrate to `anyio.Lock`.

### P1-11. Encapsulation break in session upload path
**File:** `api/routers/sessions.py:440-446`
Reaches into `session_store._sessions` and calls `_save_to_disk` directly, bypassing the per-session lock used elsewhere. Add `SessionStore.replace_session(id, session)` that locks correctly.

### P1-12. TTL is a public lie
TTL was removed (`api/services/session_store.py:38, 254-274` — `extend_ttl` and `_cleanup_expired` are no-ops) but it is still advertised in:
- `api/routers/sessions.py:29-43, 118` — docstring + `PATCH /sessions/{id}/ttl` endpoint
- `api/API_ENDPOINTS.md:32, 41-42, 57-58, 88, 97, 863`
- `api/README.md:66, 147, 280-281`
- `api/example_client.py:25, 175`
- `alchemist-web/src/api/types.ts:13, 31, 37, 42`
- `alchemist-web/src/App.tsx:191` (sends `ttl_hours: 24`)

Pick one: implement it, or rip the field, endpoint, and types out. Any external integrator reading the docs will be misled — and questions about "what's the TTL for?" at the meeting are quite plausible.

---

## P2 — Polish

- **P2-1.** ✅ FIXED. Debug `console.log('=== Render State ===')` block at `alchemist-web/src/App.tsx:337-340` removed.
- **P2-2.** 63 `console.*` calls in the React app, most ungated. `api/client.ts:28,44` is already DEV-gated — apply the same pattern (`useSessionEvents.ts:62,72,135,142`, `useLockStatus.ts:54,64,107,114`, `AcquisitionPanel.tsx:123,151,549`).
- **P2-3.** `App.tsx` makes 8 raw `fetch(...)` calls (lines 63, 81, 130, 163, 216, 254, 275, 293), bypassing the 30s-timeout axios client + interceptors. Fold into `api/endpoints/*.ts` + React Query hooks. `alchemist-web/src/components/api.ts` is a parallel mini-client used by only two files — merge in.
- **P2-4.** 55 occurrences of `: any` in the frontend; `useExperiments.ts` mutations all type errors as `any`. Define `ApiError = AxiosError<{detail: string; error_type: string}>` and replace.
- **P2-5.** React Query hooks lack `staleTime` — `useSession(sessionId)` refetches on every focus/mount. Add `staleTime: 5_000` in `useSessions.ts`, `useExperiments.ts`, `useVariables.ts`.
- **P2-6.** Inconsistent error envelopes: most responses go through `error_handlers.py` (`{detail, error_type, status_code}`), but raw `HTTPException` is still raised at `api/routers/llm.py:43-45`, `api/routers/sessions.py:461-464, 481-484`. Clients keying off `error_type` will break.
- **P2-7.** Stale design docs at the repo surface (drift from current code — TTL was the smoking gun):
  - `alchemist-web/PHASE2_PLAN.md`, `INTEGRATION_GPR.md`, `CONTOUR_PLOT_IMPLEMENTATION.md`, `SESSION_PERSISTENCE_FIXES.md`, `SESSION_SAVE_LOAD_FEATURE.md`, `QUICK_REFERENCE.md`
  - `alchemist-web/src/components/session-components.md`
  - Move under `docs/history/` or delete. Keep only `README.md` and `SETUP.md` at the surface.
- **P2-8.** `EventEmitter` (`alchemist_core/events.py`) iterates `self._listeners[event]` directly (not a copy). `once()` calls `self.off(...)` inside the iteration → potential `RuntimeError: list changed size during iteration` if a once-listener fires synchronously. Also no lock around subscribe/unsubscribe.
  **Fix:** `for cb in list(self._listeners.get(event, [])): ...`; add a lock.
- **P2-9.** `BoTorchModel.train` CV path has no guard for `len(y) < n_splits` or NaNs in `X`/`y`. Small-data demos crash inside GPyTorch with confusing tracebacks. Add explicit pre-checks at the top of `train`.
- **P2-10.** `predict` return-type changed to dict without renaming — older callers that expected `(mean, std)` silently break (P0-1 is one). Consider `predict` → `predict_all` rename; or keep tuple for SO and add `predict_all` for MO.
- **P2-11.** `cache/recovery/` contains ~50 stale recovery JSONs going back to Feb 2026; `logs/` has 10+ `.log` files. On first launch the "Unsaved work detected" banner (`App.tsx:127-140`) may surprise you on stage. Wipe before the demo or add a cleanup helper.
- **P2-12.** WS endpoint (`api/routers/websocket.py:34-71`) doesn't verify `session_id` exists in `session_store` before `accept()`. Clean-up nit, not a vuln.
- **P2-13.** `ExperimentManager.add_experiment` swallows `Exception` when syncing `_current_iteration` from row (`experiment_manager.py:72-75`); log at minimum.
- **P2-14.** `add_experiments_batch` validates required columns against search-space names only — silently accepts batches missing the `target_columns` (`experiment_manager.py:84-90`).
- **P2-15.** `qExpectedHypervolumeImprovement` is now deprecated in BoTorch (test warnings show "use `qLogExpectedHypervolumeImprovement` instead"). Plan a migration; mention only if asked at the talk.
- **P2-16.** `pyDOE==0.9.5` pinned to an old version in `pyproject.toml:49`. The package was renamed `pyDOE2`/`pyDOE3` years ago. Audit whether the pin is still required.
- **P2-17.** Several `ConvergenceWarning`s in sklearn GP tests (length-scale at upper bound); not a bug, but expanding the kernel bounds in defaults would suppress the noise.

---

## Security / deployment

CORS + zero authentication: `session_id` is the only access token, and `GET /api/v1/sessions` (`api/routers/sessions.py`) enumerates all live session IDs. On localhost this is fine. Before any non-loopback deploy:
1. Confirm `ALLOWED_ORIGINS` is restrictive (not `*`).
2. Add at least a token-based auth dependency on session routes.
3. Document the threat model in `SETUP.md` so external integrators don't expose ports naively.

---

## Demo Readiness Checklist

Order from highest demo-payoff to lowest. All achievable in a single working day.

**Must do (~1 hour):**

1. ✅ **DONE** — Fix `find_optimum` dict-unpack (P0-1). 3 regression tests added; full suite green.
2. ✅ **DONE** — Replace global `cholesky_jitter(1e-2)` with `_CHOLESKY_JITTER = 1e-4` scoped via context manager (P0-3).
3. ✅ **DONE** — Delete the four `console.log('=== Render State ===')` lines in `App.tsx` (P2-1).
4. Wipe `cache/recovery/` and trim `logs/` so the demo machine starts clean.
5. Pre-flight: run `alchemist-web` cold, walk through your demo script, check the browser console for noise.

**Should do (~2-4 hours):**

6. Decide on the TTL story (P1-12). Easiest: remove `ttl_hours` from `App.tsx:191` and `types.ts`; delete the `PATCH .../ttl` route; strip the references from `api/README.md` and `api/API_ENDPOINTS.md`.
7. Dedupe `set_config` in `session.py` (P1-1) — pure cleanup, but a duplicate method in a 6200-line file is a question waiting to happen if anyone opens the source on stage.
8. Wrap `session.train_model(...)` in `run_in_threadpool` (P1-9). If a viewer opens two browser tabs you don't want one freezing the other.
9. Validate `goal` on single-objective (P1-3). Quick `raise TypeError` if list.

**Nice to have (~half day):**

10. Fix constraint sign (P0-2) — only matters if someone asks specifically about constraints, but easy to land.
11. Add `qEHVI + outcome_constraints` warning (P1-2).
12. Move stale design docs out of `alchemist-web/` root (P2-7) — `git mv` to `docs/history/`.

**Defer past this meeting:**

- P1-5, P1-6 (bounds ordering, derived-variable perf) — touches hot code paths and merits its own PR cycle with proper test coverage.
- P2-10 `predict` API rename — semver breaking, plan for 0.4.
- BoTorch `qLogEHVI` migration (P2-15).
- Async lock migration (P1-10) — has its own plan doc already.

---

## What was not reviewed

- `alchemist_core/visualization/` internals beyond test-time signals
- `ui/` (desktop CustomTkinter app) — excluded from coverage per `pyproject.toml`
- `docker/` build configuration
- `.github/workflows/` CI specifics
- The `wiki/` QMD collection (QMD MCP server was hanging on a model fetch during this review)
- Detailed read of `api/routers/experiments.py` (721 lines) and `api/routers/acquisition.py` past imports
