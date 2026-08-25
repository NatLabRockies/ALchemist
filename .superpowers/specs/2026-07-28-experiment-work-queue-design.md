# ALchemist Experiment Work Queue — Design

**Date:** 2026-07-28
**Status:** Design spec (pre-implementation). Awaiting human review before writing the implementation plan.
**Repo:** ALchemist (`api/`, `alchemist_core/`, `alchemist-web/`)
**Upstream decision doc:** `nanoparticle-hmi/docs/superpowers/specs/2026-07-27-autonomous-monitoring-control-ux-decision.md` (§3.1–§3.3)

---

## 1. Problem

ALchemist's staged-experiments facility cannot represent live run-state, which blocks any trustworthy live monitoring/queue UI. Three concrete, code-verified problems:

1. **No per-item run-state.** `alchemist_core/session.py:111` stores `staged_experiments` as a flat `list[dict]`. `move_staged_to_experiments(outputs: List[float])` (`session.py:590`) maps outputs 1:1 by list order, and `POST .../experiments/staged/complete` (`api/routers/experiments.py:634`) completes the whole batch all-or-nothing. There are no per-item IDs, no status field, and no per-item completion. A consumer running a queue item-by-item cannot report "item 3 running, item 4 failed"; ALchemist learns nothing until the whole batch completes, and a mid-run failure discards the successes.

2. **Per-item reason is silently lost.** `GET .../experiments/staged` collapses `_reason` to the first item's value (`experiments.py:600-601`). Staging two suggestions with different rationales loses the distinction. (Note: the batch *move* at `session.py:642` already reads per-item `_reason`; the loss is in the read/response path and the batch-level `reason` override.)

3. **The objective is an unlabeled bare float.** Consumers complete with a raw scalar; ALchemist labels it `"Output"` everywhere (parity plots, metrics-over-time, contour axes) with no unit or provenance. If a consumer changes what the scalar means mid-campaign, nothing flags the now-discontinuous objective.

### Load-bearing constraint

ALchemist is a **general-purpose, domain-agnostic** active-learning toolkit with multiple future consumers. **No consumer-specific concepts** (reactors, spectroscopy, bands, detectors, MQTT) may leak in. Every mechanism here must make sense for an arbitrary black-box optimization target. The objective label is therefore an **opaque display string** the consumer sets; ALchemist stores and displays it but never parses it.

---

## 2. Approach

Evolve `staged_experiments` **in place** into a real work queue. One source of truth, backed by a new `ExperimentQueue` class. New per-item endpoints are additive; the legacy flat endpoints become a thin, deprecated compatibility layer over the new model. Per-item reason falls out naturally from per-item structure (fixes problem 2).

Rejected alternatives: a *parallel* queue alongside untouched staged endpoints (two overlapping concepts, drift); a *hard replace* (immediate break for the current HMI consumer).

---

## 3. Data Model

New module `alchemist_core/queue.py`.

### 3.1 `QueueItem` (dataclass)

| Field | Type | Notes |
|---|---|---|
| `id` | `str` | Server-generated UUID, assigned on stage |
| `inputs` | `dict[str, Any]` | Variable values only (no `_`-prefixed metadata) |
| `reason` | `str \| None` | **Per-item** — fixes problem 2 |
| `status` | `Literal["pending","running","done","failed"]` | |
| `output` | `float \| list[float] \| None` | Set on completion; `list` for multi-objective |
| `noise` | `float \| list[float] \| None` | Optional measurement uncertainty |
| `error` | `str \| None` | Set on fail |
| `dataset_ref` | `int \| None` | Iteration/index of the dataset row created on `done` |
| `staged_at` | `str \| None` | ISO timestamp |
| `started_at` | `str \| None` | ISO timestamp (set on `start`) |
| `completed_at` | `str \| None` | ISO timestamp (set on `done`/`failed`) |

### 3.2 State machine

```
pending ──start──▶ running ──complete──▶ done
   │                  │
   ├──complete────────┴──────────────────▶ done      (running is optional)
   │
   └──fail / running──fail──▶ failed
```

- `running` is **optional**: a consumer may complete straight from `pending`.
- Terminal items (`done`/`failed`) **persist as run history until purged**.
- Illegal transitions (e.g. completing an already-`done` item, starting a terminal item) raise `ValueError`.

### 3.3 `ExperimentQueue` (class)

Owns the ordered item list, an `RLock` (moved off `session._lock` into the queue), all transitions, and event emission via an injected `EventEmitter`. Independently unit-testable without a session.

Public methods (indicative):
`stage(inputs, reason) -> QueueItem` · `stage_many(...)` · `get(id)` · `list(status=None)` ·
`start(id)` · `complete(id, output, noise=None)` · `fail(id, error)` ·
`delete(id)` (pending-only) · `purge()` (terminal-only) · `pending_items()`.

`stage` strips `_`-prefixed metadata from `inputs` and lifts any legacy `_reason` into the `reason` field.

### 3.4 `OptimizationSession` integration

- `session.queue: ExperimentQueue` replaces the raw `staged_experiments` list.
- Legacy methods (`add_staged_experiment`, `get_staged_experiments`, `clear_staged_experiments`, `move_staged_to_experiments`) are reimplemented as **delegations** to the queue, preserving their existing signatures/behavior (with the documented delete semantics change).
- On `complete`, the queue calls back into the session's `add_experiment(inputs, output, noise, reason, iteration)` and records the resulting row index in `dataset_ref`.

---

## 4. API Surface

All under `/api/v1/sessions/{session_id}`.

### 4.1 New per-item endpoints

| Method + path | Purpose |
|---|---|
| `POST /experiments/queue` | Stage one or more items. Body: `{items: [{inputs, reason?}]}`. **Returns assigned IDs** (`[{id, inputs, reason, status}]`) so a consumer can map them to its own identifiers. |
| `GET /experiments/queue` | List all items with full per-item state. Optional `?status=pending\|running\|done\|failed` filter. The UI's poll/resync endpoint. |
| `GET /experiments/queue/{id}` | Single item. 404 if unknown. |
| `POST /experiments/queue/{id}/start` | `pending → running`. 409 if not `pending`. |
| `POST /experiments/queue/{id}/complete` | Body: `{outputs, noise?, iteration?, expected_objective_label?, force?}`. `→ done`; adds to dataset; sets `dataset_ref`. 409 on illegal transition or objective-label mismatch (see §5). |
| `POST /experiments/queue/{id}/fail` | Body: `{error}`. `→ failed`. Does **not** touch the dataset. |
| `DELETE /experiments/queue/{id}` | Remove a single **pending** item. 409 if `running`/`done`/`failed`. |
| `POST /experiments/queue/purge` | Remove terminal (`done`/`failed`) items. Returns `n_purged`. |

`outputs` is a list to support multi-objective; single-objective sends `[value]`. Length must match `session.n_objectives`.

**Consumer↔ID correlation:** ALchemist assigns UUIDs and returns them from the stage call. The consumer builds and owns any `{alchemist_item_id ↔ its own experiment_id}` mapping. ALchemist never learns the consumer's identifier — stays domain-agnostic.

### 4.2 Legacy compatibility layer (deprecated)

Marked `deprecated=True` in OpenAPI; behavior otherwise preserved except where noted.

| Method + path | Behavior against new model |
|---|---|
| `POST /experiments/staged/batch` | Stages items; `reason` now stored per-item on each `QueueItem`. |
| `GET /experiments/staged` | Returns clean inputs (unchanged shape). Adds a parallel `reasons: [...]` list so per-item reason is available; the scalar `reason` field remains = first item's value for back-compat. |
| `DELETE /experiments/staged` | **Clears `pending` only** (behavior change — see §7). |
| `POST /experiments/staged/complete` | Completes all `pending` items in stage order, `outputs` mapped 1:1, internally calling per-item complete. **409 if any `running`/`done`/`failed` item exists** (ambiguous under the new model). |

---

## 5. Objective Label & Mid-Campaign Guard

### 5.1 Storage

Per-objective map on the session, keyed by target column name:

```python
session.objective_metadata: dict[str, {"label": str, "unit": str | None}]
# e.g. {"Output": {"label": "area_carbonyl_1987", "unit": "a.u."}}
```

- **Opaque** to ALchemist — stored/displayed, never parsed.
- Domain-agnostic; works for single- and multi-objective (keyed by `session.objective_names`).
- Serialized in `save_session` (session `version` bumped). `load_session` tolerates its absence (older sessions → empty map).

### 5.2 Endpoints

| Method + path | Purpose |
|---|---|
| `GET /objective-metadata` | Current label/unit map. |
| `PUT /objective-metadata` | Set/update: `{objective_name: {label, unit?}}`. Writes an audit entry (§6). |

### 5.3 Display integration

Parity plots, metrics-over-time, and contour axes resolve each objective's display string as `label (unit)`, falling back to the raw column name (today's `"Output"`). Purely a display concern; the numeric pipeline is untouched.

### 5.4 Mid-campaign guard

On `complete`, the body may carry `expected_objective_label` (itself a map `{objective_name: label}` for multi-objective).

- If provided and any entry does not match the session's current label for that objective → **HTTP 409**, unless `force: true`.
- Any actual label change via `PUT` → always written to the audit log (old/new values).
- If the consumer sends nothing → ALchemist just records; no block.
- ALchemist only compares opaque strings; it never interprets them.

---

## 6. Events & Audit

### 6.1 Events (core `EventEmitter` → API `broadcast_to_session` over the existing WebSocket)

The WebSocket infra already exists (`api/routers/websocket.py`, `broadcast_to_session`) and already pushes `experiments_updated` / `model_trained`.

- `queue_item_updated` — per transition: `{item_id, status, reason, output?, error?}`. UI updates one row without refetching.
- `queue_updated` — coarse ping: `{n_pending, n_running, n_done, n_failed}`. Cheap resync trigger.
- Reconnecting client does one `GET /experiments/queue` to fully resync.
- Existing `experiments_updated` / `model_trained` retained (fire when a `done` item lands in the dataset / auto-train runs).

### 6.2 Audit (`alchemist_core/audit_log.py`)

- New audit entry type `objective_label_changed` (old/new per objective).
- Per-item reason survives into the dataset row via `add_experiment(reason=...)` (already supported); the `QueueItem` retains `dataset_ref` back to that row.
- **No** audit entry per queue transition — consistent with the "only explicit lock-ins log" philosophy; the queue itself is the live record. (Avoids audit spam.)

---

## 7. Migration & Back-Compat

- Session JSON `version` bump. `staged_experiments` serialized as `QueueItem` dicts.
- `load_session` migrates the old flat `[{...}]` form (and any `_reason`) into `QueueItem`s with `status="pending"`, empty timestamps.
- `objective_metadata` absent in old sessions → empty map; display falls back to raw column names (no visible change).

### Explicit breaking / behavior changes (call out at review)

1. **`DELETE /experiments/staged` now clears `pending` only** (was: clear everything). Protects a live run's records. A consumer that relied on wiping running/terminal items must use per-item `DELETE` / `purge`.
2. **Legacy batch `POST /experiments/staged/complete` returns 409 when any new-style `running`/`done`/`failed` item exists.** A consumer mixing the batch path with per-item endpoints must pick one.

Everything else on the legacy path preserves its contract; new endpoints are purely additive.

---

## 8. Testing (TDD)

1. **`ExperimentQueue` unit tests** — legal + illegal transitions; per-item reason preserved; thread-safety; `delete` pending-only; `purge` terminal-only; multi-objective `outputs`; timestamps set correctly.
2. **Session-level tests** — legacy method delegation; serialize/load round-trip; old-format migration; `objective_metadata` storage; guard 409/`force`; `dataset_ref` correctness.
3. **API tests** — each new endpoint; legacy compat (1:1 batch, 409 on mixed status, pending-only clear, `reasons` list on GET); event emission; WebSocket broadcast payloads.
4. **Resync test** — subscribe → drive transitions → reconnect → `GET /experiments/queue` matches the observed final state.

---

## 9. Out of Scope (flagged, belongs in the consumer)

- Any reactor/spectroscopy/band/detector/MQTT concept.
- What the objective scalar *means* or how it's reduced from raw signals — the consumer owns reduction and sets the opaque label.
- The consumer's own `experiment_id` ↔ ALchemist `item_id` mapping.
- Timeout/watchdog policy for stuck `running` items — the consumer decides when to `fail` an item.
- The cross-boundary **logging / error-provenance** decision (decision doc §3.4: shared vs. distinct audit log). It is deferred by the decision doc to the downstream controller/UX spec, not this API step. This spec only adds the `objective_label_changed` audit entry and preserves existing per-item reason provenance.

---

## 10. Consumers

- **alchemist-web** (`alchemist-web/src`): the intended live queue surface. Subscribes to `queue_item_updated` / `queue_updated`; resyncs via `GET /experiments/queue`. UI work is a separate downstream program step.
- **Autonomous controller** (out of repo): drives the per-item endpoints (`start` → `complete`/`fail`), sets the objective label, and uses `expected_objective_label` to guard mid-campaign target switches. Controller-side work is a separate downstream step.
