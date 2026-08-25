# Suggested-vs-Actual Provenance

**Date:** 2026-07-28
**Context:** Inductive RCC deployment (Wilson/Anna). When a result is recorded, the model's *suggested* conditions are discarded — only the *actual* conditions + a `Reason` strategy label survive in the dataset. This lets actuals deviate arbitrarily from the suggestion with no per-row record ("claim it was EI even if you significantly deviated"). We want a durable, per-data-point record answering: **"what did the model suggest, and what did we actually run?"**

## Problem

Two disconnected paths exist:
1. **The queue** (`QueueItem`: `id` uuid, `inputs`=suggested, `reason`, `status`, `output`) with a `complete(id, output)` lifecycle designed for exactly this.
2. **What the web app does:** bypasses the queue — calls `add_experiment` directly from the *actual* values, then DELETEs the whole staged queue and re-stages the remainder. The queue item's `id` and suggested `inputs` are thrown away, never linked to the created row.

So the suggested→actual pairing per experiment is lost. The `acquisition_locked` audit entry captures suggestions at the *batch* level but has no join key to the specific dataset row that resulted, and records no delta.

## Solution

Route web add-point through the queue **`complete()` lifecycle**. The queue item already holds suggested inputs keyed by uuid; extend completion to also capture the actual inputs and emit a full provenance record to the audit log and the session file. Fixes the architectural disconnect as a side benefit and sets up the future queue-editing UI.

### Data flow (recording a result)
1. Dialog knows the queue-item `id` for the suggestion being recorded.
2. On save → `POST /experiments/queue/{item_id}/complete` with `{ actual_inputs, output, noise?, iteration }`.
3. `queue.complete()`:
   - reads the item's **suggested** inputs (stored as `item.inputs`),
   - records the dataset row from **actual** inputs via `add_experiment`, stamping a hidden `ProvenanceId = item_id` column,
   - computes per-variable **delta** (numeric: actual − suggested; categorical/discrete: `"unchanged"` or `"suggested→actual"`),
   - writes a provenance record to the audit log + a `provenance` section of the session file,
   - marks the item `done`.

## Data model

Provenance lives in three places, each with a distinct role:

### 1. Dataset row — join key only
A hidden `ProvenanceId` column = the queue-item uuid. The ONLY dataset change. Must be added to `metadata_cols` at **every** site that builds the model input matrix `X`, so it never becomes a GP feature. Excluded from the Add Point dialog variable rows and hidden in the experiments table.

### 2. Session file — new top-level `provenance` section
A list of full records, serialized in `save_session`, restored in `_load_session_impl` (backward-compatible: missing key → empty, like `staged_experiments`). Each record:
```json
{
  "id": "<queue-item uuid>",
  "iteration": 1,
  "strategy": "qEI",
  "acq_params": { "goal": "maximize", "n_suggestions": 5 },
  "suggested": { "Alkaline Species": "K", "Mole %": 1.0 },
  "actual":    { "Alkaline Species": "K", "Mole %": 1.2 },
  "delta":     { "Mole %": 0.2, "Alkaline Species": "unchanged" },
  "output": 0.42,
  "noise": null,
  "timestamp": "2026-07-28T..."
}
```
For a **manually-added** experiment (no staged suggestion): `suggested: null`, `strategy: "Manual"`, fresh `ProvenanceId`. Every row gets a record so the "what was suggested" answer is uniformly available.

### 3. Audit log — one `experiment_completed` entry per record
Same payload, via the existing `AuditEntry` mechanism (hashed, persisted). Makes the reproducibility log answer "suggested vs run" independently of the session file.

**delta:** numeric → `actual − suggested`; categorical/discrete → `"unchanged"` or `"suggested→actual"`. Computed once at completion (durable, not recomputed).

**strategy/acq_params source (unambiguous):** the queue item carries `reason` (the strategy name) — that is the source of `strategy`. For `acq_params`, the backend looks up the most recent `acquisition_locked` audit entry whose `iteration` matches the item's iteration; if none is found (e.g. manually staged), `acq_params` is `{}`. The frontend is NOT relied upon for provenance content — it only supplies `actual_inputs`, `output`, `noise`, `iteration`, and the `item_id`.

## API changes

- **`QueueItem`:** original `inputs` remain the suggested snapshot. At completion, store `actual_inputs`, `delta`, `output`, `noise`, `iteration`, `strategy`, `acq_params`, `timestamp`.
- **`queue.complete(item_id, actual_inputs, output, noise=None, iteration=None)`:** builds delta from `item.inputs` vs `actual_inputs`; calls `session.add_experiment(actual_inputs, output, ..., provenance_id=item_id)`; emits provenance record to audit log + session `provenance` list; marks item `done`.
- **`QueueCompleteRequest`:** extend to accept `actual_inputs`, `noise`, `iteration` (currently only `outputs`).
- **Manual-add path:** `add_experiment` endpoint still handles no-suggestion entries; writes a provenance record with `suggested: null`, `strategy: "Manual"`, fresh `ProvenanceId`.
- **New read endpoints:** `GET /experiments/provenance` and `GET /experiments/provenance/{provenance_id}`.

## UI (minimal — no clutter)

- Add Point dialog: **no visual change** (already shows read-only suggested + editable actual). Only plumbing changes — submit to `complete(item_id, actual_inputs, ...)` instead of the direct add + delete/restage dance, passing the queue-item id being recorded.
- `ProvenanceId` hidden everywhere.
- No new panel now. Provenance is queryable via the endpoint and present in session file + audit log. A "view provenance for this row" affordance can come later with the queue-editing UI.

## Testing

- **Core:** `complete()` produces a record with correct suggested/actual/delta; `ProvenanceId` on the row; round-trips through save/load; **never appears in model `X`** (explicit test that trains a model with provenance present and asserts it's not a feature).
- **API:** complete endpoint stores actual+delta; manual add creates a `strategy: "Manual"` record; `GET /provenance` returns records.
- **Frontend:** dialog submits to complete with item id + actual inputs (extend the existing dialog test).

## Key risk

`ProvenanceId` leaking into the model `X`. Audit **every** `metadata_cols` / `drop(columns=...)` site (one at `experiment_manager.py:146-153`; likely 2-3 more for CV / multi-objective paths) and add a regression test that trains a model and asserts `ProvenanceId` is not a feature.

## Out of scope (separate cycles)

- Queue clear/delete/reorder UI (this migrates the web app onto the real `complete()` lifecycle, which that work builds on, but does not add the editing UI).
- A dedicated provenance-viewing panel in the web app.
- Mid-batch reload restoring `Iteration` on staged items (pre-existing minor gap).
