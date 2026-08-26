# ALchemist API Endpoints Reference

**Base URL**: `http://localhost:8000/api/v1`

**Interactive Documentation**: http://localhost:8000/api/docs

---

## Table of Contents

- [Sessions](#sessions)
- [Variables](#variables)
- [Experiments](#experiments)
- [Audit Log](#audit-log)
- [Control Channel](#control-channel)
- [Constraints](#constraints)
- [Models](#models)
- [Acquisition](#acquisition)

---

## Sessions

Manage optimization session lifecycle.

### Create Session

```http
POST /sessions
```

**Request Body**: not required (empty body or `{}` accepted)

**Response** (201 Created):
```json
{
  "session_id": "abc-123-def-456",
  "created_at": "2025-11-18T10:00:00Z"
}
```

### Get Session Info

```http
GET /sessions/{session_id}
```

**Response** (200 OK):
```json
{
  "session_id": "abc-123-def-456",
  "created_at": "2025-11-18T10:00:00Z",
  "variable_count": 3,
  "experiment_count": 15,
  "model_trained": true
}
```

### Get Session State

```http
GET /sessions/{session_id}/state
```

**Purpose**: Lightweight endpoint for monitoring autonomous optimization progress.

**Response** (200 OK):
```json
{
  "session_id": "abc-123-def-456",
  "n_variables": 3,
  "n_experiments": 15,
  "model_trained": true,
  "last_suggestion": {
    "temperature": 385.5,
    "flow_rate": 4.2,
    "catalyst": "A"
  }
}
```

### Delete Session

```http
DELETE /sessions/{session_id}
```

**Response** (204 No Content)

### Export Session

```http
GET /sessions/{session_id}/export
```

**Response**: Binary file (`.pkl` pickle file)

### Import Session

```http
POST /sessions/import
```

**Request**: Multipart form with `.pkl` file

**Response** (201 Created): Returns new session info

---

## Variables

Define and manage search space variables.

### Add Variable

```http
POST /sessions/{session_id}/variables
```

**Request Body** (Continuous/Real):
```json
{
  "name": "temperature",
  "type": "real",
  "min": 300,
  "max": 500,
  "unit": "K",
  "description": "Reactor temperature"
}
```

**Request Body** (Integer):
```json
{
  "name": "cycles",
  "type": "integer",
  "min": 1,
  "max": 10
}
```

**Request Body** (Categorical):
```json
{
  "name": "catalyst",
  "type": "categorical",
  "categories": ["A", "B", "C"]
}
```

**Response** (200 OK):
```json
{
  "message": "Variable added successfully",
  "variable_count": 3
}
```

### List Variables

```http
GET /sessions/{session_id}/variables
```

**Response** (200 OK):
```json
{
  "variables": [
    {
      "name": "temperature",
      "type": "real",
      "bounds": [300, 500],
      "unit": "K"
    },
    {
      "name": "catalyst",
      "type": "categorical",
      "categories": ["A", "B", "C"]
    }
  ],
  "count": 2
}
```

### Get Variable Details

```http
GET /sessions/{session_id}/variables/{variable_name}
```

**Response** (200 OK): Returns single variable details

### Delete Variable

```http
DELETE /sessions/{session_id}/variables/{variable_name}
```

**Response** (204 No Content)

---

## Experiments

Manage experimental data.

### Generate Initial Design (DoE)

```http
POST /sessions/{session_id}/initial-design
```

**Purpose**: Generate space-filling experimental designs for initial exploration.

**Request Body**:
```json
{
  "method": "lhs",
  "n_points": 10,
  "random_seed": 42,
  "lhs_criterion": "maximin"
}
```

**Methods**:
- `lhs` - Latin Hypercube Sampling (recommended)
- `sobol` - Sobol quasi-random sequences
- `halton` - Halton sequences
- `hammersly` - Hammersly sequences
- `random` - Uniform random sampling

**LHS Criteria** (for `method="lhs"`):
- `maximin` - Maximize minimum distance
- `correlation` - Minimize correlation
- `ratio` - Optimize aspect ratio

**Response** (200 OK):
```json
{
  "points": [
    {"temperature": 350.2, "flow_rate": 4.5, "catalyst": "A"},
    {"temperature": 420.8, "flow_rate": 7.2, "catalyst": "B"},
    {"temperature": 385.5, "flow_rate": 2.1, "catalyst": "C"}
  ],
  "method": "lhs",
  "n_points": 3
}
```

### Add Single Experiment

```http
POST /sessions/{session_id}/experiments
```

**Query Parameters**:
- `auto_train` (boolean, default: false) - Automatically retrain model after adding data
- `training_backend` (string, optional) - "sklearn" or "botorch"
- `training_kernel` (string, optional) - Kernel type

**Request Body**:
```json
{
  "inputs": {
    "temperature": 350,
    "flow_rate": 4.5,
    "catalyst": "A"
  },
  "output": 0.85,
  "noise": 0.01
}
```

**Response** (200 OK):
```json
{
  "message": "Experiment added successfully",
  "n_experiments": 16,
  "model_trained": true,
  "training_metrics": {
    "rmse": 0.045,
    "r2": 0.92,
    "backend": "sklearn"
  }
}
```

### Add Batch Experiments

```http
POST /sessions/{session_id}/experiments/batch
```

**Query Parameters**:
- `auto_train` (boolean, default: false)
- `training_backend` (string, optional)
- `training_kernel` (string, optional)

**Request Body**:
```json
{
  "experiments": [
    {
      "inputs": {"temperature": 350, "flow_rate": 4.5, "catalyst": "A"},
      "output": 0.85
    },
    {
      "inputs": {"temperature": 400, "flow_rate": 6.0, "catalyst": "B"},
      "output": 0.92
    }
  ]
}
```

**Response** (200 OK):
```json
{
  "message": "Batch of 2 experiments added successfully",
  "n_experiments": 18,
  "model_trained": false
}
```

### Upload Experiments from CSV

```http
POST /sessions/{session_id}/experiments/upload
```

**Query Parameters**:
- `target_column` (string, default: "Output") - Name of output column

**Request**: Multipart form with CSV file

**CSV Format**:
```csv
temperature,flow_rate,catalyst,Output
350,4.5,A,0.85
400,6.0,B,0.92
375,5.2,C,0.88
```

**Response** (200 OK):
```json
{
  "message": "Uploaded 3 experiments successfully",
  "n_experiments": 21
}
```

### List All Experiments

```http
GET /sessions/{session_id}/experiments
```

**Response** (200 OK):
```json
{
  "experiments": [
    {"temperature": 350, "flow_rate": 4.5, "catalyst": "A", "Output": 0.85},
    {"temperature": 400, "flow_rate": 6.0, "catalyst": "B", "Output": 0.92}
  ],
  "n_experiments": 2
}
```

### Get Experiment Summary

```http
GET /sessions/{session_id}/experiments/summary
```

**Response** (200 OK):
```json
{
  "n_experiments": 21,
  "has_data": true,
  "has_noise": false,
  "target_stats": {
    "min": 0.65,
    "max": 0.95,
    "mean": 0.82,
    "std": 0.08
  },
  "feature_names": ["temperature", "flow_rate", "catalyst"]
}
```

### Work Queue

The work queue is the current facility for staging suggestions and running them one item at a time. Each item has a server-assigned `id`, a per-item `reason`, and a `status` that moves through `pending → running → done | failed`. Terminal (`done`/`failed`) items persist as run history until purged, so a UI can render "item 3 running, item 4 failed" during a live campaign.

A `QueueItem` has this shape:

```json
{
  "id": "3f2a...uuid",
  "inputs": {"temperature": 375.2, "flow_rate": 5.8, "catalyst": "B"},
  "reason": "qEI",
  "status": "pending",
  "output": null,
  "noise": null,
  "error": null,
  "dataset_ref": null,
  "staged_at": "2026-07-28T10:15:00",
  "started_at": null,
  "completed_at": null
}
```

- `dataset_ref` is the **insertion-order index** of the dataset row created on completion. It is a provenance snapshot, not a stable key — do not use it to dereference a row after later edits/reloads.
- `output`/`noise` are scalars (single-objective). Multi-objective completion through the queue is not yet supported.

**Real-time updates**: transitions emit WebSocket events on `GET /ws/sessions/{session_id}` — `queue_item_updated` (`{item_id, status, reason, output, error}`) per item and a coarse `queue_updated`. A reconnecting client resyncs with a single `GET .../experiments/queue`.

#### Stage Queue Items

```http
POST /sessions/{session_id}/experiments/queue
```

**Purpose**: Stage one or more items. The response returns the assigned `id`s so a consumer can map them to its own identifiers.

**Request Body**:
```json
{
  "items": [
    {"inputs": {"temperature": 375.2, "flow_rate": 5.8, "catalyst": "B"}, "reason": "qEI"},
    {"inputs": {"temperature": 412.5, "flow_rate": 3.2, "catalyst": "A"}, "reason": "qEI"}
  ]
}
```

**Response** (200 OK): a `QueueListResponse` with the full queue:
```json
{
  "items": [ { "id": "…", "inputs": {…}, "reason": "qEI", "status": "pending", … } ],
  "n_pending": 2, "n_running": 0, "n_done": 0, "n_failed": 0
}
```

Returns 400 if no variables are defined.

#### List / Get Queue

```http
GET /sessions/{session_id}/experiments/queue
GET /sessions/{session_id}/experiments/queue?status=pending
GET /sessions/{session_id}/experiments/queue/{item_id}
```

**Purpose**: List all items (optionally filtered by `status`) or fetch one. The list endpoint is the poll/resync surface for a UI. `GET .../{item_id}` returns 404 for an unknown id.

#### Start / Complete / Fail an Item

```http
POST /sessions/{session_id}/experiments/queue/{item_id}/start
POST /sessions/{session_id}/experiments/queue/{item_id}/complete
POST /sessions/{session_id}/experiments/queue/{item_id}/fail
```

- **start**: `pending → running`.
- **complete**: `→ done`; adds the result to the dataset (sets `dataset_ref`). Body:
  ```json
  {
    "outputs": [0.87],
    "noise": [0.02],
    "expected_objective_label": {"Output": "carbonyl_1987"},
    "force": false
  }
  ```
  `outputs` must contain exactly one value (multi-objective completion is rejected with 400). If `expected_objective_label` is provided and does not match the session's current objective label, the request is refused with **409** unless `force: true` (see Objective Metadata).

  **Query parameter**: `auto_train` (bool, **default `false`**) — retrain the surrogate after the item lands, once the dataset has ≥5 rows. It is a query parameter, **not** a body field.

  ⚠️ **Note the default.** The deprecated `complete_staged_experiments` took `auto_train` too, and autonomous consumers typically passed `true`. A consumer migrating to the queue that forgets this parameter completes every item successfully and **never retrains** — the loop keeps suggesting from a stale surrogate, and nothing in any response says so.
- **fail**: `→ failed` with `{"error": "..."}`. Does not touch the dataset.

Status codes: **404** if the id is unknown (including if a concurrent consumer deleted it), **409** on an illegal transition or objective-label mismatch. Each returns the updated `QueueItem`.

#### Delete / Purge

```http
DELETE /sessions/{session_id}/experiments/queue/{item_id}
POST   /sessions/{session_id}/experiments/queue/purge
```

- **delete**: removes a single **pending** item (409 if it is running/done/failed; 404 if unknown). Returns the updated queue.
- **purge**: removes all terminal (`done`/`failed`) items. Returns `{"message": "...", "n_purged": N}`.

### Objective Metadata

An opaque, per-objective display label/unit that ALchemist stores and shows (parity-plot axes, etc.) but never parses. The consumer sets it (e.g. the meaning of the completed scalar); ALchemist treats it as a display string, keeping the toolkit domain-agnostic.

```http
GET /sessions/{session_id}/objective-metadata
PUT /sessions/{session_id}/objective-metadata
```

**PUT Request Body** (`{objective_name: {label, unit?}}`, merged per field):
```json
{"metadata": {"Output": {"label": "carbonyl_1987", "unit": "a.u."}}}
```

**Response** (both verbs): `{"metadata": {"Output": {"label": "carbonyl_1987", "unit": "a.u."}}}`.

Changes are recorded in the audit log. The `expected_objective_label`/`force` fields on queue completion (above) let a consumer guard against the objective's meaning changing mid-campaign.

### Deprecated: Staged Experiments (legacy)

The flat `staged` endpoints below are **deprecated** in favor of the Work Queue. They remain functional as a compatibility layer over the same underlying queue, with two behavior changes noted below. New consumers should use `.../experiments/queue`.

#### Stage Experiment (deprecated)

```http
POST /sessions/{session_id}/experiments/staged
```

**Request Body**:
```json
{"inputs": {"temperature": 375.2, "flow_rate": 5.8, "catalyst": "B"}, "reason": "qEI"}
```

**Response** (200 OK):
```json
{"message": "Experiment staged successfully", "n_staged": 1, "staged_inputs": {"temperature": 375.2, "flow_rate": 5.8, "catalyst": "B"}}
```

#### Stage Multiple Experiments (deprecated)

```http
POST /sessions/{session_id}/experiments/staged/batch
```

**Request Body**:
```json
{
  "experiments": [
    {"temperature": 375.2, "flow_rate": 5.8, "catalyst": "B"},
    {"temperature": 412.5, "flow_rate": 3.2, "catalyst": "A"}
  ],
  "reason": "qEI batch"
}
```

The `reason` is now stored **per item** on each queued item.

#### Get Staged Experiments (deprecated)

```http
GET /sessions/{session_id}/experiments/staged
```

Returns the **pending** items. Per-item reasons are now available in the `reasons` list (positionally aligned with `experiments`); the scalar `reason` remains the first item's value for backward compatibility.

**Response** (200 OK):
```json
{
  "experiments": [
    {"temperature": 375.2, "flow_rate": 5.8, "catalyst": "B"},
    {"temperature": 412.5, "flow_rate": 3.2, "catalyst": "A"}
  ],
  "n_staged": 2,
  "reason": "qEI",
  "reasons": ["qEI", "qEI"]
}
```

#### Clear Staged Experiments (deprecated)

```http
DELETE /sessions/{session_id}/experiments/staged
```

**Behavior change**: now clears **pending items only** (was: all items). This protects a live run's `running`/`done`/`failed` records. Use per-item `DELETE .../queue/{id}` or `POST .../queue/purge` for the rest.

```json
{"message": "Staged experiments cleared", "n_cleared": 2}
```

#### Complete Staged Experiments (deprecated)

```http
POST /sessions/{session_id}/experiments/staged/complete
```

**Purpose**: Complete all **pending** items in stage order, mapping `outputs` 1:1.

**Behavior change**: returns **409** if any item is currently `running` (the batch path would silently skip it). Terminal `done`/`failed` items do **not** block, so repeated stage→complete cycles keep working.

**Query Parameters**: `auto_train` (bool), `training_backend` (string), `training_kernel` (string).

**Request Body**:
```json
{"outputs": [0.87, 0.91], "noises": [0.02, 0.01], "iteration": 5, "reason": "qEI"}
```

Note: `iteration` is accepted for back-compat but ignored (iteration is auto-assigned).

**Response** (200 OK):
```json
{
  "message": "Staged experiments completed and added to dataset",
  "n_added": 2,
  "n_experiments": 23,
  "model_trained": true,
  "training_metrics": {"rmse": 0.042, "r2": 0.94, "backend": "sklearn"}
}
```

---


## Audit Log

Every session keeps an append-only audit log of decisions and configuration
changes. It is what a methods section is reconstructed from.

### Read the log

```http
GET /sessions/{session_id}/audit
GET /sessions/{session_id}/audit/export
```

- **audit**: the entries as JSON.
- **export**: the same trail rendered as markdown, for pasting into a
  manuscript or a report.

### Lock a decision

```http
POST /sessions/{session_id}/audit/lock
```

Body: `{"lock_type": "data" | "model" | "acquisition", "notes": "...", ...}`.
An `acquisition` lock additionally requires `strategy`, `parameters` and
`suggestions`.

⚠️ **`lock_type` is a closed enum.** This endpoint records *decisions*, not
arbitrary events — a value outside those three is rejected by validation
before it reaches any handler.

### Configuration changes

```http
GET /sessions/{session_id}/audit/config-changes
```

Returns `changes[]`, each entry carrying `timestamp`, `component`, `old`,
`new` and `iteration` — the provenance surface a monitoring consumer uses to
show *what changed and when* over the life of a campaign.

### Append an arbitrary event

```http
POST /sessions/{session_id}/audit/event
```

Body:

```json
{
  "entry_type": "cycle_started",
  "parameters": {"queue_item": "q1", "experiment": "exp-abc"},
  "notes": "controller"
}
```

Returns `{"entry": {...}}` — the entry as it was appended.

A thin wrapper over `AuditLog.log_event`. This is how an external consumer
puts its own run events on the shared timeline, so that a controller's
process log and ALchemist's inference trail can be stitched into one
history.

`entry_type` is an **opaque string**, deliberately *not* the lock endpoint's
closed `data|model|acquisition` enum — that enum records decisions, and
reusing it here is what previously made an audit mirror unbuildable.
ALchemist stores `entry_type` and never parses it. Constraints: non-empty,
≤ 64 characters; `notes` ≤ 2000 characters. An empty `entry_type` is
rejected with 422.

---


## Control Channel

One opaque coordination record per session, used by a human and by whichever
consumer is driving that session to tell each other what they want and what
is actually happening.

⚠️ **ALchemist stores and serves this record and never acts on it.** There is
no retry, no escalation, no timeout-triggered behaviour. A consumer polls the
record and decides for itself. Writing `requested: "pause"` does not stop
anything by itself and never can.

### Read the record

```http
GET /sessions/{session_id}/control
```

Returns all seven fields at the top level:

```json
{
  "requested": "run",
  "requested_at": null,
  "requested_by": null,
  "reported": "idle",
  "reported_at": null,
  "reported_by": null,
  "detail": null
}
```

- `requested` — `"run"` | `"pause"`. What a human is asking for.
- `reported` — `"idle"` | `"running"` | `"paused"` | `"failed"`. What the
  driving consumer says it is actually doing.
- `reported_at` doubles as a **heartbeat**: it moves on every report, so an
  observer can tell "quiet for 40 s" from "paused" — the difference between
  an unknown state and a known one.
- `requested_by` / `reported_by` are opaque display labels. **Not identity,
  not authorization.**

### Write one half of the record

```http
PUT /sessions/{session_id}/control
```

A body carries **exactly one half**. A human writes the requested half:

```json
{"requested": "pause", "requested_by": "caleb"}
```

The driving consumer writes the reported half:

```json
{"reported": "paused", "reported_by": "ctl@reactor", "detail": "held after q3"}
```

Both forms return the full `ControlResponse`.

The two halves have different writers and stay disjoint:

- A body carrying **both** `requested` and `reported` → **400**. Letting one
  writer set both halves would let a UI manufacture an acknowledgment it
  never received.
- A body carrying **neither** → **400**.
- A value outside the enums above → **422** (or 400 from the core validator).

This is what makes *"requested, but not yet acknowledged"* representable: a
UI reads `reported` for what is true and `requested` for what was asked, and
physically cannot report a state just because a button was clicked.

### Events and audit

Every accepted write broadcasts `control_changed` over the session
WebSocket, **heartbeats included** — a browser derives staleness from
`reported_at`, so a silent heartbeat would make a healthy consumer look
progressively deader.

Audit is deliberately **asymmetric** to that: an entry (`control_requested`
or `control_reported`) is written only when a value actually *changes*. At a
10 s heartbeat an 8-hour campaign is ~2900 reports; auditing each would bury
the entries that matter in the trail that `audit/export` renders for a
methods section.

---


## Constraints

### Input constraints

Linear constraints over the *input* variables, applied when generating
suggestions. Configured through the session API.

### ⛔ Outcome constraints exist in core but are not REST-exposed

`OptimizationSession.add_outcome_constraint(objective_name, bound_type, value)`
(`alchemist_core/session.py:416`) is real, and is wired into acquisition as
BoTorch constraint callables — this is genuine constrained Bayesian
optimization, not a stub.

Two limits an HTTP consumer must know:

1. **No endpoint sets one.** The API surface has no route to
   `add_outcome_constraint`, so a REST client cannot register an outcome
   constraint at all.
2. **The constrained quantity must be a modeled output column.** A constraint
   on something the model does not predict cannot be expressed.

Consumers needing "maximize A subject to B ≤ x" over HTTP must therefore
**encode the constraint into the single scalar they report** — for example
by reporting a fixed penalty value when the constraint is violated. That is
a real limitation, not a stylistic choice, and it is the gap that pushed the
AutoProc controller to a penalty-scalar objective.

---


## Models

Train surrogate models and make predictions.

### Train Model

```http
POST /sessions/{session_id}/model/train
```

**Request Body** (sklearn):
```json
{
  "backend": "sklearn",
  "kernel": "RBF",
  "kernel_params": {},
  "input_transform": "standard",
  "output_transform": "standard",
  "calibration_enabled": false
}
```

**Request Body** (BoTorch):
```json
{
  "backend": "botorch",
  "kernel": "Matern",
  "kernel_params": {
    "nu": 2.5
  },
  "input_transform": "normalize",
  "output_transform": "standardize"
}
```

**Backends**:
- `sklearn` - scikit-learn GPR (simple, fast)
- `botorch` - BoTorch/PyTorch GPR (advanced, qMC acquisition)

**Kernels**:
- `RBF` - Radial Basis Function
- `Matern` - Matérn kernel (nu: 0.5, 1.5, 2.5, inf)
- `RationalQuadratic` - Rational Quadratic

**Response** (200 OK):
```json
{
  "success": true,
  "backend": "sklearn",
  "kernel": "RBF",
  "hyperparameters": {
    "length_scale": 1.23,
    "noise_variance": 0.01
  },
  "metrics": {
    "rmse": 0.045,
    "mae": 0.032,
    "r2": 0.92,
    "mape": 3.8
  },
  "message": "Model trained successfully"
}
```

### Get Model Info

```http
GET /sessions/{session_id}/model
```

**Response** (200 OK):
```json
{
  "backend": "sklearn",
  "hyperparameters": {
    "length_scale": 1.23,
    "noise_variance": 0.01
  },
  "metrics": {
    "rmse": 0.045,
    "r2": 0.92
  },
  "is_trained": true
}
```

### Make Predictions

```http
POST /sessions/{session_id}/model/predict
```

**Request Body**:
```json
{
  "inputs": [
    {"temperature": 375, "flow_rate": 5.0, "catalyst": "A"},
    {"temperature": 425, "flow_rate": 6.5, "catalyst": "B"}
  ]
}
```

**Response** (200 OK):
```json
{
  "predictions": [
    {
      "inputs": {"temperature": 375, "flow_rate": 5.0, "catalyst": "A"},
      "prediction": 0.87,
      "uncertainty": 0.05
    },
    {
      "inputs": {"temperature": 425, "flow_rate": 6.5, "catalyst": "B"},
      "prediction": 0.91,
      "uncertainty": 0.03
    }
  ],
  "n_predictions": 2
}
```

---

## Acquisition

Generate next experiment suggestions using acquisition functions.

### Get Suggestions

```http
POST /sessions/{session_id}/acquisition/suggest
```

**Request Body**:
```json
{
  "strategy": "qEI",
  "goal": "maximize",
  "n_suggestions": 3,
  "xi": 0.01,
  "kappa": 2.0
}
```

**Strategies**:
- `EI` / `qEI` - Expected Improvement (recommended)
- `PI` / `qPI` - Probability of Improvement
- `UCB` / `qUCB` - Upper Confidence Bound
- `qNIPV` - Negative Integrated Posterior Variance (exploration)

**Parameters**:
- `xi` (float, default: 0.01) - Exploration-exploitation trade-off for EI/PI
- `kappa` (float, default: 2.0) - Exploration-exploitation trade-off for UCB
- `n_suggestions` (int, default: 1) - Number of points to suggest

**Response** (200 OK):
```json
{
  "suggestions": [
    {"temperature": 385, "flow_rate": 4.2, "catalyst": "A"},
    {"temperature": 410, "flow_rate": 7.5, "catalyst": "C"},
    {"temperature": 365, "flow_rate": 3.8, "catalyst": "B"}
  ],
  "n_suggestions": 3
}
```

### Find Optimum

```http
POST /sessions/{session_id}/acquisition/find-optimum
```

**Request Body**:
```json
{
  "goal": "maximize"
}
```

**Response** (200 OK):
```json
{
  "optimum": {
    "temperature": 425,
    "flow_rate": 6.8,
    "catalyst": "B"
  },
  "predicted_value": 0.94,
  "predicted_std": 0.02,
  "goal": "maximize"
}
```

---

## Visualization Endpoints

### Get Contour Plot Data

```http
POST /sessions/{session_id}/visualizations/contour
```

**Request Body**:
```json
{
  "x_var": "temperature",
  "y_var": "flow_rate",
  "fixed_values": {
    "catalyst": "A"
  },
  "grid_resolution": 50,
  "include_experiments": true,
  "include_suggestions": true
}
```

**Response** (200 OK): Grid data for contour plotting

### Get Parity Plot Data

```http
GET /sessions/{session_id}/visualizations/parity?calibrated=false
```

**Response** (200 OK):
```json
{
  "y_true": [0.85, 0.92, 0.88],
  "y_pred": [0.84, 0.93, 0.87],
  "y_std": [0.05, 0.03, 0.04],
  "metrics": {
    "rmse": 0.045,
    "mae": 0.032,
    "r2": 0.92,
    "mape": 3.8
  },
  "bounds": [0.6, 1.0],
  "calibrated": false
}
```

### Get Metrics Over Time

```http
GET /sessions/{session_id}/visualizations/metrics?calibrated=false
```

**Response** (200 OK):
```json
{
  "training_sizes": [5, 6, 7, 8, 9, 10],
  "rmse": [0.12, 0.09, 0.07, 0.06, 0.05, 0.045],
  "mae": [0.09, 0.07, 0.05, 0.04, 0.03, 0.032],
  "r2": [0.75, 0.82, 0.87, 0.90, 0.91, 0.92],
  "mape": [8.5, 6.2, 5.1, 4.5, 4.0, 3.8]
}
```

### Get Q-Q Plot Data

```http
GET /sessions/{session_id}/visualizations/qq?calibrated=false
```

**Response** (200 OK): Data for Q-Q plot of standardized residuals

### Get Calibration Curve Data

```http
GET /sessions/{session_id}/visualizations/calibration?calibrated=false
```

**Response** (200 OK): Nominal vs empirical coverage data

### Get Model Hyperparameters

```http
GET /sessions/{session_id}/visualizations/hyperparameters
```

**Response** (200 OK):
```json
{
  "hyperparameters": {
    "length_scale": 1.23,
    "noise_variance": 0.01
  },
  "backend": "sklearn",
  "kernel": "RBF",
  "input_transform": "standard",
  "output_transform": "standard",
  "calibration_enabled": false,
  "calibration_factor": null
}
```

---

## Error Responses

All endpoints return consistent error responses:

**400 Bad Request**:
```json
{
  "detail": "Invalid request: Missing required field 'output'"
}
```

**404 Not Found**:
```json
{
  "detail": "Session abc-123 not found or expired"
}
```

**500 Internal Server Error**:
```json
{
  "detail": "Model training failed: Insufficient data"
}
```

---

## Rate Limiting

Currently no rate limiting. Consider adding for production deployments.

---

## Authentication

Currently no authentication. Add JWT or API keys for production if needed.

---

## CORS

CORS is enabled for:
- `http://localhost:3000` (Create React App)
- `http://localhost:5173` (Vite dev server)
- `http://localhost:5174` (Vite dev server alternate)

Configure additional origins in `api/main.py` if needed.

---

## Status Codes Summary

| Code | Meaning |
|------|---------|
| 200 | OK - Request successful |
| 201 | Created - Resource created successfully |
| 204 | No Content - Deletion successful |
| 400 | Bad Request - Invalid input |
| 404 | Not Found - Session/resource not found |
| 500 | Internal Server Error - Server-side error |

---

## Changelog

### v0.2.1 (November 18, 2025)
- Added `/initial-design` endpoint for DoE generation
- Added `/state` endpoint for lightweight monitoring
- Added `auto_train` parameter to experiment endpoints
- Enhanced documentation with autonomous workflow examples

### v0.2.0 (October 31, 2025)
- Initial FastAPI implementation
- 19 endpoints across 5 routers
- Full Session API integration
- Auto-generated OpenAPI documentation
