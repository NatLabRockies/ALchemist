# ALchemist REST API: routes (0.4.0)

This file is a route index. The field-level reference is generated from the running server and lives at:

- **Swagger UI**: http://localhost:8000/api/docs (ReDoc: http://localhost:8000/api/redoc; schema: http://localhost:8000/api/openapi.json)
- **Documentation site**: [REST API](https://natlabrockies.github.io/ALchemist/reference/rest-api/), built from `docs/reference/openapi.json` (refresh it with `python docs/scripts/export_openapi.py`)
- **How to use the routes**: [The autonomous loop](https://natlabrockies.github.io/ALchemist/automation/loop/), [Staging and the work queue](https://natlabrockies.github.io/ALchemist/automation/queue/), [Provenance](https://natlabrockies.github.io/ALchemist/automation/provenance/) and the "REST" tabs of the guide pages.

Base URL: `http://localhost:8000/api/v1`. Every route below is under it unless it starts with `/`. Session routes take the session ID in the path; a session the server cannot read answers 503, an unknown one 404.

## What changed in 0.4.0

- **Autosave.** Every successful `POST`, `PUT`, `PATCH` or `DELETE` under `/sessions/{session_id}/` saves the session to `cache/sessions`. The response header `X-ALchemist-Saved` is `true` or `false`; a failed write is logged and does not fail the request. Sessions load lazily, the first time their ID is used, and a reloaded or imported session rebuilds its model from its last training call.
- **Input constraints.** `GET`/`POST /sessions/{id}/constraints` and `DELETE .../constraints/{constraint_name}`. `GET /variables/export?include_constraints=true` returns the `{variables, constraints}` document, and `POST /variables/load` accepts either that document or a bare variable list; either is validated whole.
- **Designs.** `POST /initial-design` and `POST /optimal-design` responses carry a `feasibility` block (`null` without constraints); `POST /initial-design` accepts `allow_infeasible`. A constrained classical design that can no longer estimate its model is a 400 with `error_type: DesignNotEstimableError`; an empty feasible region is `InfeasibleRegionError`. New `POST /split-plot-design`.
- **Variables.** `role` (`controllable` | `observed`) and `changes` (`easy` | `hard` | `very_hard`). Bounds must be finite. Adding a variable after results are recorded is refused (400) unless its column is already in the data.
- **Work queue and provenance.** `POST /experiments/queue` stages items (validated before any is staged), `.../queue/{item_id}/start`, `/complete` (honours `iteration`) and `/fail`; `POST /experiments/record-batch` completes staged items and adds new rows in one request under one iteration; `GET /experiments/provenance` and `/{provenance_id}` return the suggested-vs-actual records. Batch and CSV-upload rows get provenance records too. The flat `staged/*` routes remain as a deprecated layer over the queue.
- **Acquisition.** `POST /acquisition/suggest` takes `hold`, `observed` and returns `iteration` and `diagnostics`; `xi` and `kappa` have no defaults (the core's apply when omitted), and `kappa` is BoTorch's `beta` for UCB and qUCB. A one-point strategy with `n_suggestions > 1` is a 400. `qEI` and `qUCB` honour `goal: minimize`. `POST /acquisition/find-optimum` takes `observed`, respects input constraints, and is a 400 on a multi-objective session.
- **Models.** `POST /model/train` takes `optimizer` (scikit-learn) and defaults BoTorch to `normalize`/`standardize`; `GET /model` returns `training_config`, the saved arguments of the last training call. The parity, Q-Q and calibration routes take `?target=` on a multi-objective model.
- **Session info.** `GET /sessions/{id}` includes `metadata` (name, description, tags, author, last_modified) and `target_columns`; `author` is accepted by `PATCH /metadata` and `POST /sessions`. `GET /sessions/{id}/state` reports `model_backend`, `last_suggestion`, `last_acquisition_value` and `last_goal`.
- **Contour.** `POST /visualizations/contour` responses carry `output_range`.
- **Errors.** An unknown `/api/...` path is a 404. The recovery routes accept only a plain session ID.

## Routes


### Sessions, recovery, audit log, lock and control

| Method | Path | Summary |
|---|---|---|
| `DELETE` | `/recovery/cleanup` | Cleanup Old Recoveries |
| `GET` | `/recovery/list` | List Recovery Sessions |
| `POST` | `/recovery/{session_id}/restore` | Restore Recovery Session |
| `POST` | `/sessions` | Create Session |
| `POST` | `/sessions/import` | Import Session |
| `POST` | `/sessions/upload` | Upload Session |
| `DELETE` | `/sessions/{session_id}` | Delete Session |
| `GET` | `/sessions/{session_id}` | Get Session Info |
| `GET` | `/sessions/{session_id}/audit` | Get Audit Log |
| `POST` | `/sessions/{session_id}/audit/event` | Post Audit Event |
| `GET` | `/sessions/{session_id}/audit/export` | Export Audit Markdown |
| `POST` | `/sessions/{session_id}/audit/lock` | Lock Decision |
| `DELETE` | `/sessions/{session_id}/backup` | Clear Recovery Backup |
| `POST` | `/sessions/{session_id}/backup` | Create Recovery Backup |
| `GET` | `/sessions/{session_id}/control` | Get Control |
| `PUT` | `/sessions/{session_id}/control` | Update Control |
| `GET` | `/sessions/{session_id}/download` | Download Session |
| `GET` | `/sessions/{session_id}/export` | Export Session |
| `DELETE` | `/sessions/{session_id}/lock` | Unlock Session |
| `GET` | `/sessions/{session_id}/lock` | Get Lock Status |
| `POST` | `/sessions/{session_id}/lock` | Lock Session |
| `GET` | `/sessions/{session_id}/metadata` | Get Metadata |
| `PATCH` | `/sessions/{session_id}/metadata` | Update Metadata |
| `POST` | `/sessions/{session_id}/save` | Save Session Server Side |
| `GET` | `/sessions/{session_id}/state` | Get Session State |

### Variables and constraints

| Method | Path | Summary |
|---|---|---|
| `GET` | `/sessions/{session_id}/constraints` | List Constraints |
| `POST` | `/sessions/{session_id}/constraints` | Add Constraint |
| `DELETE` | `/sessions/{session_id}/constraints/{constraint_name}` | Delete Constraint |
| `GET` | `/sessions/{session_id}/variables` | List Variables |
| `POST` | `/sessions/{session_id}/variables` | Add Variable |
| `GET` | `/sessions/{session_id}/variables/export` | Export Variables To Json |
| `POST` | `/sessions/{session_id}/variables/load` | Load Variables From File |
| `DELETE` | `/sessions/{session_id}/variables/{variable_name}` | Delete Variable |
| `PUT` | `/sessions/{session_id}/variables/{variable_name}` | Update Variable |

### Experiments, designs, work queue and provenance

| Method | Path | Summary |
|---|---|---|
| `GET` | `/sessions/{session_id}/audit/config-changes` | Get Config Changes |
| `GET` | `/sessions/{session_id}/experiments` | List Experiments |
| `POST` | `/sessions/{session_id}/experiments` | Add Experiment |
| `POST` | `/sessions/{session_id}/experiments/batch` | Add Experiments Batch |
| `POST` | `/sessions/{session_id}/experiments/preview` | Preview Csv Columns |
| `GET` | `/sessions/{session_id}/experiments/provenance` | List Provenance |
| `GET` | `/sessions/{session_id}/experiments/provenance/{provenance_id}` | Get Provenance Record |
| `GET` | `/sessions/{session_id}/experiments/queue` | List Queue |
| `POST` | `/sessions/{session_id}/experiments/queue` | Stage Queue Items |
| `POST` | `/sessions/{session_id}/experiments/queue/purge` | Purge Queue |
| `DELETE` | `/sessions/{session_id}/experiments/queue/{item_id}` | Delete Queue Item |
| `GET` | `/sessions/{session_id}/experiments/queue/{item_id}` | Get Queue Item |
| `POST` | `/sessions/{session_id}/experiments/queue/{item_id}/complete` | Complete Queue Item |
| `POST` | `/sessions/{session_id}/experiments/queue/{item_id}/fail` | Fail Queue Item |
| `POST` | `/sessions/{session_id}/experiments/queue/{item_id}/start` | Start Queue Item |
| `POST` | `/sessions/{session_id}/experiments/record-batch` | Record Results Batch |
| `DELETE` | `/sessions/{session_id}/experiments/staged` | Clear Staged Experiments |
| `GET` | `/sessions/{session_id}/experiments/staged` | Get Staged Experiments |
| `POST` | `/sessions/{session_id}/experiments/staged` | Stage Experiment |
| `POST` | `/sessions/{session_id}/experiments/staged/batch` | Stage Experiments Batch |
| `POST` | `/sessions/{session_id}/experiments/staged/complete` | Complete Staged Experiments |
| `GET` | `/sessions/{session_id}/experiments/summary` | Get Experiments Summary |
| `POST` | `/sessions/{session_id}/experiments/upload` | Upload Experiments |
| `POST` | `/sessions/{session_id}/initial-design` | Generate Initial Design |
| `GET` | `/sessions/{session_id}/objective-metadata` | Get Objective Metadata |
| `PUT` | `/sessions/{session_id}/objective-metadata` | Set Objective Metadata |
| `POST` | `/sessions/{session_id}/optimal-design` | Generate Optimal Design |
| `POST` | `/sessions/{session_id}/optimal-design/info` | Get Optimal Design Info |
| `POST` | `/sessions/{session_id}/split-plot-design` | Generate Split Plot Design |

### Models

| Method | Path | Summary |
|---|---|---|
| `GET` | `/sessions/{session_id}/model` | Get Model Info |
| `POST` | `/sessions/{session_id}/model/predict` | Predict |
| `POST` | `/sessions/{session_id}/model/train` | Train Model |

### Acquisition

| Method | Path | Summary |
|---|---|---|
| `POST` | `/sessions/{session_id}/acquisition/find-optimum` | Find Model Optimum |
| `POST` | `/sessions/{session_id}/acquisition/suggest` | Suggest Next Experiments |

### Visualizations

| Method | Path | Summary |
|---|---|---|
| `GET` | `/sessions/{session_id}/visualizations/calibration-curve` | Get Calibration Curve Data |
| `POST` | `/sessions/{session_id}/visualizations/contour` | Get Contour Data |
| `GET` | `/sessions/{session_id}/visualizations/hyperparameters` | Get Hyperparameters |
| `GET` | `/sessions/{session_id}/visualizations/metrics` | Get Metrics Data |
| `GET` | `/sessions/{session_id}/visualizations/parity` | Get Parity Data |
| `GET` | `/sessions/{session_id}/visualizations/qq-plot` | Get Qq Plot Data |
| `POST` | `/sessions/{session_id}/visualizations/surface` | Get Surface Data |

### AI effect suggestion

| Method | Path | Summary |
|---|---|---|
| `GET` | `/llm/config` | Get Config |
| `POST` | `/llm/config` | Update Config |
| `GET` | `/llm/ollama/models` | List Ollama Models |
| `POST` | `/llm/suggest-effects/{session_id}` | Suggest Effects |

### Server

| Method | Path | Summary |
|---|---|---|
| `GET` | `/` | Root |
| `GET` | `/health` | Health Check |

## Errors

Validation errors are 422 with the failing fields. Domain errors are 400 with `detail` and `error_type` (`ValueError`, `DesignNotEstimableError`, `InfeasibleRegionError`). A write the server could not complete is 500; a session file it cannot read is 503. There is no authentication and no rate limiting: the server listens on every interface, so restrict port 8000 on a shared network (see the Install page).
