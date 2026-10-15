# The autonomous loop

A controller drives ALchemist over REST: it asks for points, runs them on the equipment, and hands the results back. This page lists the calls in order and says which side owns each decision. [Example controller](controller.md) is the same loop as running code.

## Who owns what

ALchemist is the engine and the record. The controller is the only thing that touches the equipment.

| ALchemist | The controller |
|---|---|
| Holds the search space, the results, the model and the audit log. | Drives the equipment and decides when anything runs. |
| Proposes points: an initial design, then suggestions from the model. | Decides whether a point is safe to run, against limits of its own. |
| Tracks each point as a queue item, and keeps the suggested and the actual conditions together as provenance. | Reports the conditions that were really run and the measured objective. |
| Stores a pause request and the controller's reported state. | Reads the request, decides what to do, and reports what it is doing. |
| Retrains the model when a request says to. | Chooses the strategy and batch size, and decides when to stop. |

ALchemist never starts a cycle and never stops one. The session lock and the control record, described in [Monitoring and remote control](../guide/more/monitoring.md), are signals for a person watching the web app, not guards.

## The cycle

Paths are under `/api/v1`, and `{session_id}` is the ID the first call returns. A controller can also use a session that was made in the web app, given its ID.

### 1. Create the session and its variables

```http
POST /api/v1/sessions
```

Each variable is a separate call; [Define variables](../guide/variables.md) lists the four types.

```http
POST /api/v1/sessions/{session_id}/variables
```

```json
{"name": "temperature", "type": "real", "min": 200.0, "max": 300.0, "unit": "°C"}
```

### 2. Take the lock and report

Lock the session so the web app shows the monitor, then report the controller's state about every 10 seconds while it is alive, and read the control record between runs for a pause request. [Monitoring](../guide/more/monitoring.md#3-ask-the-controller-to-pause) has the calls.

### 3. Stage the initial design

The initial design needs no model.

```http
POST /api/v1/sessions/{session_id}/initial-design
Content-Type: application/json

{"method": "lhs", "n_points": 6, "random_seed": 7}
```

Generating a design adds nothing to the session. Staging it puts each point in the work queue, where it has an ID, a status and a reason.

```http
POST /api/v1/sessions/{session_id}/experiments/queue
Content-Type: application/json

{"items": [{"inputs": {"temperature": 271.2, "flow": 3.4}, "reason": "Initial design (LHS)"}]}
```

### 4. Run and record

For each queue item, optionally mark it as started so the monitor shows it running, then run it on the equipment.

```http
POST /api/v1/sessions/{session_id}/experiments/queue/{item_id}/start
```

When the results are in, record them in one request. Each entry names its queue item and carries the conditions that were actually run, which are rarely the suggested ones to the last digit. Give every point of a round the same `iteration`: 0 for the initial design, and for later rounds the `iteration` that `suggest` returned.

```http
POST /api/v1/sessions/{session_id}/experiments/record-batch
Content-Type: application/json

{"entries": [{"item_id": "3f1c0b52", "inputs": {"temperature": 271.0, "flow": 3.4}, "output": 0.62, "iteration": 0}]}
```

A point that failed on the equipment is marked with `POST .../experiments/queue/{item_id}/fail`, which adds nothing to the data. [Staging and the work queue](queue.md) covers both calls.

### 5. Train once, then suggest

Training sets the model the loop will keep using. It needs at least 5 results.

```http
POST /api/v1/sessions/{session_id}/model/train
Content-Type: application/json

{"backend": "botorch", "kernel": "Matern"}
```

Then ask for the next round and stage it as in step 3.

```http
POST /api/v1/sessions/{session_id}/acquisition/suggest
Content-Type: application/json

{"strategy": "qEI", "goal": "maximize", "n_suggestions": 2}
```

The response has `suggestions` and the `iteration` to record them under. Run the points and record them as in step 4, now with retraining on:

```http
POST /api/v1/sessions/{session_id}/experiments/record-batch?auto_train=true
```

Then suggest again. When the controller stops, it reports `idle` and releases the lock with `DELETE .../lock?lock_token=<token>`.

## Batches and strategies

A round of more than one point needs a batch strategy: `qEI`, `qUCB` or `qIPV`. The single-point strategies (`EI`, `PI`, `UCB` and their log forms) refuse `n_suggestions` above 1 with a 400 that names the batch strategies, and the scikit-learn backend suggests one point at a time. A batch is 1 to 10 points. [Acquisition functions](../concepts/acquisition.md#batches-by-monte-carlo) explains how the batch is scored.

!!! warning "Known issue"
    T7: `qEI` and `qUCB` always maximize, so a controller minimizing with a batch should negate its response. Details are in [Acquisition functions](../concepts/acquisition.md#batches-by-monte-carlo).

## Retraining

Nothing retrains unless a request asks. `record-batch` with `auto_train=true` retrains once, after the last row is in, and not at all while the session has fewer than 5 results. The retrain repeats the session's most recent training call: the same kernel, kernel parameters, transforms and calibration setting. A session that has never trained falls back to scikit-learn with an RBF kernel, which is why the loop trains once itself first. The `training_backend` and `training_kernel` query parameters change the backend or kernel for that retrain; a different backend starts from its own defaults, and later retrains repeat the result.

A retrain that fails does not fail the request. The results are recorded, `model_trained` comes back `false`, and the error goes to the server log. A controller that suggests without checking `model_trained` suggests from the old model, so check it whenever the session has 5 or more results. `POST /experiments` and `POST /experiments/batch` take the same query parameters.

## Restarts and resuming

All of the loop's state is in the session: the data, the queue with each item's status, the provenance records, the model's training call and the control record. A controller that restarts reads `GET /api/v1/sessions/{session_id}/experiments/queue` and finds what it left: `pending` items it had not run, `running` items it must decide about (run again, or mark failed), and the history. It remembers nothing itself.

That state lives in the server's memory. The server writes a session to disk when it is created, imported, locked or unlocked, and on `POST /api/v1/sessions/{session_id}/save`, and it loads the saved sessions when it starts, so changes since the last write are lost if the server stops. A campaign that must survive that saves after each round. Importing a session file with `POST /api/v1/sessions/import` makes a new session with a new ID, and the model is retrained from the saved settings.

!!! warning "Known issue"
    A16: import and the reload at server start-up always retrain the model, and REST has no way to skip it.

## Gaps a controller should cover

!!! warning "Known issue"
    A4: the queue and the experiment endpoints accept inputs outside the variables' bounds and keys that are not variables; an unknown key becomes a column in the data. Check every point against the equipment's limits before running it, and send only the variables the session defines.

!!! warning "Known issue"
    W34: the web app's Stage Suggestions writes a second acquisition entry to the audit log for a batch that `suggest` already logged. A controller should not post to `/audit/lock` after `suggest`, or its log shows two.

!!! info "How ALchemist does it"
    `record_results_batch` in `api/routers/experiments.py` validates every entry before it writes any and trains at most once. The retrain's settings come from `OptimizationSession.get_last_training_config`, which `train_model` updates on every success and the session file stores.
