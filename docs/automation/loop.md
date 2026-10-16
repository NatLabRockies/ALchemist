# The autonomous loop

A controller drives ALchemist over REST: it asks for points, runs them on the equipment, and hands the results back. This page lists the calls in order and says which side owns each decision. [Example controller](controller.md) is the same loop as running code.

## Who owns what

| ALchemist | The controller |
|---|---|
| Holds the search space, the results, the model and the audit log. | Drives the equipment and decides when anything runs. |
| Proposes points: an initial design, then suggestions from the model. | Decides whether a point is safe to run, against limits of its own. |
| Tracks each point as a queue item, and keeps the suggested and the actual conditions together as provenance. | Reports the conditions that were really run and the measured objective. |
| Stores a pause request and the controller's reported state. | Reads the request, decides what to do, and reports what it is doing. |
| Retrains the model when a request says to. | Chooses the strategy and batch size, and decides when to stop. |

ALchemist never starts a cycle and never stops one. The session lock and the control record, described in [Monitoring and remote control](../guide/more/monitoring.md), are signals for a person watching the web app, not guards.

## The cycle

Paths are under `/api/v1`, and `{session_id}` is the ID the first call returns. A session made in the web app works too, given its ID.

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

Lock the session so the web app shows the monitor. Then, while the controller is alive, report its state about every 10 seconds, and between runs read the control record for a pause request. [Monitoring](../guide/more/monitoring.md#3-ask-the-controller-to-pause) has the calls.

### 3. Stage the initial design

The initial design needs no model.

```http
POST /api/v1/sessions/{session_id}/initial-design
Content-Type: application/json

{"method": "lhs", "n_points": 6, "random_seed": 7}
```

Generating a design adds nothing to the session. Staging puts each point in the work queue, with an ID, a status and a reason.

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

Training sets the model the loop will keep using. BoTorch refuses fewer than 3 results; 5 or more is a sensible minimum, and 5 is what a retrain needs.

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

A round of 2 to 10 points needs `qEI`, `qUCB` or `qIPV`; the single-point strategies and the scikit-learn backend refuse `n_suggestions` above 1 with a 400 ([how a batch is scored](../concepts/acquisition.md#batches-by-monte-carlo)).

## Retraining

Nothing retrains unless a request asks. `record-batch` with `auto_train=true` retrains once, after the last row is in, and not at all while the session has fewer than 5 results. The retrain repeats the session's most recent training call: the same kernel, kernel parameters, transforms and calibration setting. A session that has never trained falls back to scikit-learn with an RBF kernel, which is why the loop trains once itself first. The `training_backend` and `training_kernel` query parameters change the backend or kernel for that retrain. A different backend starts from its own defaults, and later retrains repeat the new settings.

A retrain that fails does not fail the request. The results are recorded, `model_trained` comes back `false`, and the error goes to the server log. A controller that suggests without checking `model_trained` suggests from the old model, so check it whenever the session has 5 or more results.

## Restarts and resuming

All of the loop's state is in the session: the data, the queue with each item's status, the provenance records, the model's training call and the control record. A controller that restarts needs no memory of its own: it reads `GET /api/v1/sessions/{session_id}/experiments/queue` and finds `pending` items it had not run, `running` items it must decide about (run again, or mark failed), and the history.

The server keeps the session in memory and writes it to disk, in `cache/sessions`, after every request that changes it. Each successful response to one of those requests (status below 400) carries an `X-ALchemist-Saved` header, `true` when the write succeeded and `false` when it did not. A failed write does not turn the response into an error: the change is in the running session, and the next change saves it again. A controller that must not lose a result checks the header after recording one, or calls `POST /api/v1/sessions/{session_id}/save`, which answers 500 if the write failed. A request that changes the session and then fails is not saved: a variables load that adds `x2` and then answers 400 for a duplicate leaves `x2` in the running session, and the file gets it with the next successful change. The server writes each file in one step, so a crash during a save leaves the previous file whole.

After a restart the server loads a session the first time its ID is used, so a controller keeps using the same `session_id`. A session's file stays until you delete the session. The file holds the training call but not the fitted model, so the server rebuilds the model by repeating the session's last training call. If that retrain fails, the error goes to the server log and the session loads without a model; train it again with `POST /api/v1/sessions/{session_id}/model/train` before the next suggestion. The training call stays in the file, so the next reload tries it again. If a session's file exists but cannot be read, requests for that session answer 503, not 404, so a controller does not take the session for deleted; `DELETE /api/v1/sessions/{session_id}` still removes it. The web app also posts a recovery backup every 30 seconds while a browser has the session open; it can be restored under a new ID.

## Gaps a controller should cover

!!! warning "Known issue"
    A4: the queue and the experiment endpoints accept inputs outside the variables' bounds and keys that are not variables; an unknown key becomes a column in the data. Check every point against the equipment's limits before running it, and send only the variables the session defines.

!!! warning "Known issue"
    A23: staging does not check that every variable is present or that a category exists; a bad point is refused only when its result is recorded.

!!! info "How ALchemist does it"
    `record_results_batch` in `api/routers/experiments.py` validates every entry before it writes any and trains at most once. The retrain's settings come from `OptimizationSession.get_last_training_config`, which `train_model` updates on every success and the session file stores.
