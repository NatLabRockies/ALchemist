# Staging and the work queue

Every point a controller is about to run is a queue item. It is staged, optionally started, and then completed with the conditions that were really run, or marked failed. This page covers the item's states, the calls that move it, and the older `staged` routes that show the same list.

## Item states

| Status | Meaning | Leaves it by |
|---|---|---|
| `pending` | Staged and waiting. | `start`, a recorded result, `fail`, or `DELETE`. |
| `running` | The controller has said it is running this point. | A recorded result or `fail`. |
| `done` | A result is recorded and in the data. | `purge`. |
| `failed` | The controller reported an error; `error` holds the text. Nothing was added to the data. | `purge`. |

`start` is optional: a result can complete a `pending` item directly, but calling it lets the monitor show which point is on the equipment. A finished item cannot change state again, so stage a failed point as a new item to try it again. A refused change is a 409 with the reason, and an unknown ID is a 404.

Each item, in the list and on `GET /api/v1/sessions/{session_id}/experiments/queue/{item_id}`, has these fields.

| Field | Content |
|---|---|
| `id` | The item's ID, which becomes the `ProvenanceId` of its data row and links it to its [provenance](provenance.md) record. |
| `inputs` | The suggested conditions, one value per variable. |
| `reason` | Why the point was staged, for example `qEI`. It is written to the row's `Reason` column. |
| `status` | One of the four above. |
| `output`, `noise` | The recorded values, once `done`. |
| `error` | The failure text, once `failed`. |
| `dataset_ref` | The row's position when it was added. It can go stale; find a row by `ProvenanceId`. |
| `staged_at`, `started_at`, `completed_at` | ISO timestamps; `started_at` is empty for an item that was never started. |
| `plot` | `WholePlot`, `Subplot` and `RunOrder` for a split-plot run, or empty. |

## Stage points

```http
POST /api/v1/sessions/{session_id}/experiments/queue
Content-Type: application/json

{
  "items": [
    {"inputs": {"temperature": 271.2, "flow": 3.4}, "reason": "qEI"},
    {"inputs": {"temperature": 245.9, "flow": 6.1}, "reason": "qEI"}
  ]
}
```

The response is the whole queue: `items` with the new ones last, and the counts `n_pending`, `n_running`, `n_done` and `n_failed`. `GET .../experiments/queue` returns the same; `?status=running` filters the items and leaves the counts alone.

Rows from a split-plot design carry `WholePlot`, `Subplot` and `RunOrder`. Leave them in `inputs` and the queue moves them into the item's `plot`; a recorded result copies them to the data row, and they are never model inputs. A key that starts with an underscore is dropped, and `_reason` is used when `reason` is absent. The design's own `Reason` column is not recognised: it would stay among the `inputs`. Send it as `reason`.

!!! warning "Known issue"
    A4: staging accepts inputs outside the bounds, a missing variable, a category that does not exist, and keys that are not variables. The missing variable and the category are rejected when the result is recorded, but nothing checks bounds, and an unknown key becomes a column in the data. Validate points before staging them.

## The older staged routes

Before the queue, a session had a flat list of staged points. Those routes remain, marked deprecated in the schema, and the web app still uses two of them. They show only `pending` items.

```http
GET /api/v1/sessions/{session_id}/experiments/staged
```

| Field | Content |
|---|---|
| `experiments` | One row per pending item, in staging order: each variable by name, plus `WholePlot`, `Subplot` and `RunOrder` where the item has them. |
| `reasons`, `ids` | The reason and the queue ID of each row, in the same order. |
| `reason` | The first row's reason, kept for older clients. |
| `n_staged` | The number of rows. |

| Route | What it does |
|---|---|
| `POST .../experiments/staged/batch` | Stages `experiments` with one `reason`, or `_reason` per row, and returns the new `ids`. It refuses a row that misses a variable. |
| `DELETE .../experiments/staged` | Removes every `pending` item. A `running` item stays. |
| `POST .../experiments/staged/complete` | Pairs `outputs` with the pending items in order, with the suggested conditions as the actual ones. It is a 409 while any item is `running`. |

!!! warning "Known issue"
    A3: `staged/complete` ignores `iteration`, so the rows get the next free iteration numbers, one each. `iteration` in the body of `queue/{item_id}/complete` is ignored in the same way. Use `record-batch` when the iteration matters.

## Record results

`record-batch` records any number of results at once. Each entry is a staged item, named by `item_id`, or a point that was never staged.

```http
POST /api/v1/sessions/{session_id}/experiments/record-batch?auto_train=true
Content-Type: application/json

{
  "entries": [
    {"item_id": "3f1c0b52", "inputs": {"temperature": 271.0, "flow": 3.4}, "output": 0.62, "iteration": 3},
    {"inputs": {"temperature": 250.0, "flow": 5.0}, "output": 0.80, "reason": "Operator check"}
  ]
}
```

| Entry field | Content |
|---|---|
| `item_id` | The queue item this result completes. Leave it out for a point that was not staged; it is added as a direct entry. |
| `inputs` | The conditions actually run, with every variable. The suggested ones stay on the item and in provenance. |
| `output` | The measured objective, a finite number. |
| `noise` | Optional measurement uncertainty, not negative. |
| `iteration` | Optional. Give a round the same value; without it each row takes the next free number. |
| `reason` | Optional. A staged item keeps its own reason when this is empty. |

Every entry is checked before any is written. A bad number, a missing variable or an unknown category is a 400; an unknown item, an item already `done` or `failed`, or an item named twice is a 409. Either way nothing is recorded and the response lists the problems. On success it has `n_added`, `n_experiments`, `completed_item_ids`, `model_trained` and `training_metrics`. With `auto_train=true` the model is retrained once at the end; [The autonomous loop](loop.md#retraining) says with what.

To record one item, `POST .../queue/{item_id}/complete` takes `outputs` (a list of exactly one number), optional `noise` (a list of one), `actual_inputs` and an `auto_train` query parameter. `expected_objective_label` guards against recording for the wrong objective; a mismatch is a 409 unless `force` is true.

```http
POST /api/v1/sessions/{session_id}/experiments/queue/{item_id}/fail
Content-Type: application/json

{"error": "pump stalled"}
```

The queue records one objective value per item. `queue/{item_id}/complete` refuses a second value with a 400, and `record-batch` takes a single `output`, which on a multi-objective session is A18 on the [Multi-objective](../guide/more/multi-objective.md) page.

## Clear the queue

`DELETE .../experiments/queue/{item_id}` removes one `pending` item and gives a 409 for any other. `POST .../experiments/queue/purge` removes every `done` and `failed` item and returns `n_purged`; `pending` and `running` items stay. `DELETE .../experiments/staged` removes every `pending` item. None of these touches recorded data.

The session file keeps the queue with each item's status, so an export and import carries it over. Queue changes are broadcast on the session's WebSocket; the events are in [Monitoring](../guide/more/monitoring.md#4-events-behind-the-monitor).

=== "Python"

    The queue is `session.queue`, and `complete_experiment` is `record-batch` for one item.

    ```python
    from alchemist_core import OptimizationSession

    session = OptimizationSession()
    session.add_variable("temperature", "real", bounds=(200.0, 300.0))
    session.add_variable("flow", "real", bounds=(1.0, 10.0))

    item = session.queue.stage({"temperature": 250.0, "flow": 5.0}, reason="Initial design (LHS)")
    session.queue.start(item.id)
    session.complete_experiment(
        item.id,
        actual_inputs={"temperature": 249.0, "flow": 5.0},
        output=0.81,
        iteration=0,
    )
    row = session.experiment_manager.get_data().iloc[0]
    print(item.status, row["temperature"], row["ProvenanceId"] == item.id)

    spare = session.queue.stage({"temperature": 260.0, "flow": 4.0})
    session.queue.fail(spare.id, "pump stalled")
    print(session.queue.purge(), len(session.queue.list()))
    ```

!!! info "How ALchemist does it"
    `ExperimentQueue` in `alchemist_core/queue.py` owns the items and their transitions under a lock, and calls back into the session to write the data, so a result and its provenance record are made in one step. The REST routes are in `api/routers/experiments.py`.
