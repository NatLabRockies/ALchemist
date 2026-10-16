# Provenance

Provenance keeps what ALchemist suggested next to what was run, for every result a controller records. This page describes the record, how to read it, and what a result that was never suggested leaves behind.

## Why both are kept

A controller seldom hits a suggestion exactly. Setpoints are rounded, a pump settles a little off its target, an operator changes a value. The model should learn from what happened, so the data row holds the actual conditions and the model trains on what was run. The suggestion stays on the queue item and in a provenance record, so you can ask later how far the plant was from what was asked of it.

## What a record holds

This is the record for one point of a `qEI` batch, with its ID shortened.

```json
{
  "id": "f70f448a",
  "iteration": 2,
  "strategy": "qEI",
  "acq_params": {"goal": "maximize", "n_suggestions": 2, "xi": 0.01, "kappa": 2.0},
  "suggested": {"temperature": 261.11048981611407, "flow": 5.940193961145457},
  "actual": {"temperature": 261.1, "flow": 5.9},
  "delta": {"temperature": -0.010489816114045425, "flow": -0.040193961145456925},
  "output": 0.9248558780605115,
  "noise": null,
  "timestamp": "2026-10-14T18:19:03.421642"
}
```

| Field | Content |
|---|---|
| `id` | The queue item's ID, also stored in the data row's `ProvenanceId` column. |
| `iteration` | The iteration of the row when it was recorded. |
| `strategy` | The item's `reason`, or `Manual` when the result had no queue item (A25: always `Manual` then, whatever reason was sent). |
| `acq_params` | The parameters of the `suggest` request that produced the point, found by matching the iteration. Empty for points that came from a design. `xi` and `kappa` always appear, even for `qEI`, which ignores them (A26). |
| `suggested` | The staged inputs, or null for a result with no queue item. |
| `actual` | The inputs the result was recorded with. |
| `delta` | One entry per variable, described below. |
| `output`, `noise` | The recorded values. |
| `timestamp` | When the record was made. |

`delta` compares each variable, taking the keys of both sides.

| Value | When |
|---|---|
| A number | The variable is numeric on both sides: actual minus suggested. |
| `"unchanged"` | A non-numeric value is the same on both sides. |
| `"A→B"` | A non-numeric value changed, for example a catalyst. |
| `"dropped"` | The variable was suggested and is missing from the actual conditions. |
| `"no-suggestion"` | There was no suggested value, as for a manual result. |

`acq_params` is matched through the iteration, so give every point of a round the `iteration` that `suggest` returned when you record it. Without it each row takes the next free number, and only the first point of a batch finds its parameters; the second gets an empty `acq_params`.

## Read it

```http
GET /api/v1/sessions/{session_id}/experiments/provenance
```

The response is `records` and `n_records`, in the order the results were recorded. `GET /api/v1/sessions/{session_id}/experiments/provenance/{provenance_id}` returns one record, or a 404. To go from a data row to its record, read `ProvenanceId` from `GET /api/v1/sessions/{session_id}/experiments` and use it as the ID.

Each record is also written to the audit log as an `experiment_completed` entry, and the session file keeps the records under `provenance`.

If a plant is always a little off, the records show the pattern, and a check such as the largest `delta` per variable over the last round tells you whether the equipment is following its setpoints.

=== "Python"

    `get_provenance` returns a copy of the records, so changing it does not change the session.

    ```python
    from alchemist_core import OptimizationSession

    session = OptimizationSession()
    session.add_variable("temperature", "real", bounds=(200.0, 300.0))
    session.add_variable("flow", "real", bounds=(1.0, 10.0))
    session.add_variable("catalyst", "categorical", categories=["A", "B"])

    item = session.queue.stage({"temperature": 250.0, "flow": 5.0, "catalyst": "A"}, reason="qEI")
    session.complete_experiment(
        item.id,
        actual_inputs={"temperature": 248.0, "flow": 5.0, "catalyst": "B"},
        output=0.74,
        iteration=3,
    )

    record = session.get_provenance()[0]
    print(record["strategy"], record["iteration"])
    for name in ("temperature", "flow", "catalyst"):
        print(name, record["suggested"][name], record["actual"][name], record["delta"][name])
    ```

## Results with no suggestion

A result that was never staged is a manual entry. Over REST, `POST /api/v1/sessions/{session_id}/experiments` and a `record-batch` entry without an `item_id` each make a record with a new ID, `suggested` null and `delta` entries of `"no-suggestion"`. The `reason` you send is stored in the data row's `Reason` column.

!!! warning "Known issue"
    A25: the `strategy` of a manual record is always `Manual`; the reason you send reaches only the data row.

!!! warning "Known issue"
    A24: `POST .../experiments/batch` and `POST .../experiments/upload` add rows with no record and no `ProvenanceId`. Send the results as `record-batch` entries without an `item_id` when each should be traceable.

In Python, `add_experiment` also adds a row with no record; use `complete_experiment` on a queue item.

A controller should send what it measured, the readback of the setpoint where the equipment has one, and not the value it commanded. If the actual conditions are left out of `queue/{item_id}/complete`, the suggested ones are recorded as the actual ones, and the record shows no deviation whether or not there was one.

!!! info "How ALchemist does it"
    `OptimizationSession._on_queue_complete` in `alchemist_core/session.py` writes the data row with the actual inputs and then calls `_record_provenance`, which builds the record, compares the two sets of inputs in `_compute_delta`, and logs it. The REST helper `_add_direct_experiment` in `api/routers/experiments.py` makes the manual record.
