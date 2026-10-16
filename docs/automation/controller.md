# Example controller

This page walks through a small controller that runs a whole campaign over REST: it stages an initial design, runs it, trains a model, then suggests, runs and records rounds of two points. It is written against a synthetic instrument, so you can run it as it stands and then replace the instrument with your own equipment.

The controller does what [The autonomous loop](loop.md) describes:

1. Checks that no one else holds the lock, locks the session and reports its state, so the web app shows the monitor and a pause request has somewhere to go.
2. Stages points in the work queue and starts each one before it runs.
3. Checks every point against limits of its own before it reaches the equipment.
4. Records the conditions that were really run, with one request per round, and retrains once.
5. Marks a point that faulted as failed and carries on.
6. Reports `idle` when it finishes, or `failed` if anything else goes wrong, and releases the lock either way.

The client below is the in-process test client for ALchemist's own app, so the example needs no server. To drive a running server, replace the first lines with `import requests`, `http = requests.Session()` and `BASE = "http://localhost:8000/api/v1"`, and pass `timeout=120` to `Api`; the rest of the code does not change.

## 1. The client and the instrument

`Api` turns an error response into an exception that carries the server's message, which is usually the reason a call was refused. `Instrument` stands in for the equipment: it runs at setpoints rounded a little from the ones requested, as real equipment does, and returns a smooth response with its best point near 250 and 6. It faults on one chosen call, so the failure path runs.

```python
import math
import time

from fastapi.testclient import TestClient

from api.main import app

BASE = "/api/v1"
http = TestClient(app)


class Api:
    """A thin wrapper over a requests-style client that raises with the server's message."""

    def __init__(self, http, base, timeout=None):
        self.http, self.base, self.timeout = http, base, timeout

    def call(self, method, path, **kwargs):
        if self.timeout:
            kwargs["timeout"] = self.timeout
        response = getattr(self.http, method)(self.base + path, **kwargs)
        if response.status_code >= 400:
            raise RuntimeError(f"{method.upper()} {path}: {response.status_code} {response.text}")
        return response.json() if response.content else None


api = Api(http, BASE)

VARIABLES = [
    {"name": "temperature", "type": "real", "min": 200.0, "max": 300.0, "unit": "C"},
    {"name": "flow", "type": "real", "min": 1.0, "max": 10.0, "unit": "mL/min"},
]
LIMITS = {v["name"]: (v["min"], v["max"]) for v in VARIABLES}


class InstrumentFault(Exception):
    pass


class Instrument:
    """Stands in for the equipment: rounds the setpoints and returns a smooth response."""

    def __init__(self, fault_on_call=None):
        self.calls = 0
        self.fault_on_call = fault_on_call

    def run(self, setpoint):
        self.calls += 1
        if self.calls == self.fault_on_call:
            raise InstrumentFault("pump stalled")
        actual = {"temperature": round(setpoint["temperature"]), "flow": round(setpoint["flow"], 1)}
        output = math.exp(-((actual["temperature"] - 250) / 40) ** 2 - ((actual["flow"] - 6) / 3) ** 2)
        return actual, output
```

## 2. Limits, reports and pausing

ALchemist records a point outside the variables' bounds as given, with only a log warning, so the controller keeps its own check. In a real plant the limits are the equipment's, and they may be narrower than the search space.

`report` writes the controller's half of the control record. `wait_while_paused` reads the other half, the request a person makes with the monitor's **Pause** button, and holds the controller still while it says `pause`. It reports `paused` on every pass, which doubles as the heartbeat the monitor needs: a controller that stays silent for more than 30 seconds is shown as unknown.

```python
def check_limits(point):
    """The controller's own guard: ALchemist only warns about a point outside the variables' bounds."""
    for name, (low, high) in LIMITS.items():
        if not low <= point[name] <= high:
            raise ValueError(f"{name}={point[name]} is outside {low} to {high}")


def report(session_id, state, detail=None):
    api.call("put", f"/sessions/{session_id}/control",
             json={"reported": state, "reported_by": "example-controller", "detail": detail})


def wait_while_paused(session_id, sleep=time.sleep):
    while api.call("get", f"/sessions/{session_id}/control")["requested"] == "pause":
        report(session_id, "paused", "waiting for resume")
        sleep(10)
```

## 3. Staging and running a batch

`stage` posts the points to the queue and returns the items it created. The response lists the whole queue with the new items last, so `stage` takes the last `len(points)` of them; anything before those was already in the queue.

`run_batch` runs each item in turn. It waits out a pause, reports, marks the item started, checks the limits and runs it. A fault or a limit error marks that item failed and moves on. The results of the others go to ALchemist in one `record-batch` request, with the conditions the instrument reports back, not the ones that were staged. All of a round shares one `iteration`, which is what ties the points to the `suggest` call in provenance.

After a round that retrains, the function checks `model_trained`. A retrain that fails does not fail the request, and suggesting from an unchanged model would look like progress, so the controller stops instead.

```python
def stage(session_id, points, reason):
    queue = api.call("post", f"/sessions/{session_id}/experiments/queue",
                     json={"items": [{"inputs": p, "reason": reason} for p in points]})
    return queue["items"][-len(points):]


def run_batch(session_id, instrument, items, iteration, retrain, sleep=time.sleep):
    entries = []
    for number, item in enumerate(items, start=1):
        wait_while_paused(session_id, sleep)
        report(session_id, "running", f"{item['reason']}, point {number} of {len(items)}")
        api.call("post", f"/sessions/{session_id}/experiments/queue/{item['id']}/start")
        try:
            check_limits(item["inputs"])
            actual, output = instrument.run(item["inputs"])
        except (InstrumentFault, ValueError) as error:
            api.call("post", f"/sessions/{session_id}/experiments/queue/{item['id']}/fail",
                     json={"error": str(error)})
            continue
        entries.append({"item_id": item["id"], "inputs": actual, "output": output, "iteration": iteration})
    if not entries:
        return
    result = api.call("post", f"/sessions/{session_id}/experiments/record-batch",
                      params={"auto_train": "true" if retrain else "false"}, json={"entries": entries})
    if retrain and not result["model_trained"]:
        raise RuntimeError("The model was not retrained; refusing to suggest from a stale model")
```

## 4. The campaign

The initial design is recorded as iteration 0 without retraining, because no model has been chosen yet. The controller then trains once, with BoTorch and a Matern kernel, which sets the configuration every later retrain repeats. Each round asks for two points with `qEI`, stages them, and records them under the `iteration` the response gave.

`POST .../lock` replaces an existing lock without a word, so the campaign reads the lock first and refuses to start if another client holds it. If anything outside the instrument fails, the `except` branch reports `failed` with the message, so the monitor shows why the controller stopped, and the `finally` branch releases the lock with its token.

```python
def campaign(session_id, instrument, rounds=3, batch=2, sleep=time.sleep):
    status = api.call("get", f"/sessions/{session_id}/lock")
    if status["locked"]:
        raise RuntimeError(f"Session is already locked by {status['locked_by']}")
    lock = api.call("post", f"/sessions/{session_id}/lock", json={"locked_by": "example-controller"})
    try:
        report(session_id, "running", "initial design")
        design = api.call("post", f"/sessions/{session_id}/initial-design",
                          json={"method": "lhs", "n_points": 6, "random_seed": 7})
        items = stage(session_id, design["points"], "Initial design (LHS)")
        run_batch(session_id, instrument, items, iteration=0, retrain=False, sleep=sleep)
        api.call("post", f"/sessions/{session_id}/model/train", json={"backend": "botorch", "kernel": "Matern"})
        for _ in range(rounds):
            suggestion = api.call("post", f"/sessions/{session_id}/acquisition/suggest",
                                  json={"strategy": "qEI", "goal": "maximize", "n_suggestions": batch})
            items = stage(session_id, suggestion["suggestions"], "qEI")
            run_batch(session_id, instrument, items, iteration=suggestion["iteration"], retrain=True, sleep=sleep)
        report(session_id, "idle", "campaign finished")
    except Exception as error:
        report(session_id, "failed", str(error)[:200])
        raise
    finally:
        api.call("delete", f"/sessions/{session_id}/lock", params={"lock_token": lock["lock_token"]})
```

## 5. Run it

The session and its variables are made first, as a person could make them in the web app; the controller needs only the ID. Before the campaign starts, an analyst asks for a pause, and the stand-in `sleep` is their **Resume**. The instrument faults on its second call.

```python
session_id = api.call("post", "/sessions")["session_id"]
for variable in VARIABLES:
    api.call("post", f"/sessions/{session_id}/variables", json=variable)

api.call("put", f"/sessions/{session_id}/control", json={"requested": "pause", "requested_by": "analyst"})


def analyst_resumes(seconds):
    api.call("put", f"/sessions/{session_id}/control", json={"requested": "run", "requested_by": "analyst"})


campaign(session_id, Instrument(fault_on_call=2), sleep=analyst_resumes)

queue = api.call("get", f"/sessions/{session_id}/experiments/queue")
rows = api.call("get", f"/sessions/{session_id}/experiments")["experiments"]
records = api.call("get", f"/sessions/{session_id}/experiments/provenance")["records"]
best = max(rows, key=lambda row: row["Output"])

print("queue:", queue["n_done"], "done,", queue["n_failed"], "failed")
print("best:", best["temperature"], best["flow"], round(best["Output"], 3))
print("largest temperature deviation:", max(abs(r["delta"]["temperature"]) for r in records))
print("controller reports:", api.call("get", f"/sessions/{session_id}/control")["reported"])
assert queue["n_failed"] == 1 and len(rows) == len(records) == 11

api.call("delete", f"/sessions/{session_id}")
```

The run ends with 11 results and a provenance record for each, one failed queue item that added nothing to the data, and a control record that reads `idle`. The temperature deviations in the records are the instrument's rounding, up to half a degree.

## Changes for real equipment

| In the example | In a real controller |
|---|---|
| `Instrument.run` | Set the conditions, wait for steady state, then read back the conditions that were reached and the objective. Return the readback as `actual`. |
| `Api(..., timeout=120)` | Keep a timeout on every call to a real server, long enough for a train or a suggest on your data. `requests` waits forever without one. The in-process test client does not take one. |
| `LIMITS` | The equipment's own safe range, set independently of the search space. |
| `report` at each point | The same, and also from a timer every 10 seconds during a run that lasts longer, so the monitor does not show a long run as silence. |
| `except (InstrumentFault, ValueError)` | Whatever your equipment raises. Everything else should stop the campaign, as here. |
| The finished campaign | Add a stop rule that suits the work: a number of rounds, a target value, or a stop on a person's request. |

If the controller restarts, it reads the queue first; [The autonomous loop](loop.md#restarts-and-resuming) says what to do with `pending` and `running` items. Everything the example sends is listed on [Staging and the work queue](queue.md) and [Provenance](provenance.md), and in the [REST reference](../reference/rest-api.md).
