# Staged Experiments Workflow

The **staged experiments** system enables autonomous optimization loops where ALchemist generates experiment suggestions, an external system executes them, and results are returned programmatically. This decouples the "suggest" and "execute" steps, supporting integration with automated laboratory equipment, simulation environments, or any system that evaluates experiments asynchronously.

---

## Concept

In a standard interactive workflow, you run an experiment and immediately enter the result. In an **autonomous loop**, the process looks like:

```
ALchemist suggests → Lab robot executes → Results returned → Repeat
```

Staged experiments provide the queue mechanism for this pattern:

1. ALchemist (or you) **stages** experiment conditions into a queue
2. Your reactor/robot/simulator **retrieves** the queue and executes the experiments
3. Results are **returned** to ALchemist, which adds them to the dataset and optionally retrains
4. ALchemist **suggests** the next batch

---

## REST API Reference

All endpoints are under `/api/v1/sessions/{session_id}/experiments/staged`.

### Stage a single experiment

```http
POST /api/v1/sessions/{session_id}/experiments/staged
Content-Type: application/json

{
  "inputs": {"Temperature": 325.0, "Pressure": 8.5, "Catalyst": "Pt"},
  "reason": "qEI"
}
```

Response:
```json
{
  "message": "Experiment staged successfully",
  "n_staged": 1,
  "staged_inputs": {"Temperature": 325.0, "Pressure": 8.5, "Catalyst": "Pt"}
}
```

### Stage multiple experiments at once

```http
POST /api/v1/sessions/{session_id}/experiments/staged/batch
Content-Type: application/json

{
  "experiments": [
    {"Temperature": 325.0, "Pressure": 8.5, "Catalyst": "Pt"},
    {"Temperature": 290.0, "Pressure": 12.0, "Catalyst": "Ni"}
  ],
  "reason": "qEI batch"
}
```

### Retrieve staged experiments

```http
GET /api/v1/sessions/{session_id}/experiments/staged
```

Response:
```json
{
  "experiments": [
    {"Temperature": 325.0, "Pressure": 8.5, "Catalyst": "Pt"},
    {"Temperature": 290.0, "Pressure": 12.0, "Catalyst": "Ni"}
  ],
  "n_staged": 2,
  "reason": "qEI batch"
}
```

### Complete staged experiments with results

Once the external system has evaluated the experiments, return the results in the **same order** as the staged queue:

```http
POST /api/v1/sessions/{session_id}/experiments/staged/complete
Content-Type: application/json

{
  "outputs": [0.87, 0.79],
  "noises": [0.02, 0.03],
  "iteration": 5
}
```

This removes the experiments from the staging queue and adds them (with outputs) to the experiment dataset.

Optional query parameters:
- `auto_train=true` — retrain the model immediately after completing
- `training_backend=botorch` — model backend to use if auto-training
- `training_kernel=Matern` — kernel to use if auto-training

### Clear the staging queue

If experiments were cancelled or need to be regenerated:

```http
DELETE /api/v1/sessions/{session_id}/experiments/staged
```

---

## Web UI: Staging from Initial Design

In the web UI, after generating an Initial Design (DoE), you can click **Stage Suggestions** to push all generated points into the staging queue. The staged experiments remain in the queue until your external system retrieves and completes them.

This enables a hybrid workflow: generate the initial design interactively, then hand off execution to an automated system.

---

## Python Integration Example

Here is a minimal example of an autonomous loop using the staged experiments API:

```python
import requests
import time

BASE = "http://localhost:8000/api/v1"
SESSION_ID = "your-session-id"

def run_experiment(conditions: dict) -> float:
    """Your external experiment execution function."""
    # Replace with actual reactor/simulation call
    raise NotImplementedError("Implement this for your lab equipment")

def autonomous_loop(n_iterations: int = 10):
    for iteration in range(n_iterations):
        # 1. Request next suggestions from ALchemist
        r = requests.post(f"{BASE}/sessions/{SESSION_ID}/acquire", json={
            "strategy": "qEI",
            "goal": "maximize",
            "n_suggestions": 2
        })
        suggestions = r.json()["suggestions"]

        # 2. Stage the suggestions
        requests.post(f"{BASE}/sessions/{SESSION_ID}/experiments/staged/batch", json={
            "experiments": suggestions,
            "reason": f"qEI iteration {iteration+1}"
        })

        # 3. Retrieve and execute
        staged = requests.get(f"{BASE}/sessions/{SESSION_ID}/experiments/staged").json()
        outputs = [run_experiment(exp) for exp in staged["experiments"]]

        # 4. Complete with results and auto-retrain
        requests.post(
            f"{BASE}/sessions/{SESSION_ID}/experiments/staged/complete",
            params={"auto_train": "true"},
            json={"outputs": outputs, "iteration": iteration + 1}
        )

        print(f"Iteration {iteration+1} complete. Best so far: {max(outputs):.3f}")
```

---

## Initial Design Staging

You can also use staged experiments to queue initial DoE points before execution:

```python
import requests

BASE = "http://localhost:8000/api/v1"
SESSION_ID = "your-session-id"

# Generate initial design
r = requests.post(f"{BASE}/sessions/{SESSION_ID}/initial-design", json={
    "method": "lhs",
    "n_points": 12,
    "random_seed": 42
})
design_points = r.json()["points"]

# Stage them for execution
requests.post(f"{BASE}/sessions/{SESSION_ID}/experiments/staged/batch", json={
    "experiments": design_points,
    "reason": "Initial LHS Design"
})

# Later — after running experiments — complete them
outputs = [...]  # measured values, same order
requests.post(f"{BASE}/sessions/{SESSION_ID}/experiments/staged/complete", json={
    "outputs": outputs,
    "iteration": 0
})
```

---

## Notes

- The staging queue is **in-memory** and does not persist across server restarts. Save your session regularly if running long autonomous campaigns.
- Staged experiments do **not** appear in the experiment table until they are completed.
- The `reason` field is stored as metadata and appears in the dataset's `Reason` column after completion.
- Multiple clients can read the staged queue simultaneously, but only one should call `/complete` at a time.
