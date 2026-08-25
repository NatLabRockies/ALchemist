# Suggested-vs-Actual Provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record, per experiment, what the model suggested vs. what was actually run — durably in the session file and audit log — without cluttering the UI.

**Architecture:** Route web add-point through the queue `complete()` lifecycle. A `QueueItem` already stores the suggested inputs keyed by uuid; we capture the actual inputs at completion, stamp a hidden `ProvenanceId` (the uuid) on the dataset row, compute per-variable deltas, and emit a full provenance record to the audit log + a new `provenance` section of the session file. A shared metadata-column constant ensures `ProvenanceId` never leaks into the model input matrix `X`.

**Tech Stack:** Python 3.13 (`~/miniforge3/envs/alchemist-env/bin/python`), pandas, FastAPI/Pydantic, pytest. Frontend: React 19 + TS + Vite, vitest.

**Spec:** `docs/superpowers/specs/2026-07-28-suggested-vs-actual-provenance-design.md`

**Interpreter:** use `~/miniforge3/envs/alchemist-env/bin/python` for all Python/pytest. Run pytest with `PYTHONPATH="$PWD"` from the worktree root so the worktree's code is imported, not the editable install.

---

## File Structure

- `alchemist_core/data/experiment_manager.py` — add a `PROVENANCE_COL` constant + shared `metadata_columns()` helper; add `ProvenanceId` to it; route the 3 duplicated `X`-building blocks through the helper (modify).
- `alchemist_core/audit_log.py` — add `ProvenanceId` to its metadata exclusion set (modify, ~line 449).
- `alchemist_core/queue.py` — `QueueItem` gains `actual_inputs`, `delta`, `iteration`, `strategy`, `acq_params`, `provenance` fields; `complete()` accepts `actual_inputs` (modify).
- `alchemist_core/session.py` — `add_experiment` accepts `provenance_id`; `_on_queue_complete` uses actual inputs + builds/records provenance; new `provenance` list + serialize/restore in save/load; `complete_experiment()` helper (modify).
- `api/models/requests.py` — `QueueCompleteRequest` gains `actual_inputs` (modify).
- `api/models/responses.py` — `ProvenanceRecordResponse`, `ProvenanceListResponse` (modify).
- `api/routers/experiments.py` — complete endpoint passes `actual_inputs`; manual add writes a provenance record; new `GET /experiments/provenance` + `/{id}` (modify).
- `alchemist-web/src/components/api.ts` — `completeQueueItem` helper (modify).
- `alchemist-web/src/features/experiments/ExperimentsPanel.tsx` — submit via complete(item_id, actual_inputs) instead of add+delete/restage (modify).
- `alchemist-web/src/components/AddPointDialog.tsx` — pass through the queue-item id (modify).
- Tests: `tests/unit/core/test_provenance.py` (new), extend `tests/integration/api/test_queue_router.py`, extend `alchemist-web/src/components/AddPointDialog.test.tsx`.

---

### Task 1: Shared metadata constant + ProvenanceId exclusion (the leak fix)

There are 3 identical `metadata_cols` blocks in `experiment_manager.py` (lines ~146, ~191, ~237) and one set in `audit_log.py` (~449). DRY them into one helper and add `ProvenanceId`, so the id can never leak into `X`.

**Files:**
- Modify: `alchemist_core/data/experiment_manager.py`
- Modify: `alchemist_core/audit_log.py`
- Test: `tests/unit/core/test_provenance.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/core/test_provenance.py`:
```python
"""Provenance: ProvenanceId must never become a model feature."""
import pandas as pd
from alchemist_core.data.experiment_manager import ExperimentManager, PROVENANCE_COL


def _manager_with_provenance():
    em = ExperimentManager(target_columns=["Output"])
    df = pd.DataFrame({
        "x": [0.1, 0.2, 0.3],
        "Output": [1.0, 2.0, 3.0],
        "Iteration": [0, 1, 2],
        "Reason": ["Manual", "qEI", "qEI"],
        PROVENANCE_COL: ["id-a", "id-b", "id-c"],
    })
    em.df = df
    return em


def test_provenance_col_constant():
    assert PROVENANCE_COL == "ProvenanceId"


def test_provenance_excluded_from_model_inputs():
    em = _manager_with_provenance()
    X = em.get_input_data()  # the X-building accessor
    assert PROVENANCE_COL not in X.columns
    assert "Output" not in X.columns
    assert "Iteration" not in X.columns
    assert "Reason" not in X.columns
    assert list(X.columns) == ["x"]
```

Note: the accessor is one of three real methods that return the `X` frame:
`get_features_and_target()`, `get_features_target_and_noise()`, and
`get_features_and_targets_multi()` (in `experiment_manager.py`). Update the
test to call `get_features_and_target()` (returns `(X, y)`), and add assertions
for the other two accessors as well:
```python
def test_provenance_excluded_from_all_x_accessors():
    em = _manager_with_provenance()
    X1, _ = em.get_features_and_target()
    X2, _, _ = em.get_features_target_and_noise()
    X3, _, _ = em.get_features_and_targets_multi()
    for X in (X1, X2, X3):
        assert PROVENANCE_COL not in X.columns
        assert list(X.columns) == ["x"]
```
Replace the `em.get_input_data()` line in `test_provenance_excluded_from_model_inputs` with `X, _ = em.get_features_and_target()`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH="$PWD" ~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/test_provenance.py -v`
Expected: FAIL — `PROVENANCE_COL` not defined (ImportError).

- [ ] **Step 3: Add the constant and shared helper**

At the top of `alchemist_core/data/experiment_manager.py` (after imports, before the class), add:
```python
# Column that carries the provenance record id (queue-item uuid) for a row.
# It is metadata, never a model feature.
PROVENANCE_COL = "ProvenanceId"
```
Add a method to `ExperimentManager` (place it near the existing `X`-building methods):
```python
    def metadata_columns(self) -> list:
        """Columns that are NOT model inputs: targets + bookkeeping metadata."""
        cols = list(self.target_columns)
        for c in ("Noise", "Iteration", "Reason", PROVENANCE_COL):
            if c in self.df.columns and c not in cols:
                cols.append(c)
        return cols
```

- [ ] **Step 4: Route the 3 duplicated blocks through the helper**

In `alchemist_core/data/experiment_manager.py`, replace EACH of the three blocks that look like:
```python
            metadata_cols = self.target_columns.copy()
            if 'Noise' in self.df.columns:
                metadata_cols.append('Noise')
            if 'Iteration' in self.df.columns:
                metadata_cols.append('Iteration')
            if 'Reason' in self.df.columns:
                metadata_cols.append('Reason')
            X = self.df.drop(columns=metadata_cols)
```
with:
```python
            X = self.df.drop(columns=self.metadata_columns())
```
(There are three occurrences ~lines 146, 191, 237. Replace all three. Keep surrounding indentation.)

- [ ] **Step 5: Add ProvenanceId to the audit_log exclusion set**

In `alchemist_core/audit_log.py` ~line 449, change:
```python
            metadata_cols = {'Iteration', 'Reason', 'Output', 'Noise'}
```
to:
```python
            metadata_cols = {'Iteration', 'Reason', 'Output', 'Noise', 'ProvenanceId'}
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `PYTHONPATH="$PWD" ~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/test_provenance.py -v`
Expected: PASS.

- [ ] **Step 7: Run the existing experiment-manager + model tests to confirm no regression**

Run: `PYTHONPATH="$PWD" ~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core tests/integration/workflows -q`
Expected: all pass (the `X`-building refactor is behavior-preserving when no ProvenanceId column is present).

- [ ] **Step 8: Commit**

```bash
git add alchemist_core/data/experiment_manager.py alchemist_core/audit_log.py tests/unit/core/test_provenance.py
git commit -m "refactor(core): shared metadata_columns() + exclude ProvenanceId from model X"
```

---

### Task 2: Capture actual inputs + build provenance on queue complete (core)

**Files:**
- Modify: `alchemist_core/queue.py`
- Modify: `alchemist_core/session.py`
- Test: `tests/unit/core/test_provenance.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/core/test_provenance.py`:
```python
from alchemist_core import OptimizationSession
from alchemist_core.data.experiment_manager import PROVENANCE_COL


def _session_with_staged_suggestion():
    s = OptimizationSession()
    s.add_variable("temperature", "real", bounds=(100, 1000))
    s.add_variable("catalyst", "categorical", categories=["A", "B"])
    item = s.queue.stage(
        {"temperature": 500.0, "catalyst": "A", "_reason": "qEI"},
    )
    return s, item


def test_complete_records_actual_and_delta():
    s, item = _session_with_staged_suggestion()
    # Model suggested temperature=500/catalyst=A; actually ran 505/A.
    s.complete_experiment(
        item.id,
        actual_inputs={"temperature": 505.0, "catalyst": "A"},
        output=0.42,
    )
    # Dataset row exists with ACTUAL values and a ProvenanceId
    df = s.experiment_manager.get_data()
    assert len(df) == 1
    assert df.iloc[0]["temperature"] == 505.0
    assert df.iloc[0][PROVENANCE_COL] == item.id

    # Provenance record captured suggested, actual, delta
    recs = s.get_provenance()
    assert len(recs) == 1
    r = recs[0]
    assert r["id"] == item.id
    assert r["strategy"] == "qEI"
    assert r["suggested"] == {"temperature": 500.0, "catalyst": "A"}
    assert r["actual"] == {"temperature": 505.0, "catalyst": "A"}
    assert r["delta"]["temperature"] == 5.0
    assert r["delta"]["catalyst"] == "unchanged"
    assert r["output"] == 0.42
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH="$PWD" ~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/test_provenance.py::test_complete_records_actual_and_delta -v`
Expected: FAIL — `complete_experiment` / `get_provenance` not defined.

- [ ] **Step 3: Extend QueueItem to carry actual inputs**

In `alchemist_core/queue.py`, add fields to the `QueueItem` dataclass (after `dataset_ref`):
```python
    actual_inputs: Optional[Dict[str, Any]] = None
```
Change `complete()` signature to accept actual inputs and store them before the callback runs. Replace the current `complete` method header and the claim/callback section:
```python
    def complete(self, item_id: str, output: OutputValue,
                 noise: Optional[OutputValue] = None,
                 actual_inputs: Optional[Dict[str, Any]] = None) -> QueueItem:
```
Inside `complete`, immediately after `self._completing.add(item_id)` (still inside the first `with self._lock:` block is fine, or right after it), set:
```python
            if actual_inputs is not None:
                item.actual_inputs = dict(actual_inputs)
```
Leave the rest of `complete()` unchanged — the callback `self._complete_callback(item, output, noise)` now sees `item.actual_inputs`.

- [ ] **Step 4: Add provenance building + storage to the session**

In `alchemist_core/session.py`:

(a) Ensure a provenance store exists. In `__init__` (near `self.queue = ...`, ~line 112), add:
```python
        self.provenance: list = []
```

(b) Add `provenance_id` to `add_experiment` so the row carries it. Change the signature (~line 542):
```python
    def add_experiment(self, inputs: Dict[str, Any], output: float,
                      noise: Optional[float] = None, iteration: Optional[int] = None,
                      reason: Optional[str] = None, provenance_id: Optional[str] = None) -> None:
```
After the existing `self.experiment_manager.add_experiment(...)` call inside `add_experiment`, stamp the id onto the just-added row:
```python
        if provenance_id is not None:
            from alchemist_core.data.experiment_manager import PROVENANCE_COL
            self.experiment_manager.df.loc[
                self.experiment_manager.df.index[-1], PROVENANCE_COL
            ] = provenance_id
```

(c) Rewrite `_on_queue_complete` (~line 621) to use actual inputs, stamp the id, and record provenance:
```python
    def _on_queue_complete(self, item, output, noise):
        """Queue completion callback: add the ACTUAL-inputs row to the dataset,
        stamp its ProvenanceId, record a provenance entry, return row index."""
        actual = item.actual_inputs if item.actual_inputs is not None else item.inputs
        self.add_experiment(
            inputs=actual,
            output=output,
            noise=noise,
            reason=item.reason,
            provenance_id=item.id,
        )
        row_index = len(self.experiment_manager.df) - 1
        self._record_provenance(item, actual, output, noise)
        return row_index
```

(d) Add the provenance-record builder + delta + acq-params lookup + public accessors. Place these methods near `_on_queue_complete`:
```python
    def _compute_delta(self, suggested: dict, actual: dict) -> dict:
        delta = {}
        for k, av in actual.items():
            sv = suggested.get(k) if suggested else None
            if isinstance(av, (int, float)) and isinstance(sv, (int, float)):
                delta[k] = av - sv
            elif sv is None:
                delta[k] = "no-suggestion"
            elif av == sv:
                delta[k] = "unchanged"
            else:
                delta[k] = f"{sv}\u2192{av}"
        return delta

    def _lookup_acq_params(self, iteration) -> dict:
        """Most recent acquisition_locked audit entry matching this iteration."""
        try:
            entries = self.audit_log.get_entries("acquisition_locked")
        except Exception:
            return {}
        match = None
        for e in entries:
            params = getattr(e, "parameters", {}) or {}
            if params.get("iteration") == iteration:
                match = params
        if match is None and entries:
            match = getattr(entries[-1], "parameters", {}) or {}
        return match.get("parameters", {}) if match else {}

    def _record_provenance(self, item, actual: dict, output, noise) -> None:
        from datetime import datetime
        suggested = dict(item.inputs) if item.inputs else None
        iteration = None
        if not self.experiment_manager.df.empty and 'Iteration' in self.experiment_manager.df.columns:
            iteration = int(self.experiment_manager.df['Iteration'].iloc[-1])
        record = {
            "id": item.id,
            "iteration": iteration,
            "strategy": item.reason or "Manual",
            "acq_params": self._lookup_acq_params(iteration),
            "suggested": suggested,
            "actual": dict(actual),
            "delta": self._compute_delta(suggested, actual),
            "output": output,
            "noise": noise,
            "timestamp": datetime.now().isoformat(),
        }
        self.provenance.append(record)
        try:
            self.audit_log.log_event("experiment_completed", record)
        except Exception as e:
            logger.warning(f"Failed to audit provenance record: {e}")

    def get_provenance(self) -> list:
        """Return all provenance records (suggested vs actual per experiment)."""
        return [dict(r) for r in self.provenance]

    def complete_experiment(self, item_id: str, actual_inputs: dict,
                            output: float, noise: Optional[float] = None) -> None:
        """Complete a staged queue item using the ACTUAL run conditions,
        recording provenance (suggested vs actual)."""
        self.queue.complete(item_id, output=output, noise=noise,
                            actual_inputs=actual_inputs)
```

- [ ] **Step 5: (removed — no new audit-log method needed)**

The session's `_record_provenance` (Step 4) already calls the existing generic
`self.audit_log.log_event("experiment_completed", record)` helper
(`audit_log.py` ~line 276), which builds an `AuditEntry` and appends it. No
change to `audit_log.py` is required in this task.

- [ ] **Step 6: Run the test to verify it passes**

Run: `PYTHONPATH="$PWD" ~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/test_provenance.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add alchemist_core/queue.py alchemist_core/session.py tests/unit/core/test_provenance.py
git commit -m "feat(core): capture actual inputs + provenance record on queue complete"
```

---

### Task 3: Persist provenance across save/load

**Files:**
- Modify: `alchemist_core/session.py`
- Test: `tests/unit/core/test_provenance.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/core/test_provenance.py`:
```python
import json, tempfile
from pathlib import Path


def test_provenance_roundtrips_through_save_load():
    s, item = _session_with_staged_suggestion()
    s.complete_experiment(item.id, {"temperature": 505.0, "catalyst": "A"}, output=0.42)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        s.save_session(path)
        data = json.load(open(path))
        assert "provenance" in data
        assert len(data["provenance"]) == 1
        assert data["provenance"][0]["actual"]["temperature"] == 505.0

        loaded = OptimizationSession.load_session(path, retrain_on_load=False)
        recs = loaded.get_provenance()
        assert len(recs) == 1
        assert recs[0]["id"] == item.id
        assert recs[0]["delta"]["temperature"] == 5.0
        # ProvenanceId column also survived on the row
        from alchemist_core.data.experiment_manager import PROVENANCE_COL
        assert loaded.experiment_manager.get_data().iloc[0][PROVENANCE_COL] == item.id
    finally:
        Path(path).unlink(missing_ok=True)


def test_load_without_provenance_key_is_empty():
    """Backward compat: older files with no 'provenance' key load cleanly."""
    s = OptimizationSession()
    s.add_variable("x", "real", bounds=(0, 1))
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        s.save_session(path)
        data = json.load(open(path))
        data.pop("provenance", None)
        json.dump(data, open(path, "w"))
        loaded = OptimizationSession.load_session(path, retrain_on_load=False)
        assert loaded.get_provenance() == []
    finally:
        Path(path).unlink(missing_ok=True)
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH="$PWD" ~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/test_provenance.py -k roundtrips -v`
Expected: FAIL — no `provenance` key in saved file.

- [ ] **Step 3: Serialize provenance in save_session**

In `alchemist_core/session.py` `save_session` (~line 2164), add to the `session_data` dict alongside `'staged_experiments'`:
```python
            'provenance': [dict(r) for r in self.provenance],
```

- [ ] **Step 4: Restore provenance in _load_session_impl**

In `_load_session_impl`, near where staged is restored (~line 2433 `staged = session_data.get('staged_experiments') or []`), add:
```python
        session.provenance = list(session_data.get('provenance') or [])
```
Note: the `ProvenanceId` column is part of the experiments DataFrame and is restored automatically by the existing experiment-restore loop, because it is a normal column in `experiments.data`. Verify: the restore loop builds `inputs` by excluding target/bookkeeping columns — confirm `ProvenanceId` is passed through to the row. If the restore loop drops unknown non-variable columns, add `ProvenanceId` handling: when building `inputs` for `add_experiment`, capture `row.get('ProvenanceId')` and pass it as `provenance_id=...`. Check the loop (~line 2410-2445) and wire `provenance_id` through if needed so the column survives.

- [ ] **Step 5: Run to verify it passes**

Run: `PYTHONPATH="$PWD" ~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/test_provenance.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add alchemist_core/session.py tests/unit/core/test_provenance.py
git commit -m "feat(core): persist provenance records + ProvenanceId column across save/load"
```

---

### Task 4: API — complete with actual_inputs, manual-add provenance, read endpoints

**Files:**
- Modify: `api/models/requests.py`
- Modify: `api/models/responses.py`
- Modify: `api/routers/experiments.py`
- Test: `tests/integration/api/test_queue_router.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/integration/api/test_queue_router.py` (match the file's existing client/fixture style — inspect the top of the file for helpers like `_create_session`, `_add_variables`; reuse them):
```python
def test_complete_with_actual_inputs_records_provenance():
    session_id = _create_session()
    try:
        _add_variables(session_id)  # temperature real, etc. (reuse existing helper)
        # stage one suggested item
        stage = client.post(
            f"/api/v1/sessions/{session_id}/experiments/queue",
            json={"items": [{"inputs": {"temperature": 500.0, "pressure": 3.0}, "reason": "qEI"}]},
        )
        assert stage.status_code == 200
        item_id = stage.json()["items"][0]["id"]

        # complete with ACTUAL inputs that differ from the suggestion
        resp = client.post(
            f"/api/v1/sessions/{session_id}/experiments/queue/{item_id}/complete",
            json={"outputs": [0.42], "actual_inputs": {"temperature": 505.0, "pressure": 3.0}},
        )
        assert resp.status_code == 200

        # provenance endpoint returns the record with suggested vs actual
        prov = client.get(f"/api/v1/sessions/{session_id}/experiments/provenance")
        assert prov.status_code == 200
        records = prov.json()["records"]
        assert len(records) == 1
        r = records[0]
        assert r["suggested"]["temperature"] == 500.0
        assert r["actual"]["temperature"] == 505.0
        assert r["delta"]["temperature"] == 5.0
    finally:
        client.delete(f"/api/v1/sessions/{session_id}")
```
(If `_add_variables` in that file uses different variable names, keep the inputs consistent with whatever variables it creates. Inspect and adapt the input dicts to match. The assertion on `delta.temperature == 5.0` requires a real variable named `temperature`; if the helper uses different names, use one of its real variables and adjust the numbers.)

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH="$PWD" ~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/integration/api/test_queue_router.py::test_complete_with_actual_inputs_records_provenance -v`
Expected: FAIL — `actual_inputs` ignored / provenance endpoint 404.

- [ ] **Step 3: Add actual_inputs to QueueCompleteRequest**

In `api/models/requests.py`, add to `QueueCompleteRequest` (after `iteration`):
```python
    actual_inputs: Optional[Dict[str, Union[float, int, str]]] = Field(
        None,
        description="Actual conditions run (for provenance). Defaults to the "
                    "staged suggested inputs when omitted.",
    )
```
(Ensure `Dict`, `Union`, `Optional` are imported at the top of the file — they are, given other models use them.)

- [ ] **Step 4: Add provenance response models**

In `api/models/responses.py`, add:
```python
class ProvenanceRecordResponse(BaseModel):
    """A single suggested-vs-actual provenance record."""
    id: str
    iteration: Optional[int] = None
    strategy: str
    acq_params: Dict[str, Any] = Field(default_factory=dict)
    suggested: Optional[Dict[str, Any]] = None
    actual: Dict[str, Any]
    delta: Dict[str, Any] = Field(default_factory=dict)
    output: Optional[Any] = None
    noise: Optional[Any] = None
    timestamp: Optional[str] = None


class ProvenanceListResponse(BaseModel):
    records: List[ProvenanceRecordResponse]
    n_records: int
```

- [ ] **Step 5: Pass actual_inputs through the complete endpoint**

In `api/routers/experiments.py` `complete_queue_item` (~line 860-863), change:
```python
    output = request.outputs[0]
    noise = request.noise[0] if request.noise is not None else None
    try:
        item = session.queue.complete(item_id, output=output, noise=noise)
```
to:
```python
    output = request.outputs[0]
    noise = request.noise[0] if request.noise is not None else None
    try:
        item = session.queue.complete(
            item_id, output=output, noise=noise,
            actual_inputs=request.actual_inputs,
        )
```

- [ ] **Step 6: Add the provenance read endpoints**

In `api/routers/experiments.py`, add (near the other queue endpoints; import the new response models at the top):
```python
@router.get("/{session_id}/experiments/provenance", response_model=ProvenanceListResponse)
async def list_provenance(session_id: str,
                          session: OptimizationSession = Depends(get_session)):
    records = session.get_provenance()
    return ProvenanceListResponse(records=records, n_records=len(records))


@router.get("/{session_id}/experiments/provenance/{provenance_id}",
            response_model=ProvenanceRecordResponse)
async def get_provenance_record(session_id: str, provenance_id: str,
                                session: OptimizationSession = Depends(get_session)):
    for r in session.get_provenance():
        if r["id"] == provenance_id:
            return r
    raise HTTPException(status_code=404, detail=f"Unknown provenance id: {provenance_id}")
```

- [ ] **Step 7: Manual-add path writes a provenance record**

In `api/routers/experiments.py`, find the direct add-experiment endpoint (the one calling `session.add_experiment(inputs=experiment.inputs, ...)` ~line 78-83). After the successful `add_experiment`, record a manual provenance entry so every row has one:
```python
        # Manual entries (no staged suggestion) still get a provenance record
        # so "what was suggested?" is uniformly answerable (answer: nothing).
        import uuid as _uuid
        from alchemist_core.data.experiment_manager import PROVENANCE_COL
        manual_id = str(_uuid.uuid4())
        session.experiment_manager.df.loc[
            session.experiment_manager.df.index[-1], PROVENANCE_COL
        ] = manual_id
        try:
            session._record_provenance(
                type("_Item", (), {"id": manual_id, "inputs": None, "reason": "Manual"})(),
                dict(experiment.inputs),
                experiment.output,
                experiment.noise,
            )
        except Exception as e:
            logger.warning(f"Failed to record manual provenance: {e}")
```
Note: `_record_provenance` expects an object with `.id`, `.inputs`, `.reason`. The tiny throwaway object supplies those. If the endpoint doesn't already `import logging`/have a `logger`, use the module's existing logger (grep the file top).

- [ ] **Step 8: Run to verify it passes**

Run: `PYTHONPATH="$PWD" ~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/integration/api/test_queue_router.py -v`
Expected: PASS (new test + existing queue tests).

- [ ] **Step 9: Commit**

```bash
git add api/models/requests.py api/models/responses.py api/routers/experiments.py tests/integration/api/test_queue_router.py
git commit -m "feat(api): complete with actual_inputs, provenance read endpoints, manual-add provenance"
```

---

### Task 5: Frontend — route add-point through complete(item_id, actual_inputs)

The staged suggestions the dialog steps through must carry their queue-item id so the dialog can complete the right item. Currently `pendingSuggestions` come from `GET /experiments/staged` which returns clean inputs; it must also return ids.

**Files:**
- Modify: `api/routers/experiments.py` (staged GET returns ids)
- Modify: `alchemist-web/src/App.tsx` (tag suggestions with `_queueItemId`)
- Modify: `alchemist-web/src/components/api.ts` (add `completeQueueItem`)
- Modify: `alchemist-web/src/features/experiments/ExperimentsPanel.tsx` (use complete)
- Test: `alchemist-web/src/components/AddPointDialog.test.tsx`

- [ ] **Step 1: Staged GET returns per-item ids**

In `api/routers/experiments.py` `get_staged_experiments` (the deprecated staged GET ~line 591-618), it currently returns `clean_experiments` and `reasons`. Add an `ids` list aligned with experiments. In the `StagedExperimentsListResponse` (in `api/models/responses.py`), add `ids: List[str] = Field(default_factory=list)`. In the endpoint, build `ids = [i.id for i in pending]` and pass `ids=ids` to the response.

- [ ] **Step 2: Tag suggestions with the queue-item id on restore**

In `alchemist-web/src/App.tsx` `restoreStagedExperiments`, where it maps `stagedData.experiments` to tagged experiments (adds `_reason`), also attach the id. Change the map to zip in `stagedData.ids`:
```typescript
            const ids = stagedData.ids || [];
            const taggedExperiments = stagedData.experiments.map((exp: any, i: number) => ({
              ...exp,
              _reason: reason,
              _queueItemId: ids[i],
            }));
```

- [ ] **Step 3: Add completeQueueItem helper**

In `alchemist-web/src/components/api.ts`, add:
```typescript
export async function completeQueueItem(
  sessionId: string,
  itemId: string,
  actualInputs: Record<string, any>,
  output: number,
  noise?: number,
) {
  const body: any = { outputs: [output], actual_inputs: actualInputs };
  if (noise !== undefined) body.noise = [noise];
  const res = await fetch(
    `/api/v1/sessions/${sessionId}/experiments/queue/${itemId}/complete`,
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) },
  );
  if (!res.ok) throw new Error(`Complete failed: ${res.statusText}`);
  return res.json();
}
```

- [ ] **Step 4: ExperimentsPanel uses complete when the suggestion has a queue id**

In `alchemist-web/src/features/experiments/ExperimentsPanel.tsx` `onConfirm`, replace the current `addExperiment` + DELETE/re-stage block with: if the current suggestion has `_queueItemId`, call `completeQueueItem`; else fall back to the existing `addExperiment` path (manual). Concretely, at the top of `onConfirm`:
```typescript
                const current = pendingSuggestions[currentIndex];
                const queueItemId = current?._queueItemId;
                const { completeQueueItem, addExperiment } = await import('../../components/api');
                try {
                  if (queueItemId) {
                    const actualInputs = payload.inputs;
                    await completeQueueItem(
                      sessionId, queueItemId, actualInputs,
                      payload.output, payload.noise,
                    );
                  } else {
                    await addExperiment(sessionId, payload, options.retrain);
                  }
                  queryClient.invalidateQueries({ queryKey: ['experiments', sessionId] });
                  queryClient.invalidateQueries({ queryKey: ['experiments-summary', sessionId] });
                  queryClient.invalidateQueries({ queryKey: ['session', sessionId] });
                  const updated = pendingSuggestions.filter((_, i) => i !== currentIndex);
                  onStageSuggestions && onStageSuggestions(updated);
                  toast.success('Experiment recorded');
                  if (updated.length === 0) setAddPointOpen(false);
                  else if (currentIndex >= updated.length) setCurrentIndex(updated.length - 1);
                } catch (e: any) {
                  toast.error('Failed to record point: ' + (e?.message || String(e)));
                }
```
Remove the old DELETE `/experiments/staged` + re-stage `/experiments/staged/batch` block — completing the queue item already removes it from pending server-side, so the manual re-sync is no longer needed for the queue path. (For the manual `addExperiment` fallback, leave behavior as-is: it never had a queue item.)

- [ ] **Step 5: Extend the dialog test**

The dialog itself doesn't call complete (the panel does), but confirm the dialog still surfaces `payload.inputs` as the actual values (already tested). Add one assertion to `AddPointDialog.test.tsx` that the suggestion's non-variable id keys (e.g. `_queueItemId`) are NOT rendered as variable rows:
```typescript
  it('does not render internal keys like _queueItemId as variable rows', () => {
    renderDialog({ suggestion: { ...suggestion, _queueItemId: 'abc-123' } });
    expect(screen.queryByLabelText('_queueItemId actual')).toBeNull();
  });
```

- [ ] **Step 6: Run web tests + typecheck + build**

Run (from `alchemist-web/`):
```bash
npm install
npm test
npx tsc -b --force
npm run build
```
Expected: tests pass, tsc exit 0, build succeeds.

- [ ] **Step 7: Commit**

```bash
git add api/routers/experiments.py api/models/responses.py alchemist-web/src/App.tsx alchemist-web/src/components/api.ts alchemist-web/src/features/experiments/ExperimentsPanel.tsx alchemist-web/src/components/AddPointDialog.test.tsx
git commit -m "feat(web): record actuals via queue complete; surface queue-item ids to dialog"
```

---

### Task 6: Full verification + end-to-end provenance check

**Files:** none (verification only).

- [ ] **Step 1: Full core + API test suites**

Run: `PYTHONPATH="$PWD" ~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit tests/integration -q`
Expected: all pass (no regressions).

- [ ] **Step 2: Model-leak regression is covered**

Confirm `tests/unit/core/test_provenance.py::test_provenance_excluded_from_model_inputs` passed in Step 1. Additionally run a training smoke test with a ProvenanceId present:
```bash
PYTHONPATH="$PWD" ~/miniforge3/envs/alchemist-env/bin/python -c "
from alchemist_core import OptimizationSession
s=OptimizationSession()
s.add_variable('t','real',bounds=(0,10))
it=s.queue.stage({'t':5.0,'_reason':'qEI'})
s.complete_experiment(it.id, {'t':5.1}, output=1.0)
for _ in range(4):
    it=s.queue.stage({'t':5.0,'_reason':'qEI'}); s.complete_experiment(it.id,{'t':5.0},output=1.0)
s.train_model(backend='sklearn')
print('trained OK with ProvenanceId present; provenance records:', len(s.get_provenance()))
"
```
Expected: prints trained OK and 5 provenance records; no error about a non-numeric 'ProvenanceId' feature.

- [ ] **Step 3: End-to-end via the API stack**

```bash
PYTHONPATH="$PWD" ~/miniforge3/envs/alchemist-env/bin/python -c "
from fastapi.testclient import TestClient
from api.main import app
c=TestClient(app)
sid=c.post('/api/v1/sessions',json={'ttl_hours':1}).json()['session_id']
c.post(f'/api/v1/sessions/{sid}/variables',json={'name':'t','type':'real','min':0,'max':10})
st=c.post(f'/api/v1/sessions/{sid}/experiments/queue',json={'items':[{'inputs':{'t':5.0},'reason':'qEI'}]})
iid=st.json()['items'][0]['id']
c.post(f'/api/v1/sessions/{sid}/experiments/queue/{iid}/complete',json={'outputs':[1.0],'actual_inputs':{'t':5.3}})
recs=c.get(f'/api/v1/sessions/{sid}/experiments/provenance').json()['records']
print('records:', recs)
assert recs[0]['suggested']['t']==5.0 and recs[0]['actual']['t']==5.3 and recs[0]['delta']['t']==0.3
print('E2E provenance OK')
"
```
Expected: prints `E2E provenance OK`.

- [ ] **Step 4: Rebuild the served web bundle (per AGENTS.md gotcha)**

Run (from `alchemist-web/`): `npm run build`
Then verify the served bundle is current:
```bash
grep -o "index-[A-Za-z0-9_]*\.js" dist/index.html
ls -la dist/assets/*.js
```
Expected: fresh bundle timestamp. (User must restart the API + hard-refresh to see it.)

- [ ] **Step 5: Commit any final touch-ups (if needed)**

If nothing changed, no commit. Otherwise fix + commit.

---

## Self-Review Notes

- **Spec coverage:** queue `complete()` lifecycle (Tasks 2, 5) ✓; hidden `ProvenanceId` join key + never in `X` (Task 1, regression test) ✓; session `provenance` section save/load, backward-compatible (Task 3) ✓; audit `experiment_completed` entry (Task 2 Step 5) ✓; full record suggested+actual+delta+acq context (Task 2) ✓; manual-add → `strategy:"Manual"`, `suggested:null` (Task 4 Step 7) ✓; acq_params looked up by iteration, backend-authoritative (Task 2 `_lookup_acq_params`) ✓; read endpoints (Task 4) ✓; UI unchanged visually, only plumbing (Task 5) ✓; delta semantics numeric vs categorical (Task 2 `_compute_delta`) ✓.
- **Out of scope confirmed absent:** no queue clear/delete/reorder UI, no provenance-viewing panel, no mid-batch reload iteration restore.
- **Type/name consistency:** `PROVENANCE_COL = "ProvenanceId"` used consistently (Tasks 1–4); `complete_experiment` / `get_provenance` / `_record_provenance` / `_compute_delta` / `_lookup_acq_params` consistent (Tasks 2–4); `completeQueueItem` / `_queueItemId` consistent (Task 5); `actual_inputs` field name consistent (Tasks 2, 4, 5).
- **Known verification-dependent spots the implementer must confirm against real code (called out inline):** the public X-accessor method name in Task 1 Step 1; the audit-log entry-append method name in Task 2 Step 5; whether the load restore loop passes `ProvenanceId` through (Task 3 Step 4); the `_add_variables` helper's variable names in Task 4 Step 1.
