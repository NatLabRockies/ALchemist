# Experiment Work Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn ALchemist's flat, batch, all-or-nothing `staged_experiments` facility into a real per-item work queue (IDs + status + per-item complete/fail + per-item reason), add an opaque per-objective label/unit mechanism with a mid-campaign guard, and push per-item queue events over the existing WebSocket — while keeping the legacy staged endpoints working as a deprecated compatibility layer.

**Architecture:** A new `ExperimentQueue` class (`alchemist_core/queue.py`) owns `QueueItem`s, their status transitions, thread-safety, and event emission. `OptimizationSession` holds one queue and reimplements its legacy staged methods as delegations. The API gains additive per-item endpoints and objective-metadata endpoints; the legacy staged endpoints become a thin compat layer. Session JSON serialization migrates old flat staged lists into `QueueItem`s.

**Tech Stack:** Python 3, dataclasses, FastAPI, Pydantic v2, pytest, FastAPI `TestClient`. Interpreter: `~/miniforge3/envs/alchemist-env/bin/python`.

**Spec:** `docs/superpowers/specs/2026-07-28-experiment-work-queue-design.md`

**Conventions (verified in repo):**
- Run tests with `~/miniforge3/envs/alchemist-env/bin/python -m pytest ... -v`.
- Core unit tests live under `tests/unit/core/`; API integration tests under `tests/integration/api/` and use `from api.main import app` + `TestClient(app)` + a `session_id` fixture (`POST /api/v1/sessions` then `DELETE`).
- Request/response models: `api/models/requests.py`, `api/models/responses.py` (Pydantic v2 `BaseModel` + `ConfigDict`).
- Experiments router: `api/routers/experiments.py`; WebSocket broadcast helper `broadcast_to_session` from `.websocket`.

---

## File Structure

- **Create** `alchemist_core/queue.py` — `QueueItem` dataclass + `ExperimentQueue` class. Sole owner of queue state, transitions, thread-safety, events.
- **Modify** `alchemist_core/session.py` — hold an `ExperimentQueue`; reimplement legacy staged methods as delegations; add `objective_metadata` + guard helper; serialize/migrate in `save_session`/`load_session`.
- **Create** `tests/unit/core/test_experiment_queue.py` — `ExperimentQueue`/`QueueItem` unit tests.
- **Create** `tests/unit/core/test_objective_metadata.py` — objective-label storage + guard tests.
- **Modify** `tests/integration/workflows/test_session_save_load_state.py` — extend round-trip/migration coverage (queue form + objective metadata).
- **Modify** `api/models/requests.py` — new request models for queue + objective metadata.
- **Modify** `api/models/responses.py` — new response models (`QueueItemResponse`, `QueueListResponse`, etc.); extend `StagedExperimentsListResponse` with `reasons`.
- **Modify** `api/routers/experiments.py` — new per-item endpoints, objective-metadata endpoints, legacy compat rewrites, event broadcasts.
- **Create** `tests/integration/api/test_queue_router.py` — API tests for the new endpoints + legacy compat + events.

---

## Task 1: `QueueItem` dataclass

**Files:**
- Create: `alchemist_core/queue.py`
- Test: `tests/unit/core/test_experiment_queue.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/core/test_experiment_queue.py
from alchemist_core.queue import QueueItem


def test_queue_item_defaults():
    item = QueueItem(id="abc", inputs={"x": 1.0}, reason="EI")
    assert item.id == "abc"
    assert item.inputs == {"x": 1.0}
    assert item.reason == "EI"
    assert item.status == "pending"
    assert item.output is None
    assert item.noise is None
    assert item.error is None
    assert item.dataset_ref is None
    assert item.staged_at is not None  # set at construction
    assert item.started_at is None
    assert item.completed_at is None


def test_queue_item_to_dict_roundtrip():
    item = QueueItem(id="abc", inputs={"x": 1.0}, reason=None)
    d = item.to_dict()
    assert d["id"] == "abc"
    assert d["status"] == "pending"
    restored = QueueItem.from_dict(d)
    assert restored == item
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/test_experiment_queue.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'alchemist_core.queue'`

- [ ] **Step 3: Write minimal implementation**

```python
# alchemist_core/queue.py
"""
Experiment work queue for ALchemist.

Turns the legacy flat staged-experiments list into a per-item work queue with
IDs, per-item status, per-item reason, and per-item completion/failure. Owned
by OptimizationSession; domain-agnostic (no consumer-specific concepts).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Literal
import threading
import uuid

from alchemist_core.events import EventEmitter
from alchemist_core.config import get_logger

logger = get_logger(__name__)

QueueStatus = Literal["pending", "running", "done", "failed"]

# Numeric output can be single- or multi-objective.
OutputValue = Union[float, List[float]]


def _now_iso() -> str:
    return datetime.now().isoformat()


@dataclass
class QueueItem:
    """A single work-queue item awaiting or undergoing evaluation."""

    id: str
    inputs: Dict[str, Any]
    reason: Optional[str] = None
    status: QueueStatus = "pending"
    output: Optional[OutputValue] = None
    noise: Optional[OutputValue] = None
    error: Optional[str] = None
    dataset_ref: Optional[int] = None
    staged_at: Optional[str] = field(default_factory=_now_iso)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "QueueItem":
        return QueueItem(**data)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/test_experiment_queue.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add alchemist_core/queue.py tests/unit/core/test_experiment_queue.py
git commit -m "feat(queue): add QueueItem dataclass"
```

---

## Task 2: `ExperimentQueue` — stage / get / list

**Files:**
- Modify: `alchemist_core/queue.py`
- Test: `tests/unit/core/test_experiment_queue.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/core/test_experiment_queue.py
from alchemist_core.queue import ExperimentQueue
from alchemist_core.events import EventEmitter


def _queue():
    return ExperimentQueue(events=EventEmitter())


def test_stage_assigns_uuid_and_pending():
    q = _queue()
    item = q.stage({"x": 1.0}, reason="EI")
    assert item.id
    assert item.status == "pending"
    assert item.reason == "EI"
    assert q.get(item.id) is item


def test_stage_strips_metadata_and_lifts_reason():
    q = _queue()
    item = q.stage({"x": 1.0, "_reason": "legacy", "_foo": 2}, reason=None)
    assert item.inputs == {"x": 1.0}  # _-prefixed stripped
    assert item.reason == "legacy"    # lifted from _reason when reason is None


def test_stage_many_preserves_order():
    q = _queue()
    items = q.stage_many([{"x": 1.0}, {"x": 2.0}])
    assert [i.inputs["x"] for i in items] == [1.0, 2.0]
    assert [i.inputs["x"] for i in q.list()] == [1.0, 2.0]


def test_list_filters_by_status():
    q = _queue()
    a = q.stage({"x": 1.0})
    q.stage({"x": 2.0})
    q.start(a.id)
    assert len(q.list(status="pending")) == 1
    assert len(q.list(status="running")) == 1


def test_get_unknown_returns_none():
    assert _queue().get("nope") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/test_experiment_queue.py -v`
Expected: FAIL with `ImportError: cannot import name 'ExperimentQueue'`

- [ ] **Step 3: Write minimal implementation**

Append to `alchemist_core/queue.py`:

```python
class ExperimentQueue:
    """Ordered, thread-safe work queue of QueueItems.

    Owns all queue state and status transitions. Emits events via the injected
    EventEmitter. Completion is delegated back to a caller-supplied callback so
    the queue stays decoupled from the dataset/session.
    """

    def __init__(self, events: Optional[EventEmitter] = None):
        self._items: List[QueueItem] = []
        self._by_id: Dict[str, QueueItem] = {}
        self._lock = threading.RLock()
        self._events = events if events is not None else EventEmitter()

    # ---- staging ----

    def stage(self, inputs: Dict[str, Any], reason: Optional[str] = None) -> QueueItem:
        clean = {k: v for k, v in inputs.items() if not k.startswith("_")}
        if reason is None:
            reason = inputs.get("_reason")
        item = QueueItem(id=str(uuid.uuid4()), inputs=clean, reason=reason)
        with self._lock:
            self._items.append(item)
            self._by_id[item.id] = item
        self._emit_item(item)
        self._emit_summary()
        logger.debug("Staged queue item %s", item.id)
        return item

    def stage_many(self, items: List[Dict[str, Any]],
                   reason: Optional[str] = None) -> List[QueueItem]:
        return [self.stage(inp, reason=reason) for inp in items]

    # ---- reads ----

    def get(self, item_id: str) -> Optional[QueueItem]:
        with self._lock:
            return self._by_id.get(item_id)

    def list(self, status: Optional[QueueStatus] = None) -> List[QueueItem]:
        with self._lock:
            if status is None:
                return list(self._items)
            return [i for i in self._items if i.status == status]

    def pending_items(self) -> List[QueueItem]:
        return self.list(status="pending")

    # ---- event helpers ----

    def _emit_item(self, item: QueueItem) -> None:
        self._events.emit("queue_item_updated", {
            "item_id": item.id,
            "status": item.status,
            "reason": item.reason,
            "output": item.output,
            "error": item.error,
        })

    def _emit_summary(self) -> None:
        with self._lock:
            counts = {"pending": 0, "running": 0, "done": 0, "failed": 0}
            for i in self._items:
                counts[i.status] += 1
        self._events.emit("queue_updated", {
            "n_pending": counts["pending"],
            "n_running": counts["running"],
            "n_done": counts["done"],
            "n_failed": counts["failed"],
        })
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/test_experiment_queue.py -v`
Expected: FAIL — `test_list_filters_by_status` references `q.start(...)`, not yet implemented. Add a temporary skip is NOT allowed; instead this task's `start` dependency is implemented in Task 3. **Reorder:** remove `test_list_filters_by_status` and `test_get_unknown_returns_none`'s dependency is fine. Move `test_list_filters_by_status` to Task 3.

> Correction: In Step 1 above, delete `test_list_filters_by_status` from this task and add it in Task 3 (it needs `start`). Re-run:

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/test_experiment_queue.py -v`
Expected: PASS (all remaining)

- [ ] **Step 5: Commit**

```bash
git add alchemist_core/queue.py tests/unit/core/test_experiment_queue.py
git commit -m "feat(queue): ExperimentQueue stage/get/list with events"
```

---

## Task 3: Transitions — start / complete / fail

**Files:**
- Modify: `alchemist_core/queue.py`
- Test: `tests/unit/core/test_experiment_queue.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/core/test_experiment_queue.py
import pytest


def test_list_filters_by_status():
    q = _queue()
    a = q.stage({"x": 1.0})
    q.stage({"x": 2.0})
    q.start(a.id)
    assert len(q.list(status="pending")) == 1
    assert len(q.list(status="running")) == 1


def test_start_sets_running_and_timestamp():
    q = _queue()
    a = q.stage({"x": 1.0})
    q.start(a.id)
    assert a.status == "running"
    assert a.started_at is not None


def test_start_illegal_from_terminal_raises():
    q = _queue()
    a = q.stage({"x": 1.0})
    q.fail(a.id, "boom")
    with pytest.raises(ValueError):
        q.start(a.id)


def test_complete_from_pending_records_output_and_dataset_ref():
    q = _queue()
    captured = {}

    def on_complete(item, output, noise):
        captured["inputs"] = item.inputs
        captured["output"] = output
        return 7  # simulated dataset row index

    q.set_complete_callback(on_complete)
    a = q.stage({"x": 1.0}, reason="EI")
    q.complete(a.id, output=0.9, noise=0.01)
    assert a.status == "done"
    assert a.output == 0.9
    assert a.noise == 0.01
    assert a.dataset_ref == 7
    assert a.completed_at is not None
    assert captured["output"] == 0.9


def test_complete_from_running_ok():
    q = _queue()
    q.set_complete_callback(lambda item, output, noise: 0)
    a = q.stage({"x": 1.0})
    q.start(a.id)
    q.complete(a.id, output=1.0)
    assert a.status == "done"


def test_complete_terminal_raises():
    q = _queue()
    q.set_complete_callback(lambda item, output, noise: 0)
    a = q.stage({"x": 1.0})
    q.complete(a.id, output=1.0)
    with pytest.raises(ValueError):
        q.complete(a.id, output=2.0)


def test_fail_sets_error():
    q = _queue()
    a = q.stage({"x": 1.0})
    q.fail(a.id, "sensor error")
    assert a.status == "failed"
    assert a.error == "sensor error"
    assert a.completed_at is not None


def test_transition_unknown_id_raises():
    q = _queue()
    with pytest.raises(KeyError):
        q.start("nope")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/test_experiment_queue.py -v`
Expected: FAIL with `AttributeError: 'ExperimentQueue' object has no attribute 'start'`

- [ ] **Step 3: Write minimal implementation**

Append transition methods + the completion callback to `ExperimentQueue`:

```python
    # (add to ExperimentQueue)

    def set_complete_callback(self, callback) -> None:
        """Register callback(item, output, noise) -> dataset_ref (int|None).

        Called inside complete() so the queue stays decoupled from the dataset.
        """
        self._complete_callback = callback

    def _require(self, item_id: str) -> QueueItem:
        item = self._by_id.get(item_id)
        if item is None:
            raise KeyError(f"Unknown queue item: {item_id}")
        return item

    def start(self, item_id: str) -> QueueItem:
        with self._lock:
            item = self._require(item_id)
            if item.status != "pending":
                raise ValueError(
                    f"Cannot start item {item_id} in status '{item.status}'"
                )
            item.status = "running"
            item.started_at = _now_iso()
        self._emit_item(item)
        self._emit_summary()
        return item

    def complete(self, item_id: str, output: OutputValue,
                 noise: Optional[OutputValue] = None) -> QueueItem:
        with self._lock:
            item = self._require(item_id)
            if item.status not in ("pending", "running"):
                raise ValueError(
                    f"Cannot complete item {item_id} in status '{item.status}'"
                )
            callback = getattr(self, "_complete_callback", None)
            dataset_ref = None
            if callback is not None:
                dataset_ref = callback(item, output, noise)
            item.output = output
            item.noise = noise
            item.dataset_ref = dataset_ref
            item.status = "done"
            item.completed_at = _now_iso()
        self._emit_item(item)
        self._emit_summary()
        return item

    def fail(self, item_id: str, error: str) -> QueueItem:
        with self._lock:
            item = self._require(item_id)
            if item.status not in ("pending", "running"):
                raise ValueError(
                    f"Cannot fail item {item_id} in status '{item.status}'"
                )
            item.status = "failed"
            item.error = error
            item.completed_at = _now_iso()
        self._emit_item(item)
        self._emit_summary()
        return item
```

Add `self._complete_callback = None` to `__init__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/test_experiment_queue.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add alchemist_core/queue.py tests/unit/core/test_experiment_queue.py
git commit -m "feat(queue): start/complete/fail transitions with completion callback"
```

---

## Task 4: delete (pending-only) + purge (terminal-only)

**Files:**
- Modify: `alchemist_core/queue.py`
- Test: `tests/unit/core/test_experiment_queue.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/core/test_experiment_queue.py
def test_delete_pending_ok():
    q = _queue()
    a = q.stage({"x": 1.0})
    q.delete(a.id)
    assert q.get(a.id) is None


def test_delete_non_pending_raises():
    q = _queue()
    a = q.stage({"x": 1.0})
    q.start(a.id)
    with pytest.raises(ValueError):
        q.delete(a.id)


def test_purge_removes_only_terminal():
    q = _queue()
    q.set_complete_callback(lambda item, output, noise: 0)
    a = q.stage({"x": 1.0})
    b = q.stage({"x": 2.0})
    c = q.stage({"x": 3.0})
    q.complete(a.id, output=1.0)   # done
    q.fail(b.id, "x")              # failed
    # c stays pending
    n = q.purge()
    assert n == 2
    remaining = q.list()
    assert len(remaining) == 1
    assert remaining[0].id == c.id


def test_clear_pending_removes_only_pending():
    q = _queue()
    q.set_complete_callback(lambda item, output, noise: 0)
    a = q.stage({"x": 1.0})
    b = q.stage({"x": 2.0})
    q.complete(a.id, output=1.0)   # done
    n = q.clear_pending()
    assert n == 1
    ids = [i.id for i in q.list()]
    assert ids == [a.id]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/test_experiment_queue.py -v`
Expected: FAIL with `AttributeError: ... 'delete'`

- [ ] **Step 3: Write minimal implementation**

Append to `ExperimentQueue`:

```python
    def delete(self, item_id: str) -> None:
        with self._lock:
            item = self._require(item_id)
            if item.status != "pending":
                raise ValueError(
                    f"Cannot delete item {item_id} in status '{item.status}'; "
                    "only pending items may be deleted."
                )
            self._items.remove(item)
            del self._by_id[item_id]
        self._emit_summary()

    def purge(self) -> int:
        with self._lock:
            terminal = [i for i in self._items if i.status in ("done", "failed")]
            for i in terminal:
                self._items.remove(i)
                del self._by_id[i.id]
            n = len(terminal)
        if n:
            self._emit_summary()
        return n

    def clear_pending(self) -> int:
        with self._lock:
            pending = [i for i in self._items if i.status == "pending"]
            for i in pending:
                self._items.remove(i)
                del self._by_id[i.id]
            n = len(pending)
        if n:
            self._emit_summary()
        return n
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/test_experiment_queue.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add alchemist_core/queue.py tests/unit/core/test_experiment_queue.py
git commit -m "feat(queue): delete (pending-only), purge, clear_pending"
```

---

## Task 5: Session holds the queue + legacy method delegation

**Files:**
- Modify: `alchemist_core/session.py` (`__init__` ~line 110-113; staged methods 528-660)
- Test: `tests/unit/core/test_core_improvements.py` (existing legacy tests must keep passing)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/core/test_session_queue_delegation.py  (new file)
from alchemist_core.session import OptimizationSession
from alchemist_core.queue import ExperimentQueue


def _session():
    s = OptimizationSession()
    s.add_variable("x", "real", bounds=(0.0, 10.0))
    return s


def test_session_exposes_queue():
    s = _session()
    assert isinstance(s.queue, ExperimentQueue)


def test_add_staged_delegates_to_queue():
    s = _session()
    s.add_staged_experiment({"x": 1.0, "_reason": "EI"})
    items = s.queue.list()
    assert len(items) == 1
    assert items[0].inputs == {"x": 1.0}
    assert items[0].reason == "EI"


def test_get_staged_experiments_backcompat_shape():
    s = _session()
    s.add_staged_experiment({"x": 1.0, "_reason": "EI"})
    staged = s.get_staged_experiments()
    # Legacy shape: list of input dicts, reason carried via _reason key.
    assert staged[0]["x"] == 1.0
    assert staged[0]["_reason"] == "EI"


def test_move_staged_to_experiments_completes_pending():
    s = _session()
    s.add_staged_experiment({"x": 1.0})
    s.add_staged_experiment({"x": 2.0})
    n = s.move_staged_to_experiments(outputs=[0.5, 0.6], reason="EI")
    assert n == 2
    assert len(s.experiment_manager.df) == 2
    # completed items now terminal, no longer pending
    assert len(s.queue.pending_items()) == 0
    assert len(s.queue.list(status="done")) == 2


def test_clear_staged_experiments_pending_only():
    s = _session()
    s.add_staged_experiment({"x": 1.0})
    n = s.clear_staged_experiments()
    assert n == 1
    assert len(s.queue.list()) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/test_session_queue_delegation.py -v`
Expected: FAIL with `AttributeError: 'OptimizationSession' object has no attribute 'queue'`

- [ ] **Step 3: Write minimal implementation**

In `alchemist_core/session.py` `__init__`, replace the staged list init (around line 110-113):

```python
        # Staged experiments / work queue (workflow management)
        from alchemist_core.queue import ExperimentQueue
        self.queue = ExperimentQueue(events=self.events)
        self.queue.set_complete_callback(self._on_queue_complete)
        self.last_suggestions = []  # Most recent acquisition suggestions (for UI)
        self._lock = threading.RLock()  # Protects last_suggestions, _current_iteration

        # Objective display metadata (opaque per-objective label/unit).
        # Keyed by target column name. ALchemist stores/displays; never parses.
        self.objective_metadata: Dict[str, Dict[str, Any]] = {}
```

Add a completion-callback method and reimplement the legacy staged methods (replace the bodies of `add_staged_experiment`, `get_staged_experiments`, `clear_staged_experiments`, `move_staged_to_experiments`; keep their signatures/docstrings):

```python
    def _on_queue_complete(self, item, output, noise):
        """Queue completion callback: add to dataset, return new row index."""
        self.add_experiment(
            inputs=item.inputs,
            output=output,
            noise=noise,
            reason=item.reason,
        )
        return len(self.experiment_manager.df) - 1

    def add_staged_experiment(self, inputs: Dict[str, Any]) -> None:
        if self.search_space.variables:
            var_names = {v['name'] for v in self.search_space.variables}
            clean_keys = {k for k in inputs.keys() if not k.startswith('_')}
            missing = var_names - clean_keys
            if missing:
                raise ValueError(f"Missing search space variables in inputs: {missing}")
        self.queue.stage(inputs)
        logger.debug(f"Staged experiment: {inputs}")

    def get_staged_experiments(self) -> List[Dict[str, Any]]:
        # Legacy shape: input dicts with reason surfaced via _reason.
        out = []
        for item in self.queue.pending_items():
            d = dict(item.inputs)
            if item.reason is not None:
                d['_reason'] = item.reason
            out.append(d)
        return out

    def clear_staged_experiments(self) -> int:
        count = self.queue.clear_pending()
        if count > 0:
            logger.info(f"Cleared {count} staged experiments")
        return count

    def move_staged_to_experiments(self, outputs: List[float],
                                   noises: Optional[List[float]] = None,
                                   iteration: Optional[int] = None,
                                   reason: Optional[str] = None) -> int:
        pending = self.queue.pending_items()
        if len(outputs) != len(pending):
            raise ValueError(
                f"Number of outputs ({len(outputs)}) must match "
                f"number of staged experiments ({len(pending)})"
            )
        if noises is not None and len(noises) != len(pending):
            raise ValueError(
                f"Number of noise values ({len(noises)}) must match "
                f"number of staged experiments ({len(pending)})"
            )
        for i, item in enumerate(pending):
            # Batch-level reason overrides only if item has none.
            if reason is not None and item.reason is None:
                item.reason = reason
            noise = noises[i] if noises is not None else None
            self.queue.complete(item.id, output=outputs[i], noise=noise)
        return len(pending)
```

> Note: `iteration` in the legacy signature is retained for compatibility but the queue path lets `add_experiment` auto-assign iteration. If callers relied on an explicit iteration, keep behavior by passing it through `_on_queue_complete` via a temporary attribute is out of scope; document that batch `iteration` is now ignored in the deprecated path (record in spec §7 already covers deprecation).

- [ ] **Step 4: Run tests to verify they pass**

Run:
```
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/test_session_queue_delegation.py tests/unit/core/test_core_improvements.py tests/unit/core/acquisition/test_acquisition.py -v
```
Expected: PASS (new tests + existing legacy staged tests still green)

- [ ] **Step 5: Commit**

```bash
git add alchemist_core/session.py tests/unit/core/test_session_queue_delegation.py
git commit -m "refactor(session): back staged API with ExperimentQueue"
```

---

## Task 6: Objective metadata storage + mid-campaign guard

**Files:**
- Modify: `alchemist_core/session.py`
- Test: `tests/unit/core/test_objective_metadata.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/core/test_objective_metadata.py
import pytest
from alchemist_core.session import OptimizationSession


def _session():
    s = OptimizationSession()
    s.add_variable("x", "real", bounds=(0.0, 10.0))
    return s


def test_default_objective_metadata_empty():
    assert _session().objective_metadata == {}


def test_set_and_get_objective_metadata():
    s = _session()
    s.set_objective_metadata({"Output": {"label": "carbonyl_area", "unit": "a.u."}})
    assert s.objective_metadata["Output"]["label"] == "carbonyl_area"
    assert s.objective_metadata["Output"]["unit"] == "a.u."


def test_set_objective_metadata_audits_change():
    s = s2 = _session()
    s.set_objective_metadata({"Output": {"label": "a"}})
    before = len(s.audit_log.entries)
    s.set_objective_metadata({"Output": {"label": "b"}})
    assert len(s.audit_log.entries) == before + 1


def test_check_objective_label_match_ok():
    s = _session()
    s.set_objective_metadata({"Output": {"label": "a"}})
    # matching -> no raise
    s.check_objective_label({"Output": "a"})


def test_check_objective_label_mismatch_raises():
    s = _session()
    s.set_objective_metadata({"Output": {"label": "a"}})
    with pytest.raises(ValueError):
        s.check_objective_label({"Output": "b"})


def test_check_objective_label_none_is_noop():
    s = _session()
    s.set_objective_metadata({"Output": {"label": "a"}})
    s.check_objective_label(None)  # no raise
```

Confirm the audit-log attribute name first:

Run: `grep -n "self.entries" alchemist_core/audit_log.py`
If the list attribute is not `entries`, adjust the test + implementation to the actual name (e.g. `_entries`). Use the real attribute.

- [ ] **Step 2: Run test to verify it fails**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/test_objective_metadata.py -v`
Expected: FAIL with `AttributeError: ... 'set_objective_metadata'`

- [ ] **Step 3: Write minimal implementation**

Add to `OptimizationSession` (near the constraint API section):

```python
    def get_objective_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Return the opaque per-objective label/unit map (never parsed)."""
        return {k: dict(v) for k, v in self.objective_metadata.items()}

    def set_objective_metadata(self, metadata: Dict[str, Dict[str, Any]]) -> None:
        """Set/update opaque per-objective display metadata.

        metadata: {objective_name: {"label": str, "unit": Optional[str]}}
        Writes an audit entry recording old/new values. ALchemist never
        interprets the strings.
        """
        old = {k: dict(v) for k, v in self.objective_metadata.items()}
        for name, meta in metadata.items():
            entry = dict(self.objective_metadata.get(name, {}))
            entry["label"] = meta.get("label")
            entry["unit"] = meta.get("unit")
            self.objective_metadata[name] = entry
        try:
            self.audit_log.log_event(
                entry_type="objective_label_changed",
                parameters={"old": old, "new": self.get_objective_metadata()},
                notes="Objective label/unit updated",
            )
        except Exception as e:
            logger.warning(f"Failed to audit objective label change: {e}")

    def check_objective_label(self, expected: Optional[Dict[str, str]]) -> None:
        """Raise ValueError if any expected label does not match the current one.

        expected: {objective_name: label}. None or {} is a no-op. Comparison is
        pure opaque-string equality.
        """
        if not expected:
            return
        for name, exp_label in expected.items():
            current = self.objective_metadata.get(name, {}).get("label")
            if current != exp_label:
                raise ValueError(
                    f"Objective label mismatch for '{name}': "
                    f"expected '{exp_label}', session has '{current}'"
                )
```

> `audit_log.log_event(...)` must match the AuditLog API. Verify with `grep -n "def " alchemist_core/audit_log.py`. If the method is `add_entry` / `lock_*` only, add a minimal `log_event` method to `AuditLog` that appends an `AuditEntry.create(entry_type, parameters, notes)` to its entries list, and add a focused test for it in `tests/unit/core/test_objective_metadata.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/test_objective_metadata.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add alchemist_core/session.py tests/unit/core/test_objective_metadata.py alchemist_core/audit_log.py
git commit -m "feat(session): opaque objective_metadata + mid-campaign guard"
```

---

## Task 7: Serialize + migrate queue and objective metadata

**Files:**
- Modify: `alchemist_core/session.py` (`_serialize_staged_experiments` ~1992; `save_session` ~2043-2058; `load_session` ~2261-2264)
- Test: `tests/integration/workflows/test_session_save_load_state.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/integration/workflows/test_session_save_load_state.py
import json, tempfile, os
from alchemist_core.session import OptimizationSession


def _session_with_queue_and_meta():
    s = OptimizationSession()
    s.add_variable("temperature", "real", bounds=(100.0, 500.0))
    s.add_staged_experiment({"temperature": 150.0, "_reason": "EI"})
    s.add_staged_experiment({"temperature": 250.0})
    s.set_objective_metadata({"Output": {"label": "carbonyl", "unit": "a.u."}})
    return s


def test_save_persists_queue_items_with_reason_and_status():
    s = _session_with_queue_and_meta()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "s.json")
        s.save_session(p)
        data = json.load(open(p))
    items = data["staged_experiments"]
    assert len(items) == 2
    assert items[0]["status"] == "pending"
    assert items[0]["reason"] == "EI"
    assert data["objective_metadata"]["Output"]["label"] == "carbonyl"


def test_load_restores_queue_items():
    s = _session_with_queue_and_meta()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "s.json")
        s.save_session(p)
        loaded = OptimizationSession().load_session(p, retrain_on_load=False)
    items = loaded.queue.list()
    assert len(items) == 2
    assert items[0].reason == "EI"
    assert loaded.objective_metadata["Output"]["label"] == "carbonyl"


def test_load_migrates_old_flat_staged_list():
    s = _session_with_queue_and_meta()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "s.json")
        s.save_session(p)
        data = json.load(open(p))
        # Simulate an OLD file: flat input dicts, no queue fields, no metadata.
        data["staged_experiments"] = [{"temperature": 150.0, "_reason": "EI"},
                                       {"temperature": 250.0}]
        data.pop("objective_metadata", None)
        json.dump(data, open(p, "w"))
        loaded = OptimizationSession().load_session(p, retrain_on_load=False)
    items = loaded.queue.list()
    assert len(items) == 2
    assert items[0].status == "pending"
    assert items[0].reason == "EI"
    assert loaded.objective_metadata == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/integration/workflows/test_session_save_load_state.py -v`
Expected: FAIL (`objective_metadata` key missing; items lack `status`/`reason`)

- [ ] **Step 3: Write minimal implementation**

Replace `_serialize_staged_experiments` body (~1992):

```python
    def _serialize_staged_experiments(self) -> List[Dict[str, Any]]:
        """Serialize queue items (full per-item state)."""
        return [item.to_dict() for item in self.queue.list()]
```

In `save_session` `session_data` (add key after `'staged_experiments': ...`):

```python
            'staged_experiments': self._serialize_staged_experiments(),
            'objective_metadata': self.get_objective_metadata(),
```

Bump `'version'` to `'1.1.0'`.

In `load_session`, where staged is restored (~2261-2264), replace with a migration-aware restore:

```python
        from alchemist_core.queue import QueueItem
        staged = session_data.get('staged_experiments') or []
        session.queue._items = []
        session.queue._by_id = {}
        for raw in staged:
            if 'status' in raw and 'id' in raw:
                item = QueueItem.from_dict(raw)          # new format
            else:
                # migrate old flat input dict
                reason = raw.get('_reason')
                clean = {k: v for k, v in raw.items() if not k.startswith('_')}
                import uuid as _uuid
                item = QueueItem(id=str(_uuid.uuid4()), inputs=clean, reason=reason)
            session.queue._items.append(item)
            session.queue._by_id[item.id] = item

        session.objective_metadata = session_data.get('objective_metadata') or {}
```

> Adjust `session` vs `self` to match the surrounding code in `load_session` (it builds a local `session`/`loaded_session`). Read lines 2124-2270 and mirror the existing variable name.

- [ ] **Step 4: Run test to verify it passes**

Run:
```
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/integration/workflows/test_session_save_load_state.py -v
```
Expected: PASS (new + existing round-trip tests)

- [ ] **Step 5: Commit**

```bash
git add alchemist_core/session.py tests/integration/workflows/test_session_save_load_state.py
git commit -m "feat(session): serialize+migrate queue items and objective metadata"
```

---

## Task 8: API request/response models

**Files:**
- Modify: `api/models/requests.py`
- Modify: `api/models/responses.py`
- Test: (covered by Task 9 API tests; no standalone test)

- [ ] **Step 1: Add request models**

Append to `api/models/requests.py`:

```python
class QueueStageItem(BaseModel):
    inputs: Dict[str, Union[float, int, str]] = Field(..., description="Variable values")
    reason: Optional[str] = Field(None, description="Per-item reason/strategy")


class QueueStageRequest(BaseModel):
    items: List[QueueStageItem] = Field(..., description="Items to stage")


class QueueCompleteRequest(BaseModel):
    outputs: List[float] = Field(..., description="Objective value(s); one per objective")
    noise: Optional[List[float]] = Field(None, description="Per-objective measurement uncertainty")
    iteration: Optional[int] = Field(None, description="Iteration number (auto if None)")
    expected_objective_label: Optional[Dict[str, str]] = Field(
        None, description="{objective_name: label} guard; 409 on mismatch unless force")
    force: bool = Field(False, description="Override objective-label mismatch")


class QueueFailRequest(BaseModel):
    error: str = Field(..., description="Failure reason")


class SetObjectiveMetadataRequest(BaseModel):
    metadata: Dict[str, Dict[str, Optional[str]]] = Field(
        ..., description="{objective_name: {label, unit?}} opaque display strings")
```

- [ ] **Step 2: Add response models**

Append to `api/models/responses.py`:

```python
class QueueItemResponse(BaseModel):
    id: str
    inputs: Dict[str, Any]
    reason: Optional[str] = None
    status: str
    output: Optional[Any] = None
    noise: Optional[Any] = None
    error: Optional[str] = None
    dataset_ref: Optional[int] = None
    staged_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class QueueListResponse(BaseModel):
    items: List[QueueItemResponse]
    n_pending: int
    n_running: int
    n_done: int
    n_failed: int


class QueuePurgeResponse(BaseModel):
    message: str = "Terminal items purged"
    n_purged: int


class ObjectiveMetadataResponse(BaseModel):
    metadata: Dict[str, Dict[str, Any]]
```

Extend the existing `StagedExperimentsListResponse` with a per-item reasons list:

```python
    reasons: Optional[List[Optional[str]]] = Field(
        None, description="Per-item reasons, aligned with experiments")
```

- [ ] **Step 3: Verify import**

Run: `~/miniforge3/envs/alchemist-env/bin/python -c "from api.models.requests import QueueStageRequest, QueueCompleteRequest; from api.models.responses import QueueItemResponse, QueueListResponse; print('ok')"`
Expected: prints `ok`

- [ ] **Step 4: Commit**

```bash
git add api/models/requests.py api/models/responses.py
git commit -m "feat(api): request/response models for work queue + objective metadata"
```

---

## Task 9: New per-item queue endpoints + events

**Files:**
- Modify: `api/routers/experiments.py`
- Test: `tests/integration/api/test_queue_router.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/api/test_queue_router.py
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


@pytest.fixture
def sid():
    r = client.post("/api/v1/sessions", json={"ttl_hours": 1})
    r.raise_for_status()
    s = r.json()["session_id"]
    client.post(f"/api/v1/sessions/{s}/variables",
                json={"name": "x", "type": "real", "min": 0.0, "max": 10.0})
    yield s
    client.delete(f"/api/v1/sessions/{s}")


def test_stage_returns_ids(sid):
    r = client.post(f"/api/v1/sessions/{sid}/experiments/queue",
                    json={"items": [{"inputs": {"x": 1.0}, "reason": "EI"},
                                    {"inputs": {"x": 2.0}, "reason": "PI"}]})
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 2
    assert items[0]["id"] and items[1]["id"]
    assert items[0]["reason"] == "EI"
    assert items[1]["reason"] == "PI"  # per-item reason preserved (fixes problem 2)


def test_list_and_filter(sid):
    client.post(f"/api/v1/sessions/{sid}/experiments/queue",
                json={"items": [{"inputs": {"x": 1.0}}]})
    r = client.get(f"/api/v1/sessions/{sid}/experiments/queue")
    body = r.json()
    assert body["n_pending"] == 1
    r2 = client.get(f"/api/v1/sessions/{sid}/experiments/queue?status=pending")
    assert len(r2.json()["items"]) == 1


def test_start_complete_flow_adds_to_dataset(sid):
    r = client.post(f"/api/v1/sessions/{sid}/experiments/queue",
                    json={"items": [{"inputs": {"x": 1.0}, "reason": "EI"}]})
    item_id = r.json()["items"][0]["id"]
    assert client.post(f"/api/v1/sessions/{sid}/experiments/queue/{item_id}/start").status_code == 200
    rc = client.post(f"/api/v1/sessions/{sid}/experiments/queue/{item_id}/complete",
                     json={"outputs": [0.9]})
    assert rc.status_code == 200
    assert rc.json()["status"] == "done"
    assert rc.json()["dataset_ref"] is not None
    exps = client.get(f"/api/v1/sessions/{sid}/experiments")
    assert exps.json()["n_experiments"] == 1


def test_fail_does_not_touch_dataset(sid):
    r = client.post(f"/api/v1/sessions/{sid}/experiments/queue",
                    json={"items": [{"inputs": {"x": 1.0}}]})
    item_id = r.json()["items"][0]["id"]
    rf = client.post(f"/api/v1/sessions/{sid}/experiments/queue/{item_id}/fail",
                     json={"error": "sensor error"})
    assert rf.status_code == 200
    assert rf.json()["status"] == "failed"
    assert client.get(f"/api/v1/sessions/{sid}/experiments").json()["n_experiments"] == 0


def test_delete_pending_ok_running_409(sid):
    r = client.post(f"/api/v1/sessions/{sid}/experiments/queue",
                    json={"items": [{"inputs": {"x": 1.0}}, {"inputs": {"x": 2.0}}]})
    a, b = [i["id"] for i in r.json()["items"]]
    assert client.delete(f"/api/v1/sessions/{sid}/experiments/queue/{a}").status_code == 200
    client.post(f"/api/v1/sessions/{sid}/experiments/queue/{b}/start")
    assert client.delete(f"/api/v1/sessions/{sid}/experiments/queue/{b}").status_code == 409


def test_purge_removes_terminal(sid):
    r = client.post(f"/api/v1/sessions/{sid}/experiments/queue",
                    json={"items": [{"inputs": {"x": 1.0}}]})
    item_id = r.json()["items"][0]["id"]
    client.post(f"/api/v1/sessions/{sid}/experiments/queue/{item_id}/complete",
                json={"outputs": [1.0]})
    rp = client.post(f"/api/v1/sessions/{sid}/experiments/queue/purge")
    assert rp.json()["n_purged"] == 1
    assert client.get(f"/api/v1/sessions/{sid}/experiments/queue").json()["items"] == []


def test_complete_unknown_id_404(sid):
    r = client.post(f"/api/v1/sessions/{sid}/experiments/queue/nope/complete",
                    json={"outputs": [1.0]})
    assert r.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/integration/api/test_queue_router.py -v`
Expected: FAIL 404s (endpoints don't exist)

- [ ] **Step 3: Write minimal implementation**

Add imports in `api/routers/experiments.py`:

```python
from ..models.requests import (
    QueueStageRequest, QueueCompleteRequest, QueueFailRequest,
    SetObjectiveMetadataRequest,
)
from ..models.responses import (
    QueueItemResponse, QueueListResponse, QueuePurgeResponse,
    ObjectiveMetadataResponse,
)
```

Add a helper and endpoints:

```python
def _item_response(item) -> QueueItemResponse:
    return QueueItemResponse(**item.to_dict())


def _list_response(session) -> QueueListResponse:
    items = session.queue.list()
    counts = {"pending": 0, "running": 0, "done": 0, "failed": 0}
    for i in items:
        counts[i.status] += 1
    return QueueListResponse(
        items=[_item_response(i) for i in items],
        n_pending=counts["pending"], n_running=counts["running"],
        n_done=counts["done"], n_failed=counts["failed"],
    )


@router.post("/{session_id}/experiments/queue", response_model=QueueListResponse)
async def stage_queue_items(session_id: str, request: QueueStageRequest,
                            session: OptimizationSession = Depends(get_session)):
    if len(session.search_space.variables) == 0:
        raise NoVariablesError("No variables defined. Add variables first.")
    for it in request.items:
        session.queue.stage(dict(it.inputs), reason=it.reason)
    await broadcast_to_session(session_id, {"event": "queue_updated"})
    return _list_response(session)


@router.get("/{session_id}/experiments/queue", response_model=QueueListResponse)
async def list_queue(session_id: str, status: Optional[str] = Query(None),
                     session: OptimizationSession = Depends(get_session)):
    resp = _list_response(session)
    if status:
        resp.items = [i for i in resp.items if i.status == status]
    return resp


@router.get("/{session_id}/experiments/queue/{item_id}", response_model=QueueItemResponse)
async def get_queue_item(session_id: str, item_id: str,
                         session: OptimizationSession = Depends(get_session)):
    item = session.queue.get(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Unknown queue item: {item_id}")
    return _item_response(item)


@router.post("/{session_id}/experiments/queue/{item_id}/start", response_model=QueueItemResponse)
async def start_queue_item(session_id: str, item_id: str,
                           session: OptimizationSession = Depends(get_session)):
    if session.queue.get(item_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown queue item: {item_id}")
    try:
        item = session.queue.start(item_id)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    await broadcast_to_session(session_id, {"event": "queue_item_updated",
                                            "item_id": item_id, "status": item.status})
    return _item_response(item)


@router.post("/{session_id}/experiments/queue/{item_id}/complete", response_model=QueueItemResponse)
async def complete_queue_item(session_id: str, item_id: str, request: QueueCompleteRequest,
                              session: OptimizationSession = Depends(get_session)):
    if session.queue.get(item_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown queue item: {item_id}")
    # objective-label guard
    if request.expected_objective_label and not request.force:
        try:
            session.check_objective_label(request.expected_objective_label)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))
    outputs = request.outputs[0] if len(request.outputs) == 1 else list(request.outputs)
    noise = None
    if request.noise is not None:
        noise = request.noise[0] if len(request.noise) == 1 else list(request.noise)
    try:
        item = session.queue.complete(item_id, output=outputs, noise=noise)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    await broadcast_to_session(session_id, {"event": "queue_item_updated",
                                            "item_id": item_id, "status": item.status})
    await broadcast_to_session(session_id, {"event": "experiments_updated",
                                            "n_experiments": len(session.experiment_manager.df)})
    return _item_response(item)


@router.post("/{session_id}/experiments/queue/{item_id}/fail", response_model=QueueItemResponse)
async def fail_queue_item(session_id: str, item_id: str, request: QueueFailRequest,
                          session: OptimizationSession = Depends(get_session)):
    if session.queue.get(item_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown queue item: {item_id}")
    try:
        item = session.queue.fail(item_id, request.error)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    await broadcast_to_session(session_id, {"event": "queue_item_updated",
                                            "item_id": item_id, "status": item.status})
    return _item_response(item)


@router.delete("/{session_id}/experiments/queue/{item_id}", response_model=QueueListResponse)
async def delete_queue_item(session_id: str, item_id: str,
                            session: OptimizationSession = Depends(get_session)):
    if session.queue.get(item_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown queue item: {item_id}")
    try:
        session.queue.delete(item_id)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    await broadcast_to_session(session_id, {"event": "queue_updated"})
    return _list_response(session)


@router.post("/{session_id}/experiments/queue/purge", response_model=QueuePurgeResponse)
async def purge_queue(session_id: str, session: OptimizationSession = Depends(get_session)):
    n = session.queue.purge()
    await broadcast_to_session(session_id, {"event": "queue_updated"})
    return QueuePurgeResponse(n_purged=n)
```

> Route-ordering note: FastAPI matches in declaration order. Declare `/queue/purge` BEFORE `/queue/{item_id}` is *not* required here because `purge` is a POST and `{item_id}` GET/DELETE differ, but the POST `{item_id}/complete|start|fail` are sub-paths so no collision with `/queue/purge`. Keep `purge` and the `{item_id}` routes as written.

- [ ] **Step 4: Run test to verify it passes**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/integration/api/test_queue_router.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add api/routers/experiments.py tests/integration/api/test_queue_router.py
git commit -m "feat(api): per-item work queue endpoints with events"
```

---

## Task 10: Objective-metadata endpoints

**Files:**
- Modify: `api/routers/experiments.py`
- Test: `tests/integration/api/test_queue_router.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/integration/api/test_queue_router.py
def test_objective_metadata_roundtrip(sid):
    r = client.put(f"/api/v1/sessions/{sid}/objective-metadata",
                   json={"metadata": {"Output": {"label": "carbonyl", "unit": "a.u."}}})
    assert r.status_code == 200
    g = client.get(f"/api/v1/sessions/{sid}/objective-metadata")
    assert g.json()["metadata"]["Output"]["label"] == "carbonyl"


def test_complete_label_mismatch_409(sid):
    client.put(f"/api/v1/sessions/{sid}/objective-metadata",
               json={"metadata": {"Output": {"label": "carbonyl"}}})
    r = client.post(f"/api/v1/sessions/{sid}/experiments/queue",
                    json={"items": [{"inputs": {"x": 1.0}}]})
    item_id = r.json()["items"][0]["id"]
    rc = client.post(f"/api/v1/sessions/{sid}/experiments/queue/{item_id}/complete",
                     json={"outputs": [1.0], "expected_objective_label": {"Output": "WRONG"}})
    assert rc.status_code == 409


def test_complete_label_mismatch_force_ok(sid):
    client.put(f"/api/v1/sessions/{sid}/objective-metadata",
               json={"metadata": {"Output": {"label": "carbonyl"}}})
    r = client.post(f"/api/v1/sessions/{sid}/experiments/queue",
                    json={"items": [{"inputs": {"x": 1.0}}]})
    item_id = r.json()["items"][0]["id"]
    rc = client.post(f"/api/v1/sessions/{sid}/experiments/queue/{item_id}/complete",
                     json={"outputs": [1.0], "expected_objective_label": {"Output": "WRONG"},
                           "force": True})
    assert rc.status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/integration/api/test_queue_router.py -k objective -v`
Expected: FAIL (metadata endpoints 404)

- [ ] **Step 3: Write minimal implementation**

Add to `api/routers/experiments.py`:

```python
@router.get("/{session_id}/objective-metadata", response_model=ObjectiveMetadataResponse)
async def get_objective_metadata(session_id: str,
                                 session: OptimizationSession = Depends(get_session)):
    return ObjectiveMetadataResponse(metadata=session.get_objective_metadata())


@router.put("/{session_id}/objective-metadata", response_model=ObjectiveMetadataResponse)
async def set_objective_metadata(session_id: str, request: SetObjectiveMetadataRequest,
                                 session: OptimizationSession = Depends(get_session)):
    session.set_objective_metadata(request.metadata)
    return ObjectiveMetadataResponse(metadata=session.get_objective_metadata())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/integration/api/test_queue_router.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add api/routers/experiments.py tests/integration/api/test_queue_router.py
git commit -m "feat(api): objective-metadata endpoints"
```

---

## Task 11: Legacy compat layer — deprecate + guard mixed status

**Files:**
- Modify: `api/routers/experiments.py` (`get_staged_experiments` ~581; `clear_staged_experiments` ~613; `complete_staged_experiments` ~633)
- Test: `tests/integration/api/test_queue_router.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/integration/api/test_queue_router.py
def test_legacy_get_staged_exposes_per_item_reasons(sid):
    client.post(f"/api/v1/sessions/{sid}/experiments/staged/batch",
                json={"experiments": [{"x": 1.0}, {"x": 2.0}], "reason": "batchreason"})
    g = client.get(f"/api/v1/sessions/{sid}/experiments/staged")
    body = g.json()
    assert body["n_staged"] == 2
    assert body["reasons"] == ["batchreason", "batchreason"]


def test_legacy_batch_complete_1to1(sid):
    client.post(f"/api/v1/sessions/{sid}/experiments/staged/batch",
                json={"experiments": [{"x": 1.0}, {"x": 2.0}], "reason": "EI"})
    r = client.post(f"/api/v1/sessions/{sid}/experiments/staged/complete",
                    json={"outputs": [0.5, 0.6]})
    assert r.status_code == 200
    assert client.get(f"/api/v1/sessions/{sid}/experiments").json()["n_experiments"] == 2


def test_legacy_batch_complete_409_with_running_item(sid):
    client.post(f"/api/v1/sessions/{sid}/experiments/staged/batch",
                json={"experiments": [{"x": 1.0}, {"x": 2.0}]})
    running = client.get(f"/api/v1/sessions/{sid}/experiments/queue").json()["items"][0]["id"]
    client.post(f"/api/v1/sessions/{sid}/experiments/queue/{running}/start")
    r = client.post(f"/api/v1/sessions/{sid}/experiments/staged/complete",
                    json={"outputs": [0.5]})
    assert r.status_code == 409


def test_legacy_clear_is_pending_only(sid):
    client.post(f"/api/v1/sessions/{sid}/experiments/staged/batch",
                json={"experiments": [{"x": 1.0}, {"x": 2.0}]})
    done = client.get(f"/api/v1/sessions/{sid}/experiments/queue").json()["items"][0]["id"]
    client.post(f"/api/v1/sessions/{sid}/experiments/queue/{done}/complete",
                json={"outputs": [1.0]})
    r = client.delete(f"/api/v1/sessions/{sid}/experiments/staged")
    assert r.json()["n_cleared"] == 1  # only the pending one
    remaining = client.get(f"/api/v1/sessions/{sid}/experiments/queue").json()["items"]
    assert len(remaining) == 1 and remaining[0]["status"] == "done"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/integration/api/test_queue_router.py -k legacy -v`
Expected: FAIL (`reasons` missing; 409 not raised; clear wipes done item)

- [ ] **Step 3: Write minimal implementation**

In `get_staged_experiments` (~581), build the parallel `reasons` list and mark deprecated:

```python
@router.get("/{session_id}/experiments/staged", response_model=StagedExperimentsListResponse,
            deprecated=True)
async def get_staged_experiments(session_id: str,
                                 session: OptimizationSession = Depends(get_session)):
    pending = session.queue.pending_items()
    clean_experiments = [dict(i.inputs) for i in pending]
    reasons = [i.reason for i in pending]
    first_reason = reasons[0] if reasons else None
    return StagedExperimentsListResponse(
        experiments=clean_experiments,
        n_staged=len(pending),
        reason=first_reason,
        reasons=reasons,
    )
```

In `complete_staged_experiments` (~633), reject mixed status before completing. After fetching `staged = session.get_staged_experiments()`, add:

```python
    # New-model guard: batch path is ambiguous if any non-pending item exists.
    non_pending = [i for i in session.queue.list() if i.status != "pending"]
    if non_pending:
        raise HTTPException(
            status_code=409,
            detail="Batch complete is unavailable once items are running/done/failed. "
                   "Use the per-item /experiments/queue/{id}/complete endpoint.",
        )
```

Mark the route `deprecated=True`. Keep the rest of the body (it calls `session.move_staged_to_experiments`, which now delegates to the queue — verified in Task 5).

In `clear_staged_experiments` (~613), it already calls `session.clear_staged_experiments()`, which is now pending-only (Task 5). Just add `deprecated=True` to the route decorator.

Also add `deprecated=True` to the `POST /experiments/staged/batch` route decorator (~548).

- [ ] **Step 4: Run test to verify it passes**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/integration/api/test_queue_router.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add api/routers/experiments.py tests/integration/api/test_queue_router.py
git commit -m "feat(api): deprecate staged endpoints; per-item reasons + mixed-status guard"
```

---

## Task 12: Display integration — resolve objective label in plots

**Files:**
- Modify: `alchemist_core/visualization/` (functions that label the objective axis: parity, metrics, contour). Locate with grep.
- Test: `tests/unit/core/test_objective_metadata.py`

- [ ] **Step 1: Locate the label sites**

Run:
```
grep -rn "'Output'\|\"Output\"\|ylabel\|Output" alchemist_core/visualization/ | grep -i "label\|Output" | head -40
```
Identify where the objective axis text is set (parity plot y/x label, metrics plot y label, contour colorbar label). Note the function signatures.

- [ ] **Step 2: Write the failing test**

```python
# append to tests/unit/core/test_objective_metadata.py
from alchemist_core.session import OptimizationSession


def test_objective_display_label_helper():
    s = OptimizationSession()
    s.add_variable("x", "real", bounds=(0.0, 10.0))
    # no metadata -> falls back to raw name
    assert s.objective_display_label("Output") == "Output"
    s.set_objective_metadata({"Output": {"label": "carbonyl", "unit": "a.u."}})
    assert s.objective_display_label("Output") == "carbonyl (a.u.)"
    s.set_objective_metadata({"Output": {"label": "carbonyl", "unit": None}})
    assert s.objective_display_label("Output") == "carbonyl"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/test_objective_metadata.py -k display -v`
Expected: FAIL (`objective_display_label` missing)

- [ ] **Step 4: Write minimal implementation**

Add helper to `OptimizationSession`:

```python
    def objective_display_label(self, objective_name: str) -> str:
        """Human display string for an objective: 'label (unit)' or raw name.

        Opaque: ALchemist never parses the label/unit.
        """
        meta = self.objective_metadata.get(objective_name)
        if not meta or not meta.get("label"):
            return objective_name
        label = meta["label"]
        unit = meta.get("unit")
        return f"{label} ({unit})" if unit else label
```

Then, at each label site found in Step 1, replace the hardcoded objective column name with `session.objective_display_label(<objective_name>)` where a `session` (or its `objective_metadata`) is in scope. **If** the visualization functions do not receive the session/metadata, thread an optional `objective_labels: Optional[Dict[str,str]] = None` argument through the plot function and have `session.create_parity_plot`/etc. pass `{name: self.objective_display_label(name) for name in self.objective_names}`. Add a focused test asserting the axis label text for one plot (parity) using `fig.axes[...].get_ylabel()`.

> Keep this task minimal: if threading the label into every plot is large, split contour/metrics into a follow-up commit but still land parity here with a test.

- [ ] **Step 5: Run test to verify it passes**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/test_objective_metadata.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add alchemist_core/session.py alchemist_core/visualization tests/unit/core/test_objective_metadata.py
git commit -m "feat(viz): display opaque objective label on plot axes"
```

---

## Task 13: Resync integration test + full-suite regression

**Files:**
- Test: `tests/integration/api/test_queue_router.py`

- [ ] **Step 1: Write the resync test**

```python
# append to tests/integration/api/test_queue_router.py
def test_resync_via_get_matches_final_state(sid):
    r = client.post(f"/api/v1/sessions/{sid}/experiments/queue",
                    json={"items": [{"inputs": {"x": 1.0}}, {"inputs": {"x": 2.0}},
                                    {"inputs": {"x": 3.0}}]})
    ids = [i["id"] for i in r.json()["items"]]
    client.post(f"/api/v1/sessions/{sid}/experiments/queue/{ids[0]}/complete",
                json={"outputs": [1.0]})
    client.post(f"/api/v1/sessions/{sid}/experiments/queue/{ids[1]}/start")
    client.post(f"/api/v1/sessions/{sid}/experiments/queue/{ids[2]}/fail",
                json={"error": "x"})
    body = client.get(f"/api/v1/sessions/{sid}/experiments/queue").json()
    assert body["n_done"] == 1 and body["n_running"] == 1 and body["n_failed"] == 1
    by_id = {i["id"]: i["status"] for i in body["items"]}
    assert by_id[ids[0]] == "done"
    assert by_id[ids[1]] == "running"
    assert by_id[ids[2]] == "failed"
```

- [ ] **Step 2: Run test to verify it fails/passes**

Run: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/integration/api/test_queue_router.py::test_resync_via_get_matches_final_state -v`
Expected: PASS (all endpoints already exist)

- [ ] **Step 3: Run the full relevant suite (regression)**

Run:
```
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core tests/integration/api tests/integration/workflows -v
```
Expected: PASS. If any pre-existing staged test broke, fix the delegation (Task 5) or compat layer (Task 11) — do NOT weaken the test.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/api/test_queue_router.py
git commit -m "test(queue): resync + full-suite regression"
```

---

## Task 14: Update API docs

**Files:**
- Modify: `api/API_ENDPOINTS.md` (staged section ~L390-511)

- [ ] **Step 1: Rewrite the staged section**

Replace the staged section with: (a) the new `/experiments/queue` per-item endpoints (stage returns IDs, list/get, start/complete/fail, delete, purge), documenting the `pending/running/done/failed` lifecycle and the `queue_item_updated`/`queue_updated` WebSocket events; (b) the `/objective-metadata` GET/PUT + `expected_objective_label`/`force` guard (409 on mismatch); (c) a "Deprecated" subsection for the legacy `/experiments/staged*` endpoints noting the two behavior changes (pending-only clear; batch complete 409 when non-pending items exist) and the new per-item `reasons` field on GET.

- [ ] **Step 2: Verify no stale examples**

Run: `grep -n "staged/complete\|List\[float\]" api/API_ENDPOINTS.md`
Ensure the batch-complete example is marked deprecated and the new per-item complete example is present.

- [ ] **Step 3: Commit**

```bash
git add api/API_ENDPOINTS.md
git commit -m "docs(api): document work queue + objective metadata; deprecate staged"
```

---

## Self-Review (completed by plan author)

**Spec coverage:**
- §1 problems 1/2/3 → Tasks 1-5 (queue+reason), Task 6/10 (label+guard). ✅
- §3 data model → Tasks 1-4. ✅
- §4.1 new endpoints → Task 9; §4.2 legacy compat → Task 11. ✅
- §5 objective label + guard → Tasks 6, 10; display §5.3 → Task 12. ✅
- §6 events → Tasks 2/3 (emit) + Task 9 (broadcast); audit → Task 6. ✅
- §7 migration/back-compat → Task 7; breaking changes → Task 11. ✅
- §8 testing → every task is TDD; resync + regression → Task 13. ✅
- §10 consumers/docs → Task 14. ✅

**Placeholder scan:** Task 2 Step 4 contains a self-correction (move a test to Task 3) — acted on in Task 3 Step 1. Task 6/12 include a "verify actual attribute/label site with grep" instruction because those exact names must be read from code, not guessed; the fallback action is fully specified. No `TBD`/`TODO`/"add error handling".

**Type consistency:** `QueueItem` fields, `ExperimentQueue` method names (`stage`, `stage_many`, `get`, `list`, `pending_items`, `start`, `complete`, `fail`, `delete`, `purge`, `clear_pending`, `set_complete_callback`), and session methods (`get/set_objective_metadata`, `check_objective_label`, `objective_display_label`, `_on_queue_complete`) are used consistently across tasks. Response model `QueueListResponse` fields (`n_pending/n_running/n_done/n_failed`, `items`) match between Task 8 and Task 9.

**Known verify-in-code points (must read real names during execution):** AuditLog append method/attribute (Task 6), visualization label sites (Task 12), `load_session` local variable name (Task 7).
