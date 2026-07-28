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
        self._complete_callback = None
        self._completing: set = set()

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

    # ---- transitions ----

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
        # Validate + claim the item under the lock, but run the (potentially
        # slow, session-lock-acquiring) completion callback OUTSIDE the lock to
        # avoid a queue_lock -> session_lock ordering hazard and to avoid
        # blocking queue reads during dataset writes.
        #
        # A private `_completing` claim set (not the public status) guards
        # against a concurrent complete()/fail() on the same id running the
        # side-effecting callback twice: the second caller sees the id already
        # claimed and raises. Status is left unchanged until the terminal write
        # so observers never see a spurious intermediate state.
        with self._lock:
            item = self._require(item_id)
            if item.status not in ("pending", "running"):
                raise ValueError(
                    f"Cannot complete item {item_id} in status '{item.status}'"
                )
            if item_id in self._completing:
                raise ValueError(
                    f"Item {item_id} is already being completed"
                )
            self._completing.add(item_id)

        try:
            dataset_ref = None
            if self._complete_callback is not None:
                dataset_ref = self._complete_callback(item, output, noise)

            with self._lock:
                item.output = output
                item.noise = noise
                item.dataset_ref = dataset_ref
                item.status = "done"
                item.completed_at = _now_iso()
        finally:
            with self._lock:
                self._completing.discard(item_id)

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
                counts[i.status] = counts.get(i.status, 0) + 1
            payload = {
                "n_pending": counts["pending"],
                "n_running": counts["running"],
                "n_done": counts["done"],
                "n_failed": counts["failed"],
            }
        self._events.emit("queue_updated", payload)
