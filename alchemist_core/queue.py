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
