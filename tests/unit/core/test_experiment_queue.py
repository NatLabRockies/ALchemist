from alchemist_core.queue import QueueItem
from alchemist_core.queue import ExperimentQueue
from alchemist_core.events import EventEmitter


def _queue():
    return ExperimentQueue(events=EventEmitter())


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


def test_get_unknown_returns_none():
    assert _queue().get("nope") is None
