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


def test_transition_emits_item_event_payload():
    # Locks the UI event contract: a transition emits queue_item_updated with
    # the item's id + new status, and queue_updated with fresh counts.
    q = _queue()
    q.set_complete_callback(lambda item, output, noise: 0)
    item_events = []
    summary_events = []
    q._events.on("queue_item_updated", lambda d: item_events.append(d))
    q._events.on("queue_updated", lambda d: summary_events.append(d))
    a = q.stage({"x": 1.0})
    q.complete(a.id, output=1.0)
    # last item event reflects the completion
    assert item_events[-1]["item_id"] == a.id
    assert item_events[-1]["status"] == "done"
    # summary reflects one done, zero pending
    assert summary_events[-1]["n_done"] == 1
    assert summary_events[-1]["n_pending"] == 0
