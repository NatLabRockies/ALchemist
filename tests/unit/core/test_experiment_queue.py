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
