"""
POST /experiments/record-batch: the web Add Point dialog's "Save N & close".

Every point with an output is recorded in ONE request with at most one
retrain. Staged points complete their queue item with the ACTUAL conditions
(W19: a row must never carry another point's inputs); unstaged points are
added directly. Nothing is written unless every entry is valid.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.services import session_store

client = TestClient(app)

VARS = [
    {"name": "temperature", "type": "real", "min": 100.0, "max": 500.0},
    {"name": "pressure", "type": "real", "min": 1.0, "max": 10.0},
    {"name": "whsv", "type": "real", "min": 0.5, "max": 5.0},
    {"name": "ratio", "type": "real", "min": 1.0, "max": 4.0},
]

_rng = np.random.default_rng(11)
POINTS = [
    {
        "temperature": float(t), "pressure": float(p),
        "whsv": float(w), "ratio": float(r),
    }
    for t, p, w, r in zip(
        _rng.uniform(100, 500, 10), _rng.uniform(1, 10, 10),
        _rng.uniform(0.5, 5, 10), _rng.uniform(1, 4, 10),
    )
]


def _output(point):
    return float(point["temperature"] / 500 + point["pressure"] / 10
                 - (point["whsv"] - 2) ** 2 / 10 + point["ratio"] / 4)


def _actual(point):
    """What was really run: rounded, slightly off the suggestion."""
    return {k: round(v + 0.5, 3) for k, v in point.items()}


@pytest.fixture
def sid():
    r = client.post("/api/v1/sessions", json={"ttl_hours": 1})
    r.raise_for_status()
    s = r.json()["session_id"]
    for v in VARS:
        client.post(f"/api/v1/sessions/{s}/variables", json=v).raise_for_status()
    yield s
    client.delete(f"/api/v1/sessions/{s}")


def _stage(sid, points, reason="Initial DoE (LHS)"):
    r = client.post(f"/api/v1/sessions/{sid}/experiments/staged/batch",
                    json={"experiments": points, "reason": reason})
    assert r.status_code == 200, r.text
    return r.json()["ids"]


def _count_trainings(sid, monkeypatch):
    session = session_store.get(sid)
    calls = []
    original = session.train_model

    def counting(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(session, "train_model", counting)
    return calls


def _record(sid, entries, auto_train=False):
    return client.post(
        f"/api/v1/sessions/{sid}/experiments/record-batch",
        params={"auto_train": str(auto_train).lower()},
        json={"entries": entries},
    )


def _rows(sid):
    return client.get(f"/api/v1/sessions/{sid}/experiments").json()["experiments"]


def test_staged_batch_returns_aligned_queue_ids(sid):
    ids = _stage(sid, POINTS[:3])
    assert len(ids) == 3 and len(set(ids)) == 3
    staged = client.get(f"/api/v1/sessions/{sid}/experiments/staged").json()
    assert staged["ids"] == ids
    assert staged["experiments"] == POINTS[:3]


def test_ten_points_one_request_one_retrain_right_inputs(sid, monkeypatch):
    ids = _stage(sid, POINTS)
    calls = _count_trainings(sid, monkeypatch)
    entries = [
        {"item_id": i, "inputs": _actual(p), "output": _output(p)}
        for i, p in zip(ids, POINTS)
    ]
    r = _record(sid, entries, auto_train=True)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_added"] == 10
    assert body["n_experiments"] == 10
    assert body["completed_item_ids"] == ids
    assert body["model_trained"] is True
    assert len(calls) == 1, "retrain must happen once, after all rows are in"

    rows = _rows(sid)
    assert len(rows) == 10
    for row, p in zip(rows, POINTS):
        for k, v in _actual(p).items():
            assert row[k] == pytest.approx(v)
        assert row["Output"] == pytest.approx(_output(p))
        assert row["Reason"] == "Initial DoE (LHS)"

    queue = client.get(f"/api/v1/sessions/{sid}/experiments/queue").json()
    assert queue["n_pending"] == 0 and queue["n_done"] == 10
    prov = session_store.get(sid).get_provenance()
    assert [rec["id"] for rec in prov] == ids
    assert prov[0]["suggested"] == POINTS[0]
    assert prov[0]["actual"] == _actual(POINTS[0])


def test_points_without_output_stay_pending(sid):
    ids = _stage(sid, POINTS[:5])
    entries = [
        {"item_id": ids[i], "inputs": _actual(POINTS[i]), "output": _output(POINTS[i])}
        for i in (0, 2, 4)
    ]
    r = _record(sid, entries)
    assert r.status_code == 200, r.text
    staged = client.get(f"/api/v1/sessions/{sid}/experiments/staged").json()
    assert staged["ids"] == [ids[1], ids[3]]
    assert len(_rows(sid)) == 3


def test_no_retrain_below_minimum_or_when_not_asked(sid, monkeypatch):
    ids = _stage(sid, POINTS[:6])
    calls = _count_trainings(sid, monkeypatch)
    entries = [{"item_id": i, "inputs": p, "output": _output(p)} for i, p in zip(ids, POINTS)]
    r = _record(sid, entries[:3], auto_train=True)  # 3 rows < 5
    assert r.json()["model_trained"] is False
    r = _record(sid, entries[3:], auto_train=False)
    assert r.json()["model_trained"] is False
    assert calls == []


@pytest.mark.parametrize("bad", ["unknown", "duplicate", "done"])
def test_queue_problems_reject_everything(sid, bad):
    ids = _stage(sid, POINTS[:3])
    if bad == "done":
        client.post(f"/api/v1/sessions/{sid}/experiments/queue/{ids[2]}/complete",
                    json={"outputs": [1.0]}).raise_for_status()
    entries = [
        {"item_id": ids[0], "inputs": POINTS[0], "output": 1.0},
        {"item_id": {"unknown": "nope", "duplicate": ids[0], "done": ids[2]}[bad],
         "inputs": POINTS[1], "output": 2.0},
    ]
    n_before = len(_rows(sid))
    r = _record(sid, entries)
    assert r.status_code == 409, r.text
    assert "Nothing was recorded" in r.json()["detail"]
    assert len(_rows(sid)) == n_before
    item0 = client.get(f"/api/v1/sessions/{sid}/experiments/queue/{ids[0]}").json()
    assert item0["status"] == "pending"


def test_value_problems_reject_everything(sid):
    ids = _stage(sid, POINTS[:2])
    missing = {k: v for k, v in POINTS[1].items() if k != "ratio"}
    entries = [
        {"item_id": ids[0], "inputs": POINTS[0], "output": 1.0},
        {"item_id": ids[1], "inputs": missing, "output": 2.0, "noise": -1.0},
    ]
    r = _record(sid, entries)
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert "missing variables ['ratio']" in detail and "noise" in detail
    assert _rows(sid) == []
    assert client.get(f"/api/v1/sessions/{sid}/experiments/staged").json()["n_staged"] == 2


def test_unstaged_points_are_added_directly_with_iteration_and_reason(sid, monkeypatch):
    calls = _count_trainings(sid, monkeypatch)
    entries = [
        {"inputs": p, "output": _output(p), "iteration": 3, "reason": "qEI"}
        for p in POINTS[:5]
    ]
    r = _record(sid, entries, auto_train=True)
    assert r.status_code == 200, r.text
    assert r.json()["completed_item_ids"] == []
    assert len(calls) == 1
    rows = _rows(sid)
    assert [row["Iteration"] for row in rows] == [3] * 5
    assert [row["Reason"] for row in rows] == ["qEI"] * 5
    # Direct rows still get a provenance record ("nothing was suggested").
    prov = session_store.get(sid).get_provenance()
    assert len(prov) == 5 and all(rec["suggested"] is None for rec in prov)


def test_queue_item_honours_iteration_and_edited_reason(sid):
    ids = _stage(sid, POINTS[:2], reason="qEI")
    entries = [
        {"item_id": ids[0], "inputs": POINTS[0], "output": 1.0, "iteration": 7},
        {"item_id": ids[1], "inputs": POINTS[1], "output": 2.0, "iteration": 7,
         "reason": "qEI (rerun)"},
    ]
    r = _record(sid, entries)
    assert r.status_code == 200, r.text
    rows = _rows(sid)
    assert [row["Iteration"] for row in rows] == [7, 7]
    assert [row["Reason"] for row in rows] == ["qEI", "qEI (rerun)"]


def test_mixed_staged_and_direct_in_order(sid):
    ids = _stage(sid, POINTS[:2])
    entries = [
        {"item_id": ids[0], "inputs": POINTS[0], "output": 1.0},
        {"inputs": POINTS[5], "output": 5.0, "reason": "Manual"},
        {"item_id": ids[1], "inputs": POINTS[1], "output": 2.0},
    ]
    r = _record(sid, entries)
    assert r.status_code == 200, r.text
    rows = _rows(sid)
    assert [row["Output"] for row in rows] == [1.0, 5.0, 2.0]
    assert rows[1]["temperature"] == pytest.approx(POINTS[5]["temperature"])


def test_empty_entries_rejected(sid):
    r = _record(sid, [])
    assert r.status_code == 422


def test_experiments_batch_keeps_iteration_and_reason(sid):
    r = client.post(
        f"/api/v1/sessions/{sid}/experiments/batch",
        json={"experiments": [
            {"inputs": POINTS[0], "output": 1.0, "iteration": 4, "reason": "qEI"},
            {"inputs": POINTS[1], "output": 2.0},
        ]},
    )
    assert r.status_code == 200, r.text
    rows = _rows(sid)
    assert rows[0]["Iteration"] == 4 and rows[0]["Reason"] == "qEI"
    assert rows[1]["Reason"] == "Manual"
