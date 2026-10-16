"""Every recorded row gets a provenance record and a ProvenanceId (A24), the
manual record's strategy is the reason that was sent (A25), and a result
without an output is refused, not a 500 (F3).

Only ``POST /experiments`` went through ``_add_direct_experiment``; the
batch route and the CSV upload added rows with no record, and the manual
record's strategy was hard-coded ``"Manual"``.
"""
import io

import pytest
from fastapi.testclient import TestClient

from alchemist_core.data.experiment_manager import PROVENANCE_COL
from api.main import app

client = TestClient(app)


@pytest.fixture
def sid():
    s = client.post("/api/v1/sessions", json={}).json()["session_id"]
    assert client.post(f"/api/v1/sessions/{s}/variables",
                       json={"name": "x", "type": "real", "min": 0.0, "max": 10.0}).status_code == 200
    yield s
    client.delete(f"/api/v1/sessions/{s}")


def _rows(sid):
    return client.get(f"/api/v1/sessions/{sid}/experiments").json()["experiments"]


def _records(sid):
    r = client.get(f"/api/v1/sessions/{sid}/experiments/provenance")
    assert r.status_code == 200, r.text
    return r.json()["records"]


def test_batch_rows_get_provenance(sid):
    r = client.post(f"/api/v1/sessions/{sid}/experiments/batch", json={"experiments": [
        {"inputs": {"x": 1.0}, "output": 0.1, "reason": "Initial Design"},
        {"inputs": {"x": 2.0}, "output": 0.2},
    ]})
    assert r.status_code == 200, r.text
    rows = _rows(sid)
    ids = [row[PROVENANCE_COL] for row in rows]
    assert all(ids) and len(set(ids)) == 2
    records = {rec["id"]: rec for rec in _records(sid)}
    assert set(records) == set(ids)
    assert records[ids[0]]["strategy"] == "Initial Design"
    assert records[ids[1]]["strategy"] == "Manual"
    assert records[ids[0]]["actual"] == {"x": 1.0}
    assert records[ids[0]]["suggested"] is None


def test_uploaded_rows_get_provenance(sid):
    csv = "x,Output,Reason\n1,0.1,Initial Design\n2,0.2,Initial Design\n3,0.3,Initial Design\n"
    r = client.post(f"/api/v1/sessions/{sid}/experiments/upload",
                    files={"file": ("data.csv", io.BytesIO(csv.encode()), "text/csv")})
    assert r.status_code == 200, r.text
    rows = _rows(sid)
    ids = [row[PROVENANCE_COL] for row in rows]
    assert all(ids) and len(set(ids)) == 3
    records = {rec["id"]: rec for rec in _records(sid)}
    assert set(records) == set(ids)
    assert records[ids[2]]["actual"] == {"x": 3.0}
    assert records[ids[2]]["output"] == pytest.approx(0.3)
    assert records[ids[2]]["strategy"] == "Initial Design"


def test_manual_record_uses_the_reason_sent(sid):
    r = client.post(f"/api/v1/sessions/{sid}/experiments",
                    json={"inputs": {"x": 1.0}, "output": 0.5, "reason": "Operator pick"})
    assert r.status_code == 200, r.text
    (record,) = _records(sid)
    assert record["strategy"] == "Operator pick"
    assert _rows(sid)[0]["Reason"] == "Operator pick"


def test_manual_record_without_a_reason_is_manual(sid):
    assert client.post(f"/api/v1/sessions/{sid}/experiments",
                       json={"inputs": {"x": 1.0}, "output": 0.5}).status_code == 200
    (record,) = _records(sid)
    assert record["strategy"] == "Manual"


def test_an_experiment_without_an_output_is_refused(sid):
    r = client.post(f"/api/v1/sessions/{sid}/experiments", json={"inputs": {"x": 1.0}})
    assert r.status_code in (400, 422), r.text
    assert _rows(sid) == []
