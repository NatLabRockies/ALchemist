import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def _session_id() -> str:
    resp = client.post("/api/v1/sessions", json={})
    assert resp.status_code in (200, 201), resp.text
    sid = resp.json()["session_id"]
    client.post(f"/api/v1/sessions/{sid}/variables",
                json={"name": "x", "type": "real", "min": 0, "max": 10})
    return sid


def test_get_control_returns_defaults():
    sid = _session_id()
    body = client.get(f"/api/v1/sessions/{sid}/control").json()
    assert body["requested"] == "run"
    assert body["reported"] == "idle"


def test_put_a_request_sets_only_the_requested_half():
    sid = _session_id()
    client.put(f"/api/v1/sessions/{sid}/control",
               json={"reported": "running", "reported_by": "ctl"})
    body = client.put(f"/api/v1/sessions/{sid}/control",
                      json={"requested": "pause", "requested_by": "caleb"}).json()
    assert body["requested"] == "pause"
    assert body["reported"] == "running"      # untouched
    assert body["reported_by"] == "ctl"


def test_put_a_report_sets_only_the_reported_half():
    sid = _session_id()
    client.put(f"/api/v1/sessions/{sid}/control",
               json={"requested": "pause", "requested_by": "caleb"})
    body = client.put(f"/api/v1/sessions/{sid}/control",
                      json={"reported": "paused", "detail": "held after q3"}).json()
    assert body["reported"] == "paused"
    assert body["detail"] == "held after q3"
    assert body["requested"] == "pause"       # untouched
    assert body["requested_by"] == "caleb"


def test_a_body_carrying_both_halves_is_rejected():
    """The disjointness property at the API layer.

    Enforced here, not merely by caller discipline: a single body that could
    set both halves would let one writer fabricate an acknowledgment.
    """
    sid = _session_id()
    resp = client.put(f"/api/v1/sessions/{sid}/control",
                      json={"requested": "pause", "reported": "paused"})
    assert resp.status_code == 400


def test_an_empty_body_is_rejected():
    sid = _session_id()
    assert client.put(f"/api/v1/sessions/{sid}/control", json={}).status_code == 400


def test_an_invalid_requested_value_is_rejected():
    sid = _session_id()
    resp = client.put(f"/api/v1/sessions/{sid}/control", json={"requested": "stop"})
    assert resp.status_code in (400, 422)


def test_an_invalid_reported_value_is_rejected():
    sid = _session_id()
    resp = client.put(f"/api/v1/sessions/{sid}/control", json={"reported": "halted"})
    assert resp.status_code in (400, 422)


def test_control_is_broadcast_on_every_accepted_write_including_a_heartbeat():
    """Broadcast on EVERY write, unlike audit which only fires on a change.

    The browser derives 'heard N seconds ago' from reported_at, so a
    heartbeat that did not broadcast would make a healthy controller look
    progressively deader.
    """
    from unittest.mock import AsyncMock, patch
    sid = _session_id()
    client.put(f"/api/v1/sessions/{sid}/control", json={"reported": "running"})

    with patch("api.routers.sessions.broadcast_to_session", new=AsyncMock()) as bc:
        client.put(f"/api/v1/sessions/{sid}/control", json={"reported": "running"})
        assert bc.await_count == 1
        assert bc.await_args.args[1]["event"] == "control_changed"


def test_unknown_session_returns_404():
    assert client.get("/api/v1/sessions/does-not-exist/control").status_code == 404


def test_post_audit_event_appends_a_readable_entry():
    """Step 4's deferred Task 9. ALchemist's audit surface was read-only from
    outside apart from one closed-enum lock POST, so a consumer had no way to
    put its own run events on the shared timeline.
    """
    sid = _session_id()
    resp = client.post(
        f"/api/v1/sessions/{sid}/audit/event",
        json={"entry_type": "cycle_started",
              "parameters": {"queue_item": "q1", "experiment": "exp-abc"},
              "notes": "controller"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["entry"]["entry_type"] == "cycle_started"

    entries = client.get(f"/api/v1/sessions/{sid}/audit",
                         params={"entry_type": "cycle_started"}).json()["entries"]
    assert len(entries) == 1
    assert entries[0]["parameters"]["queue_item"] == "q1"


def test_audit_event_accepts_a_type_outside_the_lock_enum():
    """The lock endpoint's Literal["data","model","acquisition"] is exactly
    what made the mirror unbuildable. This endpoint must not inherit it.
    """
    sid = _session_id()
    for entry_type in ("validity_hold", "queue_conflict_409", "objective_configured"):
        resp = client.post(f"/api/v1/sessions/{sid}/audit/event",
                           json={"entry_type": entry_type, "parameters": {}})
        assert resp.status_code == 200, entry_type


def test_audit_event_rejects_an_empty_entry_type():
    sid = _session_id()
    resp = client.post(f"/api/v1/sessions/{sid}/audit/event",
                       json={"entry_type": "", "parameters": {}})
    assert resp.status_code == 422
