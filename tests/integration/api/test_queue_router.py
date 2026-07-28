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
    assert items[1]["reason"] == "PI"


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


def test_start_illegal_transition_409(sid):
    r = client.post(f"/api/v1/sessions/{sid}/experiments/queue",
                    json={"items": [{"inputs": {"x": 1.0}}]})
    item_id = r.json()["items"][0]["id"]
    client.post(f"/api/v1/sessions/{sid}/experiments/queue/{item_id}/complete",
                json={"outputs": [1.0]})
    # starting a done item is illegal
    assert client.post(f"/api/v1/sessions/{sid}/experiments/queue/{item_id}/start").status_code == 409


def test_complete_multi_output_rejected_400(sid):
    # Multi-objective completion is not supported end-to-end; must be rejected
    # rather than silently corrupting the dataset.
    r = client.post(f"/api/v1/sessions/{sid}/experiments/queue",
                    json={"items": [{"inputs": {"x": 1.0}}]})
    item_id = r.json()["items"][0]["id"]
    rc = client.post(f"/api/v1/sessions/{sid}/experiments/queue/{item_id}/complete",
                     json={"outputs": [0.9, 0.2]})
    assert rc.status_code == 400
    # item was not completed
    assert client.get(f"/api/v1/sessions/{sid}/experiments/queue/{item_id}").json()["status"] == "pending"


def test_complete_multi_noise_rejected_400(sid):
    r = client.post(f"/api/v1/sessions/{sid}/experiments/queue",
                    json={"items": [{"inputs": {"x": 1.0}}]})
    item_id = r.json()["items"][0]["id"]
    rc = client.post(f"/api/v1/sessions/{sid}/experiments/queue/{item_id}/complete",
                     json={"outputs": [0.9], "noise": [0.1, 0.2]})
    assert rc.status_code == 400


def test_transition_on_deleted_item_is_404_not_500(sid):
    # Simulates the check-then-act race: the item is gone by the time the
    # transition runs. The endpoint must map the queue's KeyError to 404.
    r = client.post(f"/api/v1/sessions/{sid}/experiments/queue",
                    json={"items": [{"inputs": {"x": 1.0}}]})
    item_id = r.json()["items"][0]["id"]
    assert client.delete(f"/api/v1/sessions/{sid}/experiments/queue/{item_id}").status_code == 200
    # now start/complete/fail/delete on the vanished id -> 404, never 500
    assert client.post(f"/api/v1/sessions/{sid}/experiments/queue/{item_id}/start").status_code == 404
    assert client.post(f"/api/v1/sessions/{sid}/experiments/queue/{item_id}/complete",
                       json={"outputs": [1.0]}).status_code == 404
    assert client.post(f"/api/v1/sessions/{sid}/experiments/queue/{item_id}/fail",
                       json={"error": "x"}).status_code == 404
    assert client.delete(f"/api/v1/sessions/{sid}/experiments/queue/{item_id}").status_code == 404
