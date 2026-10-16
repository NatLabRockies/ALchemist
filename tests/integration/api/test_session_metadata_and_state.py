"""Session metadata over REST (W27, W11, A7) and the fields of `/state` (A27).

`GET /sessions/{id}` had no metadata, `author` was silently dropped by
`PATCH /metadata`, `POST /sessions` ignored its body, and `/state` returned
`model_backend`, `last_suggestion` and `last_acquisition_value` as null
whatever the session had done.
"""
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.services.session_store import session_store

client = TestClient(app)


@pytest.fixture
def sid():
    s = client.post("/api/v1/sessions", json={}).json()["session_id"]
    yield s
    client.delete(f"/api/v1/sessions/{s}")


def test_create_accepts_name_description_tags_and_author():
    r = client.post("/api/v1/sessions", json={"name": "Campaign 7", "description": "RCC",
                                              "tags": ["rcc", "qEI"], "author": "Caleb"})
    assert r.status_code == 201, r.text
    s = r.json()["session_id"]
    try:
        meta = client.get(f"/api/v1/sessions/{s}/metadata").json()
        assert (meta["name"], meta["description"], meta["tags"], meta["author"]) == \
            ("Campaign 7", "RCC", ["rcc", "qEI"], "Caleb")
    finally:
        client.delete(f"/api/v1/sessions/{s}")


def test_create_still_works_without_a_body():
    r = client.post("/api/v1/sessions")
    assert r.status_code == 201, r.text
    client.delete(f"/api/v1/sessions/{r.json()['session_id']}")


def test_info_includes_the_metadata(sid):
    assert client.patch(f"/api/v1/sessions/{sid}/metadata",
                        json={"name": "n2", "tags": ["t"]}).status_code == 200
    info = client.get(f"/api/v1/sessions/{sid}").json()
    assert info["metadata"]["name"] == "n2"
    assert info["metadata"]["tags"] == ["t"]
    assert set(info["metadata"]) >= {"name", "description", "tags", "author", "last_modified"}


def test_author_is_saved_and_returned(sid):
    r = client.patch(f"/api/v1/sessions/{sid}/metadata", json={"name": "n2", "author": "Caleb"})
    assert r.status_code == 200, r.text
    assert r.json()["author"] == "Caleb"
    assert client.get(f"/api/v1/sessions/{sid}/metadata").json()["author"] == "Caleb"
    assert client.get(f"/api/v1/sessions/{sid}/export").json()["metadata"]["author"] == "Caleb"
    # a PATCH without author leaves it alone
    client.patch(f"/api/v1/sessions/{sid}/metadata", json={"description": "d"})
    assert client.get(f"/api/v1/sessions/{sid}/metadata").json()["author"] == "Caleb"


def test_state_reports_the_backend_and_the_last_suggestion(sid):
    assert client.post(f"/api/v1/sessions/{sid}/variables",
                       json={"name": "x", "type": "real", "min": 0, "max": 1}).status_code == 200
    rows = [{"inputs": {"x": i / 10}, "output": float(i)} for i in range(6)]
    assert client.post(f"/api/v1/sessions/{sid}/experiments/batch",
                       json={"experiments": rows}).status_code == 200
    state = client.get(f"/api/v1/sessions/{sid}/state").json()
    assert state["model_trained"] is False and state["model_backend"] is None
    assert client.post(f"/api/v1/sessions/{sid}/model/train",
                       json={"backend": "sklearn", "kernel": "RBF"}).status_code == 200
    state = client.get(f"/api/v1/sessions/{sid}/state").json()
    assert state["model_trained"] is True
    assert state["model_backend"] == "sklearn"
    assert state["last_suggestion"] is None and state["last_acquisition_value"] is None

    # The core sets these after a suggestion; the route reads them defensively.
    session = session_store.get(sid)
    session._last_suggestion = {"x": 0.42}
    session._last_acquisition_value = 0.025
    state = client.get(f"/api/v1/sessions/{sid}/state").json()
    assert state["last_suggestion"] == {"x": 0.42}
    assert state["last_acquisition_value"] == pytest.approx(0.025)
