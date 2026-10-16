"""Small fixes: `/download` removes its temp file, `broadcast_to_session`
survives a connection change mid-broadcast, and the Ollama model listing
only fetches http(s) URLs without userinfo.
"""
import asyncio
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

import api.routers.sessions as sessions_router
from api.main import app
from api.routers import websocket as ws

client = TestClient(app)


@pytest.fixture
def sid():
    s = client.post("/api/v1/sessions", json={"name": "Camp aign"}).json()["session_id"]
    yield s
    client.delete(f"/api/v1/sessions/{s}")


def test_download_removes_its_temp_file(sid, monkeypatch):
    made = []
    real = tempfile.NamedTemporaryFile

    def recording(*args, **kwargs):
        f = real(*args, **kwargs)
        made.append(f.name)
        return f

    monkeypatch.setattr(sessions_router.tempfile, "NamedTemporaryFile", recording)
    r = client.get(f"/api/v1/sessions/{sid}/download")
    assert r.status_code == 200, r.text
    assert r.headers["content-disposition"].endswith("Camp_aign.json")
    assert r.json()["metadata"]["name"] == "Camp aign"
    assert made and not os.path.exists(made[0])


def test_broadcast_survives_a_connection_leaving_mid_send():
    ws.active_connections.clear()

    class Leaver:
        def __init__(self, session_id):
            self.session_id = session_id
            self.sent = []

        async def send_json(self, payload):
            # As a disconnect during the await: the set changes size.
            ws.active_connections[self.session_id].discard(other)
            self.sent.append(payload)

    class Other:
        async def send_json(self, payload):
            pass

    other = Other()
    leaver = Leaver("s")
    ws.active_connections["s"] = {leaver, other}
    try:
        asyncio.run(ws.broadcast_to_session("s", {"event": "x"}))
    finally:
        ws.active_connections.clear()
    assert leaver.sent == [{"event": "x"}]


@pytest.mark.parametrize("base_url", [
    "file:///etc/passwd",
    "ftp://127.0.0.1",
    "http://user:pw@127.0.0.1:11434",
    "127.0.0.1:11434",
    "http:///api",
])
def test_ollama_models_refuses_non_http_urls(base_url):
    r = client.get("/api/v1/llm/ollama/models", params={"base_url": base_url})
    assert r.status_code == 400, r.text


def test_ollama_models_accepts_an_http_url():
    # port 9 refuses connections at once; the route reports it, not a 400
    r = client.get("/api/v1/llm/ollama/models", params={"base_url": "http://127.0.0.1:9"})
    assert r.status_code == 200, r.text
    assert r.json()["models"] == [] and r.json()["error"]
