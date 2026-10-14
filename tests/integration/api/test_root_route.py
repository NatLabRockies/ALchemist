"""`/` opens the web UI in a browser and keeps answering API info to everything else.

`alchemist-web` tells people to open http://localhost:8000, but `/` used to
return the API-info JSON even when the frontend was built, so the browser
showed JSON instead of the app.
"""
import pytest
from fastapi.testclient import TestClient

import alchemist_core
import api.main as main

BROWSER = {"accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}


@pytest.fixture
def built_frontend(tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text("<!doctype html><title>ALchemist app</title>")
    monkeypatch.setattr(main, "static_dir", tmp_path)
    return tmp_path


@pytest.fixture
def no_frontend(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "static_dir", tmp_path / "missing")


def test_browser_gets_the_web_ui(built_frontend):
    response = TestClient(main.app).get("/", headers=BROWSER)
    assert response.status_code == 200
    assert "ALchemist app" in response.text


def test_scripts_still_get_api_info(built_frontend):
    response = TestClient(main.app).get("/", headers={"accept": "application/json"})
    assert response.json()["version"] == alchemist_core.__version__


def test_default_accept_gets_api_info(built_frontend):
    assert TestClient(main.app).get("/").json()["name"] == "ALchemist API"


def test_browser_gets_api_info_when_the_frontend_is_not_built(no_frontend):
    response = TestClient(main.app).get("/", headers=BROWSER)
    assert response.json()["name"] == "ALchemist API"


def test_api_info_has_its_own_route(built_frontend):
    response = TestClient(main.app).get("/api", headers=BROWSER)
    assert response.json() == {
        "name": "ALchemist API",
        "version": alchemist_core.__version__,
        "docs": "/api/docs",
        "health": "/health",
    }


def test_health_is_unchanged(built_frontend):
    assert TestClient(main.app).get("/health", headers=BROWSER).json()["status"] == "healthy"
