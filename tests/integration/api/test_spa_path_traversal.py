"""The SPA catch-all serves only files under the static root (NEW-1, A2).

`GET /..%2f..%2fsomething` used to resolve `..` against the static directory
and return any file the server user could read -- the LLM config with its API
keys, other users' session files, `/etc/passwd`. Anything outside the root now
falls through to the SPA index (or 404), and an unknown `/api/...` path is a
real 404, not a 200 with "Not Found" in the body.
"""
from fastapi.testclient import TestClient

import api.main as main


def _client():
    return TestClient(main.app)


def _frontend(tmp_path, monkeypatch):
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html><title>ALchemist app</title>")
    (static / "favicon.ico").write_bytes(b"icon")
    (tmp_path / "secret.json").write_text('{"api_key": "sk-secret"}')
    monkeypatch.setattr(main, "static_dir", static)
    return static


def test_a_file_under_the_static_root_is_served(tmp_path, monkeypatch):
    _frontend(tmp_path, monkeypatch)
    response = _client().get("/favicon.ico")
    assert response.status_code == 200
    assert response.content == b"icon"


def test_dot_dot_segments_cannot_escape_the_static_root(tmp_path, monkeypatch):
    _frontend(tmp_path, monkeypatch)
    for path in ("/..%2fsecret.json", "/../secret.json", "/assets/..%2f..%2fsecret.json",
                 "/..%2f..%2f..%2f..%2f..%2f..%2f..%2fetc%2fpasswd"):
        response = _client().get(path)
        assert "sk-secret" not in response.text, path
        assert "root:" not in response.text, path
        assert response.status_code in (200, 404), path
        if response.status_code == 200:
            assert "ALchemist app" in response.text, path


def test_absolute_paths_are_contained(tmp_path, monkeypatch):
    _frontend(tmp_path, monkeypatch)
    response = _client().get("/" + str(tmp_path / "secret.json").lstrip("/"))
    assert "sk-secret" not in response.text


def test_unknown_api_path_is_a_real_404(tmp_path, monkeypatch):
    _frontend(tmp_path, monkeypatch)
    response = _client().get("/api/v1/nonexistent")
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


def test_missing_index_is_a_404(tmp_path, monkeypatch):
    static = tmp_path / "empty"
    static.mkdir()
    monkeypatch.setattr(main, "static_dir", static)
    assert _client().get("/some/route").status_code == 404
