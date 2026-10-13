"""Contract for the published OpenAPI schema."""
import alchemist_core
from api.main import app

SKIP = {("/", "get"), ("/health", "get")}


def _operations():
    schema = app.openapi()
    for path, ops in schema["paths"].items():
        for method, op in ops.items():
            if (path, method) not in SKIP:
                yield path, method, op


def test_schema_version_matches_package():
    assert app.openapi()["info"]["version"] == alchemist_core.__version__


def test_root_route_reports_package_version():
    from fastapi.testclient import TestClient
    assert TestClient(app).get("/").json()["version"] == alchemist_core.__version__


def test_every_operation_declares_error_responses():
    missing = []
    for path, method, op in _operations():
        for code in ("400", "404", "500"):
            ref = op.get("responses", {}).get(code, {}).get("content", {}).get("application/json", {}).get("schema", {}).get("$ref", "")
            if not ref.endswith("/ErrorResponse"):
                missing.append(f"{method.upper()} {path} {code}")
    assert missing == []


def test_every_operation_has_a_description():
    missing = [f"{m.upper()} {p}" for p, m, op in _operations() if not op.get("description", "").strip()]
    assert missing == []


def test_spa_catch_all_not_in_schema():
    assert not any("{full_path" in p for p in app.openapi()["paths"])


CONFLICT_OPERATIONS = [
    ("/api/v1/sessions/{session_id}/experiments/queue/{item_id}/start", "post"),
    ("/api/v1/sessions/{session_id}/experiments/queue/{item_id}/complete", "post"),
    ("/api/v1/sessions/{session_id}/experiments/queue/{item_id}/fail", "post"),
    ("/api/v1/sessions/{session_id}/experiments/queue/{item_id}", "delete"),
    ("/api/v1/sessions/{session_id}/experiments/staged/complete", "post"),
    ("/api/v1/sessions/{session_id}/experiments/record-batch", "post"),
]


def test_conflict_operations_declare_409():
    paths = app.openapi()["paths"]
    missing = []
    for path, method in CONFLICT_OPERATIONS:
        ref = (paths[path][method].get("responses", {}).get("409", {})
               .get("content", {}).get("application/json", {}).get("schema", {}).get("$ref", ""))
        if not ref.endswith("/ErrorResponse"):
            missing.append(f"{method.upper()} {path}")
    assert missing == []
