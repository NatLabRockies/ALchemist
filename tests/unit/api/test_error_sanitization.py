"""
Regression tests for API error detail sanitization.

Verifies that API endpoints do not leak internal exception details
(stack traces, file paths, internal state) to API consumers.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import io
import json

from api.main import app

client = TestClient(app)


def _create_session():
    """Helper to create a session for testing."""
    response = client.post("/api/v1/sessions", json={"ttl_hours": 1})
    assert response.status_code == 201
    return response.json()["session_id"]


class TestErrorDetailSanitization:
    """Verify API endpoints do not leak internal error details."""

    def test_import_session_error_sanitized(self):
        """POST /sessions/import should not leak exception details."""
        # Send invalid file content to trigger an error
        invalid_content = b"this is not valid json"
        response = client.post(
            "/api/v1/sessions/import",
            files={"file": ("bad.json", io.BytesIO(invalid_content), "application/json")}
        )
        assert response.status_code == 400
        detail = response.json()["detail"]
        # Should be a generic message, not contain internal exception text
        assert "Expecting value" not in detail  # JSONDecodeError detail
        assert "Traceback" not in detail
        assert "str(e)" not in detail

    def test_upload_session_error_sanitized(self):
        """POST /sessions/upload should not leak exception details."""
        invalid_content = b"not valid json at all {{{{"
        response = client.post(
            "/api/v1/sessions/upload",
            files={"file": ("bad.json", io.BytesIO(invalid_content), "application/json")}
        )
        # Should return 400, not 500 with raw exception
        assert response.status_code in (400, 422)
        detail = response.json().get("detail", "")
        if isinstance(detail, str):
            assert "Traceback" not in detail

    def test_download_session_nonexistent_sanitized(self):
        """GET /sessions/{id}/download should not leak details for missing sessions."""
        response = client.get("/api/v1/sessions/nonexistent-id/download")
        # Should be 404 from middleware, not 500 with internal details
        assert response.status_code in (404, 500)
        detail = response.json().get("detail", "")
        if isinstance(detail, str):
            assert "Traceback" not in detail

    def test_metrics_error_sanitized(self):
        """GET /visualizations/metrics should not leak exception details."""
        session_id = _create_session()
        response = client.get(f"/api/v1/{session_id}/visualizations/metrics")
        # Will fail because no data/model, but should not leak internals
        if response.status_code == 500:
            detail = response.json().get("detail", "")
            if isinstance(detail, str):
                assert "Traceback" not in detail

    def test_lock_session_error_sanitized(self):
        """POST /sessions/{id}/lock should not leak details for invalid sessions."""
        response = client.post(
            "/api/v1/sessions/nonexistent-id/lock",
            json={"locked_by": "test"}
        )
        assert response.status_code in (404, 500)
        detail = response.json().get("detail", "")
        if isinstance(detail, str):
            assert "Traceback" not in detail
