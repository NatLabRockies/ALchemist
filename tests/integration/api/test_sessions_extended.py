
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock, patch, mock_open
from api.main import app
from api.services import session_store
from api.dependencies import get_session
from alchemist_core.session import OptimizationSession
from alchemist_core.audit_log import AuditLog, AuditEntry
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace

client = TestClient(app)

class TestSessionsExtended:
    
    @pytest.fixture
    def mock_session_store(self):
        with patch('api.routers.sessions.session_store') as mock:
            yield mock

    @pytest.fixture
    def mock_session_dependency(self):
        mock_session = MagicMock()
        app.dependency_overrides[get_session] = lambda: mock_session
        yield mock_session
        app.dependency_overrides = {}

    def test_create_session(self, mock_session_store):
        # Setup mock
        mock_session_store.create.return_value = "test_session_id"
        mock_session_store.get_info.return_value = {
            "created_at": "2023-01-01T00:00:00",
            "expires_at": "2023-01-02T00:00:00"
        }
        
        response = client.post("/api/v1/sessions")
        
        assert response.status_code == 201
        data = response.json()
        assert data["session_id"] == "test_session_id"
        assert "created_at" in data
        assert "expires_at" in data

    def test_get_session_info_success(self, mock_session_store):
        mock_session_store.get_info.return_value = {
            "session_id": "test_id",
            "created_at": "2023-01-01T00:00:00",
            "expires_at": "2023-01-02T00:00:00",
            "last_accessed": "2023-01-01T01:00:00",
            "n_variables": 2,
            "n_experiments": 10,
            "model_trained": True,
            "search_space": {"variables": []},
            "data": {
                "n_rows": 10,
                "n_experiments": 10,
                "has_data": True
            },
            "model": {
                "type": "sklearn",
                "backend": "sklearn",
                "hyperparameters": {},
                "metrics": {},
                "is_trained": True
            }
        }
        
        response = client.get("/api/v1/sessions/test_id")
        
        assert response.status_code == 200
        assert response.json()["session_id"] == "test_id"

    def test_get_session_info_not_found(self, mock_session_store):
        mock_session_store.get_info.return_value = None
        
        response = client.get("/api/v1/sessions/non_existent")
        
        assert response.status_code == 404

    def test_get_session_state(self, mock_session_dependency):
        # Mock session object
        mock_session_dependency.search_space.variables = [1, 2]
        mock_session_dependency.experiment_manager.df = [1, 2, 3]
        mock_session_dependency.model = MagicMock()
        mock_session_dependency._last_suggestion = {"x": 1}
        
        response = client.get("/api/v1/sessions/test_id/state")
        
        assert response.status_code == 200
        data = response.json()
        assert data["n_variables"] == 2
        assert data["n_experiments"] == 3
        assert data["model_trained"] is True
        assert data["last_suggestion"] == {"x": 1}

    def test_delete_session(self, mock_session_store):
        mock_session_store.delete.return_value = True
        
        response = client.delete("/api/v1/sessions/test_id")
        
        assert response.status_code == 204
        mock_session_store.delete.assert_called_with("test_id")

    def test_delete_session_not_found(self, mock_session_store):
        mock_session_store.delete.return_value = False
        
        response = client.delete("/api/v1/sessions/test_id")
        
        assert response.status_code == 404

    def test_extend_session_removed(self):
        """The legacy /extend endpoint has been removed; should return 404/405."""
        response = client.post("/api/v1/sessions/test_id/extend?hours=48")
        assert response.status_code in (404, 405)

    def test_extend_session_not_found_removed(self):
        """The legacy /extend endpoint has been removed."""
        response = client.post("/api/v1/sessions/test_id/extend")
        assert response.status_code in (404, 405)

    def test_save_session_server_side(self, mock_session_store):
        mock_session_store.persist_session_to_disk.return_value = True
        
        response = client.post("/api/v1/sessions/test_id/save")
        
        assert response.status_code == 200
        assert response.json()["message"] == "Session persisted to server storage"

    def test_save_session_server_side_fail(self, mock_session_store):
        mock_session_store.persist_session_to_disk.return_value = False
        
        response = client.post("/api/v1/sessions/test_id/save")
        
        assert response.status_code == 404

    def test_export_session(self, mock_session_store):
        mock_session_store.export_session.return_value = '{"session": "data"}'
        
        response = client.get("/api/v1/sessions/test_id/export")
        
        assert response.status_code == 200
        assert response.content == b'{"session": "data"}'
        assert "attachment" in response.headers["content-disposition"]

    def test_export_session_not_found(self, mock_session_store):
        mock_session_store.export_session.return_value = None
        
        response = client.get("/api/v1/sessions/test_id/export")
        
        assert response.status_code == 404

    def test_import_session(self, mock_session_store):
        mock_session_store.import_session.return_value = "new_id"
        mock_session_store.get_info.return_value = {
            "created_at": "time",
            "expires_at": "time"
        }
        
        files = {'file': ('session.json', '{"data": "test"}', 'application/json')}
        response = client.post("/api/v1/sessions/import", files=files)
        
        assert response.status_code == 201
        assert response.json()["session_id"] == "new_id"

    def test_import_session_fail(self, mock_session_store):
        mock_session_store.import_session.return_value = None
        
        files = {'file': ('session.json', '{"data": "test"}', 'application/json')}
        response = client.post("/api/v1/sessions/import", files=files)
        
        assert response.status_code == 400

    def test_get_metadata(self, mock_session_dependency):
        mock_session_dependency.metadata.session_id = "test_id"
        mock_session_dependency.metadata.name = "Test Session"
        # Use strings for datetime fields if the response model expects strings
        # Or ensure datetime objects are serialized correctly
        mock_session_dependency.metadata.created_at = datetime.now().isoformat()
        mock_session_dependency.metadata.last_modified = datetime.now().isoformat()
        mock_session_dependency.metadata.description = "Desc"
        mock_session_dependency.metadata.tags = ["tag1"]
        
        response = client.get("/api/v1/sessions/test_id/metadata")
        
        assert response.status_code == 200
        assert response.json()["name"] == "Test Session"

    def test_update_metadata(self, mock_session_dependency):
        mock_session_dependency.metadata.session_id = "test_id"
        # Setup initial state
        mock_session_dependency.metadata.name = "Old Name"
        mock_session_dependency.metadata.created_at = datetime.now().isoformat()
        mock_session_dependency.metadata.last_modified = datetime.now().isoformat()
        mock_session_dependency.metadata.description = "Old Desc"
        mock_session_dependency.metadata.tags = []
        
        # Setup update behavior
        def update_side_effect(name=None, description=None, tags=None):
            if name: mock_session_dependency.metadata.name = name
            
        mock_session_dependency.update_metadata.side_effect = update_side_effect
        
        payload = {"name": "New Name", "description": "New Desc"}
        response = client.patch("/api/v1/sessions/test_id/metadata", json=payload)
        
        assert response.status_code == 200
        mock_session_dependency.update_metadata.assert_called_with(name="New Name", description="New Desc", tags=None)

    def test_get_audit_log(self, mock_session_dependency):
        entry = MagicMock()
        entry.to_dict.return_value = {
            "timestamp": "2023-01-01",
            "entry_type": "test",
            "details": {},
            "user": "user",
            "parameters": {},
            "hash": "hash",
            "notes": "notes"
        }
        mock_session_dependency.audit_log.get_entries.return_value = [entry]
        
        response = client.get("/api/v1/sessions/test_id/audit")
        
        assert response.status_code == 200
        assert len(response.json()["entries"]) == 1

    def test_lock_decision(self, mock_session_dependency):
        # Mock the return value of lock_data
        mock_entry = MagicMock()
        mock_entry.to_dict.return_value = {
            "timestamp": "2025-01-01T00:00:00",
            "entry_type": "data",
            "parameters": {},
            "hash": "test_hash",
            "notes": "test notes"
        }
        mock_session_dependency.lock_data.return_value = mock_entry
    
        payload = {
            "lock_type": "data",
            "notes": "Locked data"
        }
        response = client.post("/api/v1/sessions/test_id/audit/lock", json=payload)
    
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["entry"]["hash"] == "test_hash"
        mock_session_dependency.lock_data.assert_called()

    def test_lock_decision_acquisition_requires_fields(self, mock_session_dependency):
        payload = {"lock_type": "acquisition", "notes": "missing params"}
        response = client.post("/api/v1/sessions/test_id/audit/lock", json=payload)
        assert response.status_code == 400
        assert "requires" in response.json()["detail"].lower()

    def test_lock_decision_invalid_type(self, mock_session_dependency):
        payload = {"lock_type": "invalid", "notes": "nope"}
        response = client.post("/api/v1/sessions/test_id/audit/lock", json=payload)
        assert response.status_code == 422
        assert "input should be" in response.json()["detail"].lower()

    def test_lock_decision_value_error(self, mock_session_dependency):
        mock_session_dependency.lock_data.side_effect = ValueError("problem")
        payload = {"lock_type": "data", "notes": "bad"}
        response = client.post("/api/v1/sessions/test_id/audit/lock", json=payload)
        assert response.status_code == 400
        assert "problem" in response.json()["detail"]

    def test_export_audit_markdown(self, mock_session_dependency):
        mock_session_dependency.export_audit_markdown.return_value = "# Audit"
        response = client.get("/api/v1/sessions/test_id/audit/export")
        assert response.status_code == 200
        assert response.text == "# Audit"
        assert "attachment" in response.headers["content-disposition"].lower()

    def test_download_session_success(self, mock_session_dependency):
        def save_session_side_effect(path: str):
            Path(path).write_text('{"foo": "bar"}')

        mock_session_dependency.metadata = SimpleNamespace(name="My Session", description="", tags=[])
        mock_session_dependency.save_session.side_effect = save_session_side_effect

        response = client.get("/api/v1/sessions/test_id/download")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")
        assert "My_Session.json" in response.headers["content-disposition"]
        assert response.content == b'{"foo": "bar"}'

    def test_download_session_failure(self, mock_session_dependency):
        mock_session_dependency.metadata = SimpleNamespace(name="Broken", description="", tags=[])
        mock_session_dependency.save_session.side_effect = RuntimeError("boom")

        response = client.get("/api/v1/sessions/test_id/download")

        assert response.status_code == 500
        assert "failed to export" in response.json()["detail"].lower()

    def test_upload_session_success(self, mock_session_store):
        uploaded_session_id = "session-uploaded"
        mock_session_store._sessions = {}

        def create_side_effect(*args, **kwargs):
            mock_session_store._sessions[uploaded_session_id] = {
                "session": MagicMock(),
                "created_at": datetime.now(),
                "last_accessed": datetime.now(),
                "expires_at": datetime.now()
            }
            return uploaded_session_id

        mock_session_store.create.side_effect = create_side_effect
        mock_session_store.get_info.return_value = {
            "created_at": "2025-01-01T00:00:00",
            "expires_at": "2025-01-02T00:00:00"
        }

        loaded_session = MagicMock()
        loaded_session.metadata = SimpleNamespace(session_id="original")

        with patch("api.routers.sessions.OptimizationSession.load_session", return_value=loaded_session) as mock_load:
            files = {"file": ("session.json", b"{}", "application/json")}
            response = client.post("/api/v1/sessions/upload", files=files)

        assert response.status_code == 201
        assert response.json()["session_id"] == uploaded_session_id
        mock_load.assert_called_once()
        mock_session_store._save_to_disk.assert_called_with(uploaded_session_id)

    def test_upload_session_failure(self, mock_session_store):
        mock_session_store._sessions = {}
        mock_session_store.create.return_value = "session-fail"

        with patch("api.routers.sessions.OptimizationSession.load_session", side_effect=RuntimeError("bad file")):
            files = {"file": ("session.json", b"{}", "application/json")}
            response = client.post("/api/v1/sessions/upload", files=files)

        assert response.status_code == 400
        assert "failed to upload" in response.json()["detail"].lower()

    def test_lock_session_success(self, mock_session_store):
        mock_session_store.lock_session.return_value = {
            "locked": True,
            "locked_by": "controller",
            "locked_at": "2025-01-01T00:00:00",
            "lock_token": "token-123"
        }

        with patch("api.routers.sessions.broadcast_to_session", new=AsyncMock()) as mock_broadcast:
            response = client.post(
                "/api/v1/sessions/test_id/lock",
                json={"locked_by": "controller", "client_id": "client-1"}
            )

        assert response.status_code == 200
        assert response.json()["lock_token"] == "token-123"
        mock_session_store.lock_session.assert_called_with(session_id="test_id", locked_by="controller", client_id="client-1")
        mock_broadcast.assert_awaited_once()

    def test_lock_session_not_found(self, mock_session_store):
        mock_session_store.lock_session.side_effect = KeyError("missing")

        with patch("api.routers.sessions.broadcast_to_session", new=AsyncMock()) as mock_broadcast:
            response = client.post(
                "/api/v1/sessions/test_id/lock",
                json={"locked_by": "controller"}
            )

        assert response.status_code == 404
        mock_broadcast.assert_not_called()

    def test_lock_session_failure(self, mock_session_store):
        mock_session_store.lock_session.side_effect = Exception("boom")

        with patch("api.routers.sessions.broadcast_to_session", new=AsyncMock()) as mock_broadcast:
            response = client.post(
                "/api/v1/sessions/test_id/lock",
                json={"locked_by": "controller"}
            )

        assert response.status_code == 500
        mock_broadcast.assert_not_called()

    def test_unlock_session_success(self, mock_session_store):
        mock_session_store.unlock_session.return_value = {
            "locked": False,
            "locked_by": None,
            "locked_at": None,
            "lock_token": None
        }

        with patch("api.routers.sessions.broadcast_to_session", new=AsyncMock()) as mock_broadcast:
            response = client.delete("/api/v1/sessions/test_id/lock?lock_token=abc")

        assert response.status_code == 200
        mock_session_store.unlock_session.assert_called_with(session_id="test_id", lock_token="abc")
        mock_broadcast.assert_awaited_once()

    def test_unlock_session_forbidden(self, mock_session_store):
        mock_session_store.unlock_session.side_effect = ValueError("bad token")

        with patch("api.routers.sessions.broadcast_to_session", new=AsyncMock()):
            response = client.delete("/api/v1/sessions/test_id/lock?lock_token=abc")

        assert response.status_code == 403

    def test_unlock_session_not_found(self, mock_session_store):
        mock_session_store.unlock_session.side_effect = KeyError("missing")

        with patch("api.routers.sessions.broadcast_to_session", new=AsyncMock()):
            response = client.delete("/api/v1/sessions/test_id/lock")

        assert response.status_code == 404

    def test_unlock_session_failure(self, mock_session_store):
        mock_session_store.unlock_session.side_effect = Exception("boom")

        with patch("api.routers.sessions.broadcast_to_session", new=AsyncMock()):
            response = client.delete("/api/v1/sessions/test_id/lock")

        assert response.status_code == 500

    def test_get_lock_status_success(self, mock_session_store):
        mock_session_store.get_lock_status.return_value = {
            "locked": True,
            "locked_by": "controller",
            "locked_at": "2025-01-01T00:00:00",
            "lock_token": "token"
        }

        response = client.get("/api/v1/sessions/test_id/lock")

        assert response.status_code == 200
        assert response.json()["locked"] is True

    def test_get_lock_status_not_found(self, mock_session_store):
        mock_session_store.get_lock_status.side_effect = KeyError("missing")

        response = client.get("/api/v1/sessions/test_id/lock")

        assert response.status_code == 404

    def test_get_lock_status_failure(self, mock_session_store):
        mock_session_store.get_lock_status.side_effect = Exception("boom")

        response = client.get("/api/v1/sessions/test_id/lock")

        assert response.status_code == 500