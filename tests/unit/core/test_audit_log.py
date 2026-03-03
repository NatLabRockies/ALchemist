"""
Test suite for audit log and session management functionality.
"""

import pytest
import json
import pandas as pd
from pathlib import Path
from alchemist_core.session import OptimizationSession
from alchemist_core.audit_log import AuditLog, SessionMetadata, AuditEntry


def test_session_metadata_creation():
    """Test creating session metadata."""
    metadata = SessionMetadata.create(
        name="Test Session",
        description="A test session",
        tags=["test", "demo"]
    )
    
    assert metadata.name == "Test Session"
    assert metadata.description == "A test session"
    assert metadata.tags == ["test", "demo"]
    assert metadata.session_id is not None
    assert metadata.created_at is not None


def test_audit_log_basic():
    """Test basic audit log functionality."""
    audit_log = AuditLog()
    assert len(audit_log) == 0
    
    # Lock data (new API: pass DataFrame and optional extra_parameters)
    df = pd.DataFrame([{"temp": 0}] * 10)
    entry = audit_log.lock_data(
        df,
        notes="",
        extra_parameters={"variables": [{"name": "temp", "type": "real"}], "data_hash": "abc123"}
    )
    
    assert len(audit_log) == 1
    assert entry.entry_type == "data_locked"
    assert entry.parameters["n_experiments"] == 10
    
    # Lock model
    entry = audit_log.lock_model(
        backend="sklearn",
        kernel="matern",
        hyperparameters={"length_scale": 0.5},
        cv_metrics={"rmse": 0.15, "r2": 0.92}
    )
    
    assert len(audit_log) == 2
    assert entry.entry_type == "model_locked"


def test_audit_log_export():
    """Test audit log export to dict and markdown."""
    audit_log = AuditLog()
    
    audit_log.lock_data(
        pd.DataFrame([{}] * 5),
        notes="",
        extra_parameters={"variables": [], "data_hash": "test123"}
    )

    # Export to dict (full export)
    data = audit_log.to_dict()
    assert isinstance(data, dict)
    assert len(data.get("entries", [])) == 1
    assert data["entries"][0]["entry_type"] == "data_locked"
    
    # Export to markdown
    md = audit_log.to_markdown()
    assert "# Optimization Audit Trail" in md
    assert "Experimental Data" in md or "experimental data" in md.lower()


def test_session_with_audit_log():
    """Test OptimizationSession with audit log integration."""
    session = OptimizationSession()
    
    # Update metadata
    session.update_metadata(
        name="Test Optimization",
        description="Testing audit log",
        tags=["test"]
    )
    
    assert session.metadata.name == "Test Optimization"
    assert len(session.audit_log) == 0
    
    # Add variables and data
    session.add_variable("temperature", "real", min=100, max=300)
    session.add_experiment({"temperature": 200}, output=85.0)
    
    # Lock data
    entry = session.lock_data(notes="Initial dataset")
    assert len(session.audit_log) == 1
    assert entry.notes == "Initial dataset"
    
    # Train model
    session.train_model(backend="sklearn", kernel="rbf")
    
    # Lock model
    entry = session.lock_model(notes="Good CV performance")
    assert len(session.audit_log) == 2
    assert entry.notes == "Good CV performance"


def test_session_save_load(tmp_path):
    """Test session save and load functionality."""
    # Create session with data
    session = OptimizationSession()
    session.update_metadata(name="Save Test", description="Testing save/load")
    session.add_variable("x", "real", min=0, max=10)
    session.add_variable("y", "integer", min=1, max=5)
    session.add_experiment({"x": 5.0, "y": 3}, output=42.0)
    session.lock_data(notes="Test data")
    
    # Save session
    filepath = tmp_path / "test_session.json"
    session.save_session(str(filepath))
    
    assert filepath.exists()
    
    # Verify JSON structure
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    assert data["version"] == "1.0.0"
    assert data["metadata"]["name"] == "Save Test"
    # Audit log exported as a dict with 'entries' list
    assert len(data["audit_log"].get("entries", [])) == 1
    assert len(data["search_space"]["variables"]) == 2
    assert data["experiments"]["n_total"] == 1
    
    # Load session
    loaded_session = OptimizationSession.load_session(str(filepath))
    
    assert loaded_session.metadata.name == "Save Test"
    assert loaded_session.metadata.description == "Testing save/load"
    assert len(loaded_session.search_space.variables) == 2
    assert len(loaded_session.experiment_manager.df) == 1
    assert len(loaded_session.audit_log) == 1


def test_lock_acquisition():
    """Test locking acquisition decisions."""
    session = OptimizationSession()
    session.add_variable("x", "real", min=0, max=1)
    session.add_experiment({"x": 0.5}, output=1.0)
    
    suggestions = [
        {"x": 0.75, "acquisition_value": 0.5}
    ]
    
    entry = session.lock_acquisition(
        strategy="EI",
        parameters={"xi": 0.01, "goal": "maximize"},
        suggestions=suggestions,
        notes="Top candidate"
    )
    
    assert entry.entry_type == "acquisition_locked"
    assert entry.parameters["strategy"] == "EI"
    assert len(entry.parameters["suggestions"]) == 1


def test_audit_log_get_methods():
    """Test audit log filtering and retrieval."""
    audit_log = AuditLog()
    
    # Add multiple entries (use DataFrame for lock_data)
    audit_log.lock_data(pd.DataFrame([{}] * 5), notes="", extra_parameters={"variables": [], "data_hash": "hash1"})
    audit_log.lock_model("sklearn", "rbf", {})
    audit_log.lock_data(pd.DataFrame([{}] * 10), notes="", extra_parameters={"variables": [], "data_hash": "hash2"})
    
    # Get all entries
    all_entries = audit_log.get_entries()
    assert len(all_entries) == 3
    
    # Get by type
    data_entries = audit_log.get_entries("data_locked")
    assert len(data_entries) == 2
    
    model_entries = audit_log.get_entries("model_locked")
    assert len(model_entries) == 1
    
    # Get latest
    latest_data = audit_log.get_latest("data_locked")
    assert latest_data.parameters["n_experiments"] == 10
    
    latest_acq = audit_log.get_latest("acquisition_locked")
    assert latest_acq is None


def test_export_audit_markdown():
    """Test markdown export from session."""
    session = OptimizationSession()
    session.add_variable("temp", "real", min=100, max=300)
    session.add_experiment({"temp": 200}, output=85.0)
    
    session.lock_data()
    session.train_model(backend="sklearn", kernel="rbf")
    session.lock_model()
    
    md = session.export_audit_markdown()
    
    assert "# Optimization Audit Trail" in md
    # Ensure major sections and iteration block are present
    assert ("model" in md or "Model" in md)
    assert ("metrics" in md.lower() or "Metrics" in md)
    assert "## Optimization Iterations" in md


class TestAuditLogWarningOnFailure:
    """Regression tests: audit log failures emit warnings, not silent debug/pass."""

    def test_initial_design_audit_failure_warns(self, caplog):
        """generate_initial_design() logs warning when audit logging fails."""
        import logging

        session = OptimizationSession()
        session.add_variable("x", "real", min=0, max=1)

        # Sabotage audit_log.lock_data to raise
        original = session.audit_log.lock_data
        session.audit_log.lock_data = _mock_raise(RuntimeError("audit boom"))

        with caplog.at_level(logging.WARNING, logger="alchemist_core.session"):
            points = session.generate_initial_design(method="random", n_points=3)

        # Design still succeeds
        assert len(points) == 3
        # Warning was logged (not silently swallowed)
        assert any("Failed to add initial design to audit log" in m for m in caplog.messages)
        assert any("audit boom" in m for m in caplog.messages)

        session.audit_log.lock_data = original

    def test_lock_model_transform_extraction_warns(self, caplog):
        """lock_model() logs warning when transform type extraction fails."""
        import logging

        session = OptimizationSession()
        session.add_variable("x", "real", min=0, max=1)
        session.add_experiment({"x": 0.5}, output=1.0)
        session.train_model(backend="sklearn", kernel="rbf")

        # Sabotage: make input_transform_type a property that raises
        original_model = session.model

        class BadModel:
            """Proxy that explodes on transform type access."""
            def __getattr__(self, name):
                if name == 'input_transform_type':
                    raise RuntimeError("transform boom")
                return getattr(original_model, name)

        session.model = BadModel()

        with caplog.at_level(logging.WARNING, logger="alchemist_core.session"):
            entry = session.lock_model(notes="test")

        # lock_model still succeeds
        assert entry.entry_type == "model_locked"
        assert any("Failed to extract model transform types for audit log" in m for m in caplog.messages)

        session.model = original_model

    def test_export_audit_markdown_metadata_warns(self, caplog):
        """export_audit_markdown() logs warning when metadata.to_dict() fails."""
        import logging

        session = OptimizationSession()

        # Sabotage metadata.to_dict
        original = session.metadata.to_dict
        session.metadata.to_dict = _mock_raise(RuntimeError("metadata boom"))

        with caplog.at_level(logging.WARNING, logger="alchemist_core.session"):
            md = session.export_audit_markdown()

        # Export still succeeds (with None metadata)
        assert "# Optimization Audit Trail" in md
        assert any("Failed to get session metadata for audit markdown" in m for m in caplog.messages)

        session.metadata.to_dict = original


def _mock_raise(exc):
    """Return a callable that always raises the given exception."""
    def _raise(*args, **kwargs):
        raise exc
    return _raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
