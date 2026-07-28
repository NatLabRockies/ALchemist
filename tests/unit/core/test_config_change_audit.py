from alchemist_core.audit_log import AuditLog


def test_log_config_change_records_old_new_and_iteration():
    log = AuditLog()
    entry = log.log_config_change(
        component="model",
        old={"kernel": "Matern"},
        new={"kernel": "RBF"},
        iteration=12,
    )
    assert entry.entry_type == "config_changed"
    assert entry.parameters["component"] == "model"
    assert entry.parameters["old"] == {"kernel": "Matern"}
    assert entry.parameters["new"] == {"kernel": "RBF"}
    assert entry.parameters["iteration"] == 12
    assert entry.timestamp  # ISO timestamp auto-set
    assert log.get_entries("config_changed") == [entry]


def test_log_config_change_iteration_optional():
    log = AuditLog()
    entry = log.log_config_change(component="acquisition", old={}, new={"strategy": "qEI"})
    assert "iteration" not in entry.parameters
