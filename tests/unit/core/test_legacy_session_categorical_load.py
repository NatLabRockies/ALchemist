"""A saved session keeps every row on load, even a row whose categorical label is no longer declared."""
import json

import pytest

from alchemist_core.session import OptimizationSession


def _write_legacy_file(tmp_path):
    s = OptimizationSession()
    s.add_variable("x1", "real", min=0, max=10)
    s.add_variable("c", "categorical", values=["a", "b", "B"])
    labels = ["a", "b", "a", "B", "b"]
    for i, lab in enumerate(labels):
        s.add_experiment({"x1": float(i + 1), "c": lab}, float(i))
    path = tmp_path / "legacy.json"
    s.save_session(str(path))
    data = json.loads(path.read_text())
    edited = 0
    for var in data["search_space"]["variables"]:
        if var["name"] == "c":
            var["values"] = ["a", "b"]
            edited += 1
    assert edited == 1
    path.write_text(json.dumps(data))
    return path


def test_load_keeps_rows_with_undeclared_label(tmp_path, caplog):
    path = _write_legacy_file(tmp_path)
    with caplog.at_level("WARNING"):
        loaded = OptimizationSession.load_session(str(path), retrain_on_load=False)
    df = loaded.experiment_manager.df
    assert len(df) == 5
    assert list(df["c"]) == ["a", "b", "a", "B", "b"]
    assert not [r for r in caplog.records if "skipped" in r.getMessage()]


def test_training_on_kept_undeclared_label_still_raises(tmp_path):
    loaded = OptimizationSession.load_session(
        str(_write_legacy_file(tmp_path)), retrain_on_load=False
    )
    with pytest.raises(ValueError, match="B") as exc:
        loaded.train_model(backend="botorch")
    assert "c" in str(exc.value)


def test_add_experiment_still_rejects_undeclared_label():
    s = OptimizationSession()
    s.add_variable("x1", "real", min=0, max=10)
    s.add_variable("c", "categorical", values=["a", "b"])
    with pytest.raises(ValueError, match="B"):
        s.add_experiment({"x1": 1.0, "c": "B"}, 1.0)
    assert len(s.experiment_manager.df) == 0
