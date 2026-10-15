"""A session with a recorded training call but no model keeps the call on disk.

A reload whose retrain fails leaves ``model`` unset with the training call
still recorded. Saving that session must not erase the call, or the next
restart has nothing to replay and training falls back to the default model.
"""
import json

import pytest

from alchemist_core import OptimizationSession

CALL = {
    "backend": "botorch",
    "kernel": "Matern",
    "kernel_params": {"nu": 2.5},
    "input_transform_type": "normalize",
    "output_transform_type": "standardize",
}


def _session_without_model():
    s = OptimizationSession()
    s.add_variable("x1", "real", min=0, max=10)
    s.add_experiment({"x1": 1.0}, 1.0)
    s.add_experiment({"x1": 2.0}, 2.0)
    s._last_training_config = dict(CALL)
    assert s.model is None
    return s


def test_the_training_call_is_written_without_a_model(tmp_path):
    path = tmp_path / "s.json"
    _session_without_model().save_session(str(path))
    cfg = json.loads(path.read_text())["model_config"]
    assert cfg["training_config"] == CALL
    assert cfg["backend"] == "botorch"
    assert cfg["fitted"] is False


def test_the_training_call_round_trips_without_a_model(tmp_path):
    path = tmp_path / "s.json"
    _session_without_model().save_session(str(path))
    loaded = OptimizationSession.load_session(str(path), retrain_on_load=False)
    assert loaded.get_last_training_config() == CALL
    assert loaded.model is None


def test_a_load_replays_a_call_saved_without_a_model(tmp_path, monkeypatch):
    path = tmp_path / "s.json"
    _session_without_model().save_session(str(path))
    calls = []
    monkeypatch.setattr(OptimizationSession, "train_model",
                        lambda self, *a, **kw: calls.append(kw))
    OptimizationSession.load_session(str(path))
    assert calls == [CALL]


def test_a_trained_session_is_not_marked_unfitted(tmp_path):
    s = OptimizationSession()
    s.add_variable("x1", "real", min=0, max=10)
    for i in range(6):
        s.add_experiment({"x1": float(i)}, float(i) ** 0.5)
    s.train_model(backend="sklearn", kernel="Matern")
    path = tmp_path / "s.json"
    s.save_session(str(path))
    assert json.loads(path.read_text())["model_config"].get("fitted", True) is True


def test_no_training_call_and_no_model_writes_no_model_config(tmp_path):
    s = OptimizationSession()
    s.add_variable("x1", "real", min=0, max=10)
    path = tmp_path / "s.json"
    s.save_session(str(path))
    assert "model_config" not in json.loads(path.read_text())
