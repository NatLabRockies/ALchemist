"""load_session honours retrain_on_load in both call forms (A16).

Called on the class, ``self`` is the path and the second positional argument
lands in ``filepath``. The static branch used to read that argument as the
retrain flag only when it was positional, and ignored ``retrain_on_load=...``
by keyword, so every server load path that passed ``False`` retrained anyway.
"""
import json

import pytest

from alchemist_core import OptimizationSession


@pytest.fixture
def saved(tmp_path):
    """A saved session file that carries a model config, so a load could retrain."""
    s = OptimizationSession()
    s.add_variable("x1", "real", min=0, max=10)
    s.add_experiment({"x1": 1.0}, 1.0)
    s.add_experiment({"x1": 2.0}, 2.0)
    path = tmp_path / "s.json"
    s.save_session(str(path))
    data = json.loads(path.read_text())
    data["model_config"] = {"backend": "botorch", "kernel": "Matern", "kernel_params": {}}
    path.write_text(json.dumps(data))
    return str(path)


@pytest.fixture
def train_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(
        OptimizationSession, "train_model",
        lambda self, *args, **kwargs: calls.append(kwargs),
    )
    return calls


def test_class_form_keyword_false_does_not_retrain(saved, train_calls):
    loaded = OptimizationSession.load_session(saved, retrain_on_load=False)
    assert isinstance(loaded, OptimizationSession)
    assert train_calls == []


def test_class_form_positional_false_does_not_retrain(saved, train_calls):
    loaded = OptimizationSession.load_session(saved, False)
    assert isinstance(loaded, OptimizationSession)
    assert train_calls == []


def test_class_form_default_retrains(saved, train_calls):
    OptimizationSession.load_session(saved)
    assert len(train_calls) == 1
    assert train_calls[0]["backend"] == "botorch"


def test_class_form_explicit_true_retrains(saved, train_calls):
    OptimizationSession.load_session(saved, retrain_on_load=True)
    OptimizationSession.load_session(saved, True)
    assert len(train_calls) == 2


def test_instance_form_is_unchanged(saved, train_calls):
    target = OptimizationSession()
    target.load_session(saved, retrain_on_load=False)
    assert train_calls == []

    target.load_session(saved)
    assert len(train_calls) == 1
