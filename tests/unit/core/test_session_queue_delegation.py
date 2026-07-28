from alchemist_core.session import OptimizationSession
from alchemist_core.queue import ExperimentQueue


def _session():
    s = OptimizationSession()
    s.add_variable("x", "real", bounds=(0.0, 10.0))
    return s


def test_session_exposes_queue():
    s = _session()
    assert isinstance(s.queue, ExperimentQueue)


def test_add_staged_delegates_to_queue():
    s = _session()
    s.add_staged_experiment({"x": 1.0, "_reason": "EI"})
    items = s.queue.list()
    assert len(items) == 1
    assert items[0].inputs == {"x": 1.0}
    assert items[0].reason == "EI"


def test_get_staged_experiments_backcompat_shape():
    s = _session()
    s.add_staged_experiment({"x": 1.0, "_reason": "EI"})
    staged = s.get_staged_experiments()
    assert staged[0]["x"] == 1.0
    assert staged[0]["_reason"] == "EI"


def test_move_staged_to_experiments_completes_pending():
    s = _session()
    s.add_staged_experiment({"x": 1.0})
    s.add_staged_experiment({"x": 2.0})
    n = s.move_staged_to_experiments(outputs=[0.5, 0.6], reason="EI")
    assert n == 2
    assert len(s.experiment_manager.df) == 2
    assert len(s.queue.pending_items()) == 0
    assert len(s.queue.list(status="done")) == 2


def test_clear_staged_experiments_pending_only():
    s = _session()
    s.add_staged_experiment({"x": 1.0})
    n = s.clear_staged_experiments()
    assert n == 1
    assert len(s.queue.list()) == 0
