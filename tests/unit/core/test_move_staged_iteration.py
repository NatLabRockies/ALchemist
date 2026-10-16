"""``move_staged_to_experiments(iteration=)`` takes effect (A3).

The docstring promised one iteration for the whole batch; the parameter was
accepted and ignored, so each row took the next free number.
"""
from alchemist_core import OptimizationSession


def _staged():
    s = OptimizationSession()
    s.add_variable("x", "real", bounds=(0.0, 1.0))
    for i in range(3):
        s.add_experiment({"x": 0.1 * i}, float(i))
    s.add_staged_experiment({"x": 0.7})
    s.add_staged_experiment({"x": 0.9})
    return s


def test_iteration_is_applied_to_every_row():
    s = _staged()
    s.move_staged_to_experiments([7.0, 9.0], iteration=99)
    assert s.experiment_manager.df["Iteration"].tolist()[-2:] == [99, 99]
    assert s._completion_iterations == {}


def test_without_iteration_rows_take_the_next_free_numbers():
    s = _staged()
    s.move_staged_to_experiments([7.0, 9.0])
    assert s.experiment_manager.df["Iteration"].tolist()[-2:] == [3, 4]
