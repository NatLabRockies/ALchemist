"""``suggest_next`` records what the API's state endpoint reports (A27).

``GET /sessions/{id}`` reads ``session._last_suggestion``; nothing set it,
so the state never showed a suggestion. ``suggest_next`` now sets it to the
suggestions as a list of dicts, ``_last_acquisition_value`` to the
optimiser's acquisition value (``None`` when the backend reports none) and
``_last_goal`` to the goal.
"""
import pytest

from alchemist_core import OptimizationSession


def _trained(backend):
    s = OptimizationSession()
    s.add_variable("x", "real", bounds=(0.0, 1.0))
    s.add_variable("y", "real", bounds=(0.0, 1.0))
    pts = [(0.1, 0.2), (0.4, 0.8), (0.7, 0.3), (0.9, 0.9), (0.3, 0.6), (0.5, 0.1),
           (0.2, 0.9), (0.8, 0.6), (0.6, 0.4), (0.35, 0.75)]
    for x, y in pts:
        s.add_experiment({"x": x, "y": y}, (x - 0.3) ** 2 + (y - 0.7) ** 2)
    s.train_model(backend=backend)
    return s


def test_fresh_session_has_the_attributes():
    s = OptimizationSession()
    assert s._last_suggestion is None
    assert s._last_acquisition_value is None
    assert s._last_goal is None


def test_botorch_records_suggestion_value_and_goal():
    s = _trained("botorch")
    s.suggest_next(strategy="LogEI", goal="minimize")
    assert s._last_suggestion == s.last_suggestions
    assert isinstance(s._last_suggestion, list) and isinstance(s._last_suggestion[0], dict)
    assert {"x", "y"} <= set(s._last_suggestion[0])
    assert isinstance(s._last_acquisition_value, float)
    assert s._last_goal == "minimize"


def test_batch_records_every_point():
    s = _trained("botorch")
    s.suggest_next(strategy="qEI", goal="maximize", n_suggestions=2)
    assert len(s._last_suggestion) == 2
    assert s._last_goal == "maximize"


def test_sklearn_records_suggestion_with_no_value():
    s = _trained("sklearn")
    s.suggest_next(strategy="EI", goal="maximize")
    assert s._last_suggestion == s.last_suggestions
    assert s._last_acquisition_value is None
