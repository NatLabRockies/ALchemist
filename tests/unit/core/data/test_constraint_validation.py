"""add_constraint validates that coefficient variables are numeric.

Without this, a categorical in a constraint fails much later and far away,
inside filter_feasible, as float('some_string').
"""

import pytest

from alchemist_core.data.search_space import SearchSpace


def _space():
    s = SearchSpace()
    s.add_variable("x1", "real", min=0.0, max=10.0)
    s.add_variable("i1", "integer", min=0, max=5)
    s.add_variable("d1", "discrete", allowed_values=[1.0, 2.0, 4.0])
    s.add_variable("cat", "categorical", values=["a", "b"])
    s.add_variable("ctx", "context")
    return s


def test_categorical_coefficient_raises_at_registration():
    s = _space()
    with pytest.raises(ValueError, match="not numeric"):
        s.add_constraint("inequality", {"x1": 1.0, "cat": 1.0}, rhs=5.0)


def test_context_coefficient_raises_at_registration():
    s = _space()
    with pytest.raises(ValueError, match="not numeric"):
        s.add_constraint("inequality", {"ctx": 1.0}, rhs=5.0)


def test_error_names_the_offending_variable_and_its_type():
    s = _space()
    with pytest.raises(ValueError) as exc:
        s.add_constraint("inequality", {"cat": 1.0}, rhs=5.0)
    assert "cat" in str(exc.value)
    assert "categorical" in str(exc.value)


def test_bad_constraint_is_not_registered():
    s = _space()
    with pytest.raises(ValueError):
        s.add_constraint("inequality", {"cat": 1.0}, rhs=5.0)
    assert s.constraints == []


def test_all_numeric_types_are_accepted():
    s = _space()
    s.add_constraint("inequality", {"x1": 1.0, "i1": 1.0, "d1": 1.0}, rhs=20.0)
    assert len(s.constraints) == 1


def test_unknown_variable_still_raises_the_original_error():
    s = _space()
    with pytest.raises(ValueError, match="not found in search space"):
        s.add_constraint("inequality", {"nope": 1.0}, rhs=5.0)
