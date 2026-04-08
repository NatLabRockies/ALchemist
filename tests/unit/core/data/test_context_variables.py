"""Unit tests for SearchSpace context variable support."""
import pytest
from alchemist_core.data.search_space import SearchSpace


def _make_space():
    ss = SearchSpace()
    ss.add_variable("T", "real", min=200.0, max=320.0)
    ss.add_variable("P", "real", min=20.0, max=80.0)
    return ss


# --- add_variable("context") ---

def test_add_context_variable_stores_entry():
    ss = _make_space()
    ss.add_variable("humidity", "context")
    ctx = [v for v in ss.variables if v["name"] == "humidity"]
    assert len(ctx) == 1
    assert ctx[0] == {"name": "humidity", "type": "context"}


def test_context_variable_not_in_skopt_dimensions():
    ss = _make_space()
    ss.add_variable("humidity", "context")
    names_in_skopt = [d.name for d in ss.skopt_dimensions]
    assert "humidity" not in names_in_skopt
    assert len(ss.skopt_dimensions) == 2  # only T and P


def test_context_variable_not_in_categorical_or_discrete():
    ss = _make_space()
    ss.add_variable("humidity", "context")
    assert "humidity" not in ss.categorical_variables
    assert "humidity" not in ss.discrete_variables


def test_add_context_variable_rejects_duplicate_name():
    ss = _make_space()
    ss.add_variable("humidity", "context")
    with pytest.raises(ValueError, match="already registered"):
        ss.add_variable("humidity", "context")


def test_add_context_variable_rejects_existing_tunable_name():
    ss = _make_space()
    with pytest.raises(ValueError, match="already registered"):
        ss.add_variable("T", "context")


# --- get_variable_names() tunable-first ordering ---

def test_get_variable_names_tunable_first():
    """Tunable vars appear before context vars regardless of registration order."""
    ss = SearchSpace()
    ss.add_variable("humidity", "context")   # context registered first
    ss.add_variable("T", "real", min=200.0, max=320.0)
    ss.add_variable("P", "real", min=20.0, max=80.0)
    names = ss.get_variable_names()
    assert names.index("T") < names.index("humidity")
    assert names.index("P") < names.index("humidity")


def test_get_variable_names_no_context_unchanged():
    """Without context vars, ordering is unchanged (registration order)."""
    ss = _make_space()
    assert ss.get_variable_names() == ["T", "P"]


# --- get_tunable_variable_names() ---

def test_get_tunable_variable_names_excludes_context():
    ss = _make_space()
    ss.add_variable("humidity", "context")
    assert ss.get_tunable_variable_names() == ["T", "P"]


def test_get_tunable_variable_names_all_tunable():
    ss = _make_space()
    assert ss.get_tunable_variable_names() == ["T", "P"]


# --- get_context_variable_names() ---

def test_get_context_variable_names():
    ss = _make_space()
    ss.add_variable("humidity", "context")
    ss.add_variable("lot", "context")
    assert ss.get_context_variable_names() == ["humidity", "lot"]


def test_get_context_variable_names_empty():
    ss = _make_space()
    assert ss.get_context_variable_names() == []


# --- to_botorch_bounds() excludes context ---

def test_to_botorch_bounds_excludes_context():
    ss = _make_space()
    ss.add_variable("humidity", "context")
    bounds = ss.to_botorch_bounds()
    assert "humidity" not in bounds
    assert "T" in bounds
    assert "P" in bounds


# --- from_dict() round-trip ---

def test_from_dict_restores_context_variable():
    ss = _make_space()
    ss.add_variable("humidity", "context")
    data = ss.variables  # list of dicts including context
    ss2 = SearchSpace()
    ss2.from_dict(data)
    assert ss2.get_context_variable_names() == ["humidity"]
    assert "humidity" not in [d.name for d in ss2.skopt_dimensions]


def test_from_dict_context_not_in_skopt():
    ss = SearchSpace()
    ss.from_dict([
        {"name": "T", "type": "real", "min": 200.0, "max": 320.0},
        {"name": "humidity", "type": "context"},
    ])
    assert len(ss.skopt_dimensions) == 1
    assert ss.skopt_dimensions[0].name == "T"
