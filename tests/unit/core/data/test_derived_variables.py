"""Unit tests for SearchSpace derived variable methods."""
import pytest
from alchemist_core.data.search_space import SearchSpace


def _make_space():
    """Return a SearchSpace with two tunable variables."""
    ss = SearchSpace()
    ss.add_variable("T", "real", min=200.0, max=320.0)
    ss.add_variable("P", "real", min=20.0, max=80.0)
    return ss


def test_add_derived_variable_stores_entry():
    ss = _make_space()
    ss.add_derived_variable("T_squared", lambda row: row["T"] ** 2, input_cols=["T"])
    assert len(ss.derived_variables) == 1
    entry = ss.derived_variables[0]
    assert entry["name"] == "T_squared"
    assert entry["input_cols"] == ["T"]
    assert callable(entry["func"])


def test_add_derived_variable_does_not_add_to_tunable_vars():
    ss = _make_space()
    ss.add_derived_variable("T_squared", lambda row: row["T"] ** 2, input_cols=["T"])
    assert "T_squared" not in ss.get_variable_names()
    assert len(ss.variables) == 2  # only the two tunable vars


def test_add_derived_variable_rejects_duplicate_name():
    ss = _make_space()
    ss.add_derived_variable("feat", lambda row: 1.0, input_cols=["T"])
    with pytest.raises(ValueError, match="already registered"):
        ss.add_derived_variable("feat", lambda row: 2.0, input_cols=["T"])


def test_add_derived_variable_rejects_tunable_name():
    ss = _make_space()
    with pytest.raises(ValueError, match="already exists as a tunable variable"):
        ss.add_derived_variable("T", lambda row: row["T"] ** 2, input_cols=["T"])


def test_register_derived_variable_replaces_func():
    ss = _make_space()
    ss.add_derived_variable("feat", func=None, input_cols=["T"])
    assert ss.derived_variables[0]["func"] is None

    ss.register_derived_variable("feat", lambda row: row["T"] * 2)
    assert callable(ss.derived_variables[0]["func"])


def test_register_derived_variable_raises_for_unknown_name():
    ss = _make_space()
    with pytest.raises(ValueError, match="No derived variable"):
        ss.register_derived_variable("nonexistent", lambda row: 1.0)


def test_has_derived_variables():
    ss = _make_space()
    assert ss.has_derived_variables() is False
    ss.add_derived_variable("feat", lambda row: 1.0, input_cols=["T"])
    assert ss.has_derived_variables() is True


def test_get_derived_variable_names():
    ss = _make_space()
    ss.add_derived_variable("a", lambda row: 1.0, input_cols=["T"])
    ss.add_derived_variable("b", lambda row: 2.0, input_cols=["P"])
    assert ss.get_derived_variable_names() == ["a", "b"]


def test_derived_variables_to_dict_excludes_func():
    ss = _make_space()
    ss.add_derived_variable("feat", lambda row: 1.0, input_cols=["T"], description="my feat")
    result = ss.derived_variables_to_dict()
    assert result == [{"name": "feat", "input_cols": ["T"], "description": "my feat"}]
    assert "func" not in result[0]


def test_add_derived_variable_stub_sets_func_to_none():
    ss = _make_space()
    ss.add_derived_variable_stub("feat", input_cols=["T"], description="stub")
    assert ss.derived_variables[0]["func"] is None
    assert ss.derived_variables[0]["name"] == "feat"
