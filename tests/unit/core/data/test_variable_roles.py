"""Variable kind, role and change difficulty on SearchSpace."""
import json

import pytest

from alchemist_core.data.search_space import (
    SearchSpace, changes_of, kind_of, role_of,
)


def _space():
    s = SearchSpace()
    s.add_variable("x1", "real", min=0, max=10)
    s.add_variable("c", "categorical", values=["a", "b", "c"], changes="hard")
    s.add_variable("h1", "real", min=0, max=1, changes="very_hard")
    s.add_variable("k", "categorical", values=["p", "q"], role="observed")
    s.add_variable("z", "real", role="observed")
    s.add_variable("z_old", "context")
    return s


def test_defaults_are_controllable_and_easy_and_not_stored():
    s = SearchSpace()
    s.add_variable("x1", "real", min=0, max=10)
    assert s.variables[0] == {"name": "x1", "type": "real", "min": 0, "max": 10}
    assert role_of(s.variables[0]) == "controllable"
    assert changes_of(s.variables[0]) == "easy"


def test_non_default_values_are_stored_sparsely():
    s = _space()
    by = {v["name"]: v for v in s.variables}
    assert by["c"]["changes"] == "hard" and "role" not in by["c"]
    assert by["k"]["role"] == "observed" and "changes" not in by["k"]
    assert by["z_old"] == {"name": "z_old", "type": "context"}


def test_helpers_read_legacy_context_as_observed_real():
    v = {"name": "z_old", "type": "context"}
    assert role_of(v) == "observed"
    assert kind_of(v) == "real"
    assert changes_of(v) is None


def test_observed_variables_have_no_dimension_and_sort_last():
    s = _space()
    assert s.get_dimension_names() == ["x1", "c", "h1"]
    assert len(s.skopt_dimensions) == 3
    assert s.get_variable_names() == ["x1", "c", "h1", "k", "z", "z_old"]
    assert s.get_observed_variable_names() == ["k", "z", "z_old"]
    assert s.get_context_variable_names() == ["k", "z", "z_old"]
    assert s.get_controllable_variable_names() == ["x1", "c", "h1"]
    assert s.get_tunable_variable_names() == ["x1", "c", "h1"]


def test_observed_categorical_is_a_categorical_variable():
    s = _space()
    assert s.get_categorical_variables() == ["c", "k"]


def test_variables_by_changes():
    s = _space()
    assert [v["name"] for v in s.get_variables_by_changes("easy")] == ["x1"]
    assert [v["name"] for v in s.get_variables_by_changes("hard")] == ["c"]
    assert [v["name"] for v in s.get_variables_by_changes("very_hard")] == ["h1"]
    with pytest.raises(ValueError, match="changes"):
        s.get_variables_by_changes("medium")


@pytest.mark.parametrize("kwargs, match", [
    ({"role": "measured"}, "role"),
    ({"changes": "medium"}, "changes"),
    ({"role": "observed", "changes": "hard"}, "observed"),
])
def test_invalid_role_or_changes_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SearchSpace().add_variable("x1", "real", min=0, max=1, **kwargs)


def test_context_cannot_be_declared_controllable():
    with pytest.raises(ValueError, match="observed"):
        SearchSpace().add_variable("z", "context", role="controllable")


def test_observed_real_bounds_are_optional_but_paired():
    s = SearchSpace()
    s.add_variable("z", "real", role="observed")
    s.add_variable("w", "real", role="observed", min=0, max=5)
    s.add_variable("v", "real", role="observed", min=None, max=None)
    assert "min" not in s.variables[0] and s.variables[1]["max"] == 5
    assert "min" not in s.variables[2]
    with pytest.raises(ValueError, match="both"):
        SearchSpace().add_variable("u", "real", role="observed", min=0)


def test_observed_categorical_still_validates_values():
    with pytest.raises(ValueError, match="at least 1"):
        SearchSpace().add_variable("k", "categorical", values=[], role="observed")


@pytest.mark.parametrize("reserved", ["WholePlot", "Subplot", "RunOrder"])
def test_plot_column_names_are_reserved(reserved):
    with pytest.raises(ValueError, match="reserved"):
        SearchSpace().add_variable(reserved, "real", min=0, max=1)


def test_round_trip_through_dict_and_json(tmp_path):
    s = _space()
    data = json.loads(json.dumps(s.to_dict()))
    s2 = SearchSpace().from_dict(data)
    assert s2.to_dict() == s.to_dict()
    assert s2.get_dimension_names() == ["x1", "c", "h1"]


def test_replace_variable_can_change_role_and_changes():
    s = _space()
    s.replace_variable("x1", "real", min=0, max=10, changes="hard")
    assert changes_of(s.variables[0]) == "hard"
    s.replace_variable("x1", "real", role="observed")
    assert s.get_dimension_names() == ["c", "h1"]
    assert s.get_observed_variable_names()[0] == "x1"


@pytest.mark.parametrize("method", ["lhs", "full_factorial", "gsd"])
def test_doe_never_samples_observed_variables(method):
    from alchemist_core.utils.doe import generate_initial_design
    s = _space()
    pts = generate_initial_design(s, method=method, n_points=6, random_seed=1)
    assert pts and set(pts[0]) == {"x1", "c", "h1"}


def test_classical_design_ignores_an_observed_categorical():
    from alchemist_core.utils.doe import generate_initial_design
    s = SearchSpace()
    s.add_variable("x1", "real", min=0, max=1)
    s.add_variable("x2", "real", min=0, max=1)
    s.add_variable("k", "categorical", values=["p", "q"], role="observed")
    pts = generate_initial_design(s, method="ccd")
    assert pts and set(pts[0]) == {"x1", "x2"}


def test_session_summary_reports_role_and_changes():
    from alchemist_core.session import OptimizationSession
    sess = OptimizationSession()
    sess.add_variable("x1", "real", min=0, max=1)
    sess.add_variable("c", "categorical", values=["a", "b"], changes="hard")
    sess.add_variable("z", "real", role="observed")
    rows = {r["name"]: r for r in sess.get_search_space_summary()["variables"]}
    assert (rows["x1"]["role"], rows["x1"]["changes"]) == ("controllable", "easy")
    assert (rows["c"]["role"], rows["c"]["changes"]) == ("controllable", "hard")
    assert (rows["z"]["role"], rows["z"]["changes"]) == ("observed", None)


def test_bounds_exports_skip_observed_variables():
    s = _space()
    assert set(s.to_botorch_bounds()) == {"x1", "h1"}
    assert set(s.to_ax_space()) == {"x1", "c", "h1"}


def test_botorch_suggestion_with_an_unbounded_observed_variable():
    import numpy as np
    from alchemist_core.session import OptimizationSession
    sess = OptimizationSession()
    sess.add_variable("x1", "real", min=0, max=1)
    sess.add_variable("x2", "real", min=0, max=1)
    sess.add_variable("z", "real", role="observed")
    rng = np.random.default_rng(0)
    for _ in range(8):
        x1, x2 = rng.random(2)
        sess.add_experiment(
            {"x1": x1, "x2": x2, "z": float(rng.random())},
            output=float(-(x1 - 0.5) ** 2 - (x2 - 0.5) ** 2),
        )
    sess.train_model(backend="botorch")
    out = sess.suggest_next(strategy="EI", context={"z": 0.5})
    assert len(out) == 1
    assert {"x1", "x2"} <= set(out.columns)
