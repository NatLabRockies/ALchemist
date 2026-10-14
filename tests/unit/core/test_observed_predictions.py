"""Observed variables are conditioned on, never gridded or optimized.

Predictions over the space (find_optimum, prediction grids, plot slices) hold each
observed variable at one value: the one the caller supplies, else the latest
experiment's, else an error that says to supply it.
"""
import numpy as np
import pandas as pd
import pytest

from alchemist_core import OptimizationSession

# Fixture artifact, not behaviour under test: a third-party module emits a
# SyntaxWarning on first import.
pytestmark = pytest.mark.filterwarnings("ignore::SyntaxWarning")


def _session(z_bounds=None, legacy_context=False, n=16, seed=0):
    rng = np.random.default_rng(seed)
    s = OptimizationSession()
    s.add_variable("x1", "real", min=0, max=10)
    s.add_variable("x2", "real", min=0, max=10)
    if legacy_context:
        s.add_variable("z", "context")
    elif z_bounds is None:
        s.add_variable("z", "real", role="observed")
    else:
        s.add_variable("z", "real", min=z_bounds[0], max=z_bounds[1], role="observed")
    if not legacy_context:
        s.add_variable("k", "categorical", values=["p", "q"], role="observed")
    for i in range(n):
        last = i == n - 1
        x1, x2 = rng.uniform(0, 10, 2)
        z = 27.5 if last else rng.uniform(20, 30)
        k = "q" if last else "pq"[i % 2]
        y = -(x1 - 6) ** 2 - (x2 - 3) ** 2 + 0.5 * z + (2.0 if k == "q" else 0.0)
        row = {"x1": x1, "x2": x2, "z": z}
        if not legacy_context:
            row["k"] = k
        s.add_experiment(row, float(y))
    s.train_model(backend="botorch")
    return s


@pytest.fixture(scope="module")
def trained():
    return _session()


def test_find_optimum_conditions_on_latest_row(trained):
    best = trained.find_optimum()["x_opt"].iloc[0]
    assert best["z"] == 27.5
    assert best["k"] == "q"
    assert abs(best["x1"] - 6) < 1.5


def test_find_optimum_uses_supplied_observed_values(trained):
    best = trained.find_optimum(observed={"z": 21.0, "k": "p"})["x_opt"].iloc[0]
    assert best["z"] == 21.0
    assert best["k"] == "p"


def test_bounded_observed_variable_is_not_optimized():
    s = _session(z_bounds=(0, 100))
    best = s.find_optimum()["x_opt"].iloc[0]
    assert best["z"] == 27.5


def test_legacy_context_variable_no_longer_breaks_find_optimum():
    s = _session(legacy_context=True)
    assert s.find_optimum()["x_opt"].iloc[0]["z"] == 27.5


def test_no_experiments_and_no_value_raises_naming_the_variable():
    s = OptimizationSession()
    s.add_variable("x1", "real", min=0, max=1)
    s.add_variable("z", "real", role="observed")
    with pytest.raises(ValueError) as err:
        s._observed_values_for_prediction()
    assert "'z'" in str(err.value) and "observed" in str(err.value)
    assert s._observed_values_for_prediction({"z": 3}) == {"z": 3.0}


def test_undeclared_categorical_label_is_refused(trained):
    with pytest.raises(ValueError, match=r"\['p', 'q'\]"):
        trained.find_optimum(observed={"k": "nope"})


def test_non_finite_observed_number_is_refused(trained):
    with pytest.raises(ValueError, match="finite"):
        trained.find_optimum(observed={"z": float("nan")})


def test_controllable_variable_in_observed_is_refused(trained):
    with pytest.raises(ValueError, match="controllable"):
        trained.find_optimum(observed={"x1": 3.0})


def test_misspelled_observed_name_is_refused_not_ignored(trained):
    with pytest.raises(ValueError) as err:
        trained.find_optimum(observed={"Z": 22.0})
    assert "'Z'" in str(err.value) and "['z', 'k']" in str(err.value)


def test_observed_is_ignored_with_a_warning_for_multi_objective(tmp_path, caplog):
    s = OptimizationSession()
    s.add_variable("x1", "real", min=0, max=10)
    s.add_variable("z", "real", role="observed")
    rng = np.random.default_rng(1)
    x1 = rng.uniform(0, 10, 10)
    csv = tmp_path / "mo.csv"
    pd.DataFrame({"x1": x1, "z": 3.0 + rng.normal(0, 0.1, 10),
                  "a": x1 + rng.normal(0, 0.1, 10), "b": -x1 + rng.normal(0, 0.1, 10)}).to_csv(csv, index=False)
    s.load_data(str(csv), target_columns=["a", "b"])
    s.train_model(backend="botorch")
    with caplog.at_level("WARNING", logger="alchemist_core.session"):
        out = s.find_optimum(goal=["maximize", "maximize"], observed={"z": 3.0})
    assert out["n_pareto"] > 0
    assert any("observed= is ignored" in r.message for r in caplog.records)
    with pytest.raises(ValueError, match="Unknown observed"):
        s.find_optimum(goal=["maximize", "maximize"], observed={"nope": 1})


def test_plot_slice_fills_an_unbounded_observed_variable_with_the_latest_value(trained, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seen = []
    original = trained.predict

    def spy(df):
        seen.append(df.copy())
        return original(df)

    monkeypatch.setattr(trained, "predict", spy)
    fig = trained.plot_slice("x1", fixed_values={"x2": 3.0}, n_points=5)
    plt.close(fig)
    assert seen and (seen[0]["z"] == 27.5).all() and (seen[0]["k"] == "q").all()

    seen.clear()
    fig = trained.plot_slice("x1", fixed_values={"x2": 3.0, "z": 22.0, "k": "p"}, n_points=5)
    plt.close(fig)
    assert (seen[0]["z"] == 22.0).all() and (seen[0]["k"] == "p").all()


def test_slice_default_per_kind(trained):
    obs = {"z": 27.5, "k": "q"}
    by_name = {v["name"]: v for v in trained.search_space.variables}
    assert trained._slice_default(by_name["x1"], obs) == 5.0
    assert trained._slice_default(by_name["z"], obs) == 27.5
    assert trained._slice_default(by_name["k"], obs) == "q"
    cat = {"name": "c", "type": "categorical", "values": ["a", "b"]}
    assert trained._slice_default(cat, obs) == "a"
    disc = {"name": "d", "type": "discrete", "allowed_values": [1, 2, 4, 8]}
    assert trained._slice_default(disc, obs) == 4


def test_prediction_grid_holds_observed_columns_constant(trained):
    grid = trained._generate_prediction_grid(100)
    assert list(grid.columns) == ["x1", "x2", "z", "k"]
    assert (grid["z"] == 27.5).all() and (grid["k"] == "q").all()
    assert len(grid[["x1", "x2"]].drop_duplicates()) == 100


def test_prediction_grid_accepts_observed_override(trained):
    grid = trained._generate_prediction_grid(25, observed={"z": 21.0, "k": "p"})
    assert (grid["z"] == 21.0).all() and (grid["k"] == "p").all()
    assert len(grid) == 25


def test_prediction_grid_uses_allowed_values_of_a_discrete_variable():
    s = OptimizationSession()
    s.add_variable("x1", "real", min=0, max=1)
    s.add_variable("d", "discrete", allowed_values=[1, 2, 4, 8])
    grid = s._generate_prediction_grid(100)
    assert sorted(grid["d"].unique()) == [1, 2, 4, 8]
    assert list(grid.columns) == ["x1", "d"]
