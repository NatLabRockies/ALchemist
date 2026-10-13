# tests/unit/core/data/test_split_plot.py
"""Split-plot design generation by nesting existing DoE methods."""
from collections import Counter

import pytest

from alchemist_core.data.search_space import SearchSpace
from alchemist_core.utils.split_plot import (
    DESIGN_REASON, generate_split_plot_design,
)


def _space(three_levels=False):
    s = SearchSpace()
    s.add_variable("x1", "real", min=0, max=10)
    s.add_variable("x2", "real", min=0, max=10)
    s.add_variable("c", "categorical", values=["a", "b", "c"], changes="hard")
    s.add_variable("h1", "real", min=0, max=1, changes="very_hard" if three_levels else "hard")
    s.add_variable("z", "real", role="observed")
    return s


TWO = {"hard": {"method": "full_factorial", "n_levels": 2}, "easy": {"method": "lhs", "n_points": 5}}


def test_row_count_ids_and_columns():
    pts, info = generate_split_plot_design(_space(), TWO, replicates={"hard": 2}, random_seed=7)
    # full factorial over c (3 levels) x h1 (2 levels) = 6 settings, x2 replicates = 12 plots
    assert info["n_plots"] == {"hard": 12}
    assert len(pts) == 12 * 5 == info["n_runs"]
    assert set(pts[0]) == {"x1", "x2", "c", "h1", "WholePlot", "RunOrder", "Reason"}
    assert [p["RunOrder"] for p in pts] == list(range(1, len(pts) + 1))
    assert {p["Reason"] for p in pts} == {DESIGN_REASON}


def test_whole_plots_are_contiguous_numbered_in_run_order_and_constant_inside():
    pts, _ = generate_split_plot_design(_space(), TWO, replicates={"hard": 2}, random_seed=7)
    ids = [p["WholePlot"] for p in pts]
    assert ids == sorted(ids)
    assert ids[0] == 1 and ids[-1] == 12
    for wp in set(ids):
        rows = [p for p in pts if p["WholePlot"] == wp]
        assert len({(r["c"], r["h1"]) for r in rows}) == 1


def test_each_setting_appears_replicates_times():
    pts, _ = generate_split_plot_design(_space(), TWO, replicates={"hard": 2}, random_seed=7)
    per_plot = {p["WholePlot"]: (p["c"], p["h1"]) for p in pts}
    assert set(Counter(per_plot.values()).values()) == {2}


def test_crossed_reuses_one_easy_design_and_fresh_does_not():
    crossed, _ = generate_split_plot_design(_space(), TWO, easy_mode="crossed",
                                            order={"easy": "as_generated"}, random_seed=1)
    fresh, _ = generate_split_plot_design(_space(), TWO, easy_mode="fresh",
                                          order={"easy": "as_generated"}, random_seed=1)

    def easy_sets(pts):
        return {wp: tuple((r["x1"], r["x2"]) for r in pts if r["WholePlot"] == wp)
                for wp in {p["WholePlot"] for p in pts}}

    assert len(set(easy_sets(crossed).values())) == 1
    assert len(set(easy_sets(fresh).values())) > 1


def test_seed_reproducibility():
    a, _ = generate_split_plot_design(_space(), TWO, random_seed=11)
    b, _ = generate_split_plot_design(_space(), TWO, random_seed=11)
    assert a == b


def test_as_generated_order_keeps_whole_plot_design_order():
    pts, _ = generate_split_plot_design(_space(), TWO, order={"hard": "as_generated"}, random_seed=2)
    first_rows = [next(p for p in pts if p["WholePlot"] == wp) for wp in range(1, 7)]
    from alchemist_core.utils.doe import generate_initial_design
    sub = SearchSpace()
    sub.add_variable("c", "categorical", values=["a", "b", "c"])
    sub.add_variable("h1", "real", min=0, max=1)
    expected = generate_initial_design(sub, method="full_factorial", n_levels=2, n_center=0)
    assert [(r["c"], r["h1"]) for r in first_rows] == [(e["c"], e["h1"]) for e in expected]


def test_unreplicated_whole_plots_warn():
    _, info = generate_split_plot_design(_space(), TWO, random_seed=3)
    assert any("whole-plot error cannot be estimated" in w for w in info["warnings"])
    _, info2 = generate_split_plot_design(_space(), TWO, replicates={"hard": 2}, random_seed=3)
    assert not any("cannot be estimated" in w for w in info2["warnings"])


def test_three_levels_nest_and_number_subplots():
    designs = {
        "very_hard": {"method": "full_factorial", "n_levels": 2},
        "hard": {"method": "full_factorial"},
        "easy": {"method": "sobol", "n_points": 4},
    }
    pts, info = generate_split_plot_design(_space(three_levels=True), designs, random_seed=4)
    assert info["n_plots"] == {"very_hard": 2, "hard": 6}
    assert len(pts) == 2 * 3 * 4
    assert {"WholePlot", "Subplot"} <= set(pts[0])
    for wp in (1, 2):
        rows = [p for p in pts if p["WholePlot"] == wp]
        assert len({r["h1"] for r in rows}) == 1
        assert len({r["Subplot"] for r in rows}) == 3


@pytest.mark.parametrize("designs, match", [
    ({"easy": {"method": "lhs", "n_points": 3}}, "exactly one entry per change level"),
    ({"hard": {"method": "full_factorial"}}, "exactly one entry per change level"),
    ({**TWO, "very_hard": {"method": "lhs", "n_points": 2}}, "exactly one entry per change level"),
    ({"hard": {"method": "optimal", "n_points": 4}, "easy": TWO["easy"]}, "do not support method 'optimal'"),
    ({"hard": {"n_levels": 2}, "easy": TWO["easy"]}, "needs a 'method'"),
])
def test_invalid_design_specs_raise(designs, match):
    with pytest.raises(ValueError, match=match):
        generate_split_plot_design(_space(), designs)


def test_space_without_a_hard_variable_points_to_generate_initial_design():
    s = SearchSpace()
    s.add_variable("x1", "real", min=0, max=1)
    with pytest.raises(ValueError, match="generate_initial_design"):
        generate_split_plot_design(s, {"easy": {"method": "lhs", "n_points": 3}})


def test_run_cap():
    big = {"hard": {"method": "full_factorial", "n_levels": 5}, "easy": {"method": "random", "n_points": 1000}}
    with pytest.raises(ValueError, match="5000"):
        generate_split_plot_design(_space(), big, replicates={"hard": 3})


def test_session_method_returns_points_and_info():
    from alchemist_core.session import OptimizationSession
    s = OptimizationSession()
    s.add_variable("x1", "real", min=0, max=10)
    s.add_variable("c", "categorical", values=["a", "b"], changes="hard")
    pts, info = s.generate_split_plot_design(
        designs={"hard": {"method": "full_factorial"}, "easy": {"method": "lhs", "n_points": 3}},
        replicates={"hard": 2}, random_seed=0,
    )
    assert len(pts) == 2 * 2 * 3 and info["n_runs"] == 12


def _categorical_only_space():
    s = SearchSpace()
    s.add_variable("x1", "real", min=0, max=10)
    s.add_variable("c", "categorical", values=["a", "b", "c"], changes="hard")
    return s


CAT_ONLY = {"hard": {"method": "full_factorial"}, "easy": {"method": "lhs", "n_points": 2}}


def test_categorical_only_level_has_one_whole_plot_per_level_and_warns():
    pts, info = generate_split_plot_design(_categorical_only_space(), CAT_ONLY, random_seed=5)
    per_plot = {p["WholePlot"]: p["c"] for p in pts}
    assert sorted(per_plot.values()) == ["a", "b", "c"]
    assert info["n_plots"] == {"hard": 3}
    assert any("whole-plot error cannot be estimated" in w for w in info["warnings"])


def test_explicit_n_center_is_honored():
    designs = {"hard": {"method": "full_factorial", "n_center": 1}, "easy": CAT_ONLY["easy"]}
    _, info = generate_split_plot_design(_categorical_only_space(), designs, random_seed=5)
    assert info["n_plots"] == {"hard": 4}
