"""Constraints reduced per plot, and reference runs inside every plot."""
import pytest

from alchemist_core.data.search_space import SearchSpace
from alchemist_core.utils.constrained_region import InfeasibleRegionError
from alchemist_core.utils.split_plot import (
    REFERENCE_REASON, generate_split_plot_design,
)

DESIGNS = {"hard": {"method": "full_factorial", "n_levels": 3}, "easy": {"method": "lhs", "n_points": 8}}


def _space():
    s = SearchSpace()
    s.add_variable("x1", "real", min=0, max=10)
    s.add_variable("x2", "real", min=0, max=10)
    s.add_variable("h1", "real", min=0, max=4, changes="hard")
    return s


def test_easy_only_constraint_holds_everywhere():
    s = _space()
    s.add_constraint("inequality", {"x1": 1.0, "x2": 1.0}, 8.0)
    pts, _ = generate_split_plot_design(s, DESIGNS, random_seed=0)
    assert all(p["x1"] + p["x2"] <= 8.0 + 1e-6 for p in pts)


def test_mixed_constraint_is_reduced_per_plot():
    s = _space()
    s.add_constraint("inequality", {"h1": 1.0, "x1": 1.0}, 6.0)
    pts, _ = generate_split_plot_design(s, DESIGNS, random_seed=0)
    assert {p["h1"] for p in pts} == {0.0, 2.0, 4.0}
    assert all(p["h1"] + p["x1"] <= 6.0 + 1e-6 for p in pts)
    assert max(p["x1"] for p in pts if p["h1"] == 4.0) <= 2.0 + 1e-6


def test_plot_with_empty_inner_region_raises_naming_the_plot():
    s = _space()
    s.add_constraint("inequality", {"h1": 1.0, "x1": 1.0}, 3.0)
    with pytest.raises(InfeasibleRegionError, match="h1"):
        generate_split_plot_design(s, DESIGNS, random_seed=0)


def test_crossed_mode_refuses_a_linking_constraint():
    s = _space()
    s.add_constraint("inequality", {"h1": 1.0, "x1": 1.0}, 6.0)
    with pytest.raises(ValueError, match="crossed"):
        generate_split_plot_design(s, DESIGNS, easy_mode="crossed", random_seed=0)


def test_reference_runs_at_both_ends_of_every_plot():
    pts, info = generate_split_plot_design(_space(), DESIGNS, reference={"position": "ends"}, random_seed=0)
    for wp in {p["WholePlot"] for p in pts}:
        rows = [p for p in pts if p["WholePlot"] == wp]
        assert rows[0]["Reason"] == rows[-1]["Reason"] == REFERENCE_REASON
        assert (rows[0]["x1"], rows[0]["x2"]) == (5.0, 5.0)
        assert sum(r["Reason"] == REFERENCE_REASON for r in rows) == 2
    assert info["n_runs"] == 3 * (8 + 2)


def test_reference_at_start_with_given_values():
    pts, _ = generate_split_plot_design(
        _space(), DESIGNS, reference={"position": "start", "values": {"x1": 1.0}}, random_seed=0)
    first = pts[0]
    assert first["Reason"] == REFERENCE_REASON and first["x1"] == 1.0 and first["x2"] == 5.0


def test_reference_defaults_for_each_kind():
    s = SearchSpace()
    s.add_variable("n1", "integer", min=0, max=5)
    s.add_variable("d1", "discrete", allowed_values=[1.0, 2.0, 10.0])
    s.add_variable("e", "categorical", values=["u", "v"])
    s.add_variable("c", "categorical", values=["a", "b"], changes="hard")
    pts, _ = generate_split_plot_design(
        s, {"hard": {"method": "full_factorial"}, "easy": {"method": "lhs", "n_points": 3}},
        reference={"position": "start"}, random_seed=0)
    ref = pts[0]
    assert (ref["n1"], ref["d1"], ref["e"]) == (2, 2.0, "u")


def test_infeasible_reference_raises():
    s = _space()
    s.add_constraint("inequality", {"x1": 1.0, "x2": 1.0}, 8.0)
    with pytest.raises(ValueError, match="reference"):
        generate_split_plot_design(s, DESIGNS, reference={"position": "start"}, random_seed=0)


def test_unknown_reference_key_raises():
    with pytest.raises(ValueError, match="h1"):
        generate_split_plot_design(_space(), DESIGNS, reference={"values": {"h1": 1.0}})
