"""``remove_variable`` drops the linear constraints that name the variable (A14).

It used to leave them in place. ``filter_feasible`` sums only the columns
that are present, so after deleting ``x2`` the constraint ``x1 + x2 <= 12``
was silently enforced as ``x1 <= 12`` and every later design was screened
against a constraint nobody wrote. The dropped constraints are named in a
warning.
"""
import logging

import pytest

from alchemist_core.data.search_space import SearchSpace


def _space():
    s = SearchSpace()
    s.add_variable("x1", "real", min=0.0, max=10.0)
    s.add_variable("x2", "real", min=0.0, max=10.0)
    s.add_variable("x3", "integer", min=0, max=8)
    s.add_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=12.0, name="budget")
    s.add_constraint("equality", {"x2": 1.0, "x3": -1.0}, rhs=0.0, name="balance")
    s.add_constraint("inequality", {"x1": 1.0, "x3": 1.0}, rhs=9.0, name="keep")
    return s


def test_constraints_naming_the_variable_are_dropped(caplog):
    s = _space()
    with caplog.at_level(logging.WARNING):
        s.remove_variable("x2")
    assert [c["name"] for c in s.get_constraints()] == ["keep"]
    assert "budget" in caplog.text and "balance" in caplog.text and "x2" in caplog.text


def test_removing_an_unreferenced_variable_keeps_every_constraint(caplog):
    s = _space()
    s.add_variable("x4", "real", min=0.0, max=1.0)
    with caplog.at_level(logging.WARNING):
        s.remove_variable("x4")
    assert [c["name"] for c in s.get_constraints()] == ["budget", "balance", "keep"]
    assert "constraint" not in caplog.text.lower()


def test_designs_after_removal_are_not_screened_by_the_dropped_constraint():
    s = _space()
    s.remove_variable("x2")
    # With "budget" gone, x1 may take its full range; the leftover x1 <= 12
    # would have been vacuous anyway, so check the constraint set directly
    # and that a design still generates against what remains.
    from alchemist_core.utils.doe import generate_initial_design
    pts = generate_initial_design(s, method="lhs", n_points=6, random_seed=1)
    assert len(pts) == 6
    assert all(set(p) >= {"x1", "x3"} for p in pts)
    assert all(p["x1"] + p["x3"] <= 9.0 + 1e-9 for p in pts)
