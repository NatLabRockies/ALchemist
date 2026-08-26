"""Geometry of a feasible region defined by linear constraints and bounds.

This module knows nothing about designs, models, or optimization. It answers
purely geometric questions about the region carved out of a variable box by a
set of linear equality/inequality constraints:

- where does a point land when projected onto a constraint boundary
- what values may one variable take when the others are fixed
- where are the vertices of the feasible polytope
- how do we turn a lattice of candidate points into a feasible candidate set
  that includes points on the boundary

All functions work in **raw variable space**, matching
``SearchSpace.filter_feasible`` and ``SearchSpace.to_botorch_constraints``.
``SearchSpace.filter_feasible`` remains the single definition of feasibility;
nothing here re-implements that predicate.

Constraint convention (from ``SearchSpace.add_constraint``):
    'inequality' -> sum(c_i * x_i) <= rhs
    'equality'   -> sum(c_i * x_i) == rhs
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from alchemist_core.config import get_logger

logger = get_logger(__name__)

# Variable types that carry a numeric range and may appear in a constraint.
NUMERIC_TYPES = ("real", "integer", "discrete")

# Strict DoE tolerance, matching doe.py:200-203. A design point must not
# exceed the user's stated bound.
DOE_RTOL = 0.0
DOE_ATOL = 1e-9


def numeric_variables(search_space) -> List[Dict[str, Any]]:
    """Variables that have a numeric range, in search-space order.

    Excludes ``categorical`` (unordered) and ``context`` (no bounds at all).
    """
    return [v for v in search_space.variables if v.get("type") in NUMERIC_TYPES]


def variable_bounds(var: Dict[str, Any]) -> Tuple[float, float]:
    """Inclusive numeric bounds of a single variable.

    ``discrete`` variables use the min and max of their allowed values.

    Raises:
        ValueError: if the variable has no numeric range.
    """
    vtype = var.get("type")
    if vtype == "discrete":
        allowed = var["allowed_values"]
        return float(min(allowed)), float(max(allowed))
    if vtype in ("real", "integer"):
        return float(var["min"]), float(var["max"])
    raise ValueError(
        f"Variable '{var.get('name')}' of type '{vtype}' has no numeric bounds."
    )


def project_onto_constraint(point: Dict[str, Any],
                            constraint: Dict[str, Any]) -> Dict[str, Any]:
    """Orthogonal projection of ``point`` onto the hyperplane c.x == rhs.

    ``x' = x - c * (c.x - rhs) / ||c||^2``

    Only the keys named in the constraint's coefficients are moved; every
    other key (categoricals, context variables) is copied through untouched.
    A degenerate constraint (all-zero coefficients, or no participating key
    present in the point) returns a copy of the input.
    """
    out = dict(point)
    coeffs = constraint["coefficients"]
    names = [n for n in coeffs if n in point]
    if not names:
        return out

    c = np.array([float(coeffs[n]) for n in names], dtype=float)
    denom = float(c @ c)
    if denom == 0.0:
        return out

    x = np.array([float(point[n]) for n in names], dtype=float)
    slack = float(c @ x) - float(constraint["rhs"])
    x_new = x - c * (slack / denom)
    for n, v in zip(names, x_new):
        out[n] = float(v)
    return out


def feasible_interval(search_space, var_name: str,
                      fixed_values: Dict[str, float]) -> Optional[Tuple[float, float]]:
    """Interval of feasible values for one variable, the others held fixed.

    Each constraint ``sum(c_j x_j) <= rhs`` collapses to a one-sided bound on
    the free variable ``x_v``::

        c_v > 0   ->   x_v <= (rhs - rest) / c_v      (upper bound)
        c_v < 0   ->   x_v >= (rhs - rest) / c_v      (lower bound, sign flip)
        c_v == 0  ->   no information

    where ``rest`` is the contribution of the fixed variables. An equality
    contributes both bounds at the same value. The result is intersected with
    the variable's own bounds.

    Args:
        search_space: SearchSpace carrying variables and constraints.
        var_name: the free variable.
        fixed_values: values for the other variables. A variable named in a
            constraint but absent here contributes nothing, so the interval
            returned is a conservative superset in that case.

    Returns:
        ``(lo, hi)``, or ``None`` when the intersection is empty.

    Raises:
        ValueError: if ``var_name`` is not a numeric variable of this space.
    """
    var = next((v for v in search_space.variables if v["name"] == var_name), None)
    if var is None:
        raise ValueError(
            f"Variable '{var_name}' not found in search space. "
            f"Available: {[v['name'] for v in search_space.variables]}"
        )
    lo, hi = variable_bounds(var)

    for c in getattr(search_space, "constraints", None) or []:
        coeffs = c["coefficients"]
        if var_name not in coeffs:
            continue
        c_v = float(coeffs[var_name])
        if c_v == 0.0:
            continue

        rest = sum(
            float(coeff) * float(fixed_values[name])
            for name, coeff in coeffs.items()
            if name != var_name and name in fixed_values
        )
        limit = (float(c["rhs"]) - rest) / c_v

        if c["type"] == "equality":
            lo = max(lo, limit)
            hi = min(hi, limit)
        elif c_v > 0.0:
            hi = min(hi, limit)
        else:
            lo = max(lo, limit)

    if lo > hi:
        return None
    return (lo, hi)
