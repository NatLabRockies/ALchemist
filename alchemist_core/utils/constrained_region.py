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


def snap_to_variable(value: float, var: Dict[str, Any]) -> float:
    """Clip to bounds, then round/snap according to the variable's type."""
    lo, hi = variable_bounds(var)
    value = max(lo, min(hi, float(value)))
    if var["type"] == "integer":
        return float(int(round(value)))
    if var["type"] == "discrete":
        allowed = var["allowed_values"]
        return float(min(allowed, key=lambda a: abs(float(a) - value)))
    return float(value)


def feasible_vertices(search_space, *,
                      fixed: Optional[Dict[str, Any]] = None,
                      max_vars: int = 5,
                      rtol: float = DOE_RTOL,
                      atol: float = DOE_ATOL) -> pd.DataFrame:
    """Vertices of the feasible polytope over the numeric variables.

    A vertex is the intersection of ``n`` hyperplanes drawn from the union of
    the registered constraints and the ``2n`` variable-bound faces, where
    ``n`` is the number of numeric variables. Every combination is solved and
    kept only if the result is genuinely feasible.

    D-optimal designs push to the extremes of the feasible region, and a
    filtered rectangular lattice contains no point on a constraint boundary.
    These vertices are what put the real extremes into the candidate set.

    The enumeration is ``C(n_planes, n)``. For 3 numeric variables with one
    constraint that is ``C(7, 3) = 35`` — trivial — but it grows fast, so
    above ``max_vars`` an empty frame is returned and the caller reports the
    omission rather than silently shipping a reduced candidate set.

    Args:
        search_space: SearchSpace carrying variables and constraints.
        fixed: values for non-numeric variables (categoricals) to attach to
            every returned row, so the frame can be fed to ``filter_feasible``
            and concatenated with a candidate set.
        max_vars: numeric-variable ceiling for enumeration.
        rtol, atol: feasibility tolerance.

    Returns:
        DataFrame of feasible vertices, deduplicated. Empty when there are no
        constraints, no numeric variables, or too many numeric variables.
    """
    constraints = getattr(search_space, "constraints", None) or []
    if not constraints:
        return pd.DataFrame()

    numeric = numeric_variables(search_space)
    n = len(numeric)
    if n == 0 or n > max_vars:
        if n > max_vars:
            logger.info(
                "Skipping vertex enumeration: %d numeric variables exceeds "
                "max_vars=%d. Boundary projection still applies.", n, max_vars,
            )
        return pd.DataFrame()

    names = [v["name"] for v in numeric]

    # Each plane is (coefficient vector over `names`, rhs).
    planes: List[Tuple[np.ndarray, float]] = []
    for c in constraints:
        row = np.array([float(c["coefficients"].get(nm, 0.0)) for nm in names],
                       dtype=float)
        if np.any(row):
            planes.append((row, float(c["rhs"])))
    for i, var in enumerate(numeric):
        lo, hi = variable_bounds(var)
        face = np.zeros(n, dtype=float)
        face[i] = 1.0
        planes.append((face.copy(), lo))
        planes.append((face.copy(), hi))

    rows: List[Dict[str, Any]] = []
    for combo in itertools.combinations(range(len(planes)), n):
        A = np.array([planes[k][0] for k in combo], dtype=float)
        b = np.array([planes[k][1] for k in combo], dtype=float)
        # Skip near-parallel plane sets; they have no unique intersection.
        if abs(np.linalg.det(A)) < 1e-12:
            continue
        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(x)):
            continue

        point: Dict[str, Any] = dict(fixed or {})
        for var, value in zip(numeric, x):
            point[var["name"]] = snap_to_variable(value, var)
        rows.append(point)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Snapping and clipping can push a solved vertex back outside the region,
    # so feasibility is re-tested rather than assumed.
    df = df[search_space.filter_feasible(df, rtol=rtol, atol=atol)]
    if df.empty:
        return pd.DataFrame()

    return _dedupe(df, names)


def _dedupe(df: pd.DataFrame, numeric_names: List[str]) -> pd.DataFrame:
    """Drop duplicate rows, comparing numeric columns on a tolerance grid."""
    if df.empty:
        return df
    key = df.copy()
    for nm in numeric_names:
        if nm in key.columns:
            key[nm] = key[nm].astype(float).round(9)
    return df[~key.duplicated()].reset_index(drop=True)


class InfeasibleRegionError(ValueError):
    """No feasible candidate point could be produced for the given constraints."""


def augment_with_boundary(search_space, points: pd.DataFrame, *,
                          max_vertex_vars: int = 5,
                          rtol: float = DOE_RTOL,
                          atol: float = DOE_ATOL) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Turn a candidate lattice into a feasible candidate set with boundary points.

    A regular lattice filtered against a constraint contains no point *on* the
    constraint boundary, yet that boundary is exactly where an optimal design
    wants to place runs. This function keeps the feasible lattice points, adds
    the projections of the infeasible ones onto the constraints they violate,
    and adds the feasible region's vertices.

    Runs once per categorical combination present in ``points``; categorical
    columns are held fixed while the numeric sub-vector is moved.

    Args:
        search_space: SearchSpace carrying variables and constraints.
        points: candidate points in **raw** variable space.
        max_vertex_vars: numeric-variable ceiling for vertex enumeration.
        rtol, atol: feasibility tolerance.

    Returns:
        ``(augmented, info)``. ``info`` carries candidate-set provenance and is
        surfaced to API callers, so a reduced candidate set is never silent.

    Raises:
        InfeasibleRegionError: when nothing feasible survives.
    """
    constraints = getattr(search_space, "constraints", None) or []
    info: Dict[str, Any] = {
        "constraints_applied": [c["name"] for c in constraints],
        "n_candidates_total": int(len(points)),
        "n_candidates_feasible": int(len(points)),
        "n_boundary_added": 0,
        "n_vertices_added": 0,
        "vertex_enumeration_skipped": False,
    }
    if not constraints or points.empty:
        return points, info

    numeric = numeric_variables(search_space)
    numeric_names = [v["name"] for v in numeric]
    by_name = {v["name"]: v for v in numeric}
    cat_names = [v["name"] for v in search_space.variables
                 if v.get("type") == "categorical" and v["name"] in points.columns]

    info["vertex_enumeration_skipped"] = len(numeric) > max_vertex_vars

    # Group by categorical combination so projection only moves numeric axes.
    groups = points.groupby(cat_names, sort=False) if cat_names else [((), points)]

    collected: List[pd.DataFrame] = []
    n_feasible = 0
    n_boundary = 0
    n_vertices = 0

    for key, group in groups:
        fixed: Dict[str, Any] = {}
        if cat_names:
            key_tuple = key if isinstance(key, tuple) else (key,)
            fixed = dict(zip(cat_names, key_tuple))

        mask = search_space.filter_feasible(group, rtol=rtol, atol=atol)
        feasible = group[mask]
        infeasible = group[~mask]
        n_feasible += int(mask.sum())
        parts = [feasible]

        # Project every infeasible point onto each constraint it violates.
        projected_rows: List[Dict[str, Any]] = []
        for _idx, row in infeasible.iterrows():
            base = row.to_dict()
            for c in constraints:
                moved = project_onto_constraint(base, c)
                for nm in numeric_names:
                    if nm in moved:
                        moved[nm] = snap_to_variable(moved[nm], by_name[nm])
                projected_rows.append(moved)

        if projected_rows:
            proj_df = pd.DataFrame(projected_rows)
            # Clipping and snapping can leave a projected point violating a
            # *different* constraint, so re-test against all of them.
            proj_df = proj_df[search_space.filter_feasible(proj_df, rtol=rtol, atol=atol)]
            if not proj_df.empty:
                n_boundary += len(proj_df)
                parts.append(proj_df)

        verts = feasible_vertices(search_space, fixed=fixed,
                                  max_vars=max_vertex_vars, rtol=rtol, atol=atol)
        if not verts.empty:
            n_vertices += len(verts)
            parts.append(verts)

        non_empty_parts = [p for p in parts if not p.empty]
        # A group can end up with nothing feasible, no surviving boundary
        # projections, and no vertices (e.g. an unreachable constraint) --
        # pd.concat on an empty list raises rather than returning an empty
        # frame, so that case is handled explicitly.
        merged = (pd.concat(non_empty_parts, ignore_index=True) if non_empty_parts
                  else pd.DataFrame(columns=group.columns))
        collected.append(merged)

    if not collected:
        raise InfeasibleRegionError(
            "No feasible design candidates could be generated for the "
            f"registered input constraints ({info['constraints_applied']})."
        )

    out = pd.concat(collected, ignore_index=True)
    out = out.reindex(columns=list(points.columns))
    before = len(out)
    out = _dedupe(out, numeric_names)
    # Dedup can only remove added rows, so attribute the loss to the additions.
    removed = before - len(out)

    if out.empty:
        raise InfeasibleRegionError(
            "No feasible design candidates could be generated for the "
            f"registered input constraints ({info['constraints_applied']}). "
            "The feasible region may be empty within the variable bounds; "
            "relax the constraints or widen the bounds."
        )

    info["n_candidates_feasible"] = int(n_feasible)
    info["n_boundary_added"] = int(max(0, n_boundary - removed))
    info["n_vertices_added"] = int(n_vertices)
    logger.info(
        "Constrained candidate set: %d total -> %d feasible, +%d boundary, "
        "+%d vertices, %d final%s",
        info["n_candidates_total"], n_feasible, info["n_boundary_added"],
        n_vertices, len(out),
        " (vertex enumeration skipped)" if info["vertex_enumeration_skipped"] else "",
    )
    return out, info
