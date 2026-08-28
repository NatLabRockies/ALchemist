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


def _dedupe(df: pd.DataFrame, numeric_names: List[str],
           ignore_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Drop duplicate rows, comparing numeric columns on a tolerance grid.

    ``duplicated()`` compares full rows, so a temporary column (e.g. a
    provenance tag) that differs between two otherwise-identical rows would
    prevent them from being recognized as duplicates. ``ignore_columns``
    excludes such columns from the comparison entirely; the rows they
    belong to are still filtered as a whole -- only the *comparison* skips
    them. Duplicates are resolved by keeping the first occurrence (pandas'
    ``duplicated()`` default), so row order controls precedence.
    """
    if df.empty:
        return df
    ignore = set(ignore_columns or ())
    compare_cols = [c for c in df.columns if c not in ignore]
    key = df[compare_cols].copy()
    for nm in numeric_names:
        if nm in key.columns:
            key[nm] = key[nm].astype(float).round(9)
    return df[~key.duplicated()].reset_index(drop=True)


class InfeasibleRegionError(ValueError):
    """No feasible candidate point could be produced for the given constraints."""


# Internal-only column tagging each assembled row's provenance ("feasible",
# "boundary", or "vertex") before the final dedup pass, so the info-dict
# counts can be read off what actually survives in the returned frame
# instead of being approximated from a pre/post row-count delta. This is a
# *sentinel*, not a guaranteed-safe name: ``SearchSpace.add_variable``
# performs no name reservation, so a user variable literally named
# ``__origin__`` is legal and would collide with a hardcoded constant. The
# actual column used at runtime is computed by ``_provenance_column`` below,
# which starts from this sentinel and lengthens it until it is verifiably
# absent from the caller's own columns. The column is always dropped before
# the augmented frame is returned.
_ORIGIN_COL = "__origin__"
_ORIGIN_FEASIBLE = "feasible"
_ORIGIN_BOUNDARY = "boundary"
_ORIGIN_VERTEX = "vertex"


def _provenance_column(columns) -> str:
    """A provenance-tag column name guaranteed absent from ``columns``.

    Starts from ``_ORIGIN_COL`` and appends underscores until the candidate
    is not already used by one of the caller's own columns. This makes the
    tag collision-proof against any legal ``SearchSpace`` variable name
    (including a variable literally named ``__origin__``), without rejecting
    such a search space or silently overwriting its data.
    """
    existing = set(columns)
    name = _ORIGIN_COL
    while name in existing:
        name += "_"
    return name


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

    Every assembled row is tagged with its provenance (pre-existing feasible
    point, boundary projection, or enumerated vertex) before the final dedup
    pass, and ``n_candidates_feasible`` / ``n_boundary_added`` /
    ``n_vertices_added`` are counted from what survives that pass -- so they
    always describe the returned frame, even when an unrelated duplicate
    collision (e.g. two identical pre-existing feasible rows) removes rows
    that have nothing to do with the boundary or vertex additions. When a
    boundary projection and a vertex land on the same point, the boundary
    tag wins: parts are concatenated feasible-then-boundary-then-vertex and
    ``duplicated()`` keeps the first occurrence, so the surviving row is
    attributed to the boundary projection (it traces back to a
    caller-supplied candidate being pulled onto the constraint) rather than
    to vertex enumeration. The tag column is dropped before returning, so
    the output's columns are exactly ``points.columns``.

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

    # Computed fresh per call from the caller's actual columns, rather than
    # used as the hardcoded sentinel, so a user variable literally named
    # ``__origin__`` (or any of its lengthened variants) can never be
    # shadowed or duplicated by the internal provenance tag.
    origin_col = _provenance_column(points.columns)

    numeric = numeric_variables(search_space)
    numeric_names = [v["name"] for v in numeric]
    by_name = {v["name"]: v for v in numeric}
    cat_names = [v["name"] for v in search_space.variables
                 if v.get("type") == "categorical" and v["name"] in points.columns]

    info["vertex_enumeration_skipped"] = len(numeric) > max_vertex_vars

    # Group by categorical combination so projection only moves numeric axes.
    groups = points.groupby(cat_names, sort=False) if cat_names else [((), points)]

    collected: List[pd.DataFrame] = []

    for key, group in groups:
        fixed: Dict[str, Any] = {}
        if cat_names:
            key_tuple = key if isinstance(key, tuple) else (key,)
            fixed = dict(zip(cat_names, key_tuple))

        mask = search_space.filter_feasible(group, rtol=rtol, atol=atol)
        feasible = group[mask].copy()
        infeasible = group[~mask]
        parts: List[pd.DataFrame] = []
        if not feasible.empty:
            feasible[origin_col] = _ORIGIN_FEASIBLE
            parts.append(feasible)

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
            proj_df = proj_df[search_space.filter_feasible(proj_df, rtol=rtol, atol=atol)].copy()
            if not proj_df.empty:
                proj_df[origin_col] = _ORIGIN_BOUNDARY
                parts.append(proj_df)

        verts = feasible_vertices(search_space, fixed=fixed,
                                  max_vars=max_vertex_vars, rtol=rtol, atol=atol)
        if not verts.empty:
            verts = verts.copy()
            verts[origin_col] = _ORIGIN_VERTEX
            parts.append(verts)

        # A group can end up with nothing feasible, no surviving boundary
        # projections, and no vertices (e.g. an unreachable constraint) --
        # pd.concat on an empty list raises rather than returning an empty
        # frame, so that case is handled explicitly.
        merged = (pd.concat(parts, ignore_index=True) if parts
                  else pd.DataFrame(columns=list(group.columns) + [origin_col]))
        collected.append(merged)

    if not collected:
        raise InfeasibleRegionError(
            "No feasible design candidates could be generated for the "
            f"registered input constraints ({info['constraints_applied']})."
        )

    out = pd.concat(collected, ignore_index=True)
    out = out.reindex(columns=list(points.columns) + [origin_col])
    # The provenance tag is excluded from the duplicate comparison so two
    # rows differing only in tag (e.g. a boundary projection landing exactly
    # on an enumerated vertex) still collapse into a single surviving row.
    out = _dedupe(out, numeric_names, ignore_columns=[origin_col])

    if out.empty:
        raise InfeasibleRegionError(
            "No feasible design candidates could be generated for the "
            f"registered input constraints ({info['constraints_applied']}). "
            "The feasible region may be empty within the variable bounds; "
            "relax the constraints or widen the bounds."
        )

    # Counts reflect what survived dedup, tag by tag -- not raw pre-dedup
    # totals debited by an unrelated collision elsewhere in the frame.
    # ``out[origin_col]`` is guaranteed to resolve to a single Series here
    # (never a DataFrame) because ``origin_col`` was chosen above to be
    # absent from ``points.columns``, so there is exactly one column by
    # that name in ``out``.
    origin_counts = out[origin_col].value_counts()
    info["n_candidates_feasible"] = int(origin_counts.get(_ORIGIN_FEASIBLE, 0))
    info["n_boundary_added"] = int(origin_counts.get(_ORIGIN_BOUNDARY, 0))
    info["n_vertices_added"] = int(origin_counts.get(_ORIGIN_VERTEX, 0))
    out = out.drop(columns=[origin_col])

    logger.info(
        "Constrained candidate set: %d total -> %d feasible, +%d boundary, "
        "+%d vertices, %d final%s",
        info["n_candidates_total"], info["n_candidates_feasible"],
        info["n_boundary_added"], info["n_vertices_added"], len(out),
        " (vertex enumeration skipped)" if info["vertex_enumeration_skipped"] else "",
    )
    return out, info


def region_is_provably_empty(search_space, *, atol: float = DOE_ATOL) -> bool:
    """Whether *no* point in the variable box can satisfy every constraint.

    Answers the question ``generate_optimal_design`` already asks through
    ``augment_with_boundary`` (which raises :class:`InfeasibleRegionError`
    when nothing feasible survives) and that the space-filling path never
    asked: it used to discover emptiness only by failing to sample its way
    out of it, escalating the oversampling factor to 4096 first.

    Decided by a linear program over the **continuous relaxation** of the
    box, so the answer is a *proof* in one direction only:

    - ``True``  -- the relaxation is infeasible. Every attainable point is a
      point of the relaxation, so nothing can be feasible. Safe to raise on.
    - ``False`` -- **indeterminate**, not "non-empty". The relaxation has a
      point, but that point may be non-integral, off a ``discrete``
      variable's grid, or the solver may simply not have converged. The
      caller must fall through to sampling.

    Every doubt therefore returns ``False``: a solver status other than
    "infeasible", a constraint naming a variable this space does not carry,
    a space with no numeric variables. Reporting an *empty* region for one
    that is merely hard is the failure mode this function must not have --
    it would turn a solvable design into a spurious 400 -- so the
    indeterminate cases are spent on extra sampling rather than on a wrong
    verdict.

    ``atol`` must be the same absolute tolerance the caller will hand
    ``SearchSpace.filter_feasible`` (with ``rtol=0``), because that predicate
    is the definition of feasibility this must relax rather than contradict.
    ``filter_feasible`` accepts ``lhs <= rhs + atol``, and accepts an
    *equality* anywhere in the slab ``|lhs - rhs| <= atol`` rather than only
    on the hyperplane. Both are modelled as inequalities at that widened
    bound: modelling an equality as ``A_eq x == rhs`` would make the program
    stricter than the predicate, and a stricter program can report
    "infeasible" for a region ``filter_feasible`` would have accepted.
    """
    constraints = getattr(search_space, "constraints", None) or []
    if not constraints:
        return False

    numeric = numeric_variables(search_space)
    if not numeric:
        return False

    names = [v["name"] for v in numeric]
    index = {nm: i for i, nm in enumerate(names)}
    n = len(names)

    a_ub: List[np.ndarray] = []
    b_ub: List[float] = []
    for c in constraints:
        coeffs = c["coefficients"]
        # A term with nowhere to go is not a term that can be dropped: the
        # program would then describe a *different* constraint set than the
        # one filter_feasible applies. Refuse to answer instead.
        if any(nm not in index for nm in coeffs):
            return False
        row = np.zeros(n, dtype=float)
        for nm, coeff in coeffs.items():
            row[index[nm]] = float(coeff)
        rhs = float(c["rhs"])
        if c["type"] == "equality":
            a_ub.append(row)
            b_ub.append(rhs + atol)
            a_ub.append(-row)
            b_ub.append(-(rhs - atol))
        else:
            a_ub.append(row)
            b_ub.append(rhs + atol)

    # `discrete` and `integer` collapse to their hull, which is a superset of
    # the values those variables can actually take -- keeping the relaxation
    # a relaxation.
    bounds = [variable_bounds(v) for v in numeric]

    from scipy.optimize import linprog

    result = linprog(
        c=np.zeros(n, dtype=float),
        A_ub=np.array(a_ub, dtype=float),
        b_ub=np.array(b_ub, dtype=float),
        bounds=bounds,
        method="highs",
    )
    # 2 is HiGHS' "problem is infeasible". 0 optimal, 1 iteration limit,
    # 3 unbounded and 4 numerical difficulty all mean "no proof", and an
    # unrecognised status must mean that too.
    return result.status == 2


def region_is_provably_measure_zero(search_space) -> bool:
    """Whether the feasible region has zero volume along a continuous axis.

    An equality ``sum(c_i x_i) == rhs`` that gives a non-zero coefficient to a
    ``real`` variable pins a *continuously distributed* quantity to a single
    value. Every space-filling sampler (``random``, ``lhs``, ``sobol``,
    ``halton``, ``hammersly``) draws that axis from a continuum, so the
    chance of landing inside the ``+/-1e-9`` slab ``filter_feasible`` allows
    is around ``1e-10`` per draw. Reject-and-resample cannot finish, and
    escalating the oversampling factor to 4096 does not change that -- it
    only decides how many minutes are spent proving it.

    The region is *not* empty in this case, which is why it is reported
    separately: the caller owes the user a different message and a different
    remedy ("use ``optimal``, which places points on constraint boundaries")
    than it owes for a region with nothing in it at all.

    Restricted to ``real`` deliberately. The same equality over ``integer``
    or ``discrete`` variables lands on a lattice the samplers do visit, and
    those designs succeed today -- see
    ``test_an_equality_constraint_holds_by_hand``. Reporting them here would
    break a working design, so ``integer`` and ``discrete`` are left to the
    sampler and a ``False`` here means only "not proven", never "reachable".
    """
    real_names = {
        v["name"] for v in getattr(search_space, "variables", []) or []
        if v.get("type") == "real"
    }
    for c in getattr(search_space, "constraints", None) or []:
        if c.get("type") != "equality":
            continue
        for nm, coeff in c["coefficients"].items():
            if nm in real_names and float(coeff) != 0.0:
                return True
    return False
