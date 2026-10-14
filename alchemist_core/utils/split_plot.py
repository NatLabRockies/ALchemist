"""Split-plot designs built by nesting existing DoE methods.

Controllable variables are grouped by change difficulty. The outermost
restricted level gets a design of its own; for every setting of it, the next
level's design is generated with the outer values fixed; the easy level is
innermost. Each row carries the IDs of the plots it belongs to and its global
run order, so the design can be run block by block and analyzed with the
correct error strata elsewhere.
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from alchemist_core.data.search_space import SearchSpace
from alchemist_core.utils.constrained_region import InfeasibleRegionError
from alchemist_core.utils.doe import generate_initial_design

LEVELS = ("very_hard", "hard", "easy")
DESIGN_REASON = "Initial DoE (split-plot)"
REFERENCE_REASON = "Reference (split-plot)"
MAX_RUNS = 5000
_MODES = ("fresh", "crossed")
_ORDERS = ("random", "as_generated")
_CENTERED_METHODS = ("full_factorial", "fractional_factorial", "ccd", "box_behnken", "plackett_burman")


def _variables_at(search_space, level: str) -> List[Dict[str, Any]]:
    return search_space.get_variables_by_changes(level)


def _reduce_constraints(search_space, level: str, fixed: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Constraints that bind ``level`` once the outer values ``fixed`` are known.

    A constraint over this level's variables and outer variables only becomes
    a constraint on this level with the outer terms moved to the right-hand
    side. One that still names an inner level's variable applies later.
    """
    names = {v["name"] for v in _variables_at(search_space, level)}
    reduced = []
    for c in getattr(search_space, "constraints", []) or []:
        coeffs = c["coefficients"]
        mine = {k: float(a) for k, a in coeffs.items() if k in names}
        outer = {k: float(a) for k, a in coeffs.items() if k in fixed}
        if not mine or len(mine) + len(outer) != len(coeffs):
            continue
        rhs = float(c["rhs"]) - sum(a * float(fixed[k]) for k, a in outer.items())
        reduced.append({"type": c["type"], "coefficients": mine, "rhs": rhs, "name": c.get("name")})
    return reduced


def _links_outer(search_space, level: str, outer_levels) -> bool:
    """True when a constraint binds ``level`` together with variables of ``outer_levels``
    (and nothing else), so it is reduced per outer plot."""
    names = {v["name"] for v in _variables_at(search_space, level)}
    outer = {v["name"] for lvl in outer_levels for v in _variables_at(search_space, lvl)}
    for c in getattr(search_space, "constraints", []) or []:
        keys = set(c["coefficients"])
        rest = keys - names
        if keys & names and rest and rest <= outer:
            return True
    return False


def _level_space(search_space, level: str, fixed: Dict[str, Any]) -> SearchSpace:
    """A throwaway space holding only ``level``'s variables and the constraints
    that bind them given the outer values ``fixed``."""
    entries = [
        {k: v for k, v in var.items() if k not in ("changes", "role")}
        for var in _variables_at(search_space, level)
    ]
    space = SearchSpace().from_dict(entries)
    for c in _reduce_constraints(search_space, level, fixed):
        space.add_constraint(c["type"], c["coefficients"], c["rhs"])
    return space


def _reference_row(search_space, given: Dict[str, Any]) -> Dict[str, Any]:
    easy = _variables_at(search_space, "easy")
    names = {v["name"] for v in easy}
    unknown = set(given) - names
    if unknown:
        raise ValueError(
            f"reference values may only name easy variables {sorted(names)}; got {sorted(unknown)}."
        )
    row = {}
    for v in easy:
        name = v["name"]
        if name in given:
            row[name] = given[name]
        elif v["type"] == "categorical":
            row[name] = v["values"][0]
        elif v["type"] == "discrete":
            mid = (min(v["allowed_values"]) + max(v["allowed_values"])) / 2
            row[name] = min(v["allowed_values"], key=lambda a: (abs(a - mid), a))
        elif v["type"] == "integer":
            row[name] = int(round((v["min"] + v["max"]) / 2))
        else:
            row[name] = (v["min"] + v["max"]) / 2
    return row


def _generate(space: SearchSpace, spec: Dict[str, Any], seed: Optional[int]) -> List[Dict[str, Any]]:
    kwargs = {k: v for k, v in spec.items() if k != "method"}
    if spec["method"] in _CENTERED_METHODS:
        kwargs.setdefault("n_center", 0)
    return generate_initial_design(space, method=spec["method"], random_seed=seed, **kwargs)


def _validate(search_space, designs, replicates, easy_mode, middle_mode, order) -> List[str]:
    present = [lvl for lvl in LEVELS if _variables_at(search_space, lvl)]
    if not any(lvl in present for lvl in ("very_hard", "hard")):
        raise ValueError(
            "A split-plot design needs at least one variable with changes='hard' or "
            "'very_hard'. With every variable easy, use generate_initial_design."
        )
    if "easy" not in present:
        raise ValueError("A split-plot design needs at least one easy variable to run inside each plot.")
    if set(designs) != set(present):
        raise ValueError(
            f"designs must have exactly one entry per change level present "
            f"{present}; got {sorted(designs)}."
        )
    for level, spec in designs.items():
        if not isinstance(spec, dict) or "method" not in spec:
            raise ValueError(f"designs['{level}'] needs a 'method'.")
        if spec["method"] == "optimal":
            raise ValueError("Split-plot designs do not support method 'optimal'; choose another method.")
    for level, r in (replicates or {}).items():
        if level not in present or level == "easy":
            raise ValueError(f"replicates applies to restricted levels present {present[:-1]}; got '{level}'.")
        if int(r) < 1:
            raise ValueError(f"replicates['{level}'] must be at least 1.")
    for mode in (easy_mode, middle_mode):
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}.")
    for level, mode in (("easy", easy_mode), ("hard", middle_mode)):
        if mode == "crossed" and level in present and level != present[0] and _links_outer(search_space, level, present[:present.index(level)]):
            raise ValueError(
                f"'crossed' cannot be used for the {level} level: a constraint links it "
                f"to an outer level, so each plot needs its own design. Use 'fresh'."
            )
    for level, how in (order or {}).items():
        if level not in present or how not in _ORDERS:
            raise ValueError(f"order['{level}'] must be one of {_ORDERS} for a present level.")
    return present


def generate_split_plot_design(
    search_space,
    designs: Dict[str, Dict[str, Any]],
    replicates: Optional[Dict[str, int]] = None,
    easy_mode: str = "fresh",
    middle_mode: str = "fresh",
    reference: Optional[Dict[str, Any]] = None,
    order: Optional[Dict[str, str]] = None,
    random_seed: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generate a split-plot (or split-split-plot) design.

    Args:
        designs: one entry per change level present, e.g.
            {"hard": {"method": "full_factorial", "n_levels": 2},
             "easy": {"method": "lhs", "n_points": 10}}. Each entry is a method
            accepted by generate_initial_design (not 'optimal') plus its kwargs.
            Classical methods default to n_center=0 inside split-plot designs;
            pass n_center to add center settings.
        replicates: repeats of a restricted level's design, each a new plot
            (default 1).
        easy_mode / middle_mode: "fresh" draws a new design per outer plot;
            "crossed" reuses one design under every outer plot. middle_mode
            applies to the hard level when a very_hard level exists.
        reference: {"position": "ends" | "start", "values": {...}} adds a
            reference run of the easy variables to every innermost plot.
        order: per level, "random" (default) or "as_generated".
        random_seed: makes the whole design reproducible.

    Returns:
        (points, info). Rows carry every controllable variable plus WholePlot,
        Subplot (three levels only), RunOrder and Reason. Observed variables
        are omitted. info has levels, n_plots, n_runs, warnings and the inputs.
    """
    replicates = dict(replicates or {})
    order = dict(order or {})
    present = _validate(search_space, designs, replicates, easy_mode, middle_mode, order)
    restricted = present[:-1]
    ref_row = None
    ref_position = None
    if reference is not None:
        ref_position = reference.get("position", "ends")
        if ref_position not in ("ends", "start"):
            raise ValueError("reference['position'] must be 'ends' or 'start'.")
        ref_row = _reference_row(search_space, dict(reference.get("values") or {}))
    rng = np.random.default_rng(random_seed)

    def child_seed() -> int:
        return int(rng.integers(0, 2**31 - 1))

    def mode_for(level: str) -> str:
        return easy_mode if level == "easy" else middle_mode

    warnings: List[str] = []
    n_plots: Dict[str, int] = {lvl: 0 for lvl in restricted}
    crossed_cache: Dict[str, List[Dict[str, Any]]] = {}

    def settings_for(level: str, fixed: Dict[str, Any], outermost: bool) -> List[Dict[str, Any]]:
        space = _level_space(search_space, level, fixed)
        try:
            if not outermost and mode_for(level) == "crossed":
                if level not in crossed_cache:
                    crossed_cache[level] = _generate(space, designs[level], child_seed())
                rows = crossed_cache[level]
            else:
                rows = _generate(space, designs[level], child_seed())
        except InfeasibleRegionError as e:
            where = ", ".join(f"{k}={v}" for k, v in fixed.items()) or "the outermost level"
            raise InfeasibleRegionError(
                f"No feasible {level}-level design in the plot with {where}: {e}"
            ) from e
        rows = [dict(r) for r in rows] * int(replicates.get(level, 1))
        if level != "easy" and order.get(level, "random") == "random":
            rows = [rows[i] for i in rng.permutation(len(rows))]
        return rows

    def note_replication(level: str, rows: List[Dict[str, Any]]) -> None:
        if level == "easy" or int(replicates.get(level, 1)) > 1:
            return
        keys = [tuple(sorted(r.items())) for r in rows]
        if len(set(keys)) == len(keys):
            msg = ("each whole-plot setting appears once; whole-plot error cannot be "
                   "estimated and whole-plot effects are confounded with their plot.")
            if msg not in warnings:
                warnings.append(msg)

    blocks: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []  # (ids+fixed values, easy rows)

    def nest(depth: int, fixed: Dict[str, Any], ids: Dict[str, int]) -> None:
        level = present[depth]
        rows = settings_for(level, fixed, outermost=(depth == 0))
        note_replication(level, rows)
        if level == "easy":
            if order.get("easy", "random") == "random":
                rows = [rows[i] for i in rng.permutation(len(rows))]
            if ref_row is not None:
                check = _level_space(search_space, "easy", fixed)
                if check.constraints and not check.is_feasible(ref_row):
                    where = ", ".join(f"{k}={v}" for k, v in fixed.items())
                    raise ValueError(
                        f"The reference run {ref_row} violates a constraint in the plot "
                        f"with {where}; pass reference['values'] that are feasible."
                    )
                # a fresh dict per use: the marker is popped when rows are emitted
                rows = ([dict(ref_row, _reference=True)] + rows
                        + ([dict(ref_row, _reference=True)] if ref_position == "ends" else []))
            blocks.append(({**fixed, **ids}, rows))
            running = sum(len(r) for _, r in blocks)
            if running > MAX_RUNS:
                raise ValueError(
                    f"This design has more than {MAX_RUNS} runs (the limit). "
                    f"Reduce replicates, levels or points per plot."
                )
            return
        id_key = "WholePlot" if depth == 0 else "Subplot"
        for row in rows:
            n_plots[level] += 1
            nest(depth + 1, {**fixed, **row}, {**ids, id_key: n_plots[level]})

    nest(0, {}, {})

    points: List[Dict[str, Any]] = []
    for header, rows in blocks:
        for row in rows:
            is_ref = row.pop("_reference", False)
            points.append({**header, **row, "Reason": REFERENCE_REASON if is_ref else DESIGN_REASON})
    for i, p in enumerate(points, start=1):
        p["RunOrder"] = i

    info = {
        "levels": {lvl: [v["name"] for v in _variables_at(search_space, lvl)] for lvl in present},
        "n_plots": n_plots,
        "runs_per_plot": sorted({len(rows) for _, rows in blocks}),
        "n_runs": len(points),
        "random_seed": random_seed,
        "designs": designs,
        "replicates": replicates,
        "easy_mode": easy_mode,
        "warnings": warnings,
    }
    return points, info
