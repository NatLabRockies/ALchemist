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
from alchemist_core.utils.doe import generate_initial_design

LEVELS = ("very_hard", "hard", "easy")
DESIGN_REASON = "Initial DoE (split-plot)"
REFERENCE_REASON = "Reference (split-plot)"
MAX_RUNS = 5000
_MODES = ("fresh", "crossed")
_ORDERS = ("random", "as_generated")


def _variables_at(search_space, level: str) -> List[Dict[str, Any]]:
    return search_space.get_variables_by_changes(level)


def _level_space(search_space, level: str, fixed: Dict[str, Any]) -> SearchSpace:
    """A throwaway space holding only ``level``'s variables, as plain easy ones.

    ``fixed`` holds the outer levels' values for the plot being generated.
    Constraints are added in Task 8.
    """
    entries = []
    for var in _variables_at(search_space, level):
        entry = {k: v for k, v in var.items() if k not in ("changes", "role")}
        entries.append(entry)
    return SearchSpace().from_dict(entries)


def _generate(space: SearchSpace, spec: Dict[str, Any], seed: Optional[int]) -> List[Dict[str, Any]]:
    kwargs = {k: v for k, v in spec.items() if k != "method"}
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
        if not outermost and mode_for(level) == "crossed":
            if level not in crossed_cache:
                crossed_cache[level] = _generate(space, designs[level], child_seed())
            rows = crossed_cache[level]
        else:
            rows = _generate(space, designs[level], child_seed())
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
            points.append({**header, **row, "Reason": DESIGN_REASON})
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
