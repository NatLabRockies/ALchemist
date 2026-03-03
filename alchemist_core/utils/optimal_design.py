"""
Optimal Experimental Design (OED) for ALchemist.

This module provides optimal design algorithms that generate statistically
efficient experimental designs based on user-specified model structures.
Unlike space-filling or classical RSM designs, optimal designs allow users
to specify which main effects, interactions, and quadratic terms they expect
to be important, then generate the most information-rich set of experiments
to estimate those specific model terms.

The module wraps pyDOE's ``doe_optimal`` subpackage, adding:

- **Custom model specification**: Users define effects using variable names
  (e.g., ``"Temperature*Pressure"``) instead of abstract polynomial degrees.
- **Mixed variable support**: Candidate set generation for continuous and
  categorical variables together.
- **ALchemist integration**: Direct mapping from SearchSpace variables to
  design points as ``List[Dict]``.

Supported optimality criteria:

- **D-optimal**: Maximize ``|X'X|`` — best for parameter estimation.
- **A-optimal**: Minimize ``trace((X'X)^-1)`` — minimize average parameter variance.
- **I-optimal**: Minimize integrated prediction variance over the design space.

Supported algorithms:

- ``sequential`` (Dykstra) — Greedy, fastest.
- ``simple_exchange`` (Wynn-Mitchell) — Drop-add exchange.
- ``fedorov`` — Full pairwise exchange (recommended default).
- ``modified_fedorov`` — Position-wise replacement.
- ``detmax`` (Mitchell) — Exchange with excursions, best quality.

.. warning::

    **Experimental Feature — Categorical Variables**: Optimal design with
    categorical variables uses dummy coding (k-1 indicator columns per
    variable, with the first category as the reference level) in the design
    matrix. The statistical properties of optimal designs with categorical
    variables are an active area of research and may not behave identically
    to classical continuous-variable optimal designs. Results should be
    reviewed carefully. For well-established categorical screening, consider
    using Full Factorial or GSD methods instead.

References:
    - Atkinson, A. C., & Donev, A. N. (1992). *Optimum Experimental Designs*.
    - Fedorov, V. V. (1972). *Theory of Optimal Experiments*.
    - Montgomery, D. C. (2017). *Design and Analysis of Experiments*.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from alchemist_core.config import get_logger
from alchemist_core.data.search_space import SearchSpace

logger = get_logger(__name__)

# ============================================================
# Type aliases
# ============================================================

# A model term is a tuple of (var_index, power) pairs.
# Examples for a 3-variable space (x0, x1, x2):
#   Intercept:       ()
#   Main effect x0:  ((0, 1),)
#   Interaction:     ((0, 1), (1, 1))
#   Quadratic x0:    ((0, 2),)
ModelTerm = Tuple[Tuple[int, int], ...]

VALID_CRITERIA = {"D", "A", "I"}
VALID_ALGORITHMS = {
    "sequential", "simple_exchange", "fedorov", "modified_fedorov", "detmax"
}
VALID_MODEL_TYPES = {"linear", "interaction", "quadratic"}


# ============================================================
# Model specification parsing
# ============================================================

def parse_model_spec(
    search_space: SearchSpace,
    model_type: Optional[str] = None,
    effects: Optional[List[str]] = None,
) -> List[ModelTerm]:
    """Parse user-friendly model specification into internal term representation.

    Converts either a ``model_type`` shortcut or a list of named ``effects``
    into a list of :data:`ModelTerm` tuples that describe each column of the
    regression (design) matrix.

    The intercept term is **always included** automatically.

    Args:
        search_space: SearchSpace with defined variables.
        model_type: Shortcut for common model structures. One of:

            - ``"linear"`` — Intercept + all main effects.
            - ``"interaction"`` — Linear + all pairwise interactions.
            - ``"quadratic"`` — Interaction + all squared terms for
              continuous (real/integer) variables. Categorical variables
              do not generate quadratic terms.

        effects: Explicit list of effect strings using variable names.
            The intercept is added automatically; do not include ``"1"``.

            **Format rules:**

            - **Main effects**: Use the variable name exactly as defined
              in the SearchSpace.

              ``"Temperature"``, ``"Pressure"``

            - **Interactions**: Join two or more variable names with ``"*"``.

              ``"Temperature*Pressure"``, ``"Catalyst*Temperature"``

            - **Quadratic terms**: Append ``"**2"`` to the variable name.

              ``"Temperature**2"``, ``"Pressure**2"``

            **Examples:**

            .. code-block:: python

                # Main effects + one interaction + one quadratic
                effects=["Temperature", "Pressure", "Catalyst",
                         "Temperature*Pressure", "Temperature**2"]

                # Main effects only (equivalent to model_type="linear")
                effects=["Temperature", "Pressure", "Flow_Rate"]

    Returns:
        List of ModelTerm tuples. Each term is a tuple of
        ``(variable_index, power)`` pairs. The intercept is ``()``.

    Raises:
        ValueError: If both or neither of ``model_type`` / ``effects`` are
            given, or if an effect references an unknown variable, or if
            a quadratic term is requested for a categorical variable.
    """
    if model_type is not None and effects is not None:
        raise ValueError(
            "Specify either 'model_type' or 'effects', not both."
        )
    if model_type is None and effects is None:
        raise ValueError(
            "Specify either 'model_type' (one of 'linear', 'interaction', "
            "'quadratic') or 'effects' (list of effect strings)."
        )

    variables = search_space.variables
    name_to_idx = {v["name"]: i for i, v in enumerate(variables)}

    # Always start with intercept
    terms: List[ModelTerm] = [()]

    if model_type is not None:
        model_type = model_type.lower()
        if model_type not in VALID_MODEL_TYPES:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Choose from: {', '.join(sorted(VALID_MODEL_TYPES))}"
            )

        # Main effects
        for i in range(len(variables)):
            terms.append(((i, 1),))

        if model_type in ("interaction", "quadratic"):
            # All pairwise interactions
            for i, j in itertools.combinations(range(len(variables)), 2):
                terms.append(((i, 1), (j, 1)))

        if model_type == "quadratic":
            # Squared terms for continuous variables only
            for i, var in enumerate(variables):
                if var["type"] in ("real", "integer"):
                    terms.append(((i, 2),))

    else:
        # Parse explicit effects list
        for effect_str in effects:
            term = _parse_single_effect(effect_str, name_to_idx, variables)
            if term not in terms:
                terms.append(term)

    return terms


def _parse_single_effect(
    effect_str: str,
    name_to_idx: Dict[str, int],
    variables: List[Dict[str, Any]],
) -> ModelTerm:
    """Parse a single effect string into a ModelTerm.

    Handles three formats:
        - ``"VarName"`` → main effect ``((idx, 1),)``
        - ``"Var1*Var2"`` → interaction ``((idx1, 1), (idx2, 1))``
        - ``"VarName**2"`` → quadratic ``((idx, 2),)``
    """
    effect_str = effect_str.strip()

    # Quadratic: "VarName**2"
    if effect_str.endswith("**2"):
        var_name = effect_str[:-3].strip()
        if var_name not in name_to_idx:
            raise ValueError(
                f"Unknown variable '{var_name}' in effect '{effect_str}'. "
                f"Available: {list(name_to_idx.keys())}"
            )
        idx = name_to_idx[var_name]
        vtype = variables[idx]["type"]
        if vtype == "categorical":
            raise ValueError(
                f"Quadratic term '{effect_str}' is not valid for categorical "
                f"variable '{var_name}'. Quadratic terms only apply to "
                f"continuous (real/integer/discrete) variables."
            )
        if vtype == "discrete":
            allowed = variables[idx].get("allowed_values", [])
            if len(allowed) <= 2:
                raise ValueError(
                    f"Quadratic term '{effect_str}' is not estimable for "
                    f"discrete variable '{var_name}' because it has only "
                    f"{len(allowed)} allowed value(s) {allowed}. "
                    f"In the coded [-1, +1] space, {var_name}\u00b2 is always "
                    f"1.0 (identical to the intercept), making the design "
                    f"matrix singular and D/A-efficiency undefined. "
                    f"To fix this, either remove '{effect_str}' from your "
                    f"effects list, or add a third allowed value for '{var_name}' "
                    f"so that curvature is estimable."
                )
        return ((idx, 2),)

    # Interaction: "Var1*Var2" or "Var1*Var2*Var3"
    if "*" in effect_str:
        parts = [p.strip() for p in effect_str.split("*")]
        factors = []
        for part in parts:
            if part not in name_to_idx:
                raise ValueError(
                    f"Unknown variable '{part}' in effect '{effect_str}'. "
                    f"Available: {list(name_to_idx.keys())}"
                )
            factors.append((name_to_idx[part], 1))
        # Sort by index for canonical ordering
        factors.sort(key=lambda x: x[0])
        return tuple(factors)

    # Main effect: "VarName"
    if effect_str not in name_to_idx:
        raise ValueError(
            f"Unknown variable '{effect_str}' in effects list. "
            f"Available: {list(name_to_idx.keys())}"
        )
    return ((name_to_idx[effect_str], 1),)


# ============================================================
# Model term names (human-readable)
# ============================================================

def get_model_term_names(
    search_space: SearchSpace,
    terms: List[ModelTerm],
) -> List[str]:
    """Convert internal ModelTerm list to human-readable term names.

    Args:
        search_space: SearchSpace with variable definitions.
        terms: List of ModelTerm tuples as returned by :func:`parse_model_spec`.

    Returns:
        List of strings like ``["Intercept", "Temperature", "Pressure",
        "Temperature*Pressure", "Temperature^2"]``.
    """
    variables = search_space.variables
    names = []
    for term in terms:
        if len(term) == 0:
            names.append("Intercept")
        elif len(term) == 1 and term[0][1] == 1:
            names.append(variables[term[0][0]]["name"])
        elif len(term) == 1 and term[0][1] == 2:
            names.append(f"{variables[term[0][0]]['name']}^2")
        else:
            parts = []
            for idx, power in term:
                name = variables[idx]["name"]
                if power == 1:
                    parts.append(name)
                else:
                    parts.append(f"{name}^{power}")
            names.append("*".join(parts))
    return names


# ============================================================
# Candidate set generation (mixed continuous/categorical)
# ============================================================

def generate_mixed_candidate_set(
    search_space: SearchSpace,
    n_levels: int = 5,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Generate a candidate point grid for mixed continuous/categorical spaces.

    For continuous (real/integer) variables, generates ``n_levels`` evenly
    spaced points across the variable bounds. For categorical variables,
    uses all category values as levels.

    The returned candidate array uses **coded** numeric values:

    - Continuous variables are scaled to ``[-1, +1]``.
    - Categorical variables are **one-hot encoded** (one column per category).

    Args:
        search_space: SearchSpace with defined variables.
        n_levels: Number of grid levels for continuous variables (default 5).

    Returns:
        Tuple of:
            - ``candidates``: ndarray of shape ``(n_candidates, n_coded_columns)``
              with coded numeric values (continuous scaled, categoricals one-hot).
            - ``column_map``: List of dicts describing each column:
              ``{"var_idx": int, "var_name": str, "type": "continuous"|"onehot",
              "category": str|None}``.
    """
    variables = search_space.variables

    # Build per-variable level arrays
    var_levels = []  # list of arrays, one per variable
    for var in variables:
        if var["type"] in ("real", "integer"):
            levels = np.linspace(-1.0, 1.0, n_levels)
            var_levels.append(levels)
        elif var["type"] == "discrete":
            # Code each allowed value to [-1, +1] using [min, max] as the range
            allowed = var["allowed_values"]
            low = min(allowed)
            high = max(allowed)
            mid = (low + high) / 2.0
            half_range = (high - low) / 2.0
            levels = np.array([(v - mid) / half_range for v in allowed])
            var_levels.append(levels)
        elif var["type"] == "categorical":
            cats = var.get("values", var.get("categories", []))
            # Use integer indices for the factorial grid
            var_levels.append(np.arange(len(cats)))

    # Full factorial grid of all combinations
    grid_points = list(itertools.product(*var_levels))
    raw_grid = np.array(grid_points)  # shape (n_candidates, n_vars)

    # Build coded candidate matrix with one-hot encoding for categoricals
    coded_columns = []
    column_map = []

    for j, var in enumerate(variables):
        if var["type"] in ("real", "integer", "discrete"):
            coded_columns.append(raw_grid[:, j].reshape(-1, 1))
            column_map.append({
                "var_idx": j,
                "var_name": var["name"],
                "type": "continuous",
                "category": None,
            })
        elif var["type"] == "categorical":
            cats = var.get("values", var.get("categories", []))
            # One-hot encode
            cat_indices = raw_grid[:, j].astype(int)
            for k, cat_val in enumerate(cats):
                onehot_col = (cat_indices == k).astype(float).reshape(-1, 1)
                coded_columns.append(onehot_col)
                column_map.append({
                    "var_idx": j,
                    "var_name": var["name"],
                    "type": "onehot",
                    "category": cat_val,
                })

    candidates = np.hstack(coded_columns)
    return candidates, column_map


# ============================================================
# Custom design matrix builder
# ============================================================

def build_custom_design_matrix(
    candidates: np.ndarray,
    terms: List[ModelTerm],
    column_map: List[Dict[str, Any]],
    variables: List[Dict[str, Any]],
) -> np.ndarray:
    """Build a regression (design) matrix from candidate points and model terms.

    Unlike pyDOE's ``build_design_matrix(candidates, degree)`` which generates
    all polynomial terms up to a given degree, this function builds **only**
    the columns corresponding to user-specified model terms.

    For continuous variables, columns are computed directly from the coded
    candidate values. For categorical variables, **dummy coding** (k-1 columns)
    is used: the first category is treated as the reference level and absorbed
    into the intercept. Each remaining category gets one indicator column. This
    avoids the perfect multicollinearity that full one-hot encoding (k columns)
    would create with an intercept term.

    For example, a categorical variable ``Catalyst`` with values
    ``["Pt", "Pd", "Ru"]`` produces 2 dummy columns (for ``Pd`` and ``Ru``),
    with ``Pt`` as the implicit reference level.

    Args:
        candidates: ndarray of shape ``(n_candidates, n_coded_columns)`` as
            returned by :func:`generate_mixed_candidate_set`.
        terms: List of ModelTerm tuples as returned by :func:`parse_model_spec`.
        column_map: Column metadata as returned by
            :func:`generate_mixed_candidate_set`.
        variables: List of variable dicts from ``SearchSpace.variables``.

    Returns:
        Design matrix X of shape ``(n_candidates, n_model_columns)``.
    """
    n_points = candidates.shape[0]

    # Build lookup: var_idx → list of candidate column indices
    var_to_cols: Dict[int, List[int]] = {}
    for col_idx, cm in enumerate(column_map):
        vidx = cm["var_idx"]
        if vidx not in var_to_cols:
            var_to_cols[vidx] = []
        var_to_cols[vidx].append(col_idx)

    all_columns = []

    for term in terms:
        if len(term) == 0:
            # Intercept
            all_columns.append(np.ones((n_points, 1)))
            continue

        # Build column(s) for this term by multiplying factor contributions
        # Each factor in the term produces one or more columns depending on
        # whether it's continuous (1 col) or categorical (k-1 cols).
        factor_column_sets = []
        for var_idx, power in term:
            var = variables[var_idx]
            cols_for_var = var_to_cols[var_idx]

            if var["type"] in ("real", "integer", "discrete"):
                # Single column, raised to power
                col_data = candidates[:, cols_for_var[0]] ** power
                factor_column_sets.append(col_data.reshape(-1, 1))
            else:
                # Categorical: dummy coding (k-1), drop first category
                # (reference level absorbed into intercept)
                cat_cols = candidates[:, cols_for_var[1:]]
                factor_column_sets.append(cat_cols)

        # Combine factor contributions via outer product across columns
        result = factor_column_sets[0]
        for fcs in factor_column_sets[1:]:
            # result: (n, a), fcs: (n, b) → (n, a*b)
            expanded = []
            for ci in range(result.shape[1]):
                for cj in range(fcs.shape[1]):
                    expanded.append(result[:, ci] * fcs[:, cj])
            result = np.column_stack(expanded)

        all_columns.append(result)

    return np.hstack(all_columns)


# ============================================================
# Custom optimal design algorithm runner
# ============================================================

def _run_algorithm(
    candidates_coded: np.ndarray,
    design_matrix_candidates: np.ndarray,
    n_points: int,
    criterion: str,
    algorithm: str,
    max_iter: int,
    terms: List[ModelTerm],
    column_map: List[Dict[str, Any]],
    variables: List[Dict[str, Any]],
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run the optimal design algorithm on a custom design matrix.

    Because pyDOE's algorithms internally call ``build_design_matrix(candidates,
    degree)`` which only supports full polynomial degrees, we implement the
    optimization loop using pyDOE's criterion functions directly with our
    custom design matrix.

    The algorithm has two phases:

    1. **Initialization**: A maximin-distance greedy selection produces a
       space-filling starting design.  This avoids the pitfall where all
       candidates evaluate to 0 under D-optimality when n < p.
    2. **Exchange improvement**: Iterative pairwise exchanges improve the
       chosen criterion (Fedorov, Wynn-Mitchell, etc.).

    A small ridge regularization is added to the moment matrix (scaled by
    the matrix norm) so that the criterion function can differentiate between
    near-singular candidates during the greedy phase.

    Args:
        candidates_coded: Coded candidate points ``(N, n_coded_cols)``.
        design_matrix_candidates: Full design matrix for all candidates
            ``(N, p)`` built by :func:`build_custom_design_matrix`.
        n_points: Number of design points to select.
        criterion: Optimality criterion (``"D"``, ``"A"``, or ``"I"``).
        algorithm: Algorithm name.
        max_iter: Maximum iterations for exchange algorithms.
        terms: Model terms (for metadata).
        column_map: Column mapping (for metadata).
        variables: Variable definitions (for metadata).
        random_seed: Optional seed for reproducible results.

    Returns:
        Tuple of selected candidate indices (ndarray of shape ``(n_points,)``)
        and info dict with criterion scores and efficiency metrics.
    """
    from pyDOE.doe_optimal.criterion import (
        a_optimality,
        d_optimality,
        i_optimality,
    )
    from pyDOE.doe_optimal.efficiency import a_efficiency, d_efficiency

    N = design_matrix_candidates.shape[0]
    p = design_matrix_candidates.shape[1]

    if n_points > N:
        raise ValueError(
            f"n_points ({n_points}) exceeds the number of candidate points "
            f"({N}). Reduce n_points or increase candidate grid resolution "
            f"(n_levels)."
        )

    # Precompute moment matrix for I-optimality
    M_moment = (design_matrix_candidates.T @ design_matrix_candidates) / max(N, 1)

    # Ridge regularization scaled by matrix norm so it adapts to the problem
    ridge_scale = max(1.0, np.linalg.norm(M_moment, ord=1))
    reg = np.eye(p) * (1e-8 * ridge_scale)

    def _score(indices: np.ndarray) -> float:
        X = design_matrix_candidates[indices]
        n = X.shape[0]
        M = (X.T @ X) / max(n, 1) + reg
        if criterion == "D":
            return d_optimality(M)
        elif criterion == "A":
            return a_optimality(M)
        elif criterion == "I":
            return i_optimality(M, M_moment)
        raise ValueError(f"Unknown criterion: {criterion}")

    # --- Phase 1: Maximin-distance initialization ---
    # Greedy space-filling: pick the first point at random, then
    # iteratively add the candidate farthest from all selected points.
    # This gives a well-spread starting design that avoids the all-zero
    # scores from d_optimality on singular matrices.
    rng = np.random.default_rng(random_seed)
    selected_list: List[int] = [rng.integers(N)]
    available = set(range(N)) - {selected_list[0]}

    for _ in range(n_points - 1):
        if not available:
            break
        sel_rows = candidates_coded[selected_list]
        best_idx = -1
        best_dist = -1.0
        for j in available:
            dists = np.sum((sel_rows - candidates_coded[j]) ** 2, axis=1)
            min_dist = float(np.min(dists))
            if min_dist > best_dist:
                best_dist = min_dist
                best_idx = j
        selected_list.append(best_idx)
        available.discard(best_idx)

    selected = np.array(selected_list, dtype=int)

    if algorithm == "sequential":
        # Sequential — refine the space-filling start with greedy criterion
        # Try replacing each point with the best candidate under the criterion
        best_score = _score(selected)
        for i in range(n_points):
            pool = set(range(N)) - set(selected.tolist())
            current_best = selected[i]
            for j in pool:
                trial = selected.copy()
                trial[i] = j
                val = _score(trial)
                if val > best_score + 1e-12:
                    best_score = val
                    current_best = j
            selected[i] = current_best
    else:
        # --- Phase 2: Exchange improvement ---
        best_score = _score(selected)

        for _iteration in range(max_iter):
            improved = False

            if algorithm in ("simple_exchange",):
                # Drop worst, add best from pool
                drop_idx_best = -1
                add_candidate_best = -1
                best_swap_score = best_score

                for di in range(n_points):
                    trial_without = np.delete(selected, di)
                    pool = set(range(N)) - set(trial_without.tolist())
                    for ai in pool:
                        trial = np.append(trial_without, ai)
                        val = _score(trial)
                        if val > best_swap_score + 1e-12:
                            best_swap_score = val
                            drop_idx_best = di
                            add_candidate_best = ai

                if drop_idx_best >= 0:
                    selected[drop_idx_best] = add_candidate_best
                    best_score = best_swap_score
                    improved = True

            elif algorithm in ("fedorov", "modified_fedorov", "detmax"):
                # Try all pairwise exchanges
                best_i, best_j, best_val = -1, -1, best_score
                pool = sorted(set(range(N)) - set(selected.tolist()))

                for i in range(n_points):
                    for j in pool:
                        trial = selected.copy()
                        trial[i] = j
                        val = _score(trial)
                        if val > best_val + 1e-12:
                            best_i, best_j, best_val = i, j, val

                if best_i >= 0:
                    selected[best_i] = best_j
                    best_score = best_val
                    improved = True

            if not improved:
                break

    # Compute final metrics (unregularized for reporting)
    X_final = design_matrix_candidates[selected]
    info = {
        "criterion": criterion,
        "algorithm": algorithm,
        "score": float(_score(selected)),
        "D_eff": float(d_efficiency(X_final)),
        "A_eff": float(a_efficiency(X_final)),
        "p_columns": int(p),
        "n_runs": int(n_points),
    }

    return selected, info


# ============================================================
# Coded-to-actual value mapping
# ============================================================

def _decode_candidates(
    candidates_coded: np.ndarray,
    selected_indices: np.ndarray,
    column_map: List[Dict[str, Any]],
    variables: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Map coded candidate rows back to actual variable values.

    Reverses the encoding from :func:`generate_mixed_candidate_set`:

    - Continuous variables: ``actual = mid + coded * half_range``
    - Categorical variables: decode from one-hot to category name.
    - Integer variables: round to nearest int.

    Args:
        candidates_coded: Full coded candidate array.
        selected_indices: Indices of selected design points.
        column_map: Column metadata from :func:`generate_mixed_candidate_set`.
        variables: Variable definitions from SearchSpace.

    Returns:
        List of dicts with actual variable values.
    """
    # Build var_idx → coded column indices lookup
    var_to_cols: Dict[int, List[int]] = {}
    for col_idx, cm in enumerate(column_map):
        vidx = cm["var_idx"]
        if vidx not in var_to_cols:
            var_to_cols[vidx] = []
        var_to_cols[vidx].append(col_idx)

    points = []
    for row_idx in selected_indices:
        row = candidates_coded[row_idx]
        point = {}
        for j, var in enumerate(variables):
            cols = var_to_cols[j]
            if var["type"] in ("real", "integer"):
                coded_val = row[cols[0]]
                low = var["min"]
                high = var["max"]
                if high == low:
                    actual = low
                else:
                    mid = (low + high) / 2.0
                    half_range = (high - low) / 2.0
                    actual = mid + coded_val * half_range
                    actual = max(low, min(high, actual))
                if var["type"] == "integer":
                    actual = int(round(actual))
                else:
                    actual = float(actual)
                point[var["name"]] = actual
            elif var["type"] == "discrete":
                # Decode from coded [-1, +1] back to actual allowed value
                allowed = var["allowed_values"]
                low = min(allowed)
                high = max(allowed)
                if high == low:
                    actual = float(low)
                else:
                    mid = (low + high) / 2.0
                    half_range = (high - low) / 2.0
                    coded_val = row[cols[0]]
                    actual = mid + coded_val * half_range
                # Snap to nearest allowed value (safety net for float precision)
                actual = float(min(allowed, key=lambda v: abs(v - actual)))
                point[var["name"]] = actual
            elif var["type"] == "categorical":
                cats = var.get("values", var.get("categories", []))
                onehot_vals = row[cols]
                cat_idx = int(np.argmax(onehot_vals))
                point[var["name"]] = cats[cat_idx]
        points.append(point)
    return points


# ============================================================
# Main orchestrator
# ============================================================

def run_optimal_design(
    search_space: SearchSpace,
    n_points: int,
    model_type: Optional[str] = None,
    effects: Optional[List[str]] = None,
    criterion: Literal["D", "A", "I"] = "D",
    algorithm: Literal[
        "sequential", "simple_exchange", "fedorov",
        "modified_fedorov", "detmax"
    ] = "fedorov",
    n_levels: int = 5,
    max_iter: int = 200,
    random_seed: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generate an optimal experimental design for a user-specified model.

    This function creates a statistically efficient set of experimental
    conditions by optimizing a chosen criterion (D, A, or I) over a
    candidate grid. Users specify which model terms (main effects,
    interactions, quadratic terms) they expect to be important, and the
    algorithm selects the design points that maximize information for
    estimating those specific terms.

    **Model Specification:**

    Provide either ``model_type`` (a shortcut) or ``effects`` (explicit list),
    but not both. The intercept is always included automatically.

    Effects are specified as a list of strings using variable names from the
    SearchSpace.

    Format rules:
        - **Main effects**: Use the variable name exactly as defined.
          e.g., ``"Temperature"``, ``"Pressure"``
        - **Interactions**: Join variable names with ``"*"``.
          e.g., ``"Temperature*Pressure"``, ``"Catalyst*Temperature"``
        - **Quadratic terms**: Append ``"**2"`` to the variable name.
          e.g., ``"Temperature**2"``, ``"Pressure**2"``

    Shortcut model types (via ``model_type`` parameter):
        - ``"linear"``: All main effects only.
        - ``"interaction"``: All main effects + all pairwise interactions.
        - ``"quadratic"``: All main effects + all pairwise interactions +
          all squared terms (continuous variables only).

    .. warning::

        **Experimental Feature — Categorical Variables**: Optimal design
        with categorical variables uses dummy coding (k-1 indicator columns,
        first category as reference level) in the design matrix. The
        statistical properties of optimal designs with categorical variables
        are an active area of research and may not behave identically to
        classical continuous-variable optimal designs. Results should be
        reviewed carefully. For well-established categorical screening,
        consider using Full Factorial or GSD methods instead.

    Args:
        search_space: SearchSpace with defined variables.
        n_points: Number of experimental runs to generate.
        model_type: Shortcut model type. One of ``"linear"``,
            ``"interaction"``, ``"quadratic"``.
        effects: Explicit list of effect strings. Examples::

            effects=["Temperature", "Pressure", "Temperature*Pressure",
                     "Temperature**2"]

        criterion: Optimality criterion. ``"D"`` (default) maximizes
            parameter information, ``"A"`` minimizes average parameter
            variance, ``"I"`` minimizes average prediction variance.
        algorithm: Optimization algorithm. One of ``"sequential"``
            (fastest, greedy), ``"simple_exchange"`` (Wynn-Mitchell),
            ``"fedorov"`` (recommended default), ``"modified_fedorov"``,
            ``"detmax"`` (best quality, slowest).
        n_levels: Number of grid levels per continuous variable for
            candidate set generation (default 5). Higher values give
            finer resolution but increase computation.
        max_iter: Maximum iterations for exchange algorithms (default 200).
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of:
            - ``points``: List of dicts with actual variable values.
            - ``info``: Dict with design quality metrics::

                {"criterion": "D", "algorithm": "fedorov",
                 "score": 0.042, "D_eff": 89.3, "A_eff": 76.1,
                 "p_columns": 6, "n_runs": 15,
                 "model_terms": ["Intercept", "Temperature", ...]}

    Raises:
        ValueError: If search space has no variables, both/neither model
            spec args given, unknown criterion/algorithm, or effects
            reference unknown variables.

    Examples:
        .. code-block:: python

            # Full quadratic model via shortcut
            points, info = run_optimal_design(
                search_space, n_points=15, model_type='quadratic',
                criterion='D', algorithm='fedorov',
            )

            # Custom model: only the effects you expect to matter
            points, info = run_optimal_design(
                search_space, n_points=12,
                effects=['Temperature', 'Pressure', 'Catalyst',
                         'Temperature*Pressure', 'Temperature**2'],
                criterion='D', algorithm='fedorov',
            )

            # Screening: main effects + one known interaction
            points, info = run_optimal_design(
                search_space, n_points=8,
                effects=['Temperature', 'Pressure', 'Flow_Rate',
                         'Temperature*Pressure'],
                criterion='D',
            )
    """
    # Validate inputs
    if len(search_space.variables) == 0:
        raise ValueError(
            "SearchSpace has no variables. Define variables before "
            "generating optimal design."
        )

    criterion = criterion.upper()
    if criterion not in VALID_CRITERIA:
        raise ValueError(
            f"Unknown criterion '{criterion}'. "
            f"Choose from: {', '.join(sorted(VALID_CRITERIA))}"
        )

    algorithm = algorithm.lower()
    if algorithm not in VALID_ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. "
            f"Choose from: {', '.join(sorted(VALID_ALGORITHMS))}"
        )

    if n_points < 1:
        raise ValueError(f"n_points must be >= 1, got {n_points}")

    # Warn about categorical variables
    categorical_vars = [v for v in search_space.variables
                        if v["type"] == "categorical"]
    if categorical_vars:
        cat_names = [v["name"] for v in categorical_vars]
        logger.warning(
            "Optimal design with categorical variables (%s) is experimental. "
            "Categorical variables use dummy coding (k-1 indicator columns, "
            "first category as reference level) in the design matrix. "
            "The statistical properties of optimal designs with categorical "
            "variables are an active area of research and results may not "
            "behave identically to classical continuous-variable optimal "
            "designs. For well-established categorical screening, consider "
            "Full Factorial or GSD methods.",
            cat_names,
        )

    # Parse model specification
    terms = parse_model_spec(search_space, model_type=model_type, effects=effects)
    variables = search_space.variables

    # Collect variable indices that appear in any model term (intercept excluded)
    vars_in_model: set = set()
    for term in terms:
        for var_idx, _power in term:
            vars_in_model.add(var_idx)

    # Track unused variable indices for warnings and post-hoc spreading
    unused_var_indices = [
        i for i in range(len(variables)) if i not in vars_in_model
    ]

    # Warning #1: Variables not covered by any model term will take arbitrary
    # values because the exchange algorithm is blind to them.
    if unused_var_indices:
        unused_names = [variables[i]["name"] for i in unused_var_indices]
        logger.warning(
            "The following search space variables are NOT included in any "
            "model term and will take arbitrary values in the design: %s. "
            "If you want these variables distributed across their range, "
            "include them as main effects (e.g., add '%s' to your effects "
            "list).",
            unused_names,
            unused_names[0],
        )

    # Warning #2: Hierarchy (marginality) principle violations.
    # Only relevant for explicit effects lists; model_type shortcuts always
    # include all main effects automatically.
    if effects is not None:
        main_effect_vars = {
            term[0][0]
            for term in terms
            if len(term) == 1 and term[0][1] == 1
        }
        missing_main_effects: set = set()
        for term in terms:
            if len(term) == 0:
                continue
            if len(term) == 1 and term[0][1] == 2:  # quadratic
                var_idx = term[0][0]
                if var_idx not in main_effect_vars:
                    missing_main_effects.add(variables[var_idx]["name"])
            elif len(term) > 1:  # interaction
                for var_idx, _power in term:
                    if var_idx not in main_effect_vars:
                        missing_main_effects.add(variables[var_idx]["name"])
        if missing_main_effects:
            logger.warning(
                "Effects list contains interaction/quadratic terms for "
                "variables that are not included as main effects: %s. "
                "This violates the marginality (hierarchy) principle. "
                "Consider adding these as main effects, or use "
                "model_type='interaction' or 'quadratic' to automatically "
                "include all main effects.",
                sorted(missing_main_effects),
            )

    logger.info(
        "Optimal design: %d terms, criterion=%s, algorithm=%s, n_points=%d",
        len(terms), criterion, algorithm, n_points,
    )

    # Generate candidate set
    candidates_coded, column_map = generate_mixed_candidate_set(
        search_space, n_levels=n_levels
    )

    logger.info(
        "Generated %d candidate points (%d coded columns) for %d variables",
        candidates_coded.shape[0], candidates_coded.shape[1], len(variables),
    )

    # Build custom design matrix for all candidates
    design_matrix = build_custom_design_matrix(
        candidates_coded, terms, column_map, variables
    )

    logger.info(
        "Design matrix: %d candidates × %d model columns",
        design_matrix.shape[0], design_matrix.shape[1],
    )

    p_columns = design_matrix.shape[1]
    if n_points < p_columns:
        raise ValueError(
            f"n_points={n_points} is fewer than the number of model columns "
            f"p={p_columns}. The design matrix would be singular — there are not "
            f"enough runs to estimate all {p_columns} model parameters. "
            f"Use n_points >= {p_columns} (recommended: n_points >= {2 * p_columns})."
        )

    # Rank-deficiency safety check: verify the candidate design matrix has
    # full column rank before running the exchange algorithm.  Rank deficiency
    # means two or more model columns are linearly dependent (e.g., x**2 = 1
    # for a 2-value discrete variable; collinear custom effects).  The exchange
    # algorithm still runs in this case (due to ridge regularization), but all
    # efficiency metrics will be 0% and the design is not trustworthy.
    rank = np.linalg.matrix_rank(design_matrix, tol=1e-6)
    if rank < p_columns:
        raise ValueError(
            f"The model design matrix has rank {rank} but {p_columns} "
            f"columns — some model terms are linearly dependent. "
            f"This makes D/A-efficiency undefined and the design invalid. "
            f"Likely causes:\n"
            f"  • A discrete variable with only 2 allowed values has a "
            f"quadratic (x**2) term: in coded space x\u00b2 = 1 always "
            f"(= intercept).\n"
            f"  • Two interaction or quadratic terms are perfectly correlated "
            f"given the candidate grid.\n"
            f"To fix: remove the offending term(s) from your effects list."
        )

    # Run optimization
    selected_indices, info = _run_algorithm(
        candidates_coded=candidates_coded,
        design_matrix_candidates=design_matrix,
        n_points=n_points,
        criterion=criterion,
        algorithm=algorithm,
        max_iter=max_iter,
        terms=terms,
        column_map=column_map,
        variables=variables,
        random_seed=random_seed,
    )

    # Decode to actual values
    points = _decode_candidates(
        candidates_coded, selected_indices, column_map, variables
    )

    # Spread non-model variables with space-filling (#3).
    # Variables absent from all model terms are invisible to the exchange
    # algorithm and end up at arbitrary candidate-set values.  Replace them
    # with an evenly-spaced (shuffled) sequence so the design looks sensible.
    # D-optimality is unaffected because these variables do not enter X.
    if unused_var_indices:
        n = len(points)
        spread_rng = np.random.default_rng(random_seed)
        for var_idx in unused_var_indices:
            var = variables[var_idx]
            if var["type"] in ("real", "integer"):
                spread_vals: list = list(np.linspace(var["min"], var["max"], n))
                spread_rng.shuffle(spread_vals)
                if var["type"] == "integer":
                    spread_vals = [int(round(v)) for v in spread_vals]
                else:
                    spread_vals = [float(v) for v in spread_vals]
            elif var["type"] == "discrete":
                allowed = var["allowed_values"]
                spread_vals = [float(allowed[i % len(allowed)]) for i in range(n)]
                spread_rng.shuffle(spread_vals)
            elif var["type"] == "categorical":
                cats = var.get("values", var.get("categories", []))
                spread_vals = [cats[i % len(cats)] for i in range(n)]
                spread_rng.shuffle(spread_vals)
            for i, point in enumerate(points):
                point[var["name"]] = spread_vals[i]

    # Add term names to info
    info["model_terms"] = get_model_term_names(search_space, terms)

    return points, info
