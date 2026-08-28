"""
Design of Experiments (DoE) - Initial sampling strategies for Bayesian optimization.

This module provides methods for generating initial experimental designs before
starting the optimization loop. Supported methods:

Space-filling:
- Random sampling
- Latin Hypercube Sampling (LHS)
- Sobol sequences
- Halton sequences
- Hammersly sequences

Classical RSM:
- Full Factorial
- Fractional Factorial
- Central Composite Design (CCD)
- Box-Behnken

Screening:
- Plackett-Burman (ultra-efficient 2-level main-effect screening)
- Generalized Subset Design (fractional factorial for mixed/multi-level factors)
"""

from typing import List, Dict, Optional, Literal, Any, Tuple
from functools import reduce
import copy
import operator
import numpy as np
from skopt.sampler import Lhs, Sobol, Hammersly
from skopt.space import Real, Integer, Categorical
from alchemist_core.data.search_space import SearchSpace
# The DoE feasibility tolerance has one definition, not one per call site.
# constrained_region owns it because it is also what region_is_provably_empty
# must relax; a second spelling here is how the two silently drift apart.
from alchemist_core.utils.constrained_region import DOE_ATOL, DOE_RTOL
from alchemist_core.config import get_logger

logger = get_logger(__name__)

SPACE_FILLING_METHODS = {"random", "lhs", "sobol", "halton", "hammersly"}
CLASSICAL_METHODS = {"full_factorial", "fractional_factorial", "ccd", "box_behnken",
                     "plackett_burman", "gsd", "optimal"}

# Standard fractional factorial generators for common factor counts.
# These provide Resolution III or better designs for screening.
_DEFAULT_GENERATORS = {
    3: "a b ab",               # 2^(3-1), Resolution III
    4: "a b c abc",            # 2^(4-1), Resolution IV
    5: "a b c ab ac",          # 2^(5-2), Resolution III
    6: "a b c ab ac bc",       # 2^(6-3), Resolution III
    7: "a b c ab ac bc abc",   # 2^(7-4), Resolution III
}


class DesignNotEstimableError(ValueError):
    """A constrained classical design lost points its implied model needs."""


# The model a classical design exists to estimate. Used to decide whether a
# design that lost points to a constraint is still worth returning.
IMPLIED_MODEL = {
    "ccd": "quadratic",
    "box_behnken": "quadratic",
    "fractional_factorial": "interaction",
    "plackett_burman": "linear",
    "gsd": "linear",
    # full_factorial depends on n_levels; resolved in _implied_model_type.
}


def _implied_model_type(method: str, n_levels: int) -> str:
    """Model type a given classical design is built to estimate."""
    if method == "full_factorial":
        return "quadratic" if n_levels >= 3 else "interaction"
    return IMPLIED_MODEL.get(method, "linear")


def _as_json_native(value: Any) -> Any:
    """The Python scalar holding exactly the value of a numpy one.

    The space-filling samplers hand back numpy scalars where the classical
    construction block (see ``_coded_to_actual``) hands back Python ones, and
    only one of those four leaks was ever visible::

        real         random=np.float64  lhs/sobol/halton/hammersly=float
        integer      np.int64 on all five                     <- fatal
        discrete     np.float64 on all five                   <- silent
        categorical  np.str_ / np.int64 on all five           <- silent

    ``np.float64`` subclasses ``float`` and ``np.str_`` subclasses ``str``, so
    three of the four rows serialize by accident and pass ``isinstance``.
    ``np.int64`` does **not** subclass ``int``, so the integer row alone
    reached the JSON encoder as "unknown type" and turned every integer
    ``POST /initial-design`` into a 400. Fixing only the row that threw would
    leave the other three one encoder change away from the same failure.

    ``.item()`` is used rather than a coercion chosen from the variable's
    declared type, for two reasons.

    First, it cannot change a value. ``int(...)``/``float(...)``/``str(...)``
    keyed off ``var['type']`` would have to decide what a *categorical* is, and
    a categorical is not necessarily a string -- ``values=[1, 2, 3]`` is
    accepted and yields ``np.int64``, which ``str()`` would silently rewrite to
    ``'1'``. ``.item()`` maps every numpy scalar to its exact Python
    counterpart (``np.int64``->``int``, ``np.floating``->``float``,
    ``np.str_``->``str``, ``np.bool_``->``bool``) with no conversion and no
    reformatting; float64 round-trips bit-for-bit.

    Second, a declared-type lookup would have to index a variable list by
    position, and the only list whose positions line up with a sample is
    ``search_space.get_dimension_names()`` -- not ``search_space.variables``,
    which spans ``context`` variables that own no dimension. A type-aware
    coercion keyed off the wrong one of those would coerce values to the wrong
    type; that misalignment was a live defect when this helper was written and
    is fixed at the zip below, but the coupling is what this argument is
    about. ``.item()`` is correct regardless of how the zip pairs up.

    Non-numpy values pass through untouched, so the already-clean ``real`` and
    classical paths are byte-identical -- which is what keeps Task 1's golden
    unconstrained fixture green.
    """
    if isinstance(value, np.generic):
        return value.item()
    return value


def generate_initial_design(
    search_space: SearchSpace,
    method: Literal[
        "random", "lhs", "sobol", "halton", "hammersly",
        "full_factorial", "fractional_factorial", "ccd", "box_behnken",
        "plackett_burman", "gsd", "optimal"
    ] = "lhs",
    n_points: Optional[int] = None,
    random_seed: Optional[int] = None,
    lhs_criterion: str = "maximin",
    # Classical design parameters
    n_levels: int = 2,
    n_center: int = 1,
    generators: Optional[str] = None,
    ccd_alpha: str = "orthogonal",
    ccd_face: str = "circumscribed",
    # GSD parameters
    gsd_reduction: int = 2,
    # Optimal design parameters
    model_type: Optional[str] = None,
    effects: Optional[List[str]] = None,
    criterion: str = "D",
    algorithm: str = "fedorov",
    max_iter: int = 200,
    allow_infeasible: bool = False,
) -> List[Dict[str, Any]]:
    """
    Generate initial experimental design using specified sampling strategy.

    This function creates a set of experimental conditions to evaluate before
    starting Bayesian optimization.

    **Space-filling methods** (take n_points as input):
    - **random**: Uniform random sampling
    - **lhs**: Latin Hypercube Sampling (recommended for most cases)
    - **sobol**: Sobol quasi-random sequences (low discrepancy)
    - **halton**: Halton sequences (via Hammersly sampler)
    - **hammersly**: Hammersly sequences (low discrepancy)

    **Classical RSM methods** (run count determined by design structure):
    - **full_factorial**: All combinations of factor levels
    - **fractional_factorial**: Subset of full factorial using generators
    - **ccd**: Central Composite Design (factorial + axial + center)
    - **box_behnken**: Box-Behnken design (3+ continuous factors)

    **Screening methods** (run count determined by design structure):
    - **plackett_burman**: Ultra-efficient 2-level screening (continuous only)
    - **gsd**: Generalized Subset Design (supports mixed categorical/continuous)

    **Optimal design** (user-specified model structure):
    - **optimal**: Generate a statistically efficient design optimized for
      estimating specific model terms (main effects, interactions, quadratic
      terms). Requires ``n_points`` and either ``model_type`` or ``effects``.

    Args:
        search_space: SearchSpace object with defined variables
        method: Sampling method to use
        n_points: Number of points (required for space-filling and optimal;
            ignored for classical)
        random_seed: Random seed for reproducibility
        lhs_criterion: Criterion for LHS ("maximin", "correlation", "ratio")
        n_levels: Levels per continuous factor for full factorial (2 or 3),
            or candidate grid resolution for optimal design (default 5 for
            optimal, 2 for factorial)
        n_center: Number of center point replicates (classical designs)
        generators: Fractional factorial generator string (e.g. "a b ab")
        ccd_alpha: CCD alpha type ("orthogonal" or "rotatable")
        ccd_face: CCD face type ("circumscribed", "inscribed", or "faced")
        gsd_reduction: GSD reduction factor (>=2); larger means fewer runs
        model_type: Optimal design model shortcut. One of "linear"
            (main effects only), "interaction" (main + all pairwise), or
            "quadratic" (main + pairwise + squared terms for continuous vars).
            Used only when method="optimal".
        effects: Optimal design custom effects list. Strings using variable
            names from the SearchSpace. Format rules:

            - Main effects: ``"Temperature"``, ``"Pressure"``
            - Interactions: ``"Temperature*Pressure"``
            - Quadratic terms: ``"Temperature**2"``

            Used only when method="optimal". Specify either ``model_type``
            or ``effects``, not both.
        criterion: Optimality criterion ("D", "A", or "I").
            Used only when method="optimal". Default "D".
        algorithm: Optimal design algorithm. One of "sequential",
            "simple_exchange", "fedorov" (default), "modified_fedorov",
            "detmax". Used only when method="optimal".
        max_iter: Maximum iterations for optimal design exchange algorithms.
            Default 200. Used only when method="optimal".
        allow_infeasible: For constrained classical designs, return the
            feasible remnant with a warning even when the design's implied
            model is no longer estimable. Default False raises instead.

    Returns:
        List of dictionaries, each containing variable names and values.
        Does NOT include 'Output' column - experiments need to be evaluated.

    Raises:
        ValueError: If search_space has no variables, method is unknown,
                    or design is incompatible with the search space
    """
    # Validate inputs
    if len(search_space.variables) == 0:
        raise ValueError("SearchSpace has no variables. Define variables before generating initial design.")

    # Default n_points for space-filling methods
    if method in SPACE_FILLING_METHODS and n_points is None:
        n_points = 10

    if method in SPACE_FILLING_METHODS and n_points < 1:
        raise ValueError(f"n_points must be >= 1, got {n_points}")

    # Optimal design requires n_points
    if method == "optimal" and n_points is None:
        raise ValueError(
            "n_points is required for optimal design. Specify the number "
            "of experimental runs to generate."
        )

    # A generator owned by this call, never the process-global one.
    #
    # np.random.seed() sets global state, and the samplers below drew from it.
    # That was atomic only because POST /initial-design used to run the whole
    # generation on the event loop: once it moved to a worker thread (so that
    # one design stops blocking every other request), two concurrent seeded
    # requests interleave their draws and neither returns the design its seed
    # names. The seed is part of this API's contract, so it cannot depend on
    # whether another request happens to be in flight.
    #
    # RandomState(seed) is the same MT19937 stream np.random.seed(seed)
    # installs globally, and the samplers consume it in the same order, so
    # every previously-generated seeded design is reproduced bit-for-bit --
    # which is what keeps Task 1's golden fixture green. optimal_design.py
    # already does this with default_rng; doe.py was the last global-seed site
    # a threadpooled route could reach.
    rng = np.random.RandomState(random_seed)
    if random_seed is not None:
        logger.info(f"Set random seed to {random_seed} for reproducibility")

    # Route to appropriate method
    if method in SPACE_FILLING_METHODS:
        # A private copy of the dimensions, not the SearchSpace's own list.
        #
        # skopt's samplers mutate the Dimension objects they are handed:
        # Lhs/Sobol/Hammersly.generate does
        #     transformer = space.get_transformer()
        #     space.set_transformer("normalize")   # mutates the Dimensions
        #     ... inverse_transform(...) ...       # needs that transformer
        #     space.set_transformer(transformer)   # restores it
        # and Space(dimensions) keeps references rather than copies. Two
        # designs generated concurrently from one session therefore share
        # those objects, and if one restores the transformer while the other
        # is between "normalize" and its inverse_transform, the second gets
        # its points back still normalized -- a design silently squashed into
        # [0,1] instead of spanning the declared bounds. It is *in* bounds and
        # the right shape, so nothing downstream can notice.
        #
        # Serialized on the event loop this could not happen; POST
        # /initial-design running in a worker thread is what makes two
        # generations concurrent. Copying is what keeps that change safe.
        skopt_space = copy.deepcopy(search_space.skopt_dimensions)
        # Names paired index-for-index with skopt_dimensions, which omits
        # `context` variables. Iterating search_space.variables instead zips a
        # 2-value sample against 3 names whenever a context variable sits
        # anywhere but last: the real variable after it is dropped from the
        # design entirely and its value is emitted under the context
        # variable's name. With a constraint registered that is worse than a
        # corrupt design -- filter_feasible sums only the terms whose column
        # is present, so the reject-and-resample loop below screens the
        # truncated points against a constraint that has silently lost a term
        # and returns points violating the real one while reporting success.
        variable_names = search_space.get_dimension_names()

        def _sample(n):
            if method == "random":
                s = _random_sampling(skopt_space, n, random_state=rng)
            elif method == "lhs":
                s = _lhs_sampling(skopt_space, n, lhs_criterion, random_state=rng)
            elif method == "sobol":
                s = _sobol_sampling(skopt_space, n, random_state=rng)
            else:  # halton / hammersly
                s = _hammersly_sampling(skopt_space, n, random_state=rng)
            return [{name: _as_json_native(value)
                     for name, value in zip(variable_names, sample)}
                    for sample in s]

        has_constraints = bool(getattr(search_space, 'constraints', None))
        if not has_constraints:
            points = _sample(n_points)
        else:
            # Two searches that cannot succeed are refused before they start.
            # The loop below discovers failure only by exhausting the
            # escalation, and its last round draws n_points*4096 samples --
            # for n_points=6 that is a single 24576-point maximin LHS batch,
            # which costs minutes on its own because the criterion is
            # quadratic in the batch size. `generate_optimal_design` has
            # always raised InfeasibleRegionError up front for the empty
            # case (via augment_with_boundary); this is the space-filling
            # path learning the same check.
            #
            # Both predicates are one-directional proofs and return False
            # whenever they cannot prove their case, so a region that is
            # merely *small* -- the case that must keep working -- still
            # falls through to the loop and still succeeds. The two are
            # reported separately because "nothing is feasible" and "the
            # feasible set is a zero-volume slice" need different remedies.
            from alchemist_core.utils.constrained_region import (
                InfeasibleRegionError,
                region_is_provably_empty,
                region_is_provably_measure_zero,
            )
            if region_is_provably_empty(search_space, atol=DOE_ATOL):
                raise InfeasibleRegionError(
                    f"The registered input constraints leave no feasible point "
                    f"anywhere within the variable bounds, so no '{method}' "
                    f"design can be generated. This is the constraint set "
                    f"itself, not the value of n_points: relax the constraints "
                    f"or widen the bounds."
                )
            if region_is_provably_measure_zero(search_space):
                raise InfeasibleRegionError(
                    f"An equality constraint over continuous ('real') variables "
                    f"restricts the feasible region to a zero-volume slice, "
                    f"which the '{method}' sampler draws continuously and "
                    f"cannot land on. The region is not empty -- it simply "
                    f"cannot be reached by sampling. Use method='optimal', "
                    f"which places design points on the constraint boundaries, "
                    f"or declare the constrained variables as 'integer' or "
                    f"'discrete' so their grid intersects the constraint."
                )

            # Reject-and-resample: over-generate feasible points until we have
            # n_points. Grow the oversampling factor; give up after a cap.
            import pandas as pd
            feasible: list = []
            factor = 4
            max_factor = 4096
            while len(feasible) < n_points and factor <= max_factor:
                batch = _sample(n_points * factor)
                # Strict tolerance: continuous samplers can place points exactly
                # on the boundary, and a DOE point should not exceed the user's
                # stated bound. (Grid-feasibility elsewhere uses a relative band.)
                mask = search_space.filter_feasible(pd.DataFrame(batch), rtol=DOE_RTOL, atol=DOE_ATOL)
                feasible.extend([p for p, ok in zip(batch, mask) if ok])
                factor *= 4
            if len(feasible) < n_points:
                raise ValueError(
                    f"Could not generate {n_points} feasible '{method}' design "
                    f"points that satisfy the registered input constraints "
                    f"(found {len(feasible)}). The feasible region may be very "
                    f"small or empty within the variable bounds; relax the "
                    f"constraints or reduce n_points."
                )
            points = feasible[:n_points]

    elif method == "full_factorial":
        points = _full_factorial(search_space, n_levels=n_levels, n_center=n_center)

    elif method == "fractional_factorial":
        _validate_classical_design(search_space, method)
        points = _fractional_factorial(search_space, generators=generators, n_center=n_center)

    elif method == "ccd":
        _validate_classical_design(search_space, method)
        points = _central_composite(search_space, n_center=n_center,
                                    alpha=ccd_alpha, face=ccd_face)

    elif method == "box_behnken":
        _validate_classical_design(search_space, method, min_continuous=3)
        points = _box_behnken(search_space, n_center=n_center)

    elif method == "plackett_burman":
        _validate_classical_design(search_space, method)
        points = _plackett_burman(search_space, n_center=n_center)

    elif method == "gsd":
        points = _gsd(search_space, reduction=gsd_reduction, n_levels=n_levels)

    elif method == "optimal":
        from alchemist_core.utils.optimal_design import run_optimal_design
        # Use n_levels=5 default for optimal design candidate grid
        opt_n_levels = n_levels if n_levels > 2 else 5
        points, _info = run_optimal_design(
            search_space=search_space,
            n_points=n_points,
            model_type=model_type,
            effects=effects,
            criterion=criterion,
            algorithm=algorithm,
            n_levels=opt_n_levels,
            max_iter=max_iter,
            random_seed=random_seed,
        )

    else:
        raise ValueError(
            f"Unknown sampling method: {method}. "
            f"Choose from: {', '.join(sorted(SPACE_FILLING_METHODS | CLASSICAL_METHODS))}"
        )

    # Classical designs have fixed structure and cannot be resampled. Filter to
    # feasible rows, then decide whether the remnant is still the design it
    # claims to be. ('optimal' is exempt: its candidate set is already
    # constrained, and its model is user-specified rather than implied.)
    if (method in CLASSICAL_METHODS and method != "optimal"
            and getattr(search_space, 'constraints', None)):
        import pandas as pd
        mask = search_space.filter_feasible(pd.DataFrame(points), rtol=DOE_RTOL, atol=DOE_ATOL)
        n_feasible = int(mask.sum())
        if n_feasible == 0:
            raise ValueError(
                f"No '{method}' design points satisfy the registered input "
                f"constraints. Classical designs have fixed structure and "
                f"cannot be resampled; use a space-filling method (random, lhs, "
                f"sobol) for constrained designs, or relax the constraints."
            )

        n_dropped = len(points) - n_feasible
        surviving = [p for p, ok in zip(points, mask) if ok]

        if n_dropped > 0:
            inestimable = _inestimable_terms(search_space, surviving, method, n_levels)
            if inestimable and not allow_infeasible:
                raise DesignNotEstimableError(
                    f"{n_dropped} of {len(points)} '{method}' design points "
                    f"violate the registered input constraints and were dropped. "
                    f"The remaining {n_feasible} points can no longer estimate "
                    f"the design's implied "
                    f"{_implied_model_type(method, n_levels)} model — these "
                    f"terms became inestimable: {', '.join(inestimable)}. "
                    f"A classical design's value comes from its structure, so "
                    f"the remnant is not the design it claims to be. Use "
                    f"method='optimal' for a genuine constrained optimal "
                    f"design, or a space-filling method (random, lhs, sobol). "
                    f"Pass allow_infeasible=True to return the remnant anyway."
                )
            if inestimable:
                logger.warning(
                    "%d of %d '%s' design points were dropped and the implied "
                    "%s model is no longer estimable (%s). Returning the "
                    "remnant because allow_infeasible=True.",
                    n_dropped, len(points), method,
                    _implied_model_type(method, n_levels), ", ".join(inestimable),
                )
            else:
                logger.info(
                    "%d of %d '%s' design points were dropped to satisfy the "
                    "registered input constraints; the implied %s model remains "
                    "estimable from the remaining %d.",
                    n_dropped, len(points), method,
                    _implied_model_type(method, n_levels), n_feasible,
                )

        points = surviving

    logger.info(
        f"Generated {len(points)} initial points using {method} method "
        f"for {len(search_space.get_dimension_variables())} variables"
    )

    return points


# ============================================================
# Validation helpers
# ============================================================

def _validate_classical_design(search_space: SearchSpace, method: str,
                               min_continuous: int = 2):
    """Validate that search space is compatible with classical RSM designs."""
    continuous_vars = [v for v in search_space.variables if v['type'] in ('real', 'integer', 'discrete')]
    categorical_vars = [v for v in search_space.variables if v['type'] == 'categorical']

    if method in ("ccd", "box_behnken", "fractional_factorial", "plackett_burman"):
        if categorical_vars:
            raise ValueError(
                f"{method} does not support categorical variables. "
                f"Found categorical: {[v['name'] for v in categorical_vars]}. "
                f"Use full_factorial for mixed variable types."
            )

    if method == "box_behnken" and len(continuous_vars) < 3:
        raise ValueError(
            f"Box-Behnken design requires at least 3 continuous variables, "
            f"got {len(continuous_vars)}."
        )

    if len(continuous_vars) < min_continuous:
        raise ValueError(
            f"{method} requires at least {min_continuous} continuous variables, "
            f"got {len(continuous_vars)}."
        )


def _term_column_owners(terms: List[Any], column_map: List[Dict[str, Any]],
                        variables: List[Dict[str, Any]]) -> List[int]:
    """Design-matrix column index -> owning term index.

    A term contributes exactly one design-matrix column per continuous factor
    but *k-1* columns for a categorical factor with k categories (dummy
    coding), and the outer product of those across a term's factors — mirrors
    :func:`optimal_design.build_custom_design_matrix`'s column layout without
    recomputing column values, so a rank-deficiency finding on a specific
    matrix column can be attributed back to the term name that produced it.
    """
    var_to_cols: Dict[int, List[int]] = {}
    for col_idx, cm in enumerate(column_map):
        var_to_cols.setdefault(cm["var_idx"], []).append(col_idx)

    owners: List[int] = []
    for term_idx, term in enumerate(terms):
        if len(term) == 0:
            owners.append(term_idx)  # intercept: exactly one column
            continue
        n_cols = 1
        for var_idx, _power in term:
            var = variables[var_idx]
            if var["type"] in ("real", "integer", "discrete"):
                n_cols *= 1
            else:
                n_cols *= max(len(var_to_cols[var_idx]) - 1, 1)
        owners.extend([term_idx] * n_cols)
    return owners


def _inestimable_terms(search_space: SearchSpace, points: List[Dict[str, Any]],
                       method: str, n_levels: int) -> List[str]:
    """Model terms the surviving points can no longer estimate.

    Builds the design matrix for the method's implied model from the points
    that survived constraint filtering and compares its rank to its column
    count. A rank-deficient matrix means the design cannot estimate every term
    it was chosen for.

    Which term(s) are actually inestimable is resolved with QR decomposition
    with column pivoting (``scipy.linalg.qr(X, pivoting=True)``): the last
    ``p_columns - rank`` pivoted columns are the ones expressible as a linear
    combination of the more significant columns already selected — i.e. the
    ones the surviving points can no longer separate from the rest of the
    model. A naive "report the last few term names" heuristic can name an
    essential, fully-estimable term while missing the actual redundancy
    (verified: a column whose removal does *not* change the matrix rank is
    genuinely redundant; one whose removal drops the rank is essential, and
    must never be reported here).

    Returns an empty list when the model is fully estimable, or when the check
    cannot be performed (an unparseable model, no points) — the gate should
    never block on its own inability to judge.
    """
    # Imported here, matching the existing lazy imports at doe.py:240 and :760.
    from scipy.linalg import qr as _qr
    from alchemist_core.utils.optimal_design import (
        build_column_map,
        build_custom_design_matrix,
        encode_candidates,
        get_model_term_names,
        parse_model_spec,
    )

    if not points:
        return []

    try:
        model_type = _implied_model_type(method, n_levels)
        terms = parse_model_spec(search_space, model_type=model_type)
        # The coded basis is the dimension-bearing variables, and every index
        # into it -- including the variable indices inside ``terms`` -- must be
        # numbered off that same list. parse_model_spec above uses it too.
        model_variables = search_space.get_dimension_variables()
        column_map = build_column_map(model_variables)
        coded = encode_candidates(points, column_map, model_variables)
        X = build_custom_design_matrix(coded, terms, column_map,
                                       model_variables)
    except (ValueError, KeyError, IndexError) as e:
        logger.debug("Estimability check skipped for '%s': %s", method, e)
        return []

    p_columns = X.shape[1]
    rank = int(np.linalg.matrix_rank(X))
    if rank >= p_columns:
        return []

    # _term_column_owners mirrors build_custom_design_matrix's column-count
    # rule independently (see its docstring); if that rule ever drifts the
    # two would disagree on how many columns a term produces, and indexing
    # into a misaligned owner list would raise IndexError straight out of
    # this function -- contradicting its own contract that the gate never
    # blocks on its own inability to judge. Guard the length instead of
    # trusting the mirror: a mismatch degrades to the same graceful skip as
    # an unparseable model, not a crash.
    col_owner = _term_column_owners(terms, column_map, model_variables)
    if len(col_owner) != p_columns:
        logger.debug(
            "Estimability check skipped for '%s': column-owner mapping "
            "length %d does not match design matrix width %d.",
            method, len(col_owner), p_columns,
        )
        return []

    # The last (p_columns - rank) pivoted columns are the ones QR judges
    # reproducible from the rest -- the genuine redundancy, not merely
    # "whatever term happened to be listed last".
    _, _, pivot = _qr(X, pivoting=True)
    dependent_cols = pivot[rank:]
    dependent_term_idxs = sorted({int(col_owner[c]) for c in dependent_cols})

    names = get_model_term_names(search_space, terms)
    return [names[i] for i in dependent_term_idxs]


def _get_continuous_vars(search_space: SearchSpace) -> List[Dict[str, Any]]:
    """Return continuous (real/integer) and discrete variables (all numeric)."""
    return [v for v in search_space.variables if v['type'] in ('real', 'integer', 'discrete')]


# ============================================================
# Coded-to-actual mapping
# ============================================================

def _coded_to_actual(coded_design: np.ndarray, search_space: SearchSpace,
                     continuous_only: bool = True) -> List[Dict[str, Any]]:
    """Map coded design matrix (-1 to +1) to actual variable bounds.

    For Real variables: actual = mid + coded * half_range
    For Integer variables: same formula, then round to nearest int
    """
    if continuous_only:
        variables = _get_continuous_vars(search_space)
    else:
        variables = search_space.variables

    points = []
    for row in coded_design:
        point = {}
        for j, var in enumerate(variables):
            if var['type'] == 'discrete':
                allowed = var['allowed_values']
                low = min(allowed)
                high = max(allowed)
                mid = (low + high) / 2.0
                half_range = (high - low) / 2.0
                actual = mid + row[j] * half_range
                # Snap to nearest allowed value
                actual = float(min(allowed, key=lambda v: abs(v - actual)))
            else:
                low = var['min']
                high = var['max']
                mid = (low + high) / 2.0
                half_range = (high - low) / 2.0
                actual = mid + row[j] * half_range
                # Clamp to bounds
                actual = max(low, min(high, actual))
                if var['type'] == 'integer':
                    actual = int(round(actual))
                else:
                    actual = float(actual)
            point[var['name']] = actual
        points.append(point)

    return points


def _center_point(search_space: SearchSpace, continuous_only: bool = True) -> Dict[str, Any]:
    """Return the center point of the continuous variable space."""
    if continuous_only:
        variables = _get_continuous_vars(search_space)
    else:
        variables = [v for v in search_space.variables if v['type'] in ('real', 'integer', 'discrete')]

    point = {}
    for var in variables:
        if var['type'] == 'discrete':
            allowed = var['allowed_values']
            midval = (min(allowed) + max(allowed)) / 2.0
            # Snap to nearest allowed value
            point[var['name']] = float(min(allowed, key=lambda v: abs(v - midval)))
        else:
            mid = float((var['min'] + var['max']) / 2.0)
            if var['type'] == 'integer':
                mid = int(round(mid))
            point[var['name']] = mid
    return point


# ============================================================
# Classical design methods
# ============================================================

def _full_factorial(search_space: SearchSpace, n_levels: int = 2,
                    n_center: int = 1) -> List[Dict[str, Any]]:
    """Generate a full factorial design.

    For continuous variables, maps evenly spaced levels across the range.
    For categorical variables, uses all categories as levels.
    """
    import pyDOE

    # Dimension-bearing variables only. A ``context`` variable carries no
    # bounds, no categories and no allowed values, so the level lookups below
    # raise KeyError: 'min' on one -- and a design must not assign it a level
    # in any case, since nothing can set it.
    variables = search_space.get_dimension_variables()
    levels_per_var = []

    for var in variables:
        if var['type'] == 'categorical':
            levels_per_var.append(len(var.get('values', var.get('categories', []))))
        elif var['type'] == 'discrete':
            levels_per_var.append(len(var['allowed_values']))
        else:
            levels_per_var.append(n_levels)

    # Generate factorial design (0-indexed levels)
    design = pyDOE.fullfact(levels_per_var)

    # Map to actual values
    points = []
    for row in design:
        point = {}
        for j, var in enumerate(variables):
            level_idx = int(row[j])
            if var['type'] == 'categorical':
                cats = var.get('values', var.get('categories', []))
                point[var['name']] = cats[level_idx]
            elif var['type'] == 'discrete':
                point[var['name']] = float(var['allowed_values'][level_idx])
            else:
                low = var['min']
                high = var['max']
                n_lvl = levels_per_var[j]
                if n_lvl == 1:
                    actual = (low + high) / 2.0
                else:
                    actual = low + level_idx * (high - low) / (n_lvl - 1)
                if var['type'] == 'integer':
                    actual = int(round(actual))
                else:
                    actual = float(actual)
                point[var['name']] = actual
        points.append(point)

    # Add center point replicates
    if n_center > 0:
        center = {}
        for var in variables:
            if var['type'] == 'categorical':
                cats = var.get('values', var.get('categories', []))
                center[var['name']] = cats[0]
            elif var['type'] == 'discrete':
                allowed = var['allowed_values']
                midval = (min(allowed) + max(allowed)) / 2.0
                center[var['name']] = float(min(allowed, key=lambda v: abs(v - midval)))
            else:
                mid = float((var['min'] + var['max']) / 2.0)
                if var['type'] == 'integer':
                    mid = int(round(mid))
                center[var['name']] = mid
        for _ in range(n_center):
            points.append(dict(center))

    return points


def _fractional_factorial(search_space: SearchSpace, generators: Optional[str] = None,
                          n_center: int = 1) -> List[Dict[str, Any]]:
    """Generate a fractional factorial design (2-level).

    Uses pyDOE.fracfact() with a generator string. If no generator is provided,
    uses a standard generator based on the number of factors.
    """
    import pyDOE

    continuous_vars = _get_continuous_vars(search_space)
    n_factors = len(continuous_vars)

    if generators is None:
        if n_factors in _DEFAULT_GENERATORS:
            generators = _DEFAULT_GENERATORS[n_factors]
        else:
            # For n_factors not in lookup table, build a basic generator.
            # Use letters for main effects, generate remaining from interactions.
            base_letters = [chr(ord('a') + i) for i in range(n_factors)]
            generators = " ".join(base_letters[:n_factors])

    # Generate coded design (-1, +1)
    coded = pyDOE.fracfact(generators)

    # Map to actual values
    points = _coded_to_actual(coded, search_space)

    # Add center point replicates
    if n_center > 0:
        center = _center_point(search_space)
        for _ in range(n_center):
            points.append(dict(center))

    return points


def _central_composite(search_space: SearchSpace, n_center: int = 1,
                       alpha: str = "orthogonal",
                       face: str = "circumscribed") -> List[Dict[str, Any]]:
    """Generate a Central Composite Design (CCD).

    Combines a factorial design with axial (star) points and center points.

    Face types:
    - circumscribed (CCC): axial points extend beyond factorial bounds
    - inscribed (CCI): factorial points are interior, axials at bounds
    - faced (CCF): axial points on the faces (at bounds)
    """
    import pyDOE

    continuous_vars = _get_continuous_vars(search_space)
    n_factors = len(continuous_vars)

    # pyDOE center param: (center_factorial, center_axial)
    coded = pyDOE.ccdesign(n_factors, center=(n_center, n_center),
                           alpha=alpha, face=face)

    # For circumscribed designs, axial points extend beyond ±1.
    # Scale the entire design so axials map to the variable bounds.
    if face in ("circumscribed", "ccc"):
        max_coded = np.abs(coded).max()
        if max_coded > 1.0:
            coded = coded / max_coded

    points = _coded_to_actual(coded, search_space)
    return points


def _box_behnken(search_space: SearchSpace,
                 n_center: int = 1) -> List[Dict[str, Any]]:
    """Generate a Box-Behnken design.

    Requires 3+ continuous factors. Points are at the midpoints of edges
    of the variable space, plus center points. No corner or axial points.
    """
    import pyDOE

    continuous_vars = _get_continuous_vars(search_space)
    n_factors = len(continuous_vars)

    coded = pyDOE.bbdesign(n_factors, center=n_center)

    points = _coded_to_actual(coded, search_space)
    return points


def _plackett_burman(search_space: SearchSpace,
                     n_center: int = 1) -> List[Dict[str, Any]]:
    """Generate a Plackett-Burman design.

    Ultra-efficient 2-level screening design for identifying main effects.
    The number of runs is the next multiple of 4 above the number of factors
    (e.g., 12 runs for 11 factors). Does not estimate interactions.
    """
    import pyDOE

    continuous_vars = _get_continuous_vars(search_space)
    n_factors = len(continuous_vars)

    # pbdesign returns coded matrix with values in {-1, +1}
    coded = pyDOE.pbdesign(n_factors)

    points = _coded_to_actual(coded, search_space)

    # Add center point replicates
    if n_center > 0:
        center = _center_point(search_space)
        for _ in range(n_center):
            points.append(dict(center))

    return points


def _gsd(search_space: SearchSpace, reduction: int = 2,
         n_levels: int = 2) -> List[Dict[str, Any]]:
    """Generate a Generalized Subset Design.

    Fractional factorial for factors with >=2 levels, including categorical
    variables. Each factor is assigned a number of levels:
    - Categorical variables: number of categories
    - Continuous variables: ``n_levels`` evenly spaced values across the range

    The design is a balanced fraction of the full factorial with approximately
    (product of levels) / reduction runs.
    """
    import pyDOE

    # Dimension-bearing variables only -- see _full_factorial.
    variables = search_space.get_dimension_variables()

    # Build levels array
    levels_per_var = []
    for var in variables:
        if var['type'] == 'categorical':
            levels_per_var.append(len(var.get('values', var.get('categories', []))))
        elif var['type'] == 'discrete':
            levels_per_var.append(len(var['allowed_values']))
        else:
            levels_per_var.append(n_levels)

    # GSD requires a plain Python list of ints
    design = pyDOE.gsd(levels_per_var, reduction=reduction)

    # When n=1 (default), pyDOE returns a single ndarray
    if isinstance(design, list):
        design = design[0]

    # Map 0-indexed levels to actual values (same logic as full factorial)
    points = []
    for row in design:
        point = {}
        for j, var in enumerate(variables):
            level_idx = int(row[j])
            if var['type'] == 'categorical':
                cats = var.get('values', var.get('categories', []))
                point[var['name']] = cats[level_idx]
            elif var['type'] == 'discrete':
                point[var['name']] = float(var['allowed_values'][level_idx])
            else:
                low = var['min']
                high = var['max']
                n_lvl = levels_per_var[j]
                if n_lvl == 1:
                    actual = (low + high) / 2.0
                else:
                    actual = low + level_idx * (high - low) / (n_lvl - 1)
                if var['type'] == 'integer':
                    actual = int(round(actual))
                else:
                    actual = float(actual)
                point[var['name']] = actual
        points.append(point)

    return points


# ============================================================
# Design info metadata
# ============================================================

def get_design_info(method: str, search_space: SearchSpace,
                    n_levels: int = 2, n_center: int = 1,
                    generators: Optional[str] = None,
                    ccd_alpha: str = "orthogonal",
                    ccd_face: str = "circumscribed",
                    gsd_reduction: int = 2,
                    # Optimal design parameters
                    model_type: Optional[str] = None,
                    effects: Optional[List[str]] = None,
                    criterion: str = "D",
                    algorithm: str = "fedorov",
                    n_points: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Return metadata about the design structure for a given method.

    Returns None for space-filling methods.
    """
    if method in SPACE_FILLING_METHODS:
        return None

    continuous_vars = _get_continuous_vars(search_space)
    n_factors = len(continuous_vars)

    if method == "full_factorial":
        # The same basis _full_factorial builds the design from. Iterating
        # search_space.variables instead lets a `context` variable fall through
        # to the `else` below and contribute n_levels, so the reported run
        # count describes a larger design than the one generate_initial_design
        # returns -- and POST /initial-design puts both in one response.
        levels_list = []
        for var in search_space.get_dimension_variables():
            if var['type'] == 'categorical':
                levels_list.append(len(var.get('values', var.get('categories', []))))
            elif var['type'] == 'discrete':
                levels_list.append(len(var['allowed_values']))
            else:
                levels_list.append(n_levels)
        factorial_runs = reduce(operator.mul, levels_list, 1)
        return {
            "factorial_runs": factorial_runs,
            "center_runs": n_center,
            "total_runs": factorial_runs + n_center,
            "levels_per_factor": levels_list,
        }

    elif method == "fractional_factorial":
        import pyDOE
        if generators is None and n_factors in _DEFAULT_GENERATORS:
            generators = _DEFAULT_GENERATORS[n_factors]
        if generators:
            coded = pyDOE.fracfact(generators)
            factorial_runs = coded.shape[0]
        else:
            factorial_runs = 2 ** n_factors
        return {
            "factorial_runs": factorial_runs,
            "center_runs": n_center,
            "total_runs": factorial_runs + n_center,
            "generators": generators,
        }

    elif method == "ccd":
        factorial_runs = 2 ** n_factors
        axial_runs = 2 * n_factors
        return {
            "factorial_runs": factorial_runs,
            "axial_runs": axial_runs,
            "center_runs": n_center * 2,  # center in factorial + center in axial
            "total_runs": factorial_runs + axial_runs + n_center * 2,
            "alpha": ccd_alpha,
            "face": ccd_face,
        }

    elif method == "box_behnken":
        import pyDOE
        coded = pyDOE.bbdesign(n_factors, center=n_center)
        edge_runs = coded.shape[0] - n_center
        return {
            "edge_runs": edge_runs,
            "center_runs": n_center,
            "total_runs": coded.shape[0],
        }

    elif method == "plackett_burman":
        import pyDOE
        coded = pyDOE.pbdesign(n_factors)
        screening_runs = coded.shape[0]
        return {
            "screening_runs": screening_runs,
            "center_runs": n_center,
            "total_runs": screening_runs + n_center,
        }

    elif method == "gsd":
        import pyDOE
        # Same basis _gsd builds from -- and here the inflated list is not
        # merely reported, it is handed to pyDOE.gsd() below, so gsd_runs was
        # computed from a different design than the one returned.
        levels_list = []
        for var in search_space.get_dimension_variables():
            if var['type'] == 'categorical':
                levels_list.append(len(var.get('values', var.get('categories', []))))
            elif var['type'] == 'discrete':
                levels_list.append(len(var['allowed_values']))
            else:
                levels_list.append(n_levels)
        full_runs = reduce(operator.mul, levels_list, 1)
        design = pyDOE.gsd(levels_list, reduction=gsd_reduction)
        if isinstance(design, list):
            design = design[0]
        return {
            "full_factorial_runs": full_runs,
            "gsd_runs": design.shape[0],
            "total_runs": design.shape[0],
            "reduction": gsd_reduction,
            "levels_per_factor": levels_list,
        }

    elif method == "optimal":
        from alchemist_core.utils.optimal_design import (
            parse_model_spec, get_model_term_names
        )
        try:
            terms = parse_model_spec(search_space, model_type=model_type,
                                     effects=effects)
            term_names = get_model_term_names(search_space, terms)
            return {
                "p_columns": len(terms),
                "model_terms": term_names,
                "criterion": criterion,
                "algorithm": algorithm,
                "total_runs": n_points if n_points else "user-specified",
            }
        except ValueError:
            return None

    return None


# ============================================================
# Space-filling methods (unchanged from original)
# ============================================================

def _random_sampling(skopt_space, n_points: int, random_state=None) -> list:
    """
    Generate random samples respecting variable types.

    Handles Real, Integer, and Categorical dimensions appropriately.
    Returns list of lists to preserve mixed types.
    """
    # np.random.mtrand._rand *is* the object np.random.seed() configures, so
    # the default preserves the previous behaviour exactly for direct callers.
    rng = np.random.mtrand._rand if random_state is None else random_state
    samples_list = []

    for dim in skopt_space:
        if isinstance(dim, Categorical):
            # Random choice from categories
            samples = rng.choice(dim.categories, size=n_points)

        elif isinstance(dim, Integer):
            # Random integers in [low, high] (inclusive)
            # np.random.randint is [low, high), so add 1 to include upper bound
            samples = rng.randint(dim.low, dim.high + 1, size=n_points)

        elif isinstance(dim, Real):
            # Random floats in [low, high]
            samples = rng.uniform(dim.low, dim.high, size=n_points)

        else:
            raise ValueError(f"Unknown dimension type: {type(dim)}")

        samples_list.append(samples)

    # Transpose to get list of samples (each sample is a list of values)
    # Don't use column_stack as it converts everything to same dtype
    samples = [[samples_list[j][i] for j in range(len(samples_list))]
               for i in range(n_points)]
    return samples


def _lhs_sampling(skopt_space, n_points: int, criterion: str = "maximin",
                  random_state=None) -> list:
    """
    Generate Latin Hypercube Sampling points.

    LHS provides good space-filling properties and is generally recommended
    for initial designs in Bayesian optimization.

    Args:
        criterion: Optimization criterion
            - "maximin": maximize minimum distance between points (default)
            - "correlation": minimize correlations between dimensions
            - "ratio": minimize ratio of max to min distance
    """
    sampler = Lhs(lhs_type="classic", criterion=criterion)
    samples = sampler.generate(skopt_space, n_points, random_state=random_state)
    # skopt returns list of samples already
    return samples


def _sobol_sampling(skopt_space, n_points: int, random_state=None) -> list:
    """
    Generate Sobol quasi-random sequence points.

    Sobol sequences have low discrepancy properties, meaning they cover
    the space more uniformly than random sampling.
    """
    sampler = Sobol()
    samples = sampler.generate(skopt_space, n_points, random_state=random_state)
    # skopt returns list of samples already
    return samples


def _hammersly_sampling(skopt_space, n_points: int, random_state=None) -> list:
    """
    Generate Hammersly sequence points.

    Hammersly and Halton sequences are low-discrepancy sequences similar
    to Sobol, providing good space coverage.
    """
    sampler = Hammersly()
    samples = sampler.generate(skopt_space, n_points, random_state=random_state)
    # skopt returns list of samples already
    return samples
