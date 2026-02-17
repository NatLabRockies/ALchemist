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
import operator
import numpy as np
from skopt.sampler import Lhs, Sobol, Hammersly
from skopt.space import Real, Integer, Categorical
from alchemist_core.data.search_space import SearchSpace
from alchemist_core.config import get_logger

logger = get_logger(__name__)

SPACE_FILLING_METHODS = {"random", "lhs", "sobol", "halton", "hammersly"}
CLASSICAL_METHODS = {"full_factorial", "fractional_factorial", "ccd", "box_behnken",
                     "plackett_burman", "gsd"}

# Standard fractional factorial generators for common factor counts.
# These provide Resolution III or better designs for screening.
_DEFAULT_GENERATORS = {
    3: "a b ab",               # 2^(3-1), Resolution III
    4: "a b c abc",            # 2^(4-1), Resolution IV
    5: "a b c ab ac",          # 2^(5-2), Resolution III
    6: "a b c ab ac bc",       # 2^(6-3), Resolution III
    7: "a b c ab ac bc abc",   # 2^(7-4), Resolution III
}


def generate_initial_design(
    search_space: SearchSpace,
    method: Literal[
        "random", "lhs", "sobol", "halton", "hammersly",
        "full_factorial", "fractional_factorial", "ccd", "box_behnken",
        "plackett_burman", "gsd"
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

    Args:
        search_space: SearchSpace object with defined variables
        method: Sampling method to use
        n_points: Number of points (required for space-filling; ignored for classical)
        random_seed: Random seed for reproducibility
        lhs_criterion: Criterion for LHS ("maximin", "correlation", "ratio")
        n_levels: Levels per continuous factor for full factorial (2 or 3)
        n_center: Number of center point replicates (classical designs)
        generators: Fractional factorial generator string (e.g. "a b ab")
        ccd_alpha: CCD alpha type ("orthogonal" or "rotatable")
        ccd_face: CCD face type ("circumscribed", "inscribed", or "faced")
        gsd_reduction: GSD reduction factor (>=2); larger means fewer runs

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

    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
        logger.info(f"Set random seed to {random_seed} for reproducibility")

    # Route to appropriate method
    if method in SPACE_FILLING_METHODS:
        skopt_space = search_space.skopt_dimensions

        if method == "random":
            samples = _random_sampling(skopt_space, n_points)
        elif method == "lhs":
            samples = _lhs_sampling(skopt_space, n_points, lhs_criterion)
        elif method == "sobol":
            samples = _sobol_sampling(skopt_space, n_points)
        elif method in ("halton", "hammersly"):
            samples = _hammersly_sampling(skopt_space, n_points)

        # Convert samples to list of dicts
        variable_names = [v['name'] for v in search_space.variables]
        points = [{name: value for name, value in zip(variable_names, sample)}
                  for sample in samples]

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

    else:
        raise ValueError(
            f"Unknown sampling method: {method}. "
            f"Choose from: {', '.join(sorted(SPACE_FILLING_METHODS | CLASSICAL_METHODS))}"
        )

    logger.info(
        f"Generated {len(points)} initial points using {method} method "
        f"for {len(search_space.variables)} variables"
    )

    return points


# ============================================================
# Validation helpers
# ============================================================

def _validate_classical_design(search_space: SearchSpace, method: str,
                               min_continuous: int = 2):
    """Validate that search space is compatible with classical RSM designs."""
    continuous_vars = [v for v in search_space.variables if v['type'] in ('real', 'integer')]
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


def _get_continuous_vars(search_space: SearchSpace) -> List[Dict[str, Any]]:
    """Return only continuous (real/integer) variables."""
    return [v for v in search_space.variables if v['type'] in ('real', 'integer')]


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
        variables = [v for v in search_space.variables if v['type'] in ('real', 'integer')]

    point = {}
    for var in variables:
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

    variables = search_space.variables
    levels_per_var = []

    for var in variables:
        if var['type'] == 'categorical':
            levels_per_var.append(len(var.get('values', var.get('categories', []))))
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

    variables = search_space.variables

    # Build levels array
    levels_per_var = []
    for var in variables:
        if var['type'] == 'categorical':
            levels_per_var.append(len(var.get('values', var.get('categories', []))))
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
                    gsd_reduction: int = 2) -> Optional[Dict[str, Any]]:
    """Return metadata about the design structure for a given method.

    Returns None for space-filling methods.
    """
    if method in SPACE_FILLING_METHODS:
        return None

    continuous_vars = _get_continuous_vars(search_space)
    n_factors = len(continuous_vars)

    if method == "full_factorial":
        levels_list = []
        for var in search_space.variables:
            if var['type'] == 'categorical':
                levels_list.append(len(var.get('values', var.get('categories', []))))
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
        levels_list = []
        for var in search_space.variables:
            if var['type'] == 'categorical':
                levels_list.append(len(var.get('values', var.get('categories', []))))
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

    return None


# ============================================================
# Space-filling methods (unchanged from original)
# ============================================================

def _random_sampling(skopt_space, n_points: int) -> list:
    """
    Generate random samples respecting variable types.

    Handles Real, Integer, and Categorical dimensions appropriately.
    Returns list of lists to preserve mixed types.
    """
    samples_list = []

    for dim in skopt_space:
        if isinstance(dim, Categorical):
            # Random choice from categories
            samples = np.random.choice(dim.categories, size=n_points)

        elif isinstance(dim, Integer):
            # Random integers in [low, high] (inclusive)
            # np.random.randint is [low, high), so add 1 to include upper bound
            samples = np.random.randint(dim.low, dim.high + 1, size=n_points)

        elif isinstance(dim, Real):
            # Random floats in [low, high]
            samples = np.random.uniform(dim.low, dim.high, size=n_points)

        else:
            raise ValueError(f"Unknown dimension type: {type(dim)}")

        samples_list.append(samples)

    # Transpose to get list of samples (each sample is a list of values)
    # Don't use column_stack as it converts everything to same dtype
    samples = [[samples_list[j][i] for j in range(len(samples_list))]
               for i in range(n_points)]
    return samples


def _lhs_sampling(skopt_space, n_points: int, criterion: str = "maximin") -> list:
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
    samples = sampler.generate(skopt_space, n_points)
    # skopt returns list of samples already
    return samples


def _sobol_sampling(skopt_space, n_points: int) -> list:
    """
    Generate Sobol quasi-random sequence points.

    Sobol sequences have low discrepancy properties, meaning they cover
    the space more uniformly than random sampling.
    """
    sampler = Sobol()
    samples = sampler.generate(skopt_space, n_points)
    # skopt returns list of samples already
    return samples


def _hammersly_sampling(skopt_space, n_points: int) -> list:
    """
    Generate Hammersly sequence points.

    Hammersly and Halton sequences are low-discrepancy sequences similar
    to Sobol, providing good space coverage.
    """
    sampler = Hammersly()
    samples = sampler.generate(skopt_space, n_points)
    # skopt returns list of samples already
    return samples
