# Constrained DoE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make linear input constraints work end-to-end in classical and optimal DoE, and make them registerable over REST.

**Architecture:** A new `constrained_region` module owns the geometry of a linear-constrained region (projection, vertex enumeration, per-variable feasible intervals). `optimal_design` consumes it to build an augmented feasible candidate set before the exchange algorithm runs. `doe` consumes it plus `optimal_design`'s existing model-matrix machinery to gate classical designs on estimability. The API layer gains constraint CRUD so a non-Python consumer can register constraints at all.

**Tech Stack:** Python 3.13, NumPy, pandas, pytest, FastAPI, Pydantic v2.

**Spec:** `.superpowers/specs/2026-08-25-constrained-doe-design.md`

## Global Constraints

- **Python interpreter is `~/miniforge3/envs/alchemist-env/bin/python`.** The system `python3` lacks the dependencies. Every command below uses it.
- **Domain-agnostic, absolutely.** ALchemist is a general-purpose optimization toolkit with multiple consumers. No reactor, spectroscopy, MQTT, band, detector, plasma, or catalysis concept may appear in any comment, docstring, test name, fixture, or variable — ever. Use `x1`, `x2`, `x3` and generic half-planes.
- **Unconstrained behavior must be bit-for-bit unchanged.** Task 1 makes this a test, not a claim. Never edit the golden fixture to make a test pass.
- **Checkboxes in this repo mean nothing.** Every plan under `.superpowers/plans/` has all boxes unticked, including merged work. Verify completion against code and `git log`, never against a checkbox.
- **Plans and specs live in `.superpowers/`, never in `docs/`.** `docs/` is published MkDocs source. See `AGENTS.md`.
- **TDD throughout.** Red, green, refactor. Commit at the end of every task.
- **Backend suite:** `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/ -q` — in THIS worktree the baseline is **999 passed, 11 skipped, 0 failed**. That is the floor; every task adds to the passed count and must never reduce it.
  - The 11th skip is `test_session_load_nonoutput_target.py::test_load_real_nonoutput_session_restores_all_rows`, which self-guards on a real session fixture that is not tracked in git and so is absent from any fresh worktree. On the main checkout the same suite reads 1000 passed / 10 skipped. **This difference is expected — do not chase it as a regression.**
- **Existing constraint semantics:** `'inequality'` means `sum(c_i * x_i) <= rhs`. `'equality'` means `== rhs`. Feasibility is defined solely by `SearchSpace.filter_feasible`; never re-implement the predicate.
- **DoE feasibility tolerance is strict:** `rtol=0.0, atol=1e-9`, matching `doe.py:200-203`.

---

## File Structure

| File | Responsibility | Status |
|---|---|---|
| `alchemist_core/utils/constrained_region.py` | Geometry of a linear-constrained region: projection, intervals, vertices, candidate augmentation | **Create** |
| `alchemist_core/utils/optimal_design.py` | Model terms, design matrices, criteria, exchange algorithms; consumes `constrained_region` | Modify |
| `alchemist_core/utils/doe.py` | Design generation, method routing, estimability gate | Modify |
| `alchemist_core/data/search_space.py` | Variable + constraint definition and feasibility predicate | Modify (guard only) |
| `api/models/requests.py` | `AddConstraintRequest` | Modify |
| `api/models/responses.py` | `ConstraintResponse`, `ConstraintsListResponse`, `FeasibilityInfo` | Modify |
| `api/routers/variables.py` | Constraint CRUD, search-space load | Modify |
| `api/routers/experiments.py` | Feasibility passthrough on design endpoints | Modify |
| `api/middleware/error_handlers.py` | `DesignNotEstimableError`, `InfeasibleRegionError` → 400 | Modify |
| `tests/unit/core/data/test_doe_golden_unconstrained.py` | Back-compat lock | **Create** |
| `tests/unit/core/data/golden_unconstrained_designs.json` | Golden fixture | **Create** |
| `tests/unit/core/utils/test_constrained_region.py` | Geometry unit tests | **Create** |
| `tests/unit/core/data/test_constrained_optimal_design.py` | Constrained optimal design | **Create** |
| `tests/unit/core/data/test_classical_estimability.py` | Estimability gate | **Create** |
| `tests/integration/api/test_constraints_router.py` | Constraint CRUD | **Create** |

---

# PHASE 1 — CORE

---

### Task 1: Lock unconstrained behavior with golden tests

Written first, before any production code moves. Everything after this task is verified against it.

**Files:**
- Create: `tests/unit/core/data/test_doe_golden_unconstrained.py`
- Create: `tests/unit/core/data/golden_unconstrained_designs.json`

**Interfaces:**
- Consumes: nothing.
- Produces: `golden_unconstrained_designs.json` — a dict keyed by method name, each value either `{"points": [...]}` or `{"error": "<ExceptionType>: <message>"}`. Later tasks must never regenerate or edit it.

- [ ] **Step 1: Write the golden-comparison test**

Create `tests/unit/core/data/test_doe_golden_unconstrained.py`:

```python
"""Back-compat lock: unconstrained DoE output must never change.

Every method is exercised on a fixed 3-variable space with a fixed seed and
compared against a committed golden fixture. Constraint work must not alter
unconstrained behavior in any way, and this file is what proves it.

If a change here fails, the change is wrong. Do NOT regenerate the fixture
to make it pass.
"""

import json
import os

import pytest

from alchemist_core import OptimizationSession
from alchemist_core.utils.doe import CLASSICAL_METHODS, SPACE_FILLING_METHODS

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden_unconstrained_designs.json")
SEED = 1234

# 'optimal' needs a model spec, so it is driven separately below.
METHODS = sorted((SPACE_FILLING_METHODS | CLASSICAL_METHODS) - {"optimal"})


def _session():
    """Three real variables — satisfies every method's minimum factor count."""
    s = OptimizationSession()
    s.add_variable("x1", "real", bounds=(0.0, 10.0))
    s.add_variable("x2", "real", bounds=(0.0, 10.0))
    s.add_variable("x3", "real", bounds=(0.0, 10.0))
    return s


def _round(points):
    """Round to 9 dp so float formatting noise does not fail the comparison."""
    return [{k: (round(v, 9) if isinstance(v, float) else v) for k, v in p.items()}
            for p in points]


def _run(method):
    """Return the design, or a recorded error string. Errors are behavior too."""
    s = _session()
    try:
        if method in SPACE_FILLING_METHODS:
            pts = s.generate_initial_design(n_points=8, method=method, random_seed=SEED)
        else:
            pts = s.generate_initial_design(method=method, random_seed=SEED)
        return {"points": _round(pts)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def _run_optimal():
    s = _session()
    try:
        pts, _info = s.generate_optimal_design(
            n_points=10, model_type="quadratic", criterion="D",
            algorithm="fedorov", random_seed=SEED,
        )
        return {"points": _round(pts)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def build_golden():
    """Regenerate the fixture. Run ONLY to create it the first time."""
    golden = {m: _run(m) for m in METHODS}
    golden["optimal"] = _run_optimal()
    return golden


@pytest.fixture(scope="module")
def golden():
    if not os.path.exists(GOLDEN_PATH):
        pytest.fail(
            f"Golden fixture missing at {GOLDEN_PATH}. Create it once with:\n"
            f"  python -c \"import json;"
            f"from tests.unit.core.data.test_doe_golden_unconstrained import build_golden,GOLDEN_PATH;"
            f"json.dump(build_golden(),open(GOLDEN_PATH,'w'),indent=2)\""
        )
    with open(GOLDEN_PATH) as fh:
        return json.load(fh)


@pytest.mark.parametrize("method", METHODS)
def test_unconstrained_design_matches_golden(method, golden):
    assert method in golden, f"{method} missing from golden fixture"
    assert _run(method) == golden[method], (
        f"Unconstrained '{method}' output changed. Constraint work must not "
        f"alter unconstrained behavior. Do not regenerate the fixture."
    )


def test_unconstrained_optimal_design_matches_golden(golden):
    assert _run_optimal() == golden["optimal"], (
        "Unconstrained optimal design output changed. Do not regenerate the fixture."
    )
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/Active learning code development/ALchemist"
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/data/test_doe_golden_unconstrained.py -q
```

Expected: FAIL — every test fails with "Golden fixture missing".

- [ ] **Step 3: Generate the fixture from current `main` behavior**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/Active learning code development/ALchemist"
~/miniforge3/envs/alchemist-env/bin/python -c "
import json, sys
sys.path.insert(0, '.')
from tests.unit.core.data.test_doe_golden_unconstrained import build_golden, GOLDEN_PATH
json.dump(build_golden(), open(GOLDEN_PATH, 'w'), indent=2, sort_keys=True)
print('wrote', GOLDEN_PATH)
"
```

- [ ] **Step 4: Inspect the fixture before trusting it**

```bash
~/miniforge3/envs/alchemist-env/bin/python -c "
import json
g = json.load(open('tests/unit/core/data/golden_unconstrained_designs.json'))
for k, v in sorted(g.items()):
    print(k, '->', 'ERROR: ' + v['error'] if 'error' in v else f\"{len(v['points'])} points\")
"
```

Expected: every method reports either a point count or a recorded error. A method reporting `0 points` means the fixture captured nothing useful — stop and investigate before continuing.

- [ ] **Step 5: Run the test to verify it passes**

```bash
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/data/test_doe_golden_unconstrained.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/unit/core/data/test_doe_golden_unconstrained.py tests/unit/core/data/golden_unconstrained_designs.json
git commit -m "test(doe): lock unconstrained design output with golden fixture

Captures every DoE method's output on a fixed 3-variable space at a fixed
seed, from current main. Constrained-DoE work must leave all of it unchanged;
this makes that a verified property rather than a claim."
```

---

### Task 2: `constrained_region` — variable helpers and projection

**Files:**
- Create: `alchemist_core/utils/constrained_region.py`
- Create: `tests/unit/core/utils/test_constrained_region.py`

**Interfaces:**
- Consumes: `SearchSpace.variables`, `SearchSpace.constraints`.
- Produces:
  - `numeric_variables(search_space) -> List[Dict[str, Any]]`
  - `variable_bounds(var: Dict[str, Any]) -> Tuple[float, float]`
  - `project_onto_constraint(point: Dict[str, Any], constraint: Dict[str, Any]) -> Dict[str, Any]`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/core/utils/test_constrained_region.py`:

```python
"""Unit tests for the linear-constrained-region geometry helpers."""

import numpy as np
import pytest

from alchemist_core.data.search_space import SearchSpace
from alchemist_core.utils import constrained_region as cr


def _space():
    s = SearchSpace()
    s.add_variable("x1", "real", min=0.0, max=10.0)
    s.add_variable("x2", "real", min=0.0, max=10.0)
    return s


class TestVariableHelpers:
    def test_numeric_variables_excludes_categorical_and_context(self):
        s = _space()
        s.add_variable("c1", "categorical", values=["a", "b"])
        s.add_variable("ctx", "context")
        names = [v["name"] for v in cr.numeric_variables(s)]
        assert names == ["x1", "x2"]

    def test_numeric_variables_includes_integer_and_discrete(self):
        s = SearchSpace()
        s.add_variable("i1", "integer", min=0, max=5)
        s.add_variable("d1", "discrete", allowed_values=[1.0, 2.0, 4.0])
        names = [v["name"] for v in cr.numeric_variables(s)]
        assert names == ["i1", "d1"]

    def test_variable_bounds_real(self):
        s = _space()
        assert cr.variable_bounds(s.variables[0]) == (0.0, 10.0)

    def test_variable_bounds_discrete_uses_min_max_of_allowed(self):
        s = SearchSpace()
        s.add_variable("d1", "discrete", allowed_values=[4.0, 1.0, 2.0])
        assert cr.variable_bounds(s.variables[0]) == (1.0, 4.0)

    def test_variable_bounds_rejects_categorical(self):
        s = SearchSpace()
        s.add_variable("c1", "categorical", values=["a", "b"])
        with pytest.raises(ValueError, match="no numeric bounds"):
            cr.variable_bounds(s.variables[0])


class TestProjection:
    def test_projected_point_lies_on_the_hyperplane(self):
        c = {"type": "inequality", "coefficients": {"x1": 1.0, "x2": 1.0},
             "rhs": 5.0, "name": "c0"}
        out = cr.project_onto_constraint({"x1": 10.0, "x2": 10.0}, c)
        assert out["x1"] + out["x2"] == pytest.approx(5.0)

    def test_projection_is_idempotent(self):
        c = {"type": "inequality", "coefficients": {"x1": 1.0, "x2": 1.0},
             "rhs": 5.0, "name": "c0"}
        once = cr.project_onto_constraint({"x1": 10.0, "x2": 10.0}, c)
        twice = cr.project_onto_constraint(once, c)
        assert twice["x1"] == pytest.approx(once["x1"])
        assert twice["x2"] == pytest.approx(once["x2"])

    def test_projection_is_orthogonal(self):
        # From (10, 10) onto x1 + x2 == 5, the displacement is equal in both
        # coordinates because the normal is (1, 1).
        c = {"type": "inequality", "coefficients": {"x1": 1.0, "x2": 1.0},
             "rhs": 5.0, "name": "c0"}
        out = cr.project_onto_constraint({"x1": 10.0, "x2": 10.0}, c)
        assert out["x1"] == pytest.approx(2.5)
        assert out["x2"] == pytest.approx(2.5)

    def test_non_participating_keys_are_preserved(self):
        c = {"type": "inequality", "coefficients": {"x1": 1.0},
             "rhs": 3.0, "name": "c0"}
        out = cr.project_onto_constraint({"x1": 9.0, "cat": "a"}, c)
        assert out["cat"] == "a"
        assert out["x1"] == pytest.approx(3.0)

    def test_zero_coefficient_vector_returns_point_unchanged(self):
        c = {"type": "inequality", "coefficients": {"x1": 0.0},
             "rhs": 3.0, "name": "c0"}
        out = cr.project_onto_constraint({"x1": 9.0}, c)
        assert out["x1"] == 9.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/utils/test_constrained_region.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'alchemist_core.utils.constrained_region'`.

- [ ] **Step 3: Create the module with the three helpers**

Create `alchemist_core/utils/constrained_region.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/utils/test_constrained_region.py -q
```

Expected: PASS — 10 tests.

- [ ] **Step 5: Commit**

```bash
git add alchemist_core/utils/constrained_region.py tests/unit/core/utils/test_constrained_region.py
git commit -m "feat(core): add constrained_region with variable helpers and projection

New module for the geometry of a linear-constrained region. This commit adds
numeric_variables, variable_bounds, and orthogonal projection onto a
constraint hyperplane. filter_feasible remains the sole definition of
feasibility; nothing here re-implements it."
```

---

### Task 3: `constrained_region.feasible_interval`

**Files:**
- Modify: `alchemist_core/utils/constrained_region.py`
- Modify: `tests/unit/core/utils/test_constrained_region.py`

**Interfaces:**
- Consumes: `numeric_variables`, `variable_bounds` from Task 2.
- Produces: `feasible_interval(search_space, var_name: str, fixed_values: Dict[str, float]) -> Optional[Tuple[float, float]]` — returns `None` when the intersection is empty.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/core/utils/test_constrained_region.py`:

```python
class TestFeasibleInterval:
    def test_positive_coefficient_gives_upper_bound(self):
        # x1 + x2 <= 6, x2 fixed at 2  ->  x1 <= 4
        s = _space()
        s.add_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=6.0)
        assert cr.feasible_interval(s, "x1", {"x2": 2.0}) == pytest.approx((0.0, 4.0))

    def test_negative_coefficient_gives_lower_bound(self):
        # -x1 + x2 <= 1, x2 fixed at 5  ->  -x1 <= -4  ->  x1 >= 4
        s = _space()
        s.add_constraint("inequality", {"x1": -1.0, "x2": 1.0}, rhs=1.0)
        assert cr.feasible_interval(s, "x1", {"x2": 5.0}) == pytest.approx((4.0, 10.0))

    def test_equality_collapses_the_interval_to_a_point(self):
        # x1 + x2 == 7, x2 fixed at 3  ->  x1 == 4
        s = _space()
        s.add_constraint("equality", {"x1": 1.0, "x2": 1.0}, rhs=7.0)
        lo, hi = cr.feasible_interval(s, "x1", {"x2": 3.0})
        assert lo == pytest.approx(4.0)
        assert hi == pytest.approx(4.0)

    def test_zero_coefficient_contributes_nothing(self):
        s = _space()
        s.add_constraint("inequality", {"x1": 0.0, "x2": 1.0}, rhs=3.0)
        assert cr.feasible_interval(s, "x1", {"x2": 1.0}) == pytest.approx((0.0, 10.0))

    def test_constraint_not_naming_the_variable_is_ignored(self):
        s = _space()
        s.add_constraint("inequality", {"x2": 1.0}, rhs=3.0)
        assert cr.feasible_interval(s, "x1", {"x2": 1.0}) == pytest.approx((0.0, 10.0))

    def test_empty_intersection_returns_none(self):
        # x1 <= -5 is outside the variable's own [0, 10] bounds.
        s = _space()
        s.add_constraint("inequality", {"x1": 1.0}, rhs=-5.0)
        assert cr.feasible_interval(s, "x1", {"x2": 1.0}) is None

    def test_two_constraints_intersect(self):
        s = _space()
        s.add_constraint("inequality", {"x1": 1.0}, rhs=8.0)
        s.add_constraint("inequality", {"x1": -1.0}, rhs=-2.0)  # x1 >= 2
        assert cr.feasible_interval(s, "x1", {"x2": 0.0}) == pytest.approx((2.0, 8.0))

    def test_no_constraints_returns_variable_bounds(self):
        s = _space()
        assert cr.feasible_interval(s, "x1", {"x2": 1.0}) == pytest.approx((0.0, 10.0))

    def test_unknown_variable_raises(self):
        s = _space()
        with pytest.raises(ValueError, match="not found"):
            cr.feasible_interval(s, "nope", {})
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/utils/test_constrained_region.py::TestFeasibleInterval -q
```

Expected: FAIL — `AttributeError: module ... has no attribute 'feasible_interval'`.

- [ ] **Step 3: Implement `feasible_interval`**

Append to `alchemist_core/utils/constrained_region.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/utils/test_constrained_region.py -q
```

Expected: PASS — 19 tests.

- [ ] **Step 5: Commit**

```bash
git add alchemist_core/utils/constrained_region.py tests/unit/core/utils/test_constrained_region.py
git commit -m "feat(core): add feasible_interval to constrained_region

Solves the registered linear constraints for one variable with the others
held fixed, intersected with that variable's own bounds. Returns None when
the intersection is empty."
```

---

### Task 4: `constrained_region.feasible_vertices`

**Files:**
- Modify: `alchemist_core/utils/constrained_region.py`
- Modify: `tests/unit/core/utils/test_constrained_region.py`

**Interfaces:**
- Consumes: `numeric_variables`, `variable_bounds` from Task 2.
- Produces: `feasible_vertices(search_space, *, fixed: Optional[Dict[str, Any]] = None, max_vars: int = 5, rtol: float = DOE_RTOL, atol: float = DOE_ATOL) -> pd.DataFrame` — empty DataFrame when there are no constraints or when the numeric-variable count exceeds `max_vars`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/core/utils/test_constrained_region.py`:

```python
class TestFeasibleVertices:
    def test_triangle_vertices_are_found(self):
        # x1, x2 in [0, 10] with x1 + x2 <= 10 is the triangle
        # (0,0), (10,0), (0,10).
        s = _space()
        s.add_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=10.0)
        df = cr.feasible_vertices(s)
        found = {(round(r.x1, 6), round(r.x2, 6)) for r in df.itertuples()}
        assert {(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)} <= found

    def test_every_returned_vertex_is_feasible(self):
        s = _space()
        s.add_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=7.0)
        df = cr.feasible_vertices(s)
        assert len(df) > 0
        assert s.filter_feasible(df, rtol=0.0, atol=1e-9).all()

    def test_no_constraints_returns_empty(self):
        s = _space()
        assert cr.feasible_vertices(s).empty

    def test_above_max_vars_returns_empty(self):
        s = _space()
        s.add_variable("x3", "real", min=0.0, max=10.0)
        s.add_constraint("inequality", {"x1": 1.0, "x2": 1.0, "x3": 1.0}, rhs=10.0)
        assert cr.feasible_vertices(s, max_vars=2).empty
        assert not cr.feasible_vertices(s, max_vars=3).empty

    def test_fixed_categorical_is_attached_to_every_vertex(self):
        s = _space()
        s.add_variable("cat", "categorical", values=["a", "b"])
        s.add_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=10.0)
        df = cr.feasible_vertices(s, fixed={"cat": "b"})
        assert len(df) > 0
        assert (df["cat"] == "b").all()

    def test_integer_vertices_are_rounded_and_still_feasible(self):
        s = SearchSpace()
        s.add_variable("i1", "integer", min=0, max=10)
        s.add_variable("i2", "integer", min=0, max=10)
        s.add_constraint("inequality", {"i1": 1.0, "i2": 1.0}, rhs=7.5)
        df = cr.feasible_vertices(s)
        assert len(df) > 0
        assert (df["i1"] == df["i1"].round()).all()
        assert s.filter_feasible(df, rtol=0.0, atol=1e-9).all()

    def test_equality_constraint_vertices_lie_on_the_hyperplane(self):
        s = _space()
        s.add_constraint("equality", {"x1": 1.0, "x2": 1.0}, rhs=6.0)
        df = cr.feasible_vertices(s)
        assert len(df) > 0
        assert np.allclose(df["x1"] + df["x2"], 6.0, atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/utils/test_constrained_region.py::TestFeasibleVertices -q
```

Expected: FAIL — `AttributeError: module ... has no attribute 'feasible_vertices'`.

- [ ] **Step 3: Implement `feasible_vertices`**

Append to `alchemist_core/utils/constrained_region.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/utils/test_constrained_region.py -q
```

Expected: PASS — 26 tests.

- [ ] **Step 5: Commit**

```bash
git add alchemist_core/utils/constrained_region.py tests/unit/core/utils/test_constrained_region.py
git commit -m "feat(core): add feasible_vertices to constrained_region

Enumerates the feasible polytope's vertices by intersecting n hyperplanes
drawn from constraints plus bound faces. Snapped for integer/discrete
variables and re-tested for feasibility afterward, since snapping can push a
solved vertex back out. Capped at max_vars with the omission logged."
```

---

### Task 5: `constrained_region.augment_with_boundary`

**Files:**
- Modify: `alchemist_core/utils/constrained_region.py`
- Modify: `tests/unit/core/utils/test_constrained_region.py`

**Interfaces:**
- Consumes: everything from Tasks 2–4.
- Produces:
  - `class InfeasibleRegionError(ValueError)`
  - `augment_with_boundary(search_space, points: pd.DataFrame, *, max_vertex_vars: int = 5, rtol: float = DOE_RTOL, atol: float = DOE_ATOL) -> Tuple[pd.DataFrame, Dict[str, Any]]`
  - The returned info dict has exactly these keys: `constraints_applied` (List[str]), `n_candidates_total` (int), `n_candidates_feasible` (int), `n_boundary_added` (int), `n_vertices_added` (int), `vertex_enumeration_skipped` (bool).

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/core/utils/test_constrained_region.py`:

```python
def _lattice(s, n_levels=5):
    """Raw-space full-factorial lattice over the numeric variables."""
    import itertools as it
    numeric = cr.numeric_variables(s)
    axes = []
    for v in numeric:
        lo, hi = cr.variable_bounds(v)
        axes.append(np.linspace(lo, hi, n_levels))
    rows = [dict(zip([v["name"] for v in numeric], combo))
            for combo in it.product(*axes)]
    return pd.DataFrame(rows)


class TestAugmentWithBoundary:
    def test_every_returned_row_is_feasible(self):
        s = _space()
        s.add_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=7.0)
        out, _info = cr.augment_with_boundary(s, _lattice(s))
        assert len(out) > 0
        assert s.filter_feasible(out, rtol=0.0, atol=1e-9).all()

    def test_boundary_points_exist_that_the_filtered_lattice_lacks(self):
        s = _space()
        s.add_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=7.0)
        lattice = _lattice(s)
        filtered = lattice[s.filter_feasible(lattice, rtol=0.0, atol=1e-9)]
        out, info = cr.augment_with_boundary(s, lattice)

        # Points lying ON the constraint (sum == 7) exist after augmentation.
        on_boundary = np.isclose(out["x1"] + out["x2"], 7.0, atol=1e-6).sum()
        assert on_boundary > 0
        assert len(out) > len(filtered)
        assert info["n_boundary_added"] + info["n_vertices_added"] > 0

    def test_info_dict_has_the_documented_keys(self):
        s = _space()
        s.add_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=7.0)
        _out, info = cr.augment_with_boundary(s, _lattice(s))
        assert set(info) == {
            "constraints_applied", "n_candidates_total", "n_candidates_feasible",
            "n_boundary_added", "n_vertices_added", "vertex_enumeration_skipped",
        }
        assert info["n_candidates_total"] == 25
        assert info["constraints_applied"] == ["constraint_0"]

    def test_no_constraints_returns_input_unchanged(self):
        s = _space()
        lattice = _lattice(s)
        out, info = cr.augment_with_boundary(s, lattice)
        pd.testing.assert_frame_equal(out, lattice)
        assert info["n_boundary_added"] == 0

    def test_empty_feasible_region_raises(self):
        s = _space()
        # x1 + x2 <= -1 is unreachable inside [0, 10]^2.
        s.add_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=-1.0)
        with pytest.raises(cr.InfeasibleRegionError, match="No feasible"):
            cr.augment_with_boundary(s, _lattice(s))

    def test_equality_constraint_yields_points_on_the_hyperplane(self):
        s = _space()
        s.add_constraint("equality", {"x1": 1.0, "x2": 1.0}, rhs=6.0)
        out, _info = cr.augment_with_boundary(s, _lattice(s))
        assert len(out) > 0
        assert np.allclose(out["x1"] + out["x2"], 6.0, atol=1e-6)

    def test_runs_per_categorical_combination(self):
        s = _space()
        s.add_variable("cat", "categorical", values=["a", "b"])
        s.add_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=7.0)
        lattice = _lattice(s)
        lattice = pd.concat([
            lattice.assign(cat="a"), lattice.assign(cat="b")
        ], ignore_index=True)
        out, _info = cr.augment_with_boundary(s, lattice)
        assert set(out["cat"]) == {"a", "b"}
        assert s.filter_feasible(out, rtol=0.0, atol=1e-9).all()

    def test_vertex_skip_is_reported_not_silent(self):
        s = SearchSpace()
        for i in range(6):
            s.add_variable(f"x{i}", "real", min=0.0, max=10.0)
        s.add_constraint("inequality", {"x0": 1.0, "x1": 1.0}, rhs=15.0)
        out, info = cr.augment_with_boundary(s, _lattice(s, n_levels=2),
                                             max_vertex_vars=5)
        assert info["vertex_enumeration_skipped"] is True
        assert info["n_vertices_added"] == 0
        assert len(out) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/utils/test_constrained_region.py::TestAugmentWithBoundary -q
```

Expected: FAIL — `AttributeError: module ... has no attribute 'InfeasibleRegionError'`.

- [ ] **Step 3: Implement the error class and `augment_with_boundary`**

Append to `alchemist_core/utils/constrained_region.py`:

```python
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

        merged = pd.concat([p for p in parts if not p.empty], ignore_index=True)
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/utils/test_constrained_region.py -q
```

Expected: PASS — 34 tests.

- [ ] **Step 5: Run the full suite to confirm nothing regressed**

```bash
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/ -q
```

Expected: 1000+ passed (new tests added), 10 skipped, 0 failed.

- [ ] **Step 6: Commit**

```bash
git add alchemist_core/utils/constrained_region.py tests/unit/core/utils/test_constrained_region.py
git commit -m "feat(core): add augment_with_boundary to constrained_region

Turns a candidate lattice into a feasible candidate set that includes points
ON the constraint boundary: keeps feasible lattice points, projects the
infeasible ones onto the constraints they violate, and adds polytope
vertices. Runs per categorical combination. Returns provenance so a reduced
candidate set is reported rather than silent."
```

---

### Task 6: Encode/decode helpers in `optimal_design`

**Files:**
- Modify: `alchemist_core/utils/optimal_design.py:309-390` (extract column map), `:680-755` (decode refactor)
- Create: `tests/unit/core/data/test_candidate_encoding.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `build_column_map(variables: List[Dict[str, Any]]) -> List[Dict[str, Any]]`
  - `decode_candidates(candidates_coded, column_map, variables, selected_indices=None) -> List[Dict[str, Any]]` — when `selected_indices` is `None`, decodes every row.
  - `encode_candidates(points: List[Dict[str, Any]], column_map, variables) -> np.ndarray`
  - `_decode_candidates` is kept as a thin alias so existing internal call sites are untouched.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/core/data/test_candidate_encoding.py`:

```python
"""Round-trip tests for candidate coded/raw encoding.

The constrained-candidate pipeline decodes the coded lattice to raw variable
space, augments it there (so filter_feasible stays the single definition of
feasibility), then re-encodes. That round trip must be exact.
"""

import numpy as np
import pytest

from alchemist_core.data.search_space import SearchSpace
from alchemist_core.utils.optimal_design import (
    build_column_map,
    decode_candidates,
    encode_candidates,
    generate_mixed_candidate_set,
)


def _space():
    s = SearchSpace()
    s.add_variable("x1", "real", min=0.0, max=10.0)
    s.add_variable("x2", "integer", min=0, max=8)
    s.add_variable("d1", "discrete", allowed_values=[1.0, 2.0, 4.0])
    s.add_variable("cat", "categorical", values=["a", "b"])
    return s


def test_build_column_map_matches_generate_mixed_candidate_set():
    s = _space()
    _cand, column_map = generate_mixed_candidate_set(s, n_levels=3)
    assert build_column_map(s.variables) == column_map


def test_decode_all_rows_when_indices_omitted():
    s = _space()
    cand, column_map = generate_mixed_candidate_set(s, n_levels=3)
    points = decode_candidates(cand, column_map, s.variables)
    assert len(points) == cand.shape[0]


def test_decode_respects_explicit_indices():
    s = _space()
    cand, column_map = generate_mixed_candidate_set(s, n_levels=3)
    points = decode_candidates(cand, column_map, s.variables,
                               selected_indices=np.array([0, 2]))
    assert len(points) == 2


def test_encode_decode_round_trip_is_identity_on_the_lattice():
    s = _space()
    cand, column_map = generate_mixed_candidate_set(s, n_levels=3)
    points = decode_candidates(cand, column_map, s.variables)
    recoded = encode_candidates(points, column_map, s.variables)
    assert recoded.shape == cand.shape
    assert np.allclose(recoded, cand, atol=1e-9)


def test_encode_handles_categorical_one_hot():
    s = _space()
    column_map = build_column_map(s.variables)
    coded = encode_candidates(
        [{"x1": 10.0, "x2": 8, "d1": 4.0, "cat": "b"}], column_map, s.variables
    )
    onehot = [coded[0][i] for i, cm in enumerate(column_map) if cm["type"] == "onehot"]
    assert onehot == [0.0, 1.0]


def test_encode_maps_bounds_to_plus_minus_one():
    s = SearchSpace()
    s.add_variable("x1", "real", min=2.0, max=6.0)
    column_map = build_column_map(s.variables)
    coded = encode_candidates([{"x1": 2.0}, {"x1": 6.0}, {"x1": 4.0}],
                              column_map, s.variables)
    assert coded[0][0] == pytest.approx(-1.0)
    assert coded[1][0] == pytest.approx(1.0)
    assert coded[2][0] == pytest.approx(0.0)


def test_encode_degenerate_range_is_zero():
    s = SearchSpace()
    s.add_variable("x1", "real", min=5.0, max=5.0)
    column_map = build_column_map(s.variables)
    coded = encode_candidates([{"x1": 5.0}], column_map, s.variables)
    assert coded[0][0] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/data/test_candidate_encoding.py -q
```

Expected: FAIL — `ImportError: cannot import name 'build_column_map'`.

- [ ] **Step 3: Extract `build_column_map` and use it in `generate_mixed_candidate_set`**

In `alchemist_core/utils/optimal_design.py`, add this function immediately **above** `generate_mixed_candidate_set` (currently line 309):

```python
def build_column_map(variables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Coded-column metadata for a variable list.

    Continuous (real/integer/discrete) variables occupy one column each;
    categorical variables occupy one one-hot column per category. Extracted
    from :func:`generate_mixed_candidate_set` so the constrained-candidate
    pipeline can build a column map without generating a full lattice.

    Note that ``context`` variables produce no column, matching the existing
    behavior of :func:`generate_mixed_candidate_set`.
    """
    column_map: List[Dict[str, Any]] = []
    for j, var in enumerate(variables):
        if var["type"] in ("real", "integer", "discrete"):
            column_map.append({
                "var_idx": j,
                "var_name": var["name"],
                "type": "continuous",
                "category": None,
            })
        elif var["type"] == "categorical":
            cats = var.get("values", var.get("categories", []))
            for cat_val in cats:
                column_map.append({
                    "var_idx": j,
                    "var_name": var["name"],
                    "type": "onehot",
                    "category": cat_val,
                })
    return column_map
```

Then in `generate_mixed_candidate_set`, replace the column-map construction loop (the block starting `for j, var in enumerate(variables):` at roughly line 366, through the end of the `elif var["type"] == "categorical":` branch) with:

```python
    # Build coded candidate matrix with one-hot encoding for categoricals
    column_map = build_column_map(variables)
    coded_columns = []
    for cm in column_map:
        j = cm["var_idx"]
        if cm["type"] == "continuous":
            coded_columns.append(raw_grid[:, j].reshape(-1, 1))
        else:
            var = variables[j]
            cats = var.get("values", var.get("categories", []))
            k = cats.index(cm["category"])
            cat_indices = raw_grid[:, j].astype(int)
            coded_columns.append((cat_indices == k).astype(float).reshape(-1, 1))

    candidates = np.hstack(coded_columns)
    return candidates, column_map
```

- [ ] **Step 4: Run the column-map test to verify the extraction is faithful**

```bash
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/data/test_candidate_encoding.py::test_build_column_map_matches_generate_mixed_candidate_set -q
```

Expected: PASS.

- [ ] **Step 5: Refactor `_decode_candidates` into a public `decode_candidates`**

In `alchemist_core/utils/optimal_design.py`, change the signature at line 680 from:

```python
def _decode_candidates(
    candidates_coded: np.ndarray,
    selected_indices: np.ndarray,
    column_map: List[Dict[str, Any]],
    variables: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
```

to:

```python
def decode_candidates(
    candidates_coded: np.ndarray,
    column_map: List[Dict[str, Any]],
    variables: List[Dict[str, Any]],
    selected_indices: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
```

Immediately inside the body, before the `var_to_cols` lookup is built, insert:

```python
    if selected_indices is None:
        selected_indices = np.arange(candidates_coded.shape[0])
```

Then add this alias immediately **after** the function's closing `return points`:

```python
def _decode_candidates(candidates_coded, selected_indices, column_map, variables):
    """Backwards-compatible alias with the original positional argument order."""
    return decode_candidates(candidates_coded, column_map, variables,
                             selected_indices=selected_indices)
```

- [ ] **Step 6: Implement `encode_candidates`**

Add immediately after the `_decode_candidates` alias:

```python
def encode_candidates(
    points: List[Dict[str, Any]],
    column_map: List[Dict[str, Any]],
    variables: List[Dict[str, Any]],
) -> np.ndarray:
    """Inverse of :func:`decode_candidates` — raw values back to coded columns.

    Continuous variables map to ``[-1, +1]`` via ``(actual - mid) / half_range``
    over the variable's range (``discrete`` uses the min and max of its allowed
    values). Categorical variables become one-hot columns.

    Args:
        points: raw-space points, as dicts keyed by variable name. Accepts a
            list of dicts or anything ``pandas.DataFrame.to_dict("records")``
            produces.
        column_map: column metadata from :func:`build_column_map`.
        variables: variable dicts from ``SearchSpace.variables``.

    Returns:
        ndarray of shape ``(len(points), len(column_map))``.
    """
    rows: List[List[float]] = []
    for point in points:
        row: List[float] = []
        for cm in column_map:
            var = variables[cm["var_idx"]]
            name = var["name"]
            if cm["type"] == "onehot":
                row.append(1.0 if point.get(name) == cm["category"] else 0.0)
                continue

            value = float(point[name])
            if var["type"] == "discrete":
                allowed = var["allowed_values"]
                low, high = float(min(allowed)), float(max(allowed))
            else:
                low, high = float(var["min"]), float(var["max"])

            if high == low:
                row.append(0.0)
            else:
                mid = (low + high) / 2.0
                half_range = (high - low) / 2.0
                row.append((value - mid) / half_range)
        rows.append(row)

    return np.array(rows, dtype=float)
```

- [ ] **Step 7: Run the encoding tests to verify they pass**

```bash
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/data/test_candidate_encoding.py -q
```

Expected: PASS — 7 tests.

- [ ] **Step 8: Run the full suite, including the golden lock**

```bash
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/ -q
```

Expected: all pass. The golden test from Task 1 is the check that the `generate_mixed_candidate_set` refactor changed nothing.

- [ ] **Step 9: Commit**

```bash
git add alchemist_core/utils/optimal_design.py tests/unit/core/data/test_candidate_encoding.py
git commit -m "refactor(core): extract build_column_map, add encode_candidates

The constrained-candidate pipeline needs to decode the coded lattice to raw
variable space, augment there, and re-encode. That needs a column map without
generating a lattice, a decoder that handles all rows, and an encoder — none
of which existed.

decode_candidates supersedes _decode_candidates, which stays as an alias so
existing call sites are untouched. Golden unconstrained tests confirm the
generate_mixed_candidate_set refactor is behavior-preserving."
```

---

### Task 7: Wire boundary augmentation into `run_optimal_design`

**Files:**
- Modify: `alchemist_core/utils/optimal_design.py:988-1000`
- Create: `tests/unit/core/data/test_constrained_optimal_design.py`

**Interfaces:**
- Consumes: `constrained_region.augment_with_boundary`, `build_column_map`, `decode_candidates`, `encode_candidates`.
- Produces: `run_optimal_design`'s returned `info` dict gains a `"feasibility"` key — the info dict from `augment_with_boundary`, or `None` when unconstrained.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/core/data/test_constrained_optimal_design.py`:

```python
"""Constrained D/A/I-optimal designs select from a feasible, boundary-aware set."""

import numpy as np
import pandas as pd
import pytest

from alchemist_core import OptimizationSession
from alchemist_core.utils.constrained_region import InfeasibleRegionError

FEAS_TOL = 1e-6


def _session():
    s = OptimizationSession()
    s.add_variable("x1", "real", bounds=(0.0, 10.0))
    s.add_variable("x2", "real", bounds=(0.0, 10.0))
    s.add_variable("x3", "real", bounds=(0.0, 10.0))
    return s


def test_every_point_of_a_constrained_optimal_design_is_feasible():
    s = _session()
    s.add_input_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=12.0)
    points, _info = s.generate_optimal_design(
        n_points=10, model_type="quadratic", random_seed=7
    )
    df = pd.DataFrame(points)
    assert ((df["x1"] + df["x2"]) <= 12.0 + FEAS_TOL).all()


def test_design_reaches_the_constraint_boundary():
    """Filter-only would leave a dead band; augmentation must not."""
    s = _session()
    s.add_input_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=12.0)
    points, _info = s.generate_optimal_design(
        n_points=12, model_type="quadratic", random_seed=7
    )
    df = pd.DataFrame(points)
    closest = (12.0 - (df["x1"] + df["x2"])).min()
    # A filtered 5-level lattice on [0,10] can get no closer than 2.0 here.
    assert closest < 0.5


def test_feasibility_info_is_reported():
    s = _session()
    s.add_input_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=12.0)
    _points, info = s.generate_optimal_design(
        n_points=10, model_type="quadratic", random_seed=7
    )
    feas = info["feasibility"]
    assert feas["constraints_applied"] == ["constraint_0"]
    assert feas["n_candidates_feasible"] < feas["n_candidates_total"]
    assert feas["n_boundary_added"] + feas["n_vertices_added"] > 0


def test_unconstrained_design_reports_no_feasibility_block():
    s = _session()
    _points, info = s.generate_optimal_design(
        n_points=10, model_type="quadratic", random_seed=7
    )
    assert info["feasibility"] is None


def test_equality_constrained_design_lies_on_the_hyperplane():
    s = _session()
    s.add_input_constraint("equality", {"x1": 1.0, "x2": 1.0}, rhs=8.0)
    points, _info = s.generate_optimal_design(
        n_points=8, model_type="linear", random_seed=7
    )
    df = pd.DataFrame(points)
    assert np.allclose(df["x1"] + df["x2"], 8.0, atol=1e-5)


def test_empty_feasible_region_raises():
    s = _session()
    s.add_input_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=-1.0)
    with pytest.raises(InfeasibleRegionError):
        s.generate_optimal_design(n_points=8, model_type="linear", random_seed=7)


def test_constrained_design_beats_filter_only_on_d_efficiency():
    """The quantitative justification for augmenting rather than just filtering."""
    from alchemist_core.utils.optimal_design import (
        build_custom_design_matrix, build_column_map, encode_candidates,
        parse_model_spec,
    )

    s = _session()
    s.add_input_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=12.0)
    terms = parse_model_spec(s.search_space, model_type="quadratic")
    column_map = build_column_map(s.search_space.variables)

    def _logdet(points):
        coded = encode_candidates(points, column_map, s.search_space.variables)
        X = build_custom_design_matrix(coded, terms, column_map,
                                       s.search_space.variables)
        sign, logabsdet = np.linalg.slogdet(X.T @ X)
        return logabsdet if sign > 0 else -np.inf

    augmented, _info = s.generate_optimal_design(
        n_points=12, model_type="quadratic", random_seed=7
    )
    assert _logdet(augmented) > -np.inf
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/data/test_constrained_optimal_design.py -q
```

Expected: FAIL — designs contain infeasible points and `info` has no `"feasibility"` key.

- [ ] **Step 3: Add the augmentation step to `run_optimal_design`**

In `alchemist_core/utils/optimal_design.py`, replace the candidate-generation block at line 988:

```python
    # Generate candidate set
    candidates_coded, column_map = generate_mixed_candidate_set(
        search_space, n_levels=n_levels
    )
```

with:

```python
    # Generate candidate set
    candidates_coded, column_map = generate_mixed_candidate_set(
        search_space, n_levels=n_levels
    )

    # Constrained designs select from a feasible candidate set that includes
    # points ON the constraint boundary. A filtered lattice has none, and an
    # optimal design wants precisely the extremes of the feasible region.
    # Geometry is done in raw variable space so SearchSpace.filter_feasible
    # stays the single definition of feasibility.
    feasibility_info = None
    if getattr(search_space, "constraints", None):
        from alchemist_core.utils import constrained_region

        raw_points = decode_candidates(candidates_coded, column_map, variables)
        raw_df = pd.DataFrame(raw_points)
        raw_df, feasibility_info = constrained_region.augment_with_boundary(
            search_space, raw_df
        )
        candidates_coded = encode_candidates(
            raw_df.to_dict("records"), column_map, variables
        )
```

- [ ] **Step 4: Add the pandas import**

At the top of `alchemist_core/utils/optimal_design.py`, the import block currently reads:

```python
import numpy as np
```

Change it to:

```python
import numpy as np
import pandas as pd
```

- [ ] **Step 5: Thread `feasibility` into the returned info dict**

Find the `info` dict returned by `run_optimal_design` (populated from `_run_algorithm`'s return, just before the final `return points, info`). Immediately before that return, add:

```python
    info["feasibility"] = feasibility_info
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/data/test_constrained_optimal_design.py -q
```

Expected: PASS — 7 tests.

- [ ] **Step 7: Run the full suite**

```bash
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/ -q
```

Expected: all pass, golden lock included.

- [ ] **Step 8: Commit**

```bash
git add alchemist_core/utils/optimal_design.py tests/unit/core/data/test_constrained_optimal_design.py
git commit -m "feat(core): constrained optimal designs select from a feasible candidate set

run_optimal_design now filters and augments its candidate set before the
exchange algorithm runs, so D/A/I-optimal designs over a constrained region
are genuine constrained optimal designs rather than filtered box designs.

Previously the exchange algorithm optimized over candidates including
infeasible ones and only the selected design was filtered afterward — it
spent its budget on points it could not keep.

Equality constraints now work too: projection onto the hyperplane generates
the feasible set exactly, where filtering a lattice found almost nothing.

BREAKING: a constrained optimal design returns different points for the same
seed. This is the fix, not a regression. Unconstrained output is unchanged."
```

---

### Task 8: Feasible-interval spreading of non-model variables

**Files:**
- Modify: `alchemist_core/utils/optimal_design.py:1056-1075`
- Modify: `tests/unit/core/data/test_constrained_optimal_design.py`

**Interfaces:**
- Consumes: `constrained_region.feasible_interval`.
- Produces: no new public names. `run_optimal_design` gains a final feasibility assertion.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/core/data/test_constrained_optimal_design.py`:

```python
class TestNonModelVariableSpreading:
    def test_constrained_non_model_variable_stays_feasible(self):
        """x3 is in no model term AND in a constraint.

        The post-hoc spread step used to overwrite it across its full range
        after all feasibility work was done, writing straight through the
        constraint.
        """
        s = _session()
        s.add_input_constraint("inequality", {"x1": 1.0, "x3": 1.0}, rhs=11.0)
        points, _info = s.generate_optimal_design(
            n_points=10, effects=["x1", "x2"], random_seed=7
        )
        df = pd.DataFrame(points)
        assert ((df["x1"] + df["x3"]) <= 11.0 + FEAS_TOL).all()

    def test_constrained_non_model_variable_is_still_spread(self):
        """Feasible must not mean clumped onto a single value."""
        s = _session()
        s.add_input_constraint("inequality", {"x1": 1.0, "x3": 1.0}, rhs=11.0)
        points, _info = s.generate_optimal_design(
            n_points=10, effects=["x1", "x2"], random_seed=7
        )
        df = pd.DataFrame(points)
        assert df["x3"].nunique() > 3

    def test_two_non_model_variables_sharing_a_constraint(self):
        s = _session()
        s.add_variable("x4", "real", bounds=(0.0, 10.0))
        s.add_input_constraint("inequality", {"x3": 1.0, "x4": 1.0}, rhs=9.0)
        points, _info = s.generate_optimal_design(
            n_points=10, effects=["x1", "x2"], random_seed=7
        )
        df = pd.DataFrame(points)
        assert ((df["x3"] + df["x4"]) <= 9.0 + FEAS_TOL).all()

    def test_unconstrained_non_model_variable_is_unchanged(self):
        """No constraint touches x3, so today's spread behavior must persist."""
        s = _session()
        s.add_input_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=12.0)
        points, _info = s.generate_optimal_design(
            n_points=10, effects=["x1", "x2"], random_seed=7
        )
        df = pd.DataFrame(points)
        # An unconstrained spread variable covers its full range endpoints.
        assert df["x3"].min() == pytest.approx(0.0)
        assert df["x3"].max() == pytest.approx(10.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/data/test_constrained_optimal_design.py::TestNonModelVariableSpreading -q
```

Expected: FAIL — the first and third tests fail with infeasible sums.

- [ ] **Step 3: Replace the spreading loop**

In `alchemist_core/utils/optimal_design.py`, the block beginning at line 1056 currently reads:

```python
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
```

Replace the whole `if unused_var_indices:` block (through the end of the assignment of `spread_vals` back onto `points`) with:

```python
    if unused_var_indices:
        n = len(points)
        spread_rng = np.random.default_rng(random_seed)
        constrained_names = set()
        for c in getattr(search_space, "constraints", None) or []:
            constrained_names.update(c["coefficients"].keys())

        for var_idx in unused_var_indices:
            var = variables[var_idx]
            name = var["name"]

            if name in constrained_names:
                # This variable is invisible to the exchange algorithm but IS
                # bound by a constraint. Spreading it across its full range
                # would write straight through the constraint, undoing the
                # feasibility work above. Draw each row's value from that
                # row's own feasible interval instead: the variable still
                # looks spread, and the design stays feasible by construction.
                from alchemist_core.utils import constrained_region

                fractions = list(np.linspace(0.0, 1.0, n))
                spread_rng.shuffle(fractions)
                for i, point in enumerate(points):
                    fixed = {k: v for k, v in point.items() if k != name}
                    interval = constrained_region.feasible_interval(
                        search_space, name, fixed
                    )
                    if interval is None:
                        # Unreachable: the selected point was already feasible.
                        logger.warning(
                            "No feasible interval for non-model variable '%s' "
                            "at design row %d; keeping the selected value.",
                            name, i,
                        )
                        continue
                    lo, hi = interval
                    value = lo + fractions[i] * (hi - lo)
                    point[name] = constrained_region.snap_to_variable(value, var)
                continue

            # Unconstrained: behavior is unchanged from before.
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
            else:
                continue

            for i, point in enumerate(points):
                point[name] = spread_vals[i]
```

> **Note for the implementer:** read the existing block before replacing it. If the current code assigns `spread_vals` back onto `points` differently from the final loop shown above (for example via a categorical branch), preserve that assignment exactly for the unconstrained path. The unconstrained path must stay bit-for-bit identical, and Task 1's golden test is what proves it.

- [ ] **Step 4: Add the final feasibility assertion**

Immediately before `run_optimal_design`'s `return points, info`, add:

```python
    # A constrained optimal design returning an infeasible point is a bug.
    # Fail here rather than letting it reach a consumer.
    if getattr(search_space, "constraints", None) and points:
        final_mask = search_space.filter_feasible(
            pd.DataFrame(points), rtol=0.0, atol=1e-9
        )
        if not final_mask.all():
            raise RuntimeError(
                f"Internal error: {(~final_mask).sum()} of {len(points)} optimal "
                f"design points violate the registered input constraints after "
                f"generation. This is a bug in the constrained design pipeline."
            )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/data/test_constrained_optimal_design.py -q
```

Expected: PASS — 11 tests.

- [ ] **Step 6: Run the full suite**

```bash
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/ -q
```

Expected: all pass. Task 1's golden test is what confirms unconstrained spreading is untouched.

- [ ] **Step 7: Commit**

```bash
git add alchemist_core/utils/optimal_design.py tests/unit/core/data/test_constrained_optimal_design.py
git commit -m "fix(core): spread non-model variables within their feasible interval

Variables absent from every model term are invisible to the exchange
algorithm, so run_optimal_design overwrote them with a shuffled linspace
across their full range. When such a variable was named in a constraint, that
overwrite ran AFTER all feasibility work and wrote straight through the
constraint — every point just guaranteed feasible could come back infeasible.

Constrained non-model variables now draw each row's value from that row's own
feasible interval, so they stay spread without crossing the constraint.
Variables in no constraint are untouched. A final assertion fails loudly if
any point is infeasible."
```

---

### Task 9: Classical design estimability gate

**Files:**
- Modify: `alchemist_core/utils/doe.py:261-291`, and the `generate_initial_design` signature at `:51`
- Modify: `alchemist_core/session.py:863` (pass `allow_infeasible` through)
- Create: `tests/unit/core/data/test_classical_estimability.py`

**Interfaces:**
- Consumes: `build_column_map`, `encode_candidates`, `parse_model_spec`, `build_custom_design_matrix`, `get_model_term_names` from `optimal_design`.
- Produces:
  - `class DesignNotEstimableError(ValueError)` in `doe.py`
  - `IMPLIED_MODEL: Dict[str, str]` in `doe.py`
  - `generate_initial_design(..., allow_infeasible: bool = False)`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/core/data/test_classical_estimability.py`:

```python
"""A constrained classical design must not silently return a degraded design.

A CCD's axial points are what let it estimate quadratic terms. Dropping the
infeasible ones does not give "a CCD minus two runs" — it gives a design that
is rank-deficient for the model it claims to fit. Dropping a replicated center
point, by contrast, is harmless. The gate distinguishes the two.
"""

import pandas as pd
import pytest

from alchemist_core import OptimizationSession
from alchemist_core.utils.doe import DesignNotEstimableError

FEAS_TOL = 1e-6


def _session():
    s = OptimizationSession()
    s.add_variable("x1", "real", bounds=(0.0, 10.0))
    s.add_variable("x2", "real", bounds=(0.0, 10.0))
    s.add_variable("x3", "real", bounds=(0.0, 10.0))
    return s


def test_ccd_losing_structural_points_raises():
    s = _session()
    # Cuts off the high-x1/high-x2 corner, removing factorial and axial points.
    s.add_input_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=11.0)
    with pytest.raises(DesignNotEstimableError, match="ccd"):
        s.generate_initial_design(method="ccd", random_seed=7)


def test_raise_message_names_the_inestimable_terms():
    s = _session()
    s.add_input_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=11.0)
    with pytest.raises(DesignNotEstimableError) as exc:
        s.generate_initial_design(method="ccd", random_seed=7)
    msg = str(exc.value)
    assert "dropped" in msg
    assert "optimal" in msg  # steers toward the right tool


def test_allow_infeasible_restores_warn_and_drop():
    s = _session()
    s.add_input_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=11.0)
    points = s.generate_initial_design(
        method="ccd", random_seed=7, allow_infeasible=True
    )
    df = pd.DataFrame(points)
    assert len(df) > 0
    assert ((df["x1"] + df["x2"]) <= 11.0 + FEAS_TOL).all()


def test_design_losing_nothing_passes_unchanged():
    s = _session()
    # Satisfied everywhere in [0, 10]^3 — nothing is dropped.
    s.add_input_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=100.0)
    points = s.generate_initial_design(method="ccd", random_seed=7)
    assert len(points) > 0


def test_fully_infeasible_design_still_raises_the_original_error():
    s = _session()
    s.add_input_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=-1.0)
    with pytest.raises(ValueError, match="No 'ccd' design points"):
        s.generate_initial_design(method="ccd", random_seed=7)


def test_optimal_method_is_exempt_from_the_gate():
    """'optimal' is in CLASSICAL_METHODS but its points are already feasible."""
    s = _session()
    s.add_input_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=11.0)
    points = s.generate_initial_design(
        method="optimal", n_points=10, model_type="quadratic", random_seed=7
    )
    df = pd.DataFrame(points)
    assert ((df["x1"] + df["x2"]) <= 11.0 + FEAS_TOL).all()


def test_space_filling_methods_are_unaffected():
    s = _session()
    s.add_input_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=11.0)
    points = s.generate_initial_design(n_points=8, method="lhs", random_seed=7)
    df = pd.DataFrame(points)
    assert len(df) == 8
    assert ((df["x1"] + df["x2"]) <= 11.0 + FEAS_TOL).all()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/data/test_classical_estimability.py -q
```

Expected: FAIL — `ImportError: cannot import name 'DesignNotEstimableError'`.

- [ ] **Step 3: Add the error class and implied-model map**

In `alchemist_core/utils/doe.py`, immediately after the `_DEFAULT_GENERATORS` dict (around line 49), add:

```python
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
```

- [ ] **Step 4: Add `allow_infeasible` to the signature**

In `generate_initial_design` (line 51), add a keyword parameter. The signature ends with several method-specific keywords; add this one immediately before the closing `)`:

```python
    allow_infeasible: bool = False,
```

And document it in the docstring's `Args:` section:

```
        allow_infeasible: For constrained classical designs, return the
            feasible remnant with a warning even when the design's implied
            model is no longer estimable. Default False raises instead.
```

- [ ] **Step 5: Replace the post-hoc filter block**

In `alchemist_core/utils/doe.py`, replace the block at line 261 (from the comment `# Classical / optimal designs have fixed structure` through `points = [p for p, ok in zip(points, mask) if ok]`) with:

```python
    # Classical designs have fixed structure and cannot be resampled. Filter to
    # feasible rows, then decide whether the remnant is still the design it
    # claims to be. ('optimal' is exempt: its candidate set is already
    # constrained, and its model is user-specified rather than implied.)
    if (method in CLASSICAL_METHODS and method != "optimal"
            and getattr(search_space, 'constraints', None)):
        import pandas as pd
        mask = search_space.filter_feasible(pd.DataFrame(points), rtol=0.0, atol=1e-9)
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
```

- [ ] **Step 6: Implement `_inestimable_terms`**

Add to `alchemist_core/utils/doe.py`, in the "Validation helpers" section near `_validate_classical_design`:

```python
def _inestimable_terms(search_space: SearchSpace, points: List[Dict[str, Any]],
                       method: str, n_levels: int) -> List[str]:
    """Model terms the surviving points can no longer estimate.

    Builds the design matrix for the method's implied model from the points
    that survived constraint filtering and compares its rank to its column
    count. A rank-deficient matrix means the design cannot estimate every term
    it was chosen for.

    Returns an empty list when the model is fully estimable, or when the check
    cannot be performed (an unparseable model, no points) — the gate should
    never block on its own inability to judge.
    """
    # Imported here, matching the existing lazy imports at doe.py:240 and :760.
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
        column_map = build_column_map(search_space.variables)
        coded = encode_candidates(points, column_map, search_space.variables)
        X = build_custom_design_matrix(coded, terms, column_map,
                                       search_space.variables)
    except (ValueError, KeyError, IndexError) as e:
        logger.debug("Estimability check skipped for '%s': %s", method, e)
        return []

    p_columns = X.shape[1]
    rank = int(np.linalg.matrix_rank(X))
    if rank >= p_columns:
        return []

    # Rank-deficient: report the terms beyond the rank, which are the ones the
    # design can no longer separate.
    names = get_model_term_names(search_space, terms)
    return names[rank:] if len(names) >= p_columns else names
```

- [ ] **Step 7: Thread `allow_infeasible` through `session.generate_initial_design`**

`OptimizationSession.generate_initial_design` (`session.py:863`) already forwards `**kwargs` to `doe.generate_initial_design`, so `allow_infeasible` passes through with no code change. Verify:

```bash
~/miniforge3/envs/alchemist-env/bin/python -c "
import inspect
from alchemist_core.session import OptimizationSession
print(inspect.signature(OptimizationSession.generate_initial_design))
"
```

Expected: the signature ends with `**kwargs`. If it does not, add `allow_infeasible: bool = False` explicitly and forward it.

- [ ] **Step 8: Run tests to verify they pass**

```bash
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/data/test_classical_estimability.py -q
```

Expected: PASS — 7 tests.

- [ ] **Step 9: Run the full suite**

```bash
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/ -q
```

Expected: all pass, golden lock included.

- [ ] **Step 10: Commit**

```bash
git add alchemist_core/utils/doe.py tests/unit/core/data/test_classical_estimability.py
git commit -m "feat(core): gate constrained classical designs on model estimability

A constrained CCD previously generated all its points ignoring the fence,
dropped the infeasible ones with a log warning, and returned the remnant.
But a CCD's axial points are what let it estimate quadratic terms — dropping
them yields a design that is rank-deficient for the model it claims to fit,
handed back with a log line nobody reads. Dropping a replicated center point
is harmless. The old code treated both identically.

Now the surviving points are checked against the design's implied model
(quadratic for ccd/box_behnken, interaction for 2-level factorials, main
effects for plackett_burman/gsd). Estimable, and it returns with an INFO
log; rank-deficient, and it raises, naming the inestimable terms and
steering toward method='optimal'.

BREAKING: constrained classical designs that lost structural points now
raise. Pass allow_infeasible=True for the old behavior. 'optimal' is exempt
and unconstrained behavior is unchanged."
```

---

### Task 10: Guard non-numeric variables in `add_constraint`

**Files:**
- Modify: `alchemist_core/data/search_space.py:353-380`
- Create: `tests/unit/core/data/test_constraint_validation.py`

**Interfaces:**
- Consumes: `constrained_region.NUMERIC_TYPES`.
- Produces: `SearchSpace.add_constraint` raises `ValueError` for a non-numeric coefficient variable.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/core/data/test_constraint_validation.py`:

```python
"""add_constraint validates that coefficient variables are numeric.

Without this, a categorical in a constraint fails much later and far away,
inside filter_feasible, as float('some_string').
"""

import pytest

from alchemist_core.data.search_space import SearchSpace


def _space():
    s = SearchSpace()
    s.add_variable("x1", "real", min=0.0, max=10.0)
    s.add_variable("i1", "integer", min=0, max=5)
    s.add_variable("d1", "discrete", allowed_values=[1.0, 2.0, 4.0])
    s.add_variable("cat", "categorical", values=["a", "b"])
    s.add_variable("ctx", "context")
    return s


def test_categorical_coefficient_raises_at_registration():
    s = _space()
    with pytest.raises(ValueError, match="not numeric"):
        s.add_constraint("inequality", {"x1": 1.0, "cat": 1.0}, rhs=5.0)


def test_context_coefficient_raises_at_registration():
    s = _space()
    with pytest.raises(ValueError, match="not numeric"):
        s.add_constraint("inequality", {"ctx": 1.0}, rhs=5.0)


def test_error_names_the_offending_variable_and_its_type():
    s = _space()
    with pytest.raises(ValueError) as exc:
        s.add_constraint("inequality", {"cat": 1.0}, rhs=5.0)
    assert "cat" in str(exc.value)
    assert "categorical" in str(exc.value)


def test_bad_constraint_is_not_registered():
    s = _space()
    with pytest.raises(ValueError):
        s.add_constraint("inequality", {"cat": 1.0}, rhs=5.0)
    assert s.constraints == []


def test_all_numeric_types_are_accepted():
    s = _space()
    s.add_constraint("inequality", {"x1": 1.0, "i1": 1.0, "d1": 1.0}, rhs=20.0)
    assert len(s.constraints) == 1


def test_unknown_variable_still_raises_the_original_error():
    s = _space()
    with pytest.raises(ValueError, match="not found in search space"):
        s.add_constraint("inequality", {"nope": 1.0}, rhs=5.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/data/test_constraint_validation.py -q
```

Expected: FAIL — the first five tests fail; no `ValueError` is raised.

- [ ] **Step 3: Add the guard**

In `alchemist_core/data/search_space.py`, inside `add_constraint`, the existing validation loop reads:

```python
        var_names = self.get_variable_names()
        for var_name in coefficients:
            if var_name not in var_names:
                raise ValueError(f"Variable '{var_name}' in constraint not found in search space. "
                                 f"Available: {var_names}")
```

Replace it with:

```python
        var_names = self.get_variable_names()
        by_name = {v["name"]: v for v in self.variables}
        numeric_types = ("real", "integer", "discrete")
        for var_name in coefficients:
            if var_name not in var_names:
                raise ValueError(f"Variable '{var_name}' in constraint not found in search space. "
                                 f"Available: {var_names}")
            var_type = by_name[var_name].get("type")
            if var_type not in numeric_types:
                raise ValueError(
                    f"Variable '{var_name}' is not numeric (type '{var_type}') and "
                    f"cannot appear in a linear constraint. Constraints may only "
                    f"reference variables of type {', '.join(numeric_types)}."
                )
```

> `numeric_types` is duplicated from `constrained_region.NUMERIC_TYPES` rather than imported, because `search_space` is imported *by* `constrained_region` and importing back would create a cycle. The tuple is three literals with no behavior; a comment in each location points at the other.

Add this comment directly above the tuple:

```python
        # Mirrors constrained_region.NUMERIC_TYPES. Not imported: constrained_region
        # imports this module, so importing back would create a cycle.
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/unit/core/data/test_constraint_validation.py -q
```

Expected: PASS — 6 tests.

- [ ] **Step 5: Run the full suite**

```bash
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/ -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add alchemist_core/data/search_space.py tests/unit/core/data/test_constraint_validation.py
git commit -m "fix(core): reject non-numeric variables in add_constraint

add_constraint validated that a coefficient variable exists but not that it
has a numeric range. A categorical or context variable was accepted happily
and then failed much later inside filter_feasible as float() on a string,
far from the cause.

BREAKING: such a constraint now raises at registration. It was already
non-functional; it just failed later and less clearly."
```

---

## PHASE 1 CHECKPOINT

Stop here for review before starting Phase 2.

- [ ] Run the full suite: `~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/ -q`
- [ ] Confirm the golden lock passes — unconstrained behavior is provably unchanged
- [ ] Confirm the pass count is ≥ 1000 plus the new tests, with 0 failures
- [ ] Report to Caleb: what shipped, the three breaking changes, and anything surprising

---

# PHASE 2 — API

---

### Task 11: Constraint CRUD endpoints

**Files:**
- Modify: `api/models/requests.py`, `api/models/responses.py`, `api/routers/variables.py`
- Create: `tests/integration/api/test_constraints_router.py`

**Interfaces:**
- Consumes: `SearchSpace.add_constraint`, `SearchSpace.get_constraints`.
- Produces:
  - `AddConstraintRequest` with fields `constraint_type: Literal["inequality","equality"]`, `coefficients: Dict[str, float]`, `rhs: float`, `name: Optional[str]`
  - `ConstraintResponse` with `message: str`, `constraint: Dict[str, Any]`
  - `ConstraintsListResponse` with `constraints: List[Dict[str, Any]]`, `n_constraints: int`
  - Routes `POST`/`GET` `/{session_id}/constraints`, `DELETE /{session_id}/constraints/{constraint_name}`

- [ ] **Step 1: Write the failing tests**

Create `tests/integration/api/test_constraints_router.py`:

```python
"""Constraint CRUD over REST.

Before this, constraints could only be set from Python, so a non-Python
consumer could not use the constraint feature at all.

Routers mount under /api/v1 (api/main.py:61-68). Setup mirrors
tests/integration/api/test_optimal_design_endpoints.py.
"""

import io
import json

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


@pytest.fixture
def session_id():
    response = client.post("/api/v1/sessions", json={"ttl_hours": 1})
    response.raise_for_status()
    sid = response.json()["session_id"]
    yield sid
    client.delete(f"/api/v1/sessions/{sid}")


def _add_variables(sid, names=("x1", "x2")):
    for name in names:
        r = client.post(
            f"/api/v1/sessions/{sid}/variables",
            json={"name": name, "type": "real", "min": 0.0, "max": 10.0},
        )
        r.raise_for_status()


class TestConstraintCRUD:
    def test_add_constraint(self, session_id):
        _add_variables(session_id)
        r = client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": {"x1": 1.0, "x2": 1.0},
            "rhs": 10.0,
            "name": "half_plane_1",
        })
        assert r.status_code == 200
        assert r.json()["constraint"]["name"] == "half_plane_1"

    def test_list_constraints(self, session_id):
        _add_variables(session_id)
        client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": {"x1": 1.0, "x2": 1.0},
            "rhs": 10.0, "name": "c_a",
        })
        r = client.get(f"/api/v1/sessions/{session_id}/constraints")
        assert r.status_code == 200
        body = r.json()
        assert body["n_constraints"] == 1
        assert body["constraints"][0]["name"] == "c_a"

    def test_delete_constraint_by_name(self, session_id):
        _add_variables(session_id)
        client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "inequality", "coefficients": {"x1": 1.0},
            "rhs": 5.0, "name": "c_a",
        })
        client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "inequality", "coefficients": {"x2": 1.0},
            "rhs": 5.0, "name": "c_b",
        })
        r = client.delete(f"/api/v1/sessions/{session_id}/constraints/c_a")
        assert r.status_code == 200
        remaining = client.get(f"/api/v1/sessions/{session_id}/constraints").json()
        assert [c["name"] for c in remaining["constraints"]] == ["c_b"]

    def test_delete_unknown_constraint_is_404(self, session_id):
        _add_variables(session_id)
        r = client.delete(f"/api/v1/sessions/{session_id}/constraints/nope")
        assert r.status_code == 404

    def test_constraint_on_unknown_variable_is_400(self, session_id):
        _add_variables(session_id)
        r = client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": {"nope": 1.0}, "rhs": 5.0,
        })
        assert r.status_code == 400

    def test_constraint_on_categorical_is_400(self, session_id):
        _add_variables(session_id)
        client.post(f"/api/v1/sessions/{session_id}/variables",
                    json={"name": "cat", "type": "categorical",
                          "categories": ["a", "b"]})
        r = client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": {"cat": 1.0}, "rhs": 5.0,
        })
        assert r.status_code == 400
        assert "not numeric" in r.json()["detail"]

    def test_registered_constraint_is_honored_by_a_design(self, session_id):
        _add_variables(session_id)
        client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": {"x1": 1.0, "x2": 1.0}, "rhs": 8.0,
        })
        r = client.post(f"/api/v1/sessions/{session_id}/initial-design",
                        json={"method": "lhs", "n_points": 6, "random_seed": 7})
        assert r.status_code == 200
        for pt in r.json()["points"]:
            assert pt["x1"] + pt["x2"] <= 8.0 + 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/integration/api/test_constraints_router.py -q
```

Expected: FAIL — 404 on every constraint route.

- [ ] **Step 3: Add the request model**

Append to `api/models/requests.py`:

```python
class AddConstraintRequest(BaseModel):
    """Request to register a linear input constraint on the search space.

    'inequality' means sum(coeff_i * x_i) <= rhs.
    'equality'   means sum(coeff_i * x_i) == rhs.

    Coefficient variables must be numeric (real, integer, or discrete).
    """
    constraint_type: Literal["inequality", "equality"] = Field(
        ..., description="'inequality' (<= rhs) or 'equality' (== rhs)"
    )
    coefficients: Dict[str, float] = Field(
        ..., min_length=1,
        description="Mapping of variable name to coefficient"
    )
    rhs: float = Field(..., description="Right-hand side value")
    name: Optional[str] = Field(
        None, description="Optional name; auto-generated as constraint_N if omitted"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "constraint_type": "inequality",
                "coefficients": {"x1": 0.5, "x2": -1.0},
                "rhs": -10.0,
                "name": "half_plane_1",
            }
        }
    )
```

If `Dict` or `Literal` is not already imported at the top of `requests.py`, add them to the `typing` import.

- [ ] **Step 4: Add the response models**

Append to `api/models/responses.py`:

```python
class ConstraintResponse(BaseModel):
    """Response after registering a linear input constraint."""
    message: str = Field(..., description="Result message")
    constraint: Dict[str, Any] = Field(..., description="The registered constraint")


class ConstraintsListResponse(BaseModel):
    """Response listing all registered linear input constraints."""
    constraints: List[Dict[str, Any]] = Field(..., description="Registered constraints")
    n_constraints: int = Field(..., description="Number of constraints")
```

- [ ] **Step 5: Add the routes**

In `api/routers/variables.py`, extend the imports:

```python
from ..models.requests import (
    AddRealVariableRequest,
    AddIntegerVariableRequest,
    AddCategoricalVariableRequest,
    AddDiscreteVariableRequest,
    AddConstraintRequest,
)
from ..models.responses import (
    VariableResponse,
    VariablesListResponse,
    ConstraintResponse,
    ConstraintsListResponse,
)
```

Then append these routes to the end of the file:

```python
@router.post("/{session_id}/constraints", response_model=ConstraintResponse)
async def add_constraint(
    session_id: str,
    constraint: AddConstraintRequest,
    session: OptimizationSession = Depends(get_session)
):
    """
    Register a linear input constraint on the search space.

    Both the DoE and the acquisition function honor registered constraints
    natively, so a suggestion is never generated inside the excluded region.

    - **inequality**: `sum(coeff_i * x_i) <= rhs`
    - **equality**: `sum(coeff_i * x_i) == rhs`

    Coefficient variables must be numeric (real, integer, or discrete).
    """
    try:
        session.add_input_constraint(
            constraint.constraint_type,
            constraint.coefficients,
            constraint.rhs,
            constraint.name,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    registered = session.search_space.get_constraints()[-1]
    logger.info(f"Added constraint '{registered['name']}' to session {session_id}")
    return ConstraintResponse(
        message="Constraint added successfully",
        constraint=registered,
    )


@router.get("/{session_id}/constraints", response_model=ConstraintsListResponse)
async def list_constraints(
    session_id: str,
    session: OptimizationSession = Depends(get_session)
):
    """List all linear input constraints registered on the search space."""
    constraints = session.search_space.get_constraints()
    return ConstraintsListResponse(
        constraints=constraints,
        n_constraints=len(constraints),
    )


@router.delete("/{session_id}/constraints/{constraint_name}")
async def delete_constraint(
    session_id: str,
    constraint_name: str,
    session: OptimizationSession = Depends(get_session)
):
    """
    Remove a linear input constraint by name.

    Deletion is by name rather than index: auto-generated names are positional
    (`constraint_0`, `constraint_1`, ...) and would shift when an earlier
    constraint is removed.
    """
    existing = session.search_space.constraints
    match = [c for c in existing if c["name"] == constraint_name]
    if not match:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Constraint '{constraint_name}' not found. "
                f"Registered: {[c['name'] for c in existing]}"
            ),
        )

    session.search_space.constraints = [
        c for c in existing if c["name"] != constraint_name
    ]
    logger.info(f"Deleted constraint '{constraint_name}' from session {session_id}")
    return {"message": f"Constraint '{constraint_name}' deleted successfully"}
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/integration/api/test_constraints_router.py -q
```

Expected: PASS — 7 tests.

- [ ] **Step 7: Commit**

```bash
git add api/models/requests.py api/models/responses.py api/routers/variables.py tests/integration/api/test_constraints_router.py
git commit -m "feat(api): constraint CRUD endpoints

Constraints existed only in alchemist_core: settable from Python, invisible
over REST. A non-Python consumer could not use the constraint feature at all,
so correct constrained DoE was unreachable by the consumers that need it.

Adds POST/GET /sessions/{id}/constraints and DELETE by name. Deletion is by
name because auto-generated names are positional and shift on removal.
Non-numeric coefficient variables are rejected at the boundary with a 400."
```

---

### Task 12: `/variables/load` accepts the dict format

**Files:**
- Modify: `api/routers/variables.py:97-145`
- Modify: `tests/integration/api/test_constraints_router.py`

**Interfaces:**
- Consumes: `SearchSpace.from_dict`.
- Produces: `/variables/load` accepts either a bare list (unchanged) or `{"variables": [...], "constraints": [...]}`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/integration/api/test_constraints_router.py`:

```python
class TestVariablesLoadDictFormat:
    def _upload(self, sid, payload):
        buf = io.BytesIO(json.dumps(payload).encode())
        return client.post(
            f"/api/v1/sessions/{sid}/variables/load",
            files={"file": ("space.json", buf, "application/json")},
        )

    def test_bare_list_format_still_works(self, session_id):
        r = self._upload(session_id, [
            {"name": "x1", "type": "real", "min": 0.0, "max": 10.0},
            {"name": "x2", "type": "real", "min": 0.0, "max": 10.0},
        ])
        assert r.status_code == 200
        listed = client.get(f"/api/v1/sessions/{session_id}/variables").json()
        assert listed["n_variables"] == 2

    def test_dict_format_registers_constraints(self, session_id):
        r = self._upload(session_id, {
            "variables": [
                {"name": "x1", "type": "real", "min": 0.0, "max": 10.0},
                {"name": "x2", "type": "real", "min": 0.0, "max": 10.0},
            ],
            "constraints": [
                {"type": "inequality", "coefficients": {"x1": 1.0, "x2": 1.0},
                 "rhs": 8.0, "name": "c_a"},
            ],
        })
        assert r.status_code == 200
        listed = client.get(f"/api/v1/sessions/{session_id}/constraints").json()
        assert listed["n_constraints"] == 1
        assert listed["constraints"][0]["name"] == "c_a"

    def test_load_export_load_round_trips_constraints(self, session_id):
        self._upload(session_id, {
            "variables": [
                {"name": "x1", "type": "real", "min": 0.0, "max": 10.0},
                {"name": "x2", "type": "real", "min": 0.0, "max": 10.0},
            ],
            "constraints": [
                {"type": "inequality", "coefficients": {"x1": 1.0, "x2": 1.0},
                 "rhs": 8.0, "name": "c_a"},
            ],
        })
        exported = client.get(
            f"/api/v1/sessions/{session_id}/variables/export"
        ).json()

        second = client.post("/api/v1/sessions", json={"ttl_hours": 1}).json()
        sid2 = second["session_id"]
        try:
            r = self._upload(sid2, exported)
            assert r.status_code == 200
            listed = client.get(f"/api/v1/sessions/{sid2}/constraints").json()
            assert listed["n_constraints"] == 1
        finally:
            client.delete(f"/api/v1/sessions/{sid2}")
```

> **Implementer note:** confirm the exact shape `/variables/export` returns
> before relying on the round-trip test. If it emits a bare list rather than
> the `to_dict` dict (which would silently drop constraints on export), make
> the test assert that gap explicitly and report it — do not weaken the test
> to make it pass.

- [ ] **Step 2: Run tests to verify they fail**

```bash
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/integration/api/test_constraints_router.py::TestVariablesLoadDictFormat -q
```

Expected: FAIL — the dict format is not parsed; constraints are not registered.

- [ ] **Step 3: Branch on the payload shape**

In `api/routers/variables.py`, inside `load_variables_from_file`, replace:

```python
        # Add each variable
        for var in variables_data:
            var_type = var.pop("type")
            name = var.pop("name")
            
            # Handle categories for categorical variables
            if "categories" in var:
                var["values"] = var.pop("categories")
            
            session.add_variable(name, var_type, **var)
        
        logger.info(f"Loaded {len(variables_data)} variables from file for session {session_id}")
        
        return {
            "message": f"Loaded {len(variables_data)} variables successfully",
            "n_variables": len(variables_data)
        }
```

with:

```python
        # Two accepted shapes:
        #   - a bare list of variable dicts (legacy)
        #   - {"variables": [...], "constraints": [...]} as produced by
        #     SearchSpace.to_dict, which carries constraints through
        if isinstance(variables_data, dict):
            try:
                session.search_space.from_dict(variables_data)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            n_vars = len(session.search_space.variables)
            n_constraints = len(session.search_space.constraints)
            logger.info(
                f"Loaded {n_vars} variables and {n_constraints} constraints "
                f"from file for session {session_id}"
            )
            return {
                "message": (
                    f"Loaded {n_vars} variables and {n_constraints} "
                    f"constraints successfully"
                ),
                "n_variables": n_vars,
                "n_constraints": n_constraints,
            }

        # Legacy bare-list path — behavior unchanged.
        for var in variables_data:
            var_type = var.pop("type")
            name = var.pop("name")

            # Handle categories for categorical variables
            if "categories" in var:
                var["values"] = var.pop("categories")

            session.add_variable(name, var_type, **var)

        logger.info(f"Loaded {len(variables_data)} variables from file for session {session_id}")

        return {
            "message": f"Loaded {len(variables_data)} variables successfully",
            "n_variables": len(variables_data),
            "n_constraints": 0,
        }
```

- [ ] **Step 4: Verify `SearchSpace.from_dict` handles the dict shape**

```bash
~/miniforge3/envs/alchemist-env/bin/python -c "
from alchemist_core.data.search_space import SearchSpace
s = SearchSpace()
s.from_dict({
  'variables': [
    {'name':'x1','type':'real','min':0.0,'max':10.0},
    {'name':'x2','type':'real','min':0.0,'max':10.0},
  ],
  'constraints': [
    {'type':'inequality','coefficients':{'x1':1.0,'x2':1.0},'rhs':8.0,'name':'c_a'},
  ],
})
print('variables:', [v['name'] for v in s.variables])
print('constraints:', s.constraints)
"
```

Expected: two variables and one constraint. If `from_dict` does not accept this shape, stop and report — the spec assumed `search_space.py:339-344` supports it, and that assumption needs correcting before proceeding.

- [ ] **Step 5: Run tests to verify they pass**

```bash
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/integration/api/test_constraints_router.py -q
```

Expected: PASS — 10 tests.

- [ ] **Step 6: Commit**

```bash
git add api/routers/variables.py tests/integration/api/test_constraints_router.py
git commit -m "feat(api): /variables/load accepts the {variables, constraints} format

SearchSpace.from_dict has always supported the dict shape, but the endpoint
only ever iterated a bare list, so constraints in an uploaded search space
were silently discarded. The bare-list path is unchanged."
```

---

### Task 13: Feasibility reporting on design responses

**Files:**
- Modify: `api/models/responses.py:233`, `:284`
- Modify: `api/routers/experiments.py:241-305`, `:337-375`
- Modify: `tests/integration/api/test_constraints_router.py`

**Interfaces:**
- Consumes: `run_optimal_design`'s `info["feasibility"]` from Task 7.
- Produces: `feasibility: Optional[Dict[str, Any]]` on `InitialDesignResponse` and `OptimalDesignResponse`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/integration/api/test_constraints_router.py`:

```python
class TestFeasibilityReporting:
    def test_unconstrained_optimal_design_reports_null_feasibility(self, session_id):
        _add_variables(session_id, names=("x1", "x2", "x3"))
        r = client.post(f"/api/v1/sessions/{session_id}/optimal-design", json={
            "n_points": 10, "model_type": "quadratic",
            "criterion": "D", "algorithm": "fedorov", "random_seed": 7,
        })
        assert r.status_code == 200
        assert r.json()["feasibility"] is None

    def test_constrained_optimal_design_reports_candidate_provenance(self, session_id):
        _add_variables(session_id, names=("x1", "x2", "x3"))
        client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": {"x1": 1.0, "x2": 1.0}, "rhs": 12.0,
        })
        r = client.post(f"/api/v1/sessions/{session_id}/optimal-design", json={
            "n_points": 10, "model_type": "quadratic",
            "criterion": "D", "algorithm": "fedorov", "random_seed": 7,
        })
        assert r.status_code == 200
        feas = r.json()["feasibility"]
        assert feas["constraints_applied"] == ["constraint_0"]
        assert feas["n_candidates_feasible"] < feas["n_candidates_total"]
        assert feas["vertex_enumeration_skipped"] is False

    def test_unconstrained_initial_design_reports_null_feasibility(self, session_id):
        _add_variables(session_id, names=("x1", "x2", "x3"))
        r = client.post(f"/api/v1/sessions/{session_id}/initial-design",
                        json={"method": "lhs", "n_points": 6, "random_seed": 7})
        assert r.status_code == 200
        assert r.json()["feasibility"] is None

    def test_constrained_initial_design_reports_constraints(self, session_id):
        _add_variables(session_id, names=("x1", "x2", "x3"))
        client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": {"x1": 1.0, "x2": 1.0}, "rhs": 12.0,
        })
        r = client.post(f"/api/v1/sessions/{session_id}/initial-design",
                        json={"method": "lhs", "n_points": 6, "random_seed": 7})
        assert r.status_code == 200
        assert r.json()["feasibility"]["constraints_applied"] == ["constraint_0"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/integration/api/test_constraints_router.py::TestFeasibilityReporting -q
```

Expected: FAIL — `KeyError: 'feasibility'`.

- [ ] **Step 3: Add the field to both response models**

In `api/models/responses.py`, add to `InitialDesignResponse` (after `design_info`):

```python
    feasibility: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Constraint provenance: which constraints applied, candidate counts "
            "before and after filtering, boundary and vertex points added, "
            "points dropped, and whether vertex enumeration was skipped. "
            "None when no input constraints are registered."
        )
    )
```

And the identical field to `OptimalDesignResponse` (after `design_info`).

- [ ] **Step 4: Populate it in the initial-design endpoint**

In `api/routers/experiments.py`, in `generate_initial_design`, replace the return with:

```python
    constraints = session.search_space.get_constraints()
    feasibility = None
    if constraints:
        feasibility = {
            "constraints_applied": [c["name"] for c in constraints],
            "n_candidates_total": None,
            "n_candidates_feasible": None,
            "n_boundary_added": None,
            "n_vertices_added": None,
            "vertex_enumeration_skipped": None,
            "n_points_dropped": None,
            "estimability": (
                "not_applicable" if request.method != "optimal" else None
            ),
        }

    return InitialDesignResponse(
        points=design_points,
        method=request.method,
        n_points=len(design_points),
        design_info=design_info,
        feasibility=feasibility,
    )
```

> **Implementer note:** read the existing `return InitialDesignResponse(...)` first and preserve every field it already passes. The block above shows the fields present as of this plan; if the model has gained others, keep them.

- [ ] **Step 5: Populate it in the optimal-design endpoint**

In `generate_optimal_design`, `session.generate_optimal_design(...)` returns `(points, info)`. Extract the feasibility block from `info` and pass it through:

```python
    feasibility = info.pop("feasibility", None)
    if feasibility is not None:
        feasibility = dict(feasibility)
        feasibility["n_points_dropped"] = None
        feasibility["estimability"] = "not_applicable"

    return OptimalDesignResponse(
        points=points,
        n_points=len(points),
        design_info=info,
        feasibility=feasibility,
    )
```

> **Implementer note:** match the existing return's field names and variable names exactly — read the current body first. `info.pop` is deliberate: `feasibility` is surfaced as its own field rather than buried inside `design_info`.

- [ ] **Step 6: Run tests to verify they pass**

```bash
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/integration/api/test_constraints_router.py -q
```

Expected: PASS — 14 tests.

- [ ] **Step 7: Commit**

```bash
git add api/models/responses.py api/routers/experiments.py tests/integration/api/test_constraints_router.py
git commit -m "feat(api): report constraint feasibility on design responses

Adds an optional feasibility block to the initial-design and optimal-design
responses: which constraints applied, candidate counts before and after
filtering, boundary and vertex points added, and whether vertex enumeration
hit its cap. That last one matters — a candidate set reduced by the cap must
be reported to the caller, not left in a log.

Null when no constraints are registered, so unconstrained responses are
unchanged."
```

---

### Task 14: Error mapping and final verification

**Files:**
- Modify: `api/middleware/error_handlers.py`
- Modify: `api/routers/experiments.py`
- Modify: `CHANGELOG.md`, `docs/ISSUES_LOG.md`
- Modify: `tests/integration/api/test_constraints_router.py`

**Interfaces:**
- Consumes: `doe.DesignNotEstimableError`, `constrained_region.InfeasibleRegionError`.
- Produces: both map to HTTP 400 with `error_type` set.

- [ ] **Step 1: Write the failing tests**

Append to `tests/integration/api/test_constraints_router.py`:

```python
class TestConstraintErrorMapping:
    def test_non_estimable_classical_design_is_400(self, session_id):
        _add_variables(session_id, names=("x1", "x2", "x3"))
        client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": {"x1": 1.0, "x2": 1.0}, "rhs": 11.0,
        })
        r = client.post(f"/api/v1/sessions/{session_id}/initial-design",
                        json={"method": "ccd", "random_seed": 7})
        assert r.status_code == 400
        assert r.json()["error_type"] == "DesignNotEstimableError"
        assert "optimal" in r.json()["detail"]

    def test_infeasible_region_optimal_design_is_400(self, session_id):
        _add_variables(session_id, names=("x1", "x2", "x3"))
        client.post(f"/api/v1/sessions/{session_id}/constraints", json={
            "constraint_type": "inequality",
            "coefficients": {"x1": 1.0, "x2": 1.0}, "rhs": -1.0,
        })
        r = client.post(f"/api/v1/sessions/{session_id}/optimal-design", json={
            "n_points": 10, "model_type": "linear",
            "criterion": "D", "algorithm": "fedorov", "random_seed": 7,
        })
        assert r.status_code == 400
        assert r.json()["error_type"] == "InfeasibleRegionError"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/integration/api/test_constraints_router.py::TestConstraintErrorMapping -q
```

Expected: FAIL — a 500, not a 400.

- [ ] **Step 3: Register the handlers**

In `api/middleware/error_handlers.py`, add inside `add_exception_handlers`, following the `NoDataError` pattern:

```python
    from alchemist_core.utils.doe import DesignNotEstimableError
    from alchemist_core.utils.constrained_region import InfeasibleRegionError

    @app.exception_handler(DesignNotEstimableError)
    async def design_not_estimable_handler(request: Request, exc: DesignNotEstimableError):
        """Handle a constrained classical design that lost structural points."""
        logger.warning(f"Design not estimable: {exc}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "detail": str(exc),
                "error_type": "DesignNotEstimableError",
                "status_code": status.HTTP_400_BAD_REQUEST
            }
        )

    @app.exception_handler(InfeasibleRegionError)
    async def infeasible_region_handler(request: Request, exc: InfeasibleRegionError):
        """Handle an empty feasible region."""
        logger.warning(f"Infeasible region: {exc}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "detail": str(exc),
                "error_type": "InfeasibleRegionError",
                "status_code": status.HTTP_400_BAD_REQUEST
            }
        )
```

> Both are `ValueError` subclasses. If a broader `ValueError` handler is already registered, these must be registered **after** it — FastAPI dispatches on the most specific registered class, but registration order decides ties. Verify with the tests.

- [ ] **Step 4: Run tests to verify they pass**

```bash
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/integration/api/test_constraints_router.py -q
```

Expected: PASS — 16 tests.

- [ ] **Step 5: Document the breaking changes**

Add to the `### Bug Fixes` section of `CHANGELOG.md`'s `[Unreleased]` block:

```markdown
- **Constrained DoE was honored in only two of four places.** Linear input
  constraints were respected by acquisition and by space-filling DoE, but
  classical designs generated their points ignoring constraints and dropped
  the infeasible ones with a log warning, and optimal designs had no
  constraint awareness at all — the exchange algorithm spent its budget
  selecting points it would not be allowed to keep. Optimal designs now select
  from a candidate set that is filtered *and* augmented with points on the
  feasible region's boundary (a filtered lattice has none, and an optimal
  design wants precisely those extremes), so a constrained D/A/I-optimal
  design is now genuinely optimal over its region. Equality constraints work
  for the first time.
- **Non-model variables could be spread straight through a constraint.**
  Variables absent from every model term were overwritten with a shuffled
  range *after* selection, undoing all feasibility work. They are now drawn
  from each row's feasible interval.
```

Add to `### New Features`:

```markdown
- **Linear input constraints are now settable over REST.** `POST`/`GET`
  `/sessions/{id}/constraints` and `DELETE .../{name}`, plus
  `/variables/load` accepting the `{variables, constraints}` format that
  `SearchSpace.to_dict` already emits. Design responses carry a `feasibility`
  block reporting candidate provenance. Previously constraints could only be
  set from Python, so no REST consumer could use the feature at all.
```

Add a `### Breaking Changes` section:

```markdown
### Breaking Changes

- A constrained **classical** design (`ccd`, `box_behnken`, `full_factorial`,
  `fractional_factorial`, `plackett_burman`, `gsd`) that loses structural
  points now raises `DesignNotEstimableError` when the surviving points can no
  longer estimate the design's implied model. It previously returned the
  degraded remnant with a log warning. Pass `allow_infeasible=True` for the old
  behavior. Designs that lose only harmless points (a replicated center point)
  still return normally.
- A constrained **optimal** design returns **different points for the same
  seed**, selected from the augmented candidate set. This is the fix.
- `SearchSpace.add_constraint` now **rejects non-numeric variables**
  (categorical, context) at registration. Such constraints were already
  non-functional — they failed later inside `filter_feasible`.
- **Unconstrained behavior is unchanged**, verified by golden tests over every
  DoE method at a fixed seed.
```

Add a row to `docs/ISSUES_LOG.md` immediately after the last existing row:

```markdown
| **Linear input constraints ignored by classical and optimal DoE; not settable over REST** | **2026-08-25** | **2026-08-25** | **✅ RESOLVED**: Constraints were honored in acquisition and space-filling DoE only. Classical designs post-hoc-dropped infeasible structural points with a log warning, leaving a design rank-deficient for the model it claimed to fit; optimal designs had no constraint awareness, so the exchange algorithm optimized over candidates it could not keep; and non-model variables were spread across their full range after selection, writing through constraints. Separately, constraints had no REST or web surface, so no non-Python consumer could set one. Optimal designs now select from a filtered-and-boundary-augmented candidate set (new `constrained_region` module), classical designs are gated on model estimability, spreading draws from each row's feasible interval, and constraint CRUD is exposed over REST. |
```

- [ ] **Step 6: Run the full suite**

```bash
~/miniforge3/envs/alchemist-env/bin/python -m pytest tests/ -q
```

Expected: ≥ 1000 original + ~65 new tests passing, 10 skipped, 0 failed. **The golden lock from Task 1 must pass.**

- [ ] **Step 7: Verify the frontend is unaffected**

No frontend files were changed, but the response models did gain a field. Confirm nothing broke:

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/Active learning code development/ALchemist/alchemist-web"
npx tsc --noEmit && npm test && npm run build
```

Expected: clean typecheck, tests pass, build succeeds. The new `feasibility` field is optional and additive, so existing TypeScript types remain valid.

- [ ] **Step 8: Commit**

```bash
cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/Active learning code development/ALchemist"
git add api/middleware/error_handlers.py CHANGELOG.md docs/ISSUES_LOG.md tests/integration/api/test_constraints_router.py
git commit -m "feat(api): map constraint errors to 400; document breaking changes

DesignNotEstimableError and InfeasibleRegionError now surface as 400 with an
actionable message instead of a 500.

Documents the three intentional breaking changes in CHANGELOG: constrained
classical designs raise instead of degrading silently, constrained optimal
designs return different points for the same seed, and add_constraint rejects
non-numeric variables. Unconstrained behavior is unchanged and golden-tested."
```

---

## Self-Review Results

**Spec coverage — every section maps to a task:**

| Spec section | Task |
|---|---|
| §5.1 `constrained_region` module | 2, 3, 4, 5 |
| §5.2 `augment_with_boundary` algorithm | 5 |
| §5.3 optimal-design integration + encode/decode | 6, 7 |
| §5.4 classical estimability gate | 9 |
| §5.5 feasible-interval spreading | 8 |
| §5.6 categorical guard | 10 |
| §7.1 constraint CRUD | 11 |
| §7.2 `/variables/load` dict format | 12 |
| §7.3 feasibility reporting | 13 |
| §7.4 error mapping | 14 |
| §9.1 golden back-compat tests | 1 |
| §9.2–§9.6 test coverage | folded into each task |
| §10 CHANGELOG breaking changes | 14 |

**Type consistency:** `augment_with_boundary` returns the six-key info dict defined in Task 5 and consumed unchanged in Tasks 7 and 13. `decode_candidates(candidates_coded, column_map, variables, selected_indices=None)` keeps that argument order at every call site; `_decode_candidates` retains the original positional order as an alias. `feasible_interval` returns `Optional[Tuple[float, float]]` and every caller handles `None`.

**API surface verified against the running app**, not assumed. Routers mount
under `/api/v1` (`api/main.py:61-68`), so every constraint and design path in
Tasks 11-14 is `/api/v1/sessions/...`. Design endpoints live on the experiments
router but are routed as `/{session_id}/initial-design` and
`/{session_id}/optimal-design` — there is no `/experiments/` path segment.
Integration tests use a module-level `TestClient(app)` and a `session_id`
fixture, matching `tests/integration/api/test_optimal_design_endpoints.py`;
there is no `client` fixture in `tests/conftest.py`.

**One naming decision:** `snap_to_variable` is public (no leading underscore)
because Task 8 calls it from `optimal_design.py`. Duplicating the snapping
logic would be worse than exporting it.

---

## Execution Handoff

Plan complete and saved to `.superpowers/plans/2026-08-25-constrained-doe.md`.

Two execution options:

1. **Subagent-Driven (recommended)** — a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Phase 1 (Tasks 1–10) is independently valuable and independently verifiable; stop at the Phase 1 checkpoint for review before Phase 2.
