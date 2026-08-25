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
