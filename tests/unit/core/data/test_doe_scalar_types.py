"""Space-filling designs must return JSON-native Python scalars.

The space-filling samplers used to hand back numpy scalars where the classical
construction block hands back Python ones. Only ``np.int64`` threw -- it is the
one member of the set that does *not* subclass its Python counterpart -- so
``POST /initial-design`` returned 400 for any space containing an ``integer``
variable while three further leaks rode along silently:

    real         np.float64 under 'random' only
    integer      np.int64 on all five methods      <- the only visible one
    discrete     np.float64 on all five methods
    categorical  np.str_ / np.int64 on all five methods

The leak is asymmetric on *both* axes -- ``real`` leaks under one method out of
five -- so these tests assert on the full type x method grid rather than
sweeping one axis with the other held fixed.

Every assertion is ``type(v) is T``, never ``isinstance``: ``np.float64``
passes ``isinstance(v, float)`` and ``np.str_`` passes ``isinstance(v, str)``,
so an isinstance check would have called three of the four leaks clean.
"""

import json

import pytest

from alchemist_core import OptimizationSession
from alchemist_core.utils.doe import SPACE_FILLING_METHODS

METHODS = sorted(SPACE_FILLING_METHODS)

# name -> (registration kwargs, expected Python type, the values allowed back)
VAR_SPECS = {
    "real": (dict(var_type="real", min=0.0, max=10.0), float, None),
    "integer": (dict(var_type="integer", min=0, max=10), int, set(range(0, 11))),
    "discrete": (dict(var_type="discrete", allowed_values=[1.0, 2.0, 4.0]),
                 float, {1.0, 2.0, 4.0}),
    "categorical_str": (dict(var_type="categorical", values=["a", "b", "c"]),
                        str, {"a", "b", "c"}),
    # A categorical is not necessarily a string. This row is why the coercion
    # may not route through str(): str() would rewrite 1 to '1'.
    "categorical_int": (dict(var_type="categorical", values=[1, 2, 3]),
                        int, {1, 2, 3}),
}


def _single_var_session(spec_name):
    kwargs, _expected, _allowed = VAR_SPECS[spec_name]
    kwargs = dict(kwargs)
    s = OptimizationSession()
    s.add_variable("x1", kwargs.pop("var_type"), **kwargs)
    return s


def _all_types_session():
    """One space holding every variable type at once."""
    s = OptimizationSession()
    for i, name in enumerate(VAR_SPECS, start=1):
        kwargs = dict(VAR_SPECS[name][0])
        s.add_variable(f"x{i}", kwargs.pop("var_type"), **kwargs)
    return s


# --------------------------------------------------------------------------
# The grid: every variable type x every space-filling method.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("spec_name", sorted(VAR_SPECS))
def test_scalar_type_is_json_native(spec_name, method):
    """type(v) is the exact Python type -- on every cell of the grid."""
    _kwargs, expected, _allowed = VAR_SPECS[spec_name]
    s = _single_var_session(spec_name)
    points = s.generate_initial_design(n_points=8, method=method, random_seed=7)

    assert len(points) == 8
    for p in points:
        v = p["x1"]
        assert type(v) is expected, (
            f"{spec_name} under method={method} returned "
            f"{type(v).__name__}, expected {expected.__name__}"
        )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("spec_name", sorted(VAR_SPECS))
def test_scalar_value_is_preserved(spec_name, method):
    """Coercing the type must not alter the value.

    Guards the specific wrong fix of stringifying categoricals: '1' is not in
    the allowed set {1, 2, 3}, and 1.0 is not in {'a', 'b', 'c'}.
    """
    _kwargs, _expected, allowed = VAR_SPECS[spec_name]
    s = _single_var_session(spec_name)
    points = s.generate_initial_design(n_points=8, method=method, random_seed=7)

    for p in points:
        v = p["x1"]
        if allowed is None:  # real: bounded rather than enumerated
            assert 0.0 <= v <= 10.0
        else:
            assert v in allowed, f"{spec_name}/{method} produced {v!r}"
            # `in` alone is loose: True in {1} is True. Pin the type of the
            # matching member too.
            match = next(a for a in allowed if a == v)
            assert type(v) is type(match)


@pytest.mark.parametrize("method", METHODS)
def test_mixed_space_all_types_json_native(method):
    """All five types in one space, every method."""
    s = _all_types_session()
    points = s.generate_initial_design(n_points=8, method=method, random_seed=7)

    for p in points:
        for i, spec_name in enumerate(VAR_SPECS, start=1):
            expected = VAR_SPECS[spec_name][1]
            v = p[f"x{i}"]
            assert type(v) is expected, (
                f"x{i} ({spec_name}) under method={method} returned "
                f"{type(v).__name__}"
            )


@pytest.mark.parametrize("method", METHODS)
def test_design_is_json_serializable(method):
    """The end the bug actually surfaced at: the JSON encoder."""
    s = _all_types_session()
    points = s.generate_initial_design(n_points=8, method=method, random_seed=7)
    # json.dumps is what raised "Unable to serialize unknown type:
    # <class 'numpy.int64'>" and turned every integer design into a 400.
    encoded = json.dumps(points)
    assert json.loads(encoded) == points


@pytest.mark.parametrize("method", METHODS)
def test_integer_design_serializes_alone(method):
    """The fatal row, isolated -- an integer-only space must serialize."""
    s = _single_var_session("integer")
    points = s.generate_initial_design(n_points=8, method=method, random_seed=7)
    json.dumps(points)
    assert all(type(p["x1"]) is int for p in points)


# --------------------------------------------------------------------------
# Constraint + integer + DoE. This combination existed nowhere in the suite,
# because the endpoint 400'd on any integer variable.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("method", METHODS)
def test_constrained_design_with_integer_variable(method):
    """A registered constraint, an integer variable, and a DoE call together.

    Asserts all three at once: the design is feasible, it is the requested
    size, and the integer variable comes back as a JSON-native ``int`` on the
    constrained path -- which is a *different* code path from the
    unconstrained one (reject-and-resample, then slice).
    """
    s = OptimizationSession()
    s.add_variable("x1", "real", bounds=(0.0, 10.0))
    s.add_variable("x2", "integer", min=0, max=10)
    s.add_input_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=10.0)

    points = s.generate_initial_design(n_points=8, method=method, random_seed=7)

    assert len(points) == 8
    for p in points:
        assert type(p["x1"]) is float
        assert type(p["x2"]) is int, (
            f"constrained path leaked {type(p['x2']).__name__} for the "
            f"integer variable under method={method}"
        )
        assert p["x1"] + p["x2"] <= 10.0 + 1e-9
        assert 0 <= p["x2"] <= 10
    json.dumps(points)


def test_constrained_design_with_every_type_and_integer():
    """The constrained path with all five types present, not just integer."""
    s = _all_types_session()  # x1 real, x2 integer, x3 discrete, x4/x5 categorical
    s.add_input_constraint("inequality", {"x1": 1.0, "x2": 1.0}, rhs=10.0)

    points = s.generate_initial_design(n_points=8, method="lhs", random_seed=7)

    assert len(points) == 8
    for p in points:
        assert p["x1"] + p["x2"] <= 10.0 + 1e-9
        for i, spec_name in enumerate(VAR_SPECS, start=1):
            assert type(p[f"x{i}"]) is VAR_SPECS[spec_name][1]
    json.dumps(points)
