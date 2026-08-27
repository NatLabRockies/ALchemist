from typing import List, Dict, Any, Union, Optional, Tuple
from skopt.space import Real, Integer, Categorical
import numpy as np
import pandas as pd
import json
import re
import sys

# Auto-generated constraint names. Kept as a module constant so the generator
# and the matcher below can never drift apart.
_AUTO_CONSTRAINT_NAME = "constraint_{}"
_AUTO_CONSTRAINT_RE = re.compile(r"^constraint_(\d+)$")

# What counts as a number wherever this module demands a finite one -- a
# constraint rhs or coefficient, and a variable bound. bool is deliberately
# included (it is a subclass of int); complex, str, None and containers are
# not, and neither are Decimal or Fraction -- this is a tuple of concrete
# types, not a numbers-ABC test.
#
# np.bool_ is listed explicitly because it is *not* a subclass of either bool
# or np.integer, so the "accept numpy scalars" intent had a hole: np.True_ was
# rejected while both True and np.int64(1) were accepted, and the diagnostic
# said "of type bool" -- naming the very type the line above says is accepted.
# Accepting it is what makes the rule statable in one sentence: a finite value
# of any concrete int, float or bool type Python or numpy defines, at any
# magnitude.
_FINITE_NUMBER_TYPES = (int, float, bool, np.bool_, np.integer, np.floating)

# The members of _FINITE_NUMBER_TYPES that have non-finite values at all.
# The types left out -- int, bool, np.bool_, and every np.integer other than
# np.timedelta64 -- are finite by construction, so there is nothing to test
# them for, and testing them anyway is what broke: np.isfinite(2**64) is a
# TypeError rather than True, because a Python int outside the uint64 range
# cannot be coerced to any numpy dtype. That put a TypeError back inside a
# guard whose entire job is to convert one into a ValueError -- a 500 on
# POST /variables (no try/except, and only ValueError has a global handler)
# for an ordinary bound, and on the load branches a 400 whose text was a raw
# numpy ufunc string naming nothing.
#
# Restricting the finiteness test to these keeps np.isfinite away from every
# input that could make it raise, without narrowing what is accepted.
# math.isfinite is not the alternative: it takes 2**64 but converts through
# float() to do it, so it raises OverflowError from 2**1024 up -- the same
# defect one door further along, and equally a 500 on POST /variables.
#
# np.timedelta64 is listed because np.integer is *not* wholly finite by
# construction, contrary to what this comment claimed for one revision:
# np.timedelta64 subclasses np.signedinteger, so it is inside the accepted
# tower, and np.timedelta64('NaT') is a non-finite value of it. np.isfinite
# answers False for NaT cleanly, so one tuple entry is the whole fix; without
# it NaT registered as an ordinary integer bound, and on a real bound reached
# skopt and came back as UFuncTypeError -- a TypeError subclass leaving a
# guard whose documented sole exit is a labelled ValueError.
#
# Two invariants hold this together, both pinned in
# tests/unit/core/data/test_constraints.py: every entry here is a type the
# tower admits (by issubclass, not by tuple membership -- np.timedelta64 is
# admitted through np.integer without being an entry above), and no concrete
# type the tower *admits an instance of* can produce a non-finite instance
# without being tested for finiteness. The
# second is asserted over constructed instances rather than over tuple
# entries, because an entry-level assertion is what let np.timedelta64 through
# -- issubclass(np.integer, (int, np.integer, np.bool_)) is True while
# np.timedelta64('NaT') walks past. Adding, say, Decimal or
# np.complexfloating above without revisiting here fails it.
_MAY_BE_NON_FINITE = (float, np.floating, np.timedelta64)

# The largest finite float64. A bound of larger magnitude has no place in a
# dimension backed by float64 -- see _validate_float64_backed.
_FLOAT64_MAX = sys.float_info.max

# Which variable types build a dimension whose bounds live in float64, and
# which build one that does not. This is a property of the skopt class
# add_variable will construct, not of the value:
#
#   * skopt.Real is float64 all the way down. Real.__init__ calls
#     set_transformer(transform="identity"), and for the default uniform
#     prior that evaluates _uniform_inclusive(self.low, self.high - self.low),
#     whose body is np.nextafter(scale, scale + 1.0) -- space.py:403 -> :444
#     -> :307. So what skopt converts through float() is the *span*, not each
#     bound: `scale + 1.0` raised OverflowError three frames inside skopt for
#     a Python int wider than float64 -- an ArithmeticError, outside the
#     loaders' (ValueError, KeyError, TypeError) tuple and outside the global
#     ValueError handler, so a 500 on every path.
#
#     This comment claimed for two rounds that set_transformer builds
#     Normalize(self.low, self.high) and converts each bound. Normalize is
#     real, but it is built only under transform="normalize", which
#     add_variable never passes. The difference is not cosmetic: a per-bound
#     mental model produces a per-bound guard, and that is exactly what
#     shipped -- two bounds each individually inside the float64 range whose
#     span is not (see _validate_float64_span) walked straight past it.
#   * discrete builds a Categorical of float()-coerced values, so its entries
#     are float64-backed for the same reason and by the same conversion.
#   * skopt.Integer never converts: it keeps Python ints at full width, so an
#     integer bound has no upper magnitude at all and 2**2000 is legitimate.
#   * categorical and context take no numeric bound.
#
# The two sets must together cover every variable type that validates a bound,
# which tests/unit/core/data/test_constraints.py pins: a new bound-taking type
# added to add_variable without an entry here fails that test rather than
# silently inheriting whichever default happened to be written below.
_FLOAT64_BACKED_VAR_TYPES = frozenset({"real", "discrete"})
_ARBITRARY_PRECISION_VAR_TYPES = frozenset({"integer"})

# Members of the accepted tower that carry a unit rather than a magnitude, so
# they cannot be compared against a float at all: np.timedelta64 lands inside
# np.integer, and `np.timedelta64(5, 's') <= 1.79e308` raises UFuncTypeError
# instead of answering. Nothing float64-backed can hold one, so it is refused
# on that ground rather than on magnitude -- and refused *before* the
# comparison, so the comparison itself is total over everything that reaches
# it.
_NOT_REAL_VALUED = (np.timedelta64,)


def _type_name(value: Any) -> str:
    """Type name qualified by module for anything outside ``builtins``.

    ``type(np.True_).__name__`` is the bare string ``'bool'``, and several
    numpy scalar types shadow a builtin name this way. An unqualified name in a
    rejection message therefore reads as a claim about the builtin, which is
    how ``np.True_`` came to be refused as "of type bool" while ``bool`` was
    documented as accepted.
    """
    cls = type(value)
    if cls.__module__ in ("builtins", None):
        return cls.__name__
    return f"{cls.__module__}.{cls.__name__}"


def _validate_finite_number(value: Any, label: str) -> None:
    """Raise ValueError unless ``value`` is a finite number.

    The type check has to come first. ``np.isfinite(None)`` raises
    ``TypeError: ufunc 'isfinite' not supported for the input types``, not
    ValueError -- so a non-numeric value escaped the documented contract of
    :meth:`SearchSpace.add_constraint`, escaped
    :meth:`OptimizationSession.add_input_constraint` which documents the same,
    and escaped ``api/routers/variables.py``, which catches only ValueError and
    would have returned a 500.

    It was unreachable through ``POST /constraints`` (``rhs: float`` coerces
    first) until ``/variables/load`` began registering constraints straight out
    of an uploaded JSON file, where ``"rhs": null`` is ordinary.

    Numeric strings are rejected rather than coerced: accepting ``"3.0"`` would
    admit a whole file of quoted numbers and store types the DoE does not
    expect downstream.

    :meth:`SearchSpace.add_variable` applies the same check to variable bounds,
    for the reason recorded in :meth:`add_constraint`: skopt's ``low >= high``
    test is ``False`` for ``NaN``, so a non-finite bound registered cleanly and
    then made *every* export of that session a 400 -- ``json.dumps`` refuses
    ``nan``/``inf`` under ``allow_nan=False``, which is what FastAPI's
    ``JSONResponse`` uses. Neither export shape could get the space back out,
    so the session was unrecoverable through the API. ``json.load`` accepts the
    bare ``NaN``/``Infinity`` literals by default, so such a file is an
    ordinary upload rather than a hostile one.

    That reason is finiteness, not JSON-representability, and the two coincide
    only over the types REST can deliver. A parsed JSON document yields Python
    scalars, and the only Python scalar in the accepted tower that
    ``json.dumps(allow_nan=False)`` refuses is a non-finite float -- an ``int``
    is emitted at any width, ``2**70`` included. The numpy half of the tower
    does not coincide: ``np.float64`` serializes, but ``np.bool_`` and
    ``np.integer`` raise "Object of type int64 is not JSON serializable". They
    are accepted anyway. They reach here only from a Python caller already
    holding numpy scalars -- never from a request body or an uploaded file --
    so they cannot cause the failure above, and narrowing the tower to exclude
    them would refuse existing callers for a reason this guard does not have.
    """
    if not isinstance(value, _FINITE_NUMBER_TYPES):
        raise ValueError(
            f"{label} must be a finite number, got {_magnitude_repr(value)} "
            f"of type {_type_name(value)}"
        )
    # Only the types that have non-finite values are asked about their
    # finiteness. See _MAY_BE_NON_FINITE: np.isfinite raises TypeError, not
    # False, for a Python int outside the uint64 range, and this guard exists
    # precisely so that no caller of it has to catch a TypeError.
    if isinstance(value, _MAY_BE_NON_FINITE) and not np.isfinite(value):
        raise ValueError(f"{label} must be finite, got {value}")


# The longest repr any rejection message from this module will carry. Past it
# the value is described rather than printed.
_MAX_REPR_CHARS = 120


def _magnitude_repr(value: Any) -> str:
    """``repr`` for a rejection message, bounded in length whatever it is given.

    Two separate reasons, and the second is why this is not only an int helper.

    CPython 3.11+ caps int-to-string conversion at
    ``sys.get_int_max_str_digits()`` (4300 by default) and raises ValueError
    past it -- from inside the very message meant to explain the rejection. A
    bound that long is being refused for its magnitude anyway, so beyond 100
    digits it is described rather than printed. That ceiling is reachable only
    from a Python caller holding the int already: ``json.loads`` applies the
    same cap while parsing, so a file carrying a 4400-digit literal raises its
    own ValueError in the parser and never reaches this module at all.

    Nothing else here can raise, but plenty can be long. ``real max='1' * 5000``
    is refused by :func:`_validate_finite_number` for its *type*, and printing
    its repr put five thousand characters into a 400 body -- no exception, just
    bulk. Anything past ``_MAX_REPR_CHARS`` is therefore truncated with its full
    length stated, which is the diagnostic part of a value that long.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        bits = value.bit_length()
        if bits > 332:  # ~100 decimal digits
            # Not f"{sign}an integer ...": that read "-an integer of 2001 bits".
            if value < 0:
                return f"a negative integer of {bits} bits"
            return f"an integer of {bits} bits"
    text = repr(value)
    if len(text) > _MAX_REPR_CHARS:
        return f"{text[:_MAX_REPR_CHARS]}... ({len(text)} characters)"
    return text


def _validate_float64_backed(value: Any, label: str) -> None:
    """Raise ValueError unless a float64-backed dimension can hold ``value``.

    This is the structural half of the bound guard, and the reason it exists is
    that three rounds of naming the exception instead of the question each
    removed one exception type and revealed the next -- ZeroDivisionError, then
    TypeError, then OverflowError, every one a 500 where the endpoint documents
    400. The question is not "which exception does skopt raise for this value"
    but "can the dimension that is about to be constructed represent it": a
    ``real`` bound of ``2**1024`` is not a float64 value, so ``skopt.Real``
    cannot hold it, and it is refused here with the variable and key on it
    rather than discovered three frames down inside ``Real.set_transformer``.

    Answered by comparison, never by conversion. ``float(2**1024)`` raises
    OverflowError and ``np.isfinite(2**64)`` raises TypeError, but
    ``2**1024 <= sys.float_info.max`` is exact and total for a Python int of
    any width -- CPython compares int against float without converting either.
    The same comparison is safe for every other type the tower admits, once
    ``np.timedelta64`` is taken out of its way (see ``_NOT_REAL_VALUED``), so
    no input can make this function raise anything but the labelled ValueError
    it is documented to raise.

    The line is drawn at ``sys.float_info.max`` -- the float64 range -- not at
    ``2**1024 - 2**970``, which is where ``float()`` itself starts overflowing.
    The band between them is about ``2**970`` wide and contains only integers
    that have no float64 value of their own and merely round down to
    ``sys.float_info.max``. Refusing them is deliberate: "the bound must be a
    float64" is one statable rule, while "the bound must be something float()
    happens to round" is a rule about CPython's rounding mode.

    ``integer`` bounds do not come here at all. ``skopt.Integer`` keeps Python
    ints at full width and never converts, so ``2**64``, ``2**1024`` and
    ``2**2000`` remain legitimate integer bounds. Constraint ``rhs`` values and
    coefficients do not come here either -- they build no dimension, they are
    checked by ``_validate_finite_number`` directly, and they stay accepted at
    every magnitude.
    """
    if isinstance(value, _NOT_REAL_VALUED):
        raise ValueError(
            f"{label} must be a real number for a float-backed dimension, "
            f"got {value!r} of type {_type_name(value)}"
        )
    if not -_FLOAT64_MAX <= value <= _FLOAT64_MAX:
        raise ValueError(
            f"{label} must be within the float64 range this variable's "
            f"dimension can hold (magnitude at most {_FLOAT64_MAX!r}), "
            f"got {_magnitude_repr(value)}"
        )


def _as_python_scalar(value: Any) -> Any:
    """The Python scalar holding exactly the value of a numpy one.

    Only :func:`_validate_float64_span` needs this, and it needs it to stay
    total. Every numpy member of the accepted tower has an exact Python
    counterpart -- ``np.integer`` and ``np.bool_`` are ints, ``np.floating`` is
    a float -- but numpy's own ``-`` between one of them and a Python int does
    not merely lose the answer, it refuses to give one::

        np.int64(0) - 2**1023   OverflowError: int too large to convert to C long
        np.True_  - False       TypeError: numpy boolean subtract ... not supported
        np.uint8(0) - (-1)      OverflowError: -1 out of bounds for uint8

    Every operand above is a value ``_validate_bound`` already accepted, so a
    span check written on the raw operands raises from inside a guard whose
    only documented exit is a labelled ValueError -- the same shape of
    unexamined assumption that produced the three rounds before this one.
    Taken through here, the same three subtractions are ordinary arithmetic.

    Python scalars pass through untouched, so ``int - int`` stays exact and
    ``float - float`` stays IEEE: for the only spellings the load path can
    deliver, the expression below is character-for-character the one skopt
    evaluates.

    The numpy tests come first and that ordering is load-bearing.
    ``np.float64`` *is* a subclass of ``float`` (``np.int64`` and ``np.bool_``
    are not subclasses of ``int`` and ``bool``, which is what makes the trap
    easy to miss), so a Python-scalar test written first returns an
    ``np.float64`` untouched and the subtraction is numpy's after all --
    ``np.float64(1.7e308) - np.float64(-1.7e308)`` emits "overflow encountered
    in scalar subtract", which is an exception for any caller who promotes
    warnings. Caught by the exhaustive sweep in
    ``tests/unit/core/data/test_constraints.py``, which is what that sweep is
    for.
    """
    if isinstance(value, (np.bool_, np.integer)):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _validate_float64_span(low: Any, high: Any, var_name: str) -> None:
    """Raise ValueError unless ``high - low`` is a float64 value too.

    ``_validate_float64_backed`` asks its question of one bound at a time.
    ``skopt.Real`` does not: ``set_transformer`` computes
    ``_uniform_inclusive(self.low, self.high - self.low)``, so the quantity it
    has to be able to represent is the **span**. Two bounds each individually
    inside the float64 range can have a span that is not, and both bounds and
    the span have to be checked because neither implies the other.

    The gap had two faces and this closes both:

    * ``real min=-(2**1023) max=2**1023``. The span is ``2**1024``, and
      ``scale + 1.0`` inside ``np.nextafter`` raised ``OverflowError`` -- not a
      fourth exception *type* but the very one round 3 closed, arriving through
      a door a per-bound question could not reach, and still a 500 on an
      endpoint that documents 400.
    * ``real min=-1.7e308 max=1.7e308``. Nothing raises at all: the float
      subtraction saturates, and the dimension is built with ``scale=inf``, so
      ``Real(-1.7e308, 1.7e308).rvs(3)`` returns three identical points at the
      upper bound. A 200 and a degenerate design, which is the worse of the two
      because nothing reports it.

    **Why this cannot itself raise.** It runs only after ``_validate_bound`` has
    accepted both operands, which leaves exactly three things true of each, and
    all three are load-bearing. It is a member of the accepted tower; it is not
    ``np.timedelta64`` (``_NOT_REAL_VALUED`` refuses that before any comparison,
    so nothing here carries a unit); and its magnitude is at most
    ``_FLOAT64_MAX``. Given those, ``_as_python_scalar`` maps it to an ``int``
    or a ``float`` without conversion loss and without raising -- an ``int`` at
    most ``int(_FLOAT64_MAX)`` wide, or a float already inside the range. The
    subtraction of two such scalars is then total: ``int - int`` is exact and
    unbounded, ``float - float`` saturates to ``inf`` rather than raising, and
    the mixed case converts the int through ``PyLong_AsDouble``, which cannot
    overflow because ``int(_FLOAT64_MAX) + 1 <= _FLOAT64_MAX`` is already False
    and so no accepted int rounds past ``_FLOAT64_MAX``. The comparison that
    follows takes an int of any width against a float without converting
    either, and ``inf <= _FLOAT64_MAX`` is an ordinary ``False``. Verified by
    exhausting the tower's cross product under ``warnings.simplefilter('error')``
    rather than argued for: numpy's overflow *warnings* would otherwise be the
    fifth exception out of this guard for a caller who turns them into errors.

    Written as ``not span <= _FLOAT64_MAX`` rather than ``span > _FLOAT64_MAX``
    for the reason the finiteness test is ordered ahead of the magnitude test:
    it is the negation of the accept condition, so a value that compares False
    both ways is refused rather than admitted.

    **Where the line is drawn.** At ``_FLOAT64_MAX``, which is a shade tighter
    than where skopt actually breaks -- ``min=-1, max=int(_FLOAT64_MAX)`` has an
    exact span of ``int(_FLOAT64_MAX) + 1``, which ``float()`` rounds back down
    and skopt then handles. That band is under ``2**970`` wide and is refused
    deliberately, for the reason ``_validate_float64_backed`` already refuses
    the same band per bound: "the span must be a float64" is one statable rule,
    while "the span must be something float() happens to round" is a rule about
    CPython's rounding mode.

    ``integer`` does not come here. ``Integer.set_transformer`` computes no
    span, so ``min=-(2**2000), max=2**2000`` is a legitimate integer dimension
    and stays one. ``discrete`` does not either: its entries become a
    ``Categorical`` of floats, and a Categorical subtracts nothing.
    """
    span = _as_python_scalar(high) - _as_python_scalar(low)
    if not span <= _FLOAT64_MAX:
        raise ValueError(
            f"Variable '{var_name}' min and max are each within the float64 "
            f"range but span more than it (at most {_FLOAT64_MAX!r}), so the "
            f"dimension built from them cannot be represented: span "
            f"{_magnitude_repr(span)} from {_magnitude_repr(low)} to "
            f"{_magnitude_repr(high)}"
        )


def _validate_bound(value: Any, var_name: str, key: str, var_type: str) -> None:
    """``_validate_finite_number`` for a variable bound, labelled by variable.

    ``var_type`` is required rather than optional so that a bound cannot be
    validated without saying which dimension it is being validated *for*. That
    is the whole point of the check: what a bound may be is a property of the
    skopt class about to be constructed from it, and a call site that does not
    state the type cannot be given the right answer. See
    ``_FLOAT64_BACKED_VAR_TYPES``.

    The label names both the variable and the key, and every rejection made
    here carries it, because a labelled ValueError is the only way out of this
    guard. Three things make that true and all three are load-bearing: the type
    tuple is checked first; the finiteness test after it runs only on the types
    that cannot make ``np.isfinite`` raise; and the representability test after
    *that* compares rather than converts, so it cannot raise either. For one
    round the finiteness test ran on everything, and ``np.isfinite(2**64)``
    then left by a path with no label on it at all -- a bare TypeError naming
    neither the variable nor the key. For the next round the representability
    test did not exist, and ``Real(0, 2**1024)`` left by an OverflowError with
    no label on it either.

    Order matters between the last two. A non-finite value is rejected as
    non-finite before anything asks whether it is in range, so ``NaN`` is still
    reported as "must be finite" rather than as an out-of-range magnitude --
    ``nan <= x`` is False, which would otherwise produce a true statement for
    the wrong reason.

    What it does *not* cover is the pair. Every check here is about one value,
    and ``skopt.Real``'s question is about ``high - low``; a guard that
    validates each bound is not a guard that validates the dimension. That half
    lives in :func:`_validate_float64_span`, called beside this one from the
    ``real`` branch, and it exists because this docstring's totality claim was
    read as covering more than it does.

    The label matters most to ``POST /variables/load``: it hands a whole
    uploaded file to the core and can only report what the exception says.

    ``allowed_values`` entries are validated on both sides of their ``float()``
    coercion, so a quoted number survives there while a quoted bound is
    refused. That asymmetry is pre-existing and deliberately left alone here
    (branch item M5); narrowing it is a change to what files load, not to this
    guard.
    """
    label = f"Variable '{var_name}' {key}"
    _validate_finite_number(value, label)
    if var_type in _FLOAT64_BACKED_VAR_TYPES:
        _validate_float64_backed(value, label)


class SearchSpace:
    """
    Class for storing and managing the search space in a consistent way across backends.
    Provides methods for conversions to different formats required by different backends.
    """
    def __init__(self):
        self.variables = []  # List of variable dictionaries with metadata
        self.skopt_dimensions = []  # skopt dimensions (used by scikit-learn)
        self.categorical_variables = []  # List of categorical variable names
        self.discrete_variables = []  # List of discrete variable names
        self.constraints = []  # List of linear constraint dicts
        self.derived_variables = []  # List of derived (non-tunable) variable dicts
        # Each derived entry: {"name": str, "input_cols": List[str],
        #                      "description": str, "func": callable | None}

    def add_variable(self, name: str, var_type: str, **kwargs):
        """
        Add a variable to the search space.

        Args:
            name: Variable name
            var_type: "real", "integer", "categorical", "discrete", or "context"
            **kwargs: Additional parameters:
                - real/integer: min, max
                - categorical: values (list of strings)
                - discrete: allowed_values (list of numbers, at least 2, no duplicates)
                - context: no additional parameters required
        """
        var_type_lower = var_type.lower()

        # Guard: reject duplicate names across all variable types
        if name in [v["name"] for v in self.variables]:
            raise ValueError(
                f"Variable '{name}' is already registered."
            )

        var_dict = {"name": name, "type": var_type_lower}
        var_dict.update(kwargs)

        # Build the dimension before registering anything. The variable used to
        # be appended first, so every failure below -- a missing bound, min >
        # max, a categorical with no values, a one-element discrete, an unknown
        # type -- left a half-registered variable in self.variables with no
        # entry in self.skopt_dimensions. The two lists are positionally paired
        # for every dimension-bearing type, so the desync is silent until
        # something zips them. See get_dimension_names/get_dimension_index
        # below for the pairing rule itself, and for why the pairing is *not*
        # index-for-index between the two lists whenever a context variable is
        # registered -- a case this paragraph does not cover.
        #
        # Reachable over REST through the *bare-list* branch of
        # POST /variables/load, which appends straight into the session with no
        # dry run: a file of [x1: 0..10, x2: 9..1] returned 400 and left the
        # session holding variables=['x1','x2'] against dims=['x1'], after
        # which DELETE /variables/x1 removed the wrong dimension, the export
        # emitted a file that would not load, and POST /initial-design died on
        # an AssertionError. The dict branch is *not* the reachable path -- its
        # dry run on a throwaway SearchSpace absorbs the failure before the
        # session is touched, and a mutation restoring append-first leaves
        # every dict-path session-intact test passing.
        #
        # Every bound is validated against the variable type, because what a
        # bound may be depends on the dimension class built from it: Real is
        # float64 all the way down, Integer is arbitrary precision. Passing
        # var_type_lower rather than a literal keeps the branch and the rule it
        # is validated under from drifting apart.
        dimension = None
        if var_type_lower == "real":
            _validate_bound(kwargs["min"], name, "min", var_type_lower)
            _validate_bound(kwargs["max"], name, "max", var_type_lower)
            # And then the pair, because Real's question is about the span and
            # not about either bound alone. See _validate_float64_span.
            _validate_float64_span(kwargs["min"], kwargs["max"], name)
            dimension = Real(kwargs["min"], kwargs["max"], name=name)
        elif var_type_lower == "integer":
            _validate_bound(kwargs["min"], name, "min", var_type_lower)
            _validate_bound(kwargs["max"], name, "max", var_type_lower)
            dimension = Integer(kwargs["min"], kwargs["max"], name=name)
        elif var_type_lower == "categorical":
            values = kwargs["values"]
            # skopt divides by len(categories) to build the prior, so an empty
            # list raised ZeroDivisionError -- outside the (ValueError,
            # KeyError, TypeError) tuple the API loader catches, and therefore
            # a 500 on both load branches instead of the 400 the endpoint
            # documents. Refused here rather than in the router so the desktop
            # loader (ui/ui.py -> from_dict) is covered by the same rule.
            if values is not None and len(values) == 0:
                raise ValueError(
                    f"Categorical variable '{name}' requires 'values' with at "
                    f"least 1 value, got an empty list."
                )
            dimension = Categorical(values, name=name)
        elif var_type_lower == "discrete":
            allowed = kwargs.get("allowed_values")
            if allowed is None or len(allowed) < 2:
                raise ValueError(
                    f"Discrete variable '{name}' requires 'allowed_values' with at least 2 values."
                )
            if len(allowed) != len(set(allowed)):
                raise ValueError(
                    f"Discrete variable '{name}' has duplicate values in 'allowed_values'."
                )
            # The guard goes in front of the float() coercion as well as
            # behind it. float() is itself one of the conversions this guard
            # exists to stand in front of: it raises OverflowError for an int
            # outside the float64 range, which is neither a ValueError nor in
            # the loaders' catch tuple, so allowed_values=[1, 2**1024] in an
            # uploaded file was a 500 for exactly the reason a real bound of
            # 2**1024 was. The entries end up in a Categorical of floats, so
            # they are float64-backed and answer to the same rule.
            #
            # Only entries already of an accepted numeric type are checked
            # before the coercion, and the string "3.0" is not the whole of
            # what that leaves out. The pre-coercion guard covers members of
            # the accepted tower and nothing else, so *any* other object with
            # a __float__ reaches float() unguarded -- and one whose value is
            # outside the float64 range raises OverflowError there.
            # allowed_values=[0.5, Fraction(2**1024, 1)] is the smallest
            # example, and Decimal spells the same thing.
            #
            # Left alone deliberately, not overlooked. Neither registration
            # endpoint can deliver such an object: json.loads yields only
            # Python scalars, and POST /variables coerces through a Pydantic
            # float first. Closing it means either narrowing what discrete
            # accepts -- a change to which files load, which is branch item M5
            # and not a change to this guard -- or catching OverflowError by
            # name, which is the answer three rounds of this defect
            # established as the wrong one.
            #
            # The quoted-number asymmetry against a bound (a quoted bound is
            # refused, a quoted entry is coerced) is pre-existing and is the
            # same branch item. Whatever float() returns is validated
            # afterwards exactly as before -- that is what still catches
            # "inf", "nan", and a coerced non-number.
            #
            # Both checks run before sorting, not after: sorted() puts NaN
            # wherever the comparisons happen to land it, so an unchecked NaN
            # would also scramble the order of the values around it.
            coerced = []
            for i, value in enumerate(allowed):
                bound_key = f"allowed_values[{i}]"
                if isinstance(value, _FINITE_NUMBER_TYPES):
                    _validate_bound(value, name, bound_key, var_type_lower)
                as_float = float(value)
                _validate_bound(as_float, name, bound_key, var_type_lower)
                coerced.append(as_float)
            sorted_vals = sorted(coerced)
            var_dict["allowed_values"] = sorted_vals
            dimension = Categorical(sorted_vals, name=name)
        elif var_type_lower == "context":
            pass  # No skopt dimension; no bounds; just lives in self.variables
        else:
            raise ValueError(f"Unknown variable type: {var_type}")

        self.variables.append(var_dict)
        if dimension is not None:
            self.skopt_dimensions.append(dimension)
        if var_type_lower == "categorical":
            self.categorical_variables.append(name)
        elif var_type_lower == "discrete":
            self.discrete_variables.append(name)

    # ==================================================================
    # The pairing between self.variables and self.skopt_dimensions
    # ==================================================================
    #
    # These are two lists of *different lengths* that callers address
    # positionally against each other. add_variable's comment above frames the
    # risk as half-registration -- a variable appended without its dimension --
    # and that is one way the pairing breaks. It is not the common way.
    #
    # A ``context`` variable occupies a slot in self.variables and contributes
    # no skopt dimension at all, so on any correctly registered space that has
    # one, the two lists differ in length by construction:
    #
    #     variables        : [('c1','context'), ('x1','real'), ('x2','categorical')]
    #     skopt_dimensions : [Real(x1),         Categorical(x2)]
    #     index of x1 in variables = 1  |  index of x1 in skopt_dimensions = 0
    #
    # No care taken inside add_variable can prevent that; it is what ``context``
    # means. So an index obtained by enumerating self.variables is not an index
    # into self.skopt_dimensions and never was -- it is only accidentally equal
    # to one on a space with no context variable in front of the variable in
    # question, which is why the mistake survives casual testing.
    #
    # The two methods below are the supported way to cross between the lists.
    # Deriving a dimension position by enumerating self.variables is a defect.

    # The variable types that contribute an entry to self.skopt_dimensions.
    # This is the positive form of the rule -- the set of types add_variable
    # builds a ``dimension`` for -- rather than the negative "not context".
    # The two agree today. They are not the same statement: "tunable" is about
    # whether the optimizer varies a variable, "dimension-bearing" is about
    # whether it occupies a slot in skopt_dimensions, and a future type could
    # answer those differently. The positive form is the one that stays true.
    _DIMENSION_BEARING_TYPES = frozenset({"real", "integer", "categorical", "discrete"})

    @classmethod
    def _has_dimension(cls, var: Dict[str, Any]) -> bool:
        """Whether ``var`` contributes an entry to ``self.skopt_dimensions``."""
        return var.get("type") in cls._DIMENSION_BEARING_TYPES

    def get_dimension_names(self) -> List[str]:
        """Variable names positionally paired with ``self.skopt_dimensions``.

        ``get_dimension_names()[i]`` is the name of ``skopt_dimensions[i]``, so
        this is what any caller zipping a sampled point against variable names
        wants -- a sample drawn from ``skopt_dimensions`` has one value per
        dimension, not one per variable.

        Derived from ``self.variables`` rather than by reading ``dim.name`` off
        each dimension, deliberately: this states which variables *should* hold
        a dimension and in what order, so it remains the correct answer to
        compare a corrupted ``skopt_dimensions`` against rather than agreeing
        with it.
        """
        return [v["name"] for v in self.variables if self._has_dimension(v)]

    def get_dimension_index(self, name: str) -> Optional[int]:
        """Index into ``self.skopt_dimensions`` for variable ``name``.

        ``None`` when ``name`` carries no dimension -- either because it is not
        registered at all, or because it is registered as a type that has none
        (``context``). Both answers are "there is no dimension slot for this
        name", which is the only thing a caller indexing ``skopt_dimensions``
        can act on; callers that need to tell a missing variable from a context
        one look at ``self.variables``.
        """
        index = 0
        for var in self.variables:
            if var["name"] == name:
                return index if self._has_dimension(var) else None
            if self._has_dimension(var):
                index += 1
        return None

    def _variable_index(self, name: str) -> Optional[int]:
        """Index into ``self.variables`` for ``name``, or None if not registered."""
        for i, var in enumerate(self.variables):
            if var["name"] == name:
                return i
        return None

    def _sync_type_membership(self, name: str, var_type: Optional[str]) -> None:
        """Make the categorical/discrete name lists agree with ``var_type``.

        Both directions: a variable moving *into* a type joins that list, one
        moving *out of* it leaves. ``var_type=None`` means "no longer any type"
        and removes ``name`` from both, which is what a removal wants.

        Position preserving: a variable that keeps its type is left where it
        already sits rather than removed and re-appended. ``categorical_variables``
        is used as a column selection for the one-hot encoder, so its order is
        not arbitrary even though nothing indexes it.
        """
        for names, owning_type in (
            (self.categorical_variables, "categorical"),
            (self.discrete_variables, "discrete"),
        ):
            if var_type == owning_type:
                if name not in names:
                    names.append(name)
            elif name in names:
                names.remove(name)

    def replace_variable(self, name: str, var_type: str, **kwargs):
        """Redefine an already-registered variable in place, keeping its position.

        Same signature as ``add_variable`` -- ``name``, ``var_type``, and the
        type's own kwargs -- so a caller that has translated a payload once can
        hand it to either without a second translation table to drift.

        The new definition is built by calling ``add_variable`` on a throwaway
        space. That is the whole point of this method rather than an
        incidental way to write it: there is no second construction path, so
        every guard ``add_variable`` enforces -- the finite-bound check, the
        float64 span check, the empty-categorical and short-discrete checks,
        and any guard added after this was written -- applies to a redefinition
        for free. The prior implementations of this operation were hand-rolled
        copies of ``add_variable``'s branches with the guards omitted, which is
        exactly the rot this shape exists to prevent.

        The throwaway starts empty, so the one guard that must *not* fire here
        -- the duplicate-name check -- does not, without needing a flag to
        suppress it.

        It also makes the operation atomic. Everything that can fail happens on
        the throwaway before ``self`` is touched, so a rejected redefinition
        leaves the variable exactly as it was rather than half-overwritten.

        Ordering is preserved in both lists, which is why this is not
        ``remove_variable`` followed by ``add_variable``: that pair moves the
        variable to the end of both, and ``skopt_dimensions``' positional
        pairing makes the reordering observable to every consumer.

        Type transitions are handled in both directions, including to and from
        ``context``: a variable that gains a dimension has one inserted at the
        position its name maps to, and one that loses its dimension has it
        removed.

        Raises:
            ValueError: if ``name`` is not registered, or if the new definition
                fails any of ``add_variable``'s guards.
        """
        var_index = self._variable_index(name)
        if var_index is None:
            raise ValueError(f"Variable '{name}' is not registered.")

        staged = SearchSpace()
        staged.add_variable(name, var_type, **kwargs)
        new_var = staged.variables[0]
        new_dimension = staged.skopt_dimensions[0] if staged.skopt_dimensions else None

        # Two different things answer "does this type carry a dimension":
        # ``staged`` answers it by having built one or not, and
        # _DIMENSION_BEARING_TYPES answers it for get_dimension_index, which
        # decides *where* the dimension goes. They must agree, and if they ever
        # stop agreeing the failure is silent in the worst direction:
        # get_dimension_index returns None while new_dimension is not, and
        # ``list.insert(None, dim)`` raises TypeError -- not a ValueError, so it
        # escapes the app's global handler as an unlabelled 500 rather than a
        # 400 naming the problem. That is the shape of failure this branch spent
        # four rounds on in Task 12.
        #
        # Checked here, against the staged build and before ``self`` has been
        # touched, so the atomicity guarantee in the docstring still holds: a
        # drift is refused with the space unchanged rather than partway through
        # a replace.
        if (new_dimension is not None) != self._has_dimension(new_var):
            raise ValueError(
                f"Cannot replace variable '{name}': add_variable built "
                f"{'a dimension' if new_dimension is not None else 'no dimension'} "
                f"for type '{new_var['type']}', but _DIMENSION_BEARING_TYPES says "
                f"that type bears "
                f"{'one' if self._has_dimension(new_var) else 'none'}. "
                f"add_variable and _DIMENSION_BEARING_TYPES have drifted; the "
                f"type must be added to or removed from the set."
            )

        # Resolved by name against the *old* metadata, before anything moves.
        old_dim_index = self.get_dimension_index(name)
        if old_dim_index is not None:
            self.skopt_dimensions.pop(old_dim_index)

        self.variables[var_index] = new_var

        if new_dimension is not None:
            # Recomputed after the swap, because where the dimension belongs
            # depends on the *new* type and on how many dimension-bearing
            # variables precede it -- which is not var_index whenever a context
            # variable sits in front of it. Non-None here because the agreement
            # check above already refused the only case that could make it None.
            self.skopt_dimensions.insert(self.get_dimension_index(name), new_dimension)

        self._sync_type_membership(name, new_var["type"])

    def remove_variable(self, name: str):
        """Remove ``name`` and the dimension it owns, if it owns one.

        Every other variable keeps its position in both lists, and the pairing
        between them survives. Removing a ``context`` variable removes no
        dimension at all -- the case a single index popped from both lists gets
        wrong, by discarding some other variable's dimension.

        Constraints and derived variables that reference ``name`` are left
        alone; this method is about the two paired lists only.

        Raises:
            ValueError: if ``name`` is not registered.
        """
        var_index = self._variable_index(name)
        if var_index is None:
            raise ValueError(f"Variable '{name}' is not registered.")

        # Both positions are resolved before either list is mutated: popping
        # from self.variables first would change what get_dimension_index
        # computes for the very name being removed.
        dim_index = self.get_dimension_index(name)
        self.variables.pop(var_index)
        if dim_index is not None:
            self.skopt_dimensions.pop(dim_index)
        self._sync_type_membership(name, None)

    # Descriptive fields carried on a variable that no backend consumes: they
    # are echoed back to the user by the API and the desktop GUI and nothing
    # else. Forwarded verbatim rather than defaulted, so a file that omits them
    # produces exactly the variable dict it always did.
    _METADATA_KEYS = ("unit", "description")

    def _metadata_of(self, var: Dict[str, Any]) -> Dict[str, Any]:
        """Optional descriptive fields present on ``var``, as add_variable kwargs.

        ``from_dict`` used to pass only the fields each type needs to build its
        skopt dimension, so ``unit`` and ``description`` were dropped -- while
        the bare-list branch of ``POST /variables/load`` (which calls
        ``add_variable`` with the whole entry) and ``POST /variables`` both kept
        them. Latent until the dict shape was advertised as the round-trip
        format and as what ``save_to_json`` writes: from then on the documented
        path lost metadata that the legacy path beside it preserved, so
        export -> load -> export was not a fixed point.
        """
        return {k: var[k] for k in self._METADATA_KEYS if k in var}

    def from_dict(self, data: List[Dict[str, Any]]):
        """Load search space from a list of dictionaries (used with JSON/CSV loading)."""
        self.variables = []
        self.skopt_dimensions = []
        self.categorical_variables = []
        self.discrete_variables = []

        for var in data:
            var_type = var["type"].lower()
            metadata = self._metadata_of(var)
            if var_type in ["real", "integer"]:
                self.add_variable(
                    name=var["name"],
                    var_type=var_type,
                    min=var["min"],
                    max=var["max"],
                    **metadata,
                )
            elif var_type == "categorical":
                # Accept both 'values' (canonical) and 'categories' (alias used by
                # the web app's REST schema and variable-export endpoint) so a JSON
                # file exported from any frontend round-trips cleanly.
                values = var.get("values", var.get("categories"))
                if values is None:
                    raise ValueError(
                        f"Categorical variable '{var['name']}' is missing both "
                        f"'values' and 'categories'."
                    )
                self.add_variable(
                    name=var["name"],
                    var_type=var_type,
                    values=values,
                    **metadata,
                )
            elif var_type == "discrete":
                self.add_variable(
                    name=var["name"],
                    var_type=var_type,
                    allowed_values=var["allowed_values"],
                    **metadata,
                )
            elif var_type == "context":
                self.add_variable(
                    name=var["name"], var_type="context", **metadata
                )

        return self

    def from_skopt(self, dimensions):
        """Load search space from skopt dimensions."""
        self.variables = []
        self.skopt_dimensions = dimensions.copy()
        self.categorical_variables = []
        self.discrete_variables = []

        for dim in dimensions:
            name = dim.name
            if isinstance(dim, Real):
                self.variables.append({
                    "name": name,
                    "type": "real",
                    "min": dim.low,
                    "max": dim.high
                })
            elif isinstance(dim, Integer):
                self.variables.append({
                    "name": name,
                    "type": "integer",
                    "min": dim.low,
                    "max": dim.high
                })
            elif isinstance(dim, Categorical):
                cats = list(dim.categories)
                # Distinguish discrete (all-numeric categories) from true categorical
                try:
                    numeric_cats = [float(c) for c in cats]
                    # Heuristic: if all categories are numeric, treat as discrete
                    self.variables.append({
                        "name": name,
                        "type": "discrete",
                        "allowed_values": numeric_cats
                    })
                    self.discrete_variables.append(name)
                except (ValueError, TypeError):
                    self.variables.append({
                        "name": name,
                        "type": "categorical",
                        "values": cats
                    })
                    self.categorical_variables.append(name)

        return self

    def to_dict(self) -> List[Dict[str, Any]]:
        """Convert search space to a list of dictionaries."""
        return self.variables.copy()

    def to_skopt(self) -> List[Union[Real, Integer, Categorical]]:
        """Get skopt dimensions for scikit-learn."""
        return self.skopt_dimensions.copy()

    def to_ax_space(self) -> Dict[str, Dict[str, Any]]:
        """Convert to Ax parameter format."""
        ax_params = {}
        for var in self.variables:
            name = var["name"]
            if var["type"] == "real":
                ax_params[name] = {
                    "name": name,
                    "type": "range",
                    "bounds": [var["min"], var["max"]],
                }
            elif var["type"] == "integer":
                ax_params[name] = {
                    "name": name,
                    "type": "range",
                    "bounds": [var["min"], var["max"]],
                    "value_type": "int",
                }
            elif var["type"] == "categorical":
                ax_params[name] = {
                    "name": name,
                    "type": "choice",
                    "values": var["values"],
                }
            elif var["type"] == "discrete":
                # Ax represents discrete as a choice parameter with numeric values
                ax_params[name] = {
                    "name": name,
                    "type": "choice",
                    "values": var["allowed_values"],
                    "is_ordered": True,
                }
        return ax_params

    def to_botorch_bounds(self) -> Dict[str, np.ndarray]:
        """Create bounds in BoTorch format.

        For discrete variables, bounds span [min(allowed_values), max(allowed_values)].
        The acquisition optimizer uses these as the continuous relaxation bounds;
        the discrete constraint is enforced separately via optimize_acqf_mixed.
        """
        bounds = {}
        for var in self.variables:
            if var["type"] in ["real", "integer"]:
                bounds[var["name"]] = np.array([var["min"], var["max"]])
            elif var["type"] == "discrete":
                vals = var["allowed_values"]
                bounds[var["name"]] = np.array([min(vals), max(vals)])
        return bounds

    def get_variable_names(self) -> List[str]:
        """Get all variable names, tunable variables first, then context variables."""
        tunable = [v["name"] for v in self.variables if v.get("type") != "context"]
        context = [v["name"] for v in self.variables if v.get("type") == "context"]
        return tunable + context

    def get_tunable_variable_names(self) -> List[str]:
        """Get names of all non-context (tunable) variables in registration order."""
        return [v["name"] for v in self.variables if v.get("type") != "context"]

    def get_context_variable_names(self) -> List[str]:
        """Get names of all context (observed, non-optimized) variables in registration order."""
        return [v["name"] for v in self.variables if v.get("type") == "context"]

    def get_categorical_variables(self) -> List[str]:
        """Get list of categorical variable names."""
        return self.categorical_variables.copy()

    def get_integer_variables(self) -> List[str]:
        """Get list of integer variable names."""
        return [var["name"] for var in self.variables if var["type"] == "integer"]

    def get_discrete_variables(self) -> List[str]:
        """Get list of discrete variable names."""
        return self.discrete_variables.copy()

    def add_derived_variable(
        self,
        name: str,
        func,
        input_cols: List[str],
        description: str = "",
    ) -> None:
        """
        Register a derived (non-tunable) variable.

        Derived variables are deterministic functions of existing input variables.
        They are appended to the GP feature matrix at train and predict time, but
        the acquisition function never suggests values for them.

        Args:
            name: Column name for the derived feature.
            func: Callable with signature ``func(row: dict) -> float``.
                  Pass ``None`` when restoring a stub from a saved session.
            input_cols: Base variable names this feature depends on (for
                        documentation; the full row dict is still passed to func).
            description: Human-readable description stored in session JSON.

        Raises:
            ValueError: If name conflicts with an existing tunable variable or
                        an already-registered derived variable.
        """
        if name in [v["name"] for v in self.variables]:
            raise ValueError(f"'{name}' already exists as a tunable variable.")
        if name in [d["name"] for d in self.derived_variables]:
            raise ValueError(f"Derived variable '{name}' is already registered.")
        self.derived_variables.append({
            "name": name,
            "input_cols": list(input_cols),
            "description": description,
            "func": func,
        })

    def register_derived_variable(self, name: str, func) -> None:
        """
        Re-attach a callable to a derived variable stub after session load.

        Args:
            name: Name of the derived variable to update.
            func: Callable with signature ``func(row: dict) -> float``.

        Raises:
            ValueError: If no derived variable with the given name exists.
        """
        for dv in self.derived_variables:
            if dv["name"] == name:
                dv["func"] = func
                return
        raise ValueError(
            f"No derived variable named '{name}'. "
            f"Use add_derived_variable() to register a new one."
        )

    def add_derived_variable_stub(
        self, name: str, input_cols: List[str], description: str = ""
    ) -> None:
        """Restore a derived variable stub from session JSON (func=None)."""
        self.add_derived_variable(name=name, func=None, input_cols=input_cols, description=description)

    def has_derived_variables(self) -> bool:
        """Return True if any derived variables are registered."""
        return len(self.derived_variables) > 0

    def get_derived_variable_names(self) -> List[str]:
        """Return list of derived variable names (in registration order)."""
        return [dv["name"] for dv in self.derived_variables]

    def derived_variables_to_dict(self) -> List[Dict[str, Any]]:
        """Return serializable metadata for all derived variables (no func)."""
        return [
            {
                "name": dv["name"],
                "input_cols": dv["input_cols"],
                "description": dv["description"],
            }
            for dv in self.derived_variables
        ]

    def save_to_json(self, filepath: str):
        """Save search space to a JSON file."""
        data = {
            'variables': self.to_dict(),
            'constraints': self.constraints
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from_json(self, filepath: str):
        """Load search space from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        # Support both old format (list of variables) and new format (dict with constraints)
        if isinstance(data, list):
            return self.from_dict(data)
        else:
            self.from_dict(data.get('variables', []))
            self.constraints = data.get('constraints', [])
            return self
    
    @classmethod
    def from_json(cls, filepath: str):
        """Class method to create a SearchSpace from a JSON file."""
        instance = cls()
        return instance.load_from_json(filepath)

    def add_constraint(self, constraint_type: str, coefficients: Dict[str, float],
                       rhs: float, name: Optional[str] = None):
        """Add linear input constraint.

        Args:
            constraint_type: 'inequality' (sum(coeff_i * x_i) <= rhs) or
                             'equality' (sum(coeff_i * x_i) == rhs)
            coefficients: {variable_name: coefficient} mapping
            rhs: right-hand side value
            name: optional human-readable name. Auto-generated as
                  ``constraint_N`` when omitted. Names identify a constraint
                  for removal, so an explicit name that duplicates an existing
                  one raises ValueError.

        Raises:
            ValueError: unknown constraint_type, a coefficient variable that is
                missing or non-numeric, a non-numeric or non-finite rhs or
                coefficient, or a duplicate explicit name. Never TypeError:
                callers such as the API router catch only ValueError, so a
                ``None`` or string value arriving from a JSON file must fail
                through the documented channel.
        """
        valid_types = ('inequality', 'equality')
        if constraint_type not in valid_types:
            raise ValueError(f"constraint_type must be one of {valid_types}, got '{constraint_type}'")

        # Mirrors the finite check add_outcome_constraint has always had
        # (session.py). A NaN rhs makes every point infeasible, which sends
        # the DoE into a pathological resampling path, and a non-finite value
        # is not JSON-representable: it serializes to null, so the constraint
        # the API emits cannot be posted back.
        if not isinstance(coefficients, dict):
            raise ValueError(
                f"Constraint coefficients must be a mapping of variable name to "
                f"coefficient, got {type(coefficients).__name__}"
            )
        _validate_finite_number(rhs, "Constraint rhs")
        for var_name, coefficient in coefficients.items():
            _validate_finite_number(
                coefficient, f"Constraint coefficient for '{var_name}'"
            )

        var_names = self.get_variable_names()
        by_name = {v["name"]: v for v in self.variables}
        # Mirrors constrained_region.NUMERIC_TYPES -- keep the two in sync.
        # Not imported: alchemist_core/utils already depends on alchemist_core/data
        # (see utils/doe.py, utils/optimal_design.py), so importing constrained_region
        # here would add the reverse dependency. No import cycle exists between them.
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

        if name is None:
            name = self._next_auto_constraint_name()
        elif any(c.get('name') == name for c in self.constraints):
            raise ValueError(
                f"Constraint name '{name}' is already registered. A constraint "
                f"name is the identity used to remove it, so names must be "
                f"unique. Registered: {[c.get('name') for c in self.constraints]}"
            )

        self.constraints.append({
            'type': constraint_type,
            'coefficients': coefficients,
            'rhs': rhs,
            'name': name
        })

    def _next_auto_constraint_name(self) -> str:
        """Return an auto-generated constraint name not already in use.

        Stateless on purpose. ``save_to_json``/``load_from_json`` round-trip
        ``self.constraints`` as raw data, so a counter attribute would not
        survive a load and would resynchronize to an index already taken. The
        next index is therefore derived from the names present right now.

        The index is one past the highest ``constraint_N`` in use rather than
        the lowest free one, so an index is never recycled: a name that was
        deleted does not come back attached to a different constraint, and a
        stale client reference to it fails loudly with a 404 instead of
        silently resolving to something else.
        """
        highest = -1
        for c in self.constraints:
            match = _AUTO_CONSTRAINT_RE.match(str(c.get('name', '')))
            if match:
                highest = max(highest, int(match.group(1)))
        return _AUTO_CONSTRAINT_NAME.format(highest + 1)

    def get_constraints(self) -> List[Dict]:
        """Return list of constraint dicts."""
        return [c.copy() for c in self.constraints]

    def _constraint_scale(self, c: Dict) -> float:
        """Characteristic magnitude of a constraint, for relative tolerance."""
        scale = 0.0
        for var in self.variables:
            name = var['name']
            if name not in c['coefficients']:
                continue
            coeff = abs(float(c['coefficients'][name]))
            if 'min' in var and 'max' in var:
                rng = abs(float(var['max']) - float(var['min']))
            elif var.get('type') == 'discrete':
                vals = var.get('allowed_values', [0.0, 1.0])
                rng = abs(float(max(vals)) - float(min(vals)))
            else:
                rng = 1.0
            scale += coeff * rng
        return scale

    def filter_feasible(self, points, rtol: float = 1e-3, atol: float = 1e-6) -> np.ndarray:
        """Boolean mask: which rows satisfy ALL registered linear input constraints.

        Args:
            points: pandas DataFrame (columns are variable names) or a list of dicts.
            rtol, atol: relative/absolute tolerance. Equality is feasible when
                |lhs - rhs| <= atol + rtol * max(|rhs|, scale); inequality when
                lhs <= rhs + atol + rtol * max(|rhs|, scale).

        Returns:
            numpy boolean array of length len(points). All True if no constraints.
        """
        df = points if isinstance(points, pd.DataFrame) else pd.DataFrame(list(points))
        n = len(df)
        mask = np.ones(n, dtype=bool)
        if not self.constraints:
            return mask

        for c in self.constraints:
            lhs = np.zeros(n, dtype=float)
            any_col = False
            for var_name, coeff in c['coefficients'].items():
                if var_name in df.columns:
                    lhs = lhs + float(coeff) * df[var_name].to_numpy(dtype=float)
                    any_col = True
            if not any_col:
                continue  # constraint references no present columns; cannot judge -> skip
            rhs = float(c['rhs'])
            tol = atol + rtol * max(abs(rhs), self._constraint_scale(c))
            if c['type'] == 'equality':
                mask &= np.abs(lhs - rhs) <= tol
            else:  # inequality: lhs <= rhs
                mask &= lhs <= rhs + tol
        return mask

    def is_feasible(self, point, rtol: float = 1e-3, atol: float = 1e-6) -> bool:
        """Whether a single point (dict or 1-row DataFrame) is feasible."""
        if isinstance(point, dict):
            df = pd.DataFrame([point])
        elif isinstance(point, pd.DataFrame):
            df = point
        else:
            df = pd.DataFrame([dict(point)])
        return bool(self.filter_feasible(df, rtol=rtol, atol=atol)[0])

    def to_botorch_constraints(self, feature_names: List[str]) -> Tuple[Optional[List], Optional[List]]:
        """Convert to BoTorch format for optimize_acqf.

        Each constraint is a tuple (indices_tensor, coefficients_tensor, rhs_float).

        ALchemist convention (user-facing): inequality means
            sum(coeff_i * x_i) <= rhs

        BoTorch convention (``optimize_acqf``): inequality means
            sum(coeff_i * x_i) >= rhs

        To convert, we negate both the coefficients and the rhs for
        inequality constraints. Equality constraints are sign-symmetric and
        are passed through unchanged.

        Coordinate space: constraints are emitted in **raw variable space**,
        matching the bounds passed to ``optimize_acqf``. ``BoTorchModel`` applies
        its ``Normalize`` input transform *internally*, so the acquisition
        optimizer operates on raw-scale variables and returns raw-scale
        candidates. The stored constraints are already raw-scale, so the
        coefficients and rhs pass through unchanged (only the inequality sign is
        flipped for BoTorch's ``>=`` convention).

        Args:
            feature_names: ordered list of feature column names matching model input

        Returns:
            (inequality_constraints, equality_constraints) — each is a list of tuples
            or None if no constraints of that type exist.
        """
        import torch

        inequality_constraints = []
        equality_constraints = []

        name_to_idx = {name: i for i, name in enumerate(feature_names)}

        for c in self.constraints:
            indices = []
            coeffs = []

            for var_name, coeff in c['coefficients'].items():
                if var_name not in name_to_idx:
                    continue  # skip variables not in features (e.g. categorical)
                indices.append(name_to_idx[var_name])
                coeffs.append(float(coeff))

            if not indices:
                continue

            rhs = float(c['rhs'])

            if c['type'] == 'inequality':
                # Flip sign: ALchemist (coeff·x <= rhs) -> BoTorch (coeff·x >= rhs)
                inequality_constraints.append((
                    torch.tensor(indices, dtype=torch.long),
                    torch.tensor([-co for co in coeffs], dtype=torch.double),
                    -rhs
                ))
            else:
                equality_constraints.append((
                    torch.tensor(indices, dtype=torch.long),
                    torch.tensor(coeffs, dtype=torch.double),
                    rhs
                ))

        return (
            inequality_constraints if inequality_constraints else None,
            equality_constraints if equality_constraints else None
        )

    def __len__(self):
        return len(self.variables)
