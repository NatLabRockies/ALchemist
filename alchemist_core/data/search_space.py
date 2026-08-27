from typing import List, Dict, Any, Union, Optional, Tuple
from skopt.space import Real, Integer, Categorical
import numpy as np
import pandas as pd
import json
import re

# Auto-generated constraint names. Kept as a module constant so the generator
# and the matcher below can never drift apart.
_AUTO_CONSTRAINT_NAME = "constraint_{}"
_AUTO_CONSTRAINT_RE = re.compile(r"^constraint_(\d+)$")

# What counts as a number wherever this module demands a finite one -- a
# constraint rhs or coefficient, and a variable bound. bool is deliberately
# included (it is a subclass of int and np.isfinite handles it); complex, str,
# None and containers are not.
#
# np.bool_ is listed explicitly because it is *not* a subclass of either bool
# or np.integer, so the "accept numpy scalars" intent had a hole: np.True_ was
# rejected while both True and np.int64(1) were accepted, and the diagnostic
# said "of type bool" -- naming the very type the line above says is accepted.
# Accepting it is what makes the rule statable in one sentence: a finite
# Python or numpy real scalar.
_FINITE_NUMBER_TYPES = (int, float, bool, np.bool_, np.integer, np.floating)


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
    for the JSON-representability reason recorded in :meth:`add_constraint`:
    skopt's ``low >= high`` test is ``False`` for ``NaN``, so a non-finite
    bound registered cleanly and then made *every* export of that session a
    400 -- ``json.dumps`` refuses ``nan``/``inf`` under ``allow_nan=False``,
    which is what FastAPI's ``JSONResponse`` uses. Neither export shape could
    get the space back out, so the session was unrecoverable through the API.
    ``json.load`` accepts the bare ``NaN``/``Infinity`` literals by default, so
    such a file is an ordinary upload rather than a hostile one.
    """
    if not isinstance(value, _FINITE_NUMBER_TYPES):
        raise ValueError(
            f"{label} must be a finite number, got {value!r} "
            f"of type {_type_name(value)}"
        )
    if not np.isfinite(value):
        raise ValueError(f"{label} must be finite, got {value}")


def _validate_bound(value: Any, var_name: str, key: str) -> None:
    """``_validate_finite_number`` for a variable bound, labelled by variable.

    The label names both the variable and the key, because the caller that
    needs this most is ``POST /variables/load``: it hands a whole uploaded file
    to the core and can only report what the exception says.

    ``allowed_values`` entries reach this already coerced by ``float()``, so a
    quoted number survives there while a quoted bound is refused. That
    asymmetry is pre-existing and deliberately left alone here (branch item M5);
    narrowing it is a change to what files load, not to this guard.
    """
    _validate_finite_number(value, f"Variable '{var_name}' {key}")


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
        # for every dimension-bearing type (update_variable and delete_variable
        # in api/routers/variables.py index one by the other), so the desync is
        # silent until something zips them.
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
        dimension = None
        if var_type_lower == "real":
            _validate_bound(kwargs["min"], name, "min")
            _validate_bound(kwargs["max"], name, "max")
            dimension = Real(kwargs["min"], kwargs["max"], name=name)
        elif var_type_lower == "integer":
            _validate_bound(kwargs["min"], name, "min")
            _validate_bound(kwargs["max"], name, "max")
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
            coerced = [float(v) for v in allowed]
            # Before sorting, not after: sorted() puts NaN wherever the
            # comparisons happen to land it, so an unchecked NaN would also
            # scramble the order of the values around it.
            for i, value in enumerate(coerced):
                _validate_bound(value, name, f"allowed_values[{i}]")
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
