from typing import List, Dict, Any, Union, Optional, Tuple
from skopt.space import Real, Integer, Categorical
import numpy as np
import pandas as pd
import json

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

    def add_variable(self, name: str, var_type: str, **kwargs):
        """
        Add a variable to the search space.

        Args:
            name: Variable name
            var_type: "real", "integer", "categorical", or "discrete"
            **kwargs: Additional parameters:
                - real/integer: min, max
                - categorical: values (list of strings)
                - discrete: allowed_values (list of numbers, at least 2, no duplicates)
        """
        var_dict = {"name": name, "type": var_type.lower()}
        var_dict.update(kwargs)
        self.variables.append(var_dict)

        # Create the corresponding skopt dimension
        if var_type.lower() == "real":
            self.skopt_dimensions.append(Real(kwargs["min"], kwargs["max"], name=name))
        elif var_type.lower() == "integer":
            self.skopt_dimensions.append(Integer(kwargs["min"], kwargs["max"], name=name))
        elif var_type.lower() == "categorical":
            self.skopt_dimensions.append(Categorical(kwargs["values"], name=name))
            self.categorical_variables.append(name)
        elif var_type.lower() == "discrete":
            allowed = kwargs.get("allowed_values")
            if allowed is None or len(allowed) < 2:
                raise ValueError(
                    f"Discrete variable '{name}' requires 'allowed_values' with at least 2 values."
                )
            if len(allowed) != len(set(allowed)):
                raise ValueError(
                    f"Discrete variable '{name}' has duplicate values in 'allowed_values'."
                )
            # Store sorted for consistency; represent to skopt as Categorical of numeric values
            sorted_vals = sorted(float(v) for v in allowed)
            var_dict["allowed_values"] = sorted_vals
            self.skopt_dimensions.append(Categorical(sorted_vals, name=name))
            self.discrete_variables.append(name)
        else:
            raise ValueError(f"Unknown variable type: {var_type}")

    def from_dict(self, data: List[Dict[str, Any]]):
        """Load search space from a list of dictionaries (used with JSON/CSV loading)."""
        self.variables = []
        self.skopt_dimensions = []
        self.categorical_variables = []
        self.discrete_variables = []

        for var in data:
            var_type = var["type"].lower()
            if var_type in ["real", "integer"]:
                self.add_variable(
                    name=var["name"],
                    var_type=var_type,
                    min=var["min"],
                    max=var["max"]
                )
            elif var_type == "categorical":
                self.add_variable(
                    name=var["name"],
                    var_type=var_type,
                    values=var["values"]
                )
            elif var_type == "discrete":
                self.add_variable(
                    name=var["name"],
                    var_type=var_type,
                    allowed_values=var["allowed_values"]
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
        """Get list of all variable names."""
        return [var["name"] for var in self.variables]

    def get_categorical_variables(self) -> List[str]:
        """Get list of categorical variable names."""
        return self.categorical_variables.copy()

    def get_integer_variables(self) -> List[str]:
        """Get list of integer variable names."""
        return [var["name"] for var in self.variables if var["type"] == "integer"]

    def get_discrete_variables(self) -> List[str]:
        """Get list of discrete variable names."""
        return self.discrete_variables.copy()

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
            name: optional human-readable name
        """
        valid_types = ('inequality', 'equality')
        if constraint_type not in valid_types:
            raise ValueError(f"constraint_type must be one of {valid_types}, got '{constraint_type}'")

        var_names = self.get_variable_names()
        for var_name in coefficients:
            if var_name not in var_names:
                raise ValueError(f"Variable '{var_name}' in constraint not found in search space. "
                                 f"Available: {var_names}")

        self.constraints.append({
            'type': constraint_type,
            'coefficients': coefficients,
            'rhs': rhs,
            'name': name or f"constraint_{len(self.constraints)}"
        })

    def get_constraints(self) -> List[Dict]:
        """Return list of constraint dicts."""
        return [c.copy() for c in self.constraints]

    def to_botorch_constraints(self, feature_names: List[str]) -> Tuple[Optional[List], Optional[List]]:
        """Convert to BoTorch format for optimize_acqf.

        Each constraint is a tuple (indices_tensor, coefficients_tensor, rhs_float).
        BoTorch convention: inequality means sum(coeff_i * x_i) - rhs <= 0.

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
                coeffs.append(coeff)

            if not indices:
                continue

            constraint_tuple = (
                torch.tensor(indices, dtype=torch.long),
                torch.tensor(coeffs, dtype=torch.double),
                c['rhs']
            )

            if c['type'] == 'inequality':
                inequality_constraints.append(constraint_tuple)
            else:
                equality_constraints.append(constraint_tuple)

        return (
            inequality_constraints if inequality_constraints else None,
            equality_constraints if equality_constraints else None
        )

    def __len__(self):
        return len(self.variables)
