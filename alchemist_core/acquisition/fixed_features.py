"""Held and observed values as BoTorch ``fixed_features``.

Every variable the optimizer must not move is pinned at one value in the model's
own input space: a controllable variable the caller holds for this call, and
every observed variable at its measured value. Column indices are positions in
the model's feature order (``model.original_feature_names``), the same indices
``cat_dims`` and ``discrete_dims`` use, so nothing is ever remapped into a
reduced space.
"""
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

from alchemist_core.data.search_space import kind_of, role_of


@dataclass(frozen=True)
class FixedFeatures:
    columns: Dict[int, float]   # model column index -> encoded value
    held: Dict[str, Any]        # controllable name -> value as the user sees it
    observed: Dict[str, Any]    # observed name -> value as the user sees it


def _encode(var: Dict[str, Any], value: Any, encodings: Mapping[str, Mapping],
            where: str) -> Tuple[Any, float]:
    name = var["name"]
    kind = kind_of(var)
    if kind == "categorical":
        declared = list(var.get("values") or [])
        if value not in declared:
            same_text = [v for v in declared if str(v) == str(value)]
            if not same_text:
                raise ValueError(
                    f"{where} value {value!r} for '{name}' is not one of its "
                    f"declared values {declared}."
                )
            value = same_text[0]
        encoding = encodings.get(name) or {v: i for i, v in enumerate(declared)}
        return value, float(encoding[value])
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{where} value {value!r} for '{name}' is not a number.")
    if not math.isfinite(x):
        raise ValueError(f"{where} value for '{name}' must be finite, got {value!r}.")
    if role_of(var) == "observed":
        return x, x
    if kind == "discrete":
        allowed = [float(a) for a in var["allowed_values"]]
        if not any(math.isclose(x, a, rel_tol=1e-9, abs_tol=1e-12) for a in allowed):
            raise ValueError(
                f"Held value {x} for '{name}' is not one of its allowed values {allowed}."
            )
        return x, x
    lo, hi = float(var["min"]), float(var["max"])
    if not lo <= x <= hi:
        raise ValueError(f"Held value {x} for '{name}' is outside its bounds [{lo}, {hi}].")
    if kind == "integer":
        if x != round(x):
            raise ValueError(f"Held value {x} for integer variable '{name}' is not an integer.")
        return int(round(x)), float(round(x))
    return x, x


def build_fixed_features(search_space, feature_names: List[str],
                         encodings: Mapping[str, Mapping],
                         hold: Mapping[str, Any],
                         observed: Mapping[str, Any]) -> FixedFeatures:
    """Validate ``hold`` and ``observed`` and encode them by model column.

    ``observed`` must already be restricted to registered observed variables;
    ``hold`` may name any variable and is checked here.
    """
    by_name = {v["name"]: v for v in search_space.variables}
    columns: Dict[int, float] = {}
    held: Dict[str, Any] = {}
    obs: Dict[str, Any] = {}
    for name, value in hold.items():
        var = by_name.get(name)
        if var is None:
            raise ValueError(
                f"Cannot hold '{name}': no such variable. Registered: {sorted(by_name)}."
            )
        if role_of(var) == "observed":
            raise ValueError(
                f"'{name}' is an observed variable; pass its measured value in "
                f"observed=, not hold=."
            )
        held[name], columns[feature_names.index(name)] = _encode(var, value, encodings, "Held")
    for name, value in observed.items():
        obs[name], columns[feature_names.index(name)] = _encode(
            by_name[name], value, encodings, "Observed"
        )
    return FixedFeatures(columns=columns, held=held, observed=obs)
