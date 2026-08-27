"""
Variables router - Search space management.
"""

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from typing import Union
from ..models.requests import (
    AddRealVariableRequest,
    AddIntegerVariableRequest,
    AddCategoricalVariableRequest,
    AddDiscreteVariableRequest,
    AddConstraintRequest,
    unaddressable_name_reason,
)
from ..models.responses import (
    VariableResponse,
    VariablesListResponse,
    ConstraintResponse,
    ConstraintsListResponse,
)
from ..dependencies import get_session
from ..middleware.error_handlers import NoVariablesError
from alchemist_core.session import OptimizationSession
from alchemist_core.data.search_space import SearchSpace
import logging
import json
import tempfile
import os

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================
# /variables/load helpers
# ============================================================

# The three keys a constraint entry in an uploaded file must carry. ``name``
# is optional -- SearchSpace.add_constraint auto-generates ``constraint_N``.
_REQUIRED_CONSTRAINT_KEYS = ("type", "coefficients", "rhs")


def _reject_unaddressable(name, resource: str, collection: str) -> None:
    """400 if ``name`` could never be addressed by the DELETE route.

    ``/variables/load`` parses a raw JSON file and never constructs the request
    models, so it bypassed the validator those models carry: ``POST /variables``
    with ``name='..'`` returned 422 while ``POST /variables/load`` with the same
    name returned 200 and registered it. That is not a cosmetic gap. Clients
    apply RFC 3986 dot-segment removal before sending, so the user's follow-up
    ``DELETE .../variables/..`` is rewritten onto the session route in transit
    and destroys the entire session -- every variable, every experiment, the
    trained model -- while returning 204.

    The rule itself lives in ``api/models/requests.unaddressable_name_reason``
    and is not restated here; only the error shape differs, because an uploaded
    file is a 400 (the request itself was well formed) rather than a 422.
    """
    reason = unaddressable_name_reason(name)
    if reason is not None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"The {resource} name {name!r} cannot be addressed by "
                f"DELETE .../{collection}/<name>: {reason}. The name is the "
                f"identity used to delete the {resource}, so such a "
                f"{resource} could never be removed."
            ),
        )


def _validate_load_payload(payload):
    """Split an uploaded payload into (variables, constraints, is_dict_format).

    Everything reachable without touching the session is checked here, before
    any mutation: shape, required keys, and name addressability for both
    variables and constraints.
    """
    if isinstance(payload, dict):
        dict_format = True
        variables_data = payload.get("variables")
        constraints_data = payload.get("constraints")
        variables_data = [] if variables_data is None else variables_data
        constraints_data = [] if constraints_data is None else constraints_data
    elif isinstance(payload, list):
        dict_format = False
        variables_data = payload
        constraints_data = []
    else:
        raise HTTPException(
            status_code=400,
            detail=(
                "A search space file must contain either a JSON array of "
                "variables or a JSON object with a 'variables' key, got "
                f"{type(payload).__name__}."
            ),
        )

    if not isinstance(variables_data, list):
        raise HTTPException(
            status_code=400,
            detail=f"'variables' must be a JSON array, got {type(variables_data).__name__}.",
        )
    if not isinstance(constraints_data, list):
        raise HTTPException(
            status_code=400,
            detail=f"'constraints' must be a JSON array, got {type(constraints_data).__name__}.",
        )

    for i, var in enumerate(variables_data):
        if not isinstance(var, dict):
            raise HTTPException(
                status_code=400,
                detail=f"variables[{i}] must be a JSON object, got {type(var).__name__}.",
            )
        if "name" not in var:
            raise HTTPException(status_code=400, detail=f"variables[{i}] is missing 'name'.")
        if "type" not in var:
            raise HTTPException(
                status_code=400,
                detail=f"variables[{i}] ({var['name']!r}) is missing 'type'.",
            )
        if not isinstance(var["type"], str):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"variables[{i}] ({var['name']!r}) has a non-string 'type': "
                    f"{var['type']!r}."
                ),
            )
        _reject_unaddressable(var["name"], "variable", "variables")

    for i, constraint in enumerate(constraints_data):
        if not isinstance(constraint, dict):
            raise HTTPException(
                status_code=400,
                detail=f"constraints[{i}] must be a JSON object, got {type(constraint).__name__}.",
            )
        missing = [k for k in _REQUIRED_CONSTRAINT_KEYS if k not in constraint]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"constraints[{i}] is missing required key(s): {', '.join(missing)}.",
            )
        # An omitted or null name is legal -- it means "auto-generate".
        if constraint.get("name") is not None:
            _reject_unaddressable(constraint["name"], "constraint", "constraints")

    return variables_data, constraints_data, dict_format


def _apply_search_space(space: SearchSpace, variables_data, constraints_data) -> None:
    """Load variables and constraints into ``space``, replacing what is there.

    Constraints go through ``add_constraint`` rather than being assigned
    straight across as ``SearchSpace.load_from_json`` does. This is a REST
    write path into the search space, and the alternative would let
    ``/variables/load`` register precisely what ``POST /constraints`` rejects:
    a coefficient on a categorical or unknown variable, a duplicate name (two
    constraints sharing one delete identity), or a non-finite rhs (which makes
    every point infeasible and drives the DoE into a pathological resampling
    path).

    Constraints are replaced, not merged, for the same reason ``load_from_json``
    replaces them: ``from_dict`` discards the previous variables, so a retained
    constraint would reference variables that no longer exist and nothing
    downstream would catch it.
    """
    space.from_dict(variables_data)

    # from_dict silently ignores a variable whose 'type' it does not recognize
    # -- its branch chain has no else. Silently loading fewer variables than the
    # file contains is worse than refusing the file, so compare the counts.
    if len(space.variables) != len(variables_data):
        loaded = {v["name"] for v in space.variables}
        dropped = [v["name"] for v in variables_data if v["name"] not in loaded]
        raise ValueError(
            f"Unsupported variable type for: {dropped}. Supported types are "
            f"real, integer, categorical, discrete, context."
        )

    space.constraints = []
    for constraint in constraints_data:
        coefficients = constraint["coefficients"]
        if isinstance(coefficients, dict):
            # add_constraint stores the mapping by reference; copy it so the
            # registered constraint does not alias the parsed file.
            coefficients = dict(coefficients)
        space.add_constraint(
            constraint["type"],
            coefficients,
            constraint["rhs"],
            constraint.get("name"),
        )


def _load_error_detail(exc: Exception) -> str:
    """Turn a core-library failure into a message that names the file's fault."""
    if isinstance(exc, KeyError):
        return f"Search space file is missing required key {exc.args[0]!r}."
    if isinstance(exc, TypeError):
        return f"Search space file could not be loaded: {exc}"
    return str(exc)


@router.post("/{session_id}/variables", response_model=VariableResponse)
async def add_variable(
    session_id: str,
    variable: Union[AddRealVariableRequest, AddIntegerVariableRequest, AddCategoricalVariableRequest, AddDiscreteVariableRequest],
    session: OptimizationSession = Depends(get_session)
):
    """
    Add a variable to the search space.

    Supports four types of variables:
    - real: Continuous floating-point values
    - integer: Discrete integer values
    - categorical: Unordered named categories
    - discrete: Numerical variable restricted to specific allowed values
    """
    # Extract variable data
    var_dict = variable.model_dump()
    var_type = var_dict.pop("type")
    name = var_dict.pop("name")
    
    logger.info(f"Received variable data: {var_dict}")
    
    # Check if variable already exists
    existing_names = [v['name'] for v in session.search_space.variables]
    if name in existing_names:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail=f"Variable '{name}' already exists. Please use a different name or delete the existing variable first."
        )
    
    # Handle categories → values conversion for categorical
    if "categories" in var_dict:
        var_dict["values"] = var_dict.pop("categories")
    
    # Add variable to session
    session.add_variable(name, var_type, **var_dict)
    
    logger.info(f"Added variable '{name}' ({var_type}) to session {session_id}")
    
    return VariableResponse(
        message="Variable added successfully",
        variable={
            "name": name,
            "type": var_type,
            **var_dict
        }
    )


@router.get("/{session_id}/variables", response_model=VariablesListResponse)
async def list_variables(
    session_id: str,
    session: OptimizationSession = Depends(get_session)
):
    """
    Get all variables in the search space.
    
    Returns list of variables with their types and parameters.
    """
    summary = session.get_search_space_summary()
    
    logger.info(f"Returning variables summary: {summary}")
    
    return VariablesListResponse(
        variables=summary["variables"],
        n_variables=summary["n_variables"]
    )


@router.post("/{session_id}/variables/load")
async def load_variables_from_file(
    session_id: str,
    file: UploadFile = File(...),
    session: OptimizationSession = Depends(get_session)
):
    """
    Load a search space definition from a JSON file.

    Two shapes are accepted.

    **Bare list** (legacy). Variables are *appended* to the existing search
    space; constraints are untouched:

    ```json
    [
        {"name": "x1", "type": "real", "min": 300, "max": 500},
        {"name": "x2", "type": "categorical", "categories": ["A", "B", "C"]}
    ]
    ```

    **Dict** — the shape `SearchSpace.save_to_json` writes and
    `GET /variables/export?include_constraints=true` returns. The search space
    is *replaced*, constraints included:

    ```json
    {
        "variables": [{"name": "x1", "type": "real", "min": 0, "max": 10}],
        "constraints": [
            {"type": "inequality", "coefficients": {"x1": 3.0},
             "rhs": 8.0, "name": "c_a"}
        ]
    }
    ```

    Loaded constraints are registered through the same validation as
    `POST /constraints`, and loaded names through the same addressability rule
    as `POST /variables`. A file that fails any of it is rejected whole, with
    400 — the session is left exactly as it was, never holding half a file.
    """
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.json') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Load and parse JSON
        try:
            with open(tmp_path, 'r') as f:
                payload = json.load(f)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400, detail=f"Uploaded file is not valid JSON: {e}"
            )

        variables_data, constraints_data, dict_format = _validate_load_payload(payload)

        if dict_format:
            # Dry run on a throwaway space first. from_dict discards the
            # session's variables before it adds any, so a file that fails
            # partway through would otherwise leave the session holding a
            # fragment of it with no way to tell.
            try:
                _apply_search_space(SearchSpace(), variables_data, constraints_data)
            except (ValueError, KeyError, TypeError) as e:
                # TypeError is caught as a backstop for malformed uploaded data
                # reaching a core method that did not expect it. The one known
                # case -- a non-numeric rhs or coefficient -- now raises
                # ValueError from SearchSpace.add_constraint, pinned by
                # tests/unit/core/data/test_constraints.py.
                raise HTTPException(status_code=400, detail=_load_error_detail(e))

            # Cannot fail: the dry run above performed the identical sequence.
            _apply_search_space(session.search_space, variables_data, constraints_data)

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

        # Legacy bare-list path: variables are appended, as they always were.
        for entry in variables_data:
            var = dict(entry)
            var_type = var.pop("type")
            name = var.pop("name")

            # Handle categories for categorical variables
            if "categories" in var:
                var["values"] = var.pop("categories")

            try:
                session.add_variable(name, var_type, **var)
            except (ValueError, KeyError, TypeError) as e:
                # KeyError and TypeError were previously uncaught, so a file
                # missing a bound surfaced as a 500 rather than naming the key.
                # (A ValueError was already a 400 -- the app registers a global
                # ValueError handler -- but is caught here so every failure on
                # this path reports through one message format.)
                raise HTTPException(status_code=400, detail=_load_error_detail(e))

        logger.info(f"Loaded {len(variables_data)} variables from file for session {session_id}")
        
        return {
            "message": f"Loaded {len(variables_data)} variables successfully",
            "n_variables": len(variables_data),
            "n_constraints": 0,
        }
        
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.get("/{session_id}/variables/export")
async def export_variables_to_json(
    session_id: str,
    include_constraints: bool = False,
    session: OptimizationSession = Depends(get_session)
):
    """
    Export the search space definition to JSON.

    By default returns a bare JSON array of variables — the shape
    `SearchSpace.from_dict` consumes directly, which the desktop GUI and the
    core Python API both rely on. Constraints are **not** in that shape and are
    dropped from the default export.

    Pass `include_constraints=true` for
    `{"variables": [...], "constraints": [...]}` — the same payload
    `SearchSpace.save_to_json` writes, readable by `SearchSpace.load_from_json`
    and by `POST /variables/load`, so `load → export → load` round-trips
    constraints.

    The opt-in rather than a shape change is deliberate: the bare list is an
    existing cross-surface contract, and flipping it would break every consumer
    that loads an export through `from_dict`.
    """
    from fastapi.responses import JSONResponse
    
    summary = session.get_search_space_summary()
    variables = summary["variables"]
    
    # Convert to export format
    export_data = []
    for var in variables:
        var_dict = {
            "name": var["name"],
            "type": var["type"]
        }
        
        if var.get("bounds"):
            var_dict["min"] = var["bounds"][0]
            var_dict["max"] = var["bounds"][1]

        if var.get("allowed_values"):
            var_dict["allowed_values"] = var["allowed_values"]

        # Emit the canonical 'values' field for categorical variables. This is
        # the schema that SearchSpace.from_dict expects, so the exported JSON
        # can be loaded by the desktop GUI, the core Python API, or re-uploaded
        # via /variables/load without any field translation.
        if var.get("categories"):
            var_dict["values"] = var["categories"]
        
        # Include optional fields
        if var.get("unit"):
            var_dict["unit"] = var["unit"]
        if var.get("description"):
            var_dict["description"] = var["description"]
            
        export_data.append(var_dict)
    
    if include_constraints:
        constraints = session.search_space.get_constraints()
        content = {"variables": export_data, "constraints": constraints}
        logger.info(
            f"Exported {len(export_data)} variables and {len(constraints)} "
            f"constraints from session {session_id}"
        )
    else:
        content = export_data
        logger.info(f"Exported {len(export_data)} variables from session {session_id}")

    return JSONResponse(
        content=content,
        headers={
            "Content-Disposition": f"attachment; filename=variables_{session_id[:8]}.json"
        }
    )


@router.put("/{session_id}/variables/{variable_name}", response_model=VariableResponse)
async def update_variable(
    session_id: str,
    variable_name: str,
    variable: Union[AddRealVariableRequest, AddIntegerVariableRequest, AddCategoricalVariableRequest, AddDiscreteVariableRequest],
    session: OptimizationSession = Depends(get_session)
):
    """
    Update an existing variable in the search space.
    
    Note: Variable name cannot be changed. To rename, delete and create new.
    """
    # Extract variable data
    var_dict = variable.model_dump()
    var_type = var_dict.pop("type")
    new_name = var_dict.pop("name")
    
    logger.info(f"UPDATE: Received var_dict: {var_dict}")
    
    # Ensure name matches the path parameter
    if new_name != variable_name:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail="Variable name in request body must match the name in URL path"
        )
    
    # Find the variable
    var_index = None
    for i, var in enumerate(session.search_space.variables):
        if var['name'] == variable_name:
            var_index = i
            break
    
    if var_index is None:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=404,
            detail=f"Variable '{variable_name}' not found"
        )
    
    # Handle categories → values conversion for categorical
    if "categories" in var_dict:
        var_dict["values"] = var_dict.pop("categories")
    
    # Update the variable
    updated_var = {"name": variable_name, "type": var_type}
    updated_var.update(var_dict)
    logger.info(f"UPDATE: Final updated_var: {updated_var}")
    session.search_space.variables[var_index] = updated_var
    
    # Update the skopt dimension
    if var_type == "real":
        from skopt.space import Real
        session.search_space.skopt_dimensions[var_index] = Real(
            var_dict["min"], var_dict["max"], name=variable_name
        )
    elif var_type == "integer":
        from skopt.space import Integer
        session.search_space.skopt_dimensions[var_index] = Integer(
            var_dict["min"], var_dict["max"], name=variable_name
        )
    elif var_type == "categorical":
        from skopt.space import Categorical
        session.search_space.skopt_dimensions[var_index] = Categorical(
            var_dict["values"], name=variable_name
        )
        # Update categorical variables list
        if variable_name not in session.search_space.categorical_variables:
            session.search_space.categorical_variables.append(variable_name)
    elif var_type == "discrete":
        from skopt.space import Categorical
        sorted_vals = sorted(float(v) for v in var_dict["allowed_values"])
        var_dict["allowed_values"] = sorted_vals
        updated_var["allowed_values"] = sorted_vals
        session.search_space.skopt_dimensions[var_index] = Categorical(
            sorted_vals, name=variable_name
        )
        # Update discrete variables list
        if variable_name not in session.search_space.discrete_variables:
            session.search_space.discrete_variables.append(variable_name)
    
    logger.info(f"Updated variable '{variable_name}' ({var_type}) in session {session_id}")
    
    return VariableResponse(
        message="Variable updated successfully",
        variable={
            "name": variable_name,
            "type": var_type,
            **var_dict
        }
    )


@router.delete("/{session_id}/variables/{variable_name}")
async def delete_variable(
    session_id: str,
    variable_name: str,
    session: OptimizationSession = Depends(get_session)
):
    """
    Delete a variable from the search space.
    
    Args:
        session_id: The session ID
        variable_name: Name of the variable to delete
        
    Returns:
        Success message with updated count
    """
    # Find and remove the variable from the session's search space
    variable_found = False
    for i, var in enumerate(session.search_space.variables):
        if var['name'] == variable_name:
            # Remove from variables list
            session.search_space.variables.pop(i)
            # Remove from skopt dimensions
            session.search_space.skopt_dimensions.pop(i)
            # Remove from categorical/discrete lists if applicable
            if variable_name in session.search_space.categorical_variables:
                session.search_space.categorical_variables.remove(variable_name)
            if variable_name in session.search_space.discrete_variables:
                session.search_space.discrete_variables.remove(variable_name)
            variable_found = True
            break
    
    if not variable_found:
        raise HTTPException(status_code=404, detail=f"Variable '{variable_name}' not found")
    
    logger.info(f"Deleted variable '{variable_name}' from session {session_id}")
    
    # Get updated summary
    summary = session.get_search_space_summary()
    
    return {
        "message": f"Variable '{variable_name}' deleted successfully",
        "n_variables": summary["n_variables"]
    }


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
    Names are unique: omit `name` to get an auto-generated `constraint_N`,
    or supply one that is not already registered.
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

    Deletion is by name rather than index: an index shifts as soon as an
    earlier constraint is removed, so a client holding one would delete the
    wrong constraint. Exactly one constraint is removed per call.
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

    # Remove the first match only, never every match. add_constraint rejects a
    # duplicate name, but that is not the only way constraints get into the
    # list: SearchSpace.load_from_json assigns self.constraints straight from
    # the file and bypasses add_constraint entirely, so a loaded search space
    # can hold duplicates. A filter on != name would silently drop all of them
    # while reporting a single deletion.
    existing.remove(match[0])
    logger.info(f"Deleted constraint '{constraint_name}' from session {session_id}")
    return {"message": f"Constraint '{constraint_name}' deleted successfully"}
