"""
Variables router - Search space management.
"""

from fastapi import APIRouter, Depends, UploadFile, File, Form
from typing import Union
from ..models.requests import (
    AddRealVariableRequest,
    AddIntegerVariableRequest,
    AddCategoricalVariableRequest,
    AddDiscreteVariableRequest,
)
from ..models.responses import VariableResponse, VariablesListResponse
from ..dependencies import get_session
from ..middleware.error_handlers import NoVariablesError
from alchemist_core.session import OptimizationSession
import logging
import json
import tempfile
import os

logger = logging.getLogger(__name__)

router = APIRouter()


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
    Load search space definition from JSON file.
    
    Expected JSON format:
    [
        {"name": "temp", "type": "real", "min": 300, "max": 500},
        {"name": "catalyst", "type": "categorical", "categories": ["A", "B", "C"]}
    ]
    """
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.json') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Load and parse JSON
        with open(tmp_path, 'r') as f:
            variables_data = json.load(f)
        
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
        
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.get("/{session_id}/variables/export")
async def export_variables_to_json(
    session_id: str,
    session: OptimizationSession = Depends(get_session)
):
    """
    Export search space definition to JSON format.
    
    Returns JSON array suitable for saving to file and loading later.
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

        if var.get("categories"):
            var_dict["categories"] = var["categories"]
        
        # Include optional fields
        if var.get("unit"):
            var_dict["unit"] = var["unit"]
        if var.get("description"):
            var_dict["description"] = var["description"]
            
        export_data.append(var_dict)
    
    logger.info(f"Exported {len(export_data)} variables from session {session_id}")
    
    return JSONResponse(
        content=export_data,
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
        raise ValueError(f"Variable '{variable_name}' not found")
    
    logger.info(f"Deleted variable '{variable_name}' from session {session_id}")
    
    # Get updated summary
    summary = session.get_search_space_summary()
    
    return {
        "message": f"Variable '{variable_name}' deleted successfully",
        "n_variables": summary["n_variables"]
    }
