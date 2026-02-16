"""
Pydantic request models for API endpoints.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional, Literal, Union


# ============================================================
# Variable Models
# ============================================================

class AddRealVariableRequest(BaseModel):
    """Request to add a real-valued variable."""
    name: str = Field(..., description="Variable name")
    type: Literal["real"] = Field(default="real", description="Variable type")
    min: float = Field(..., description="Minimum value")
    max: float = Field(..., description="Maximum value")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    description: Optional[str] = Field(None, description="Variable description")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "temperature",
                "type": "real",
                "min": 300,
                "max": 500,
                "unit": "°C",
                "description": "Reaction temperature"
            }
        }
    )


class AddIntegerVariableRequest(BaseModel):
    """Request to add an integer variable."""
    name: str = Field(..., description="Variable name")
    type: Literal["integer"] = Field(default="integer", description="Variable type")
    min: int = Field(..., description="Minimum value")
    max: int = Field(..., description="Maximum value")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    description: Optional[str] = Field(None, description="Variable description")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "batch_size",
                "type": "integer",
                "min": 1,
                "max": 10,
                "unit": "batches",
                "description": "Number of batches"
            }
        }
    )


class AddCategoricalVariableRequest(BaseModel):
    """Request to add a categorical variable."""
    name: str = Field(..., description="Variable name")
    type: Literal["categorical"] = Field(default="categorical", description="Variable type")
    categories: List[str] = Field(..., description="List of category values")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    description: Optional[str] = Field(None, description="Variable description")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "catalyst",
                "type": "categorical",
                "categories": ["A", "B", "C"],
                "description": "Catalyst type"
            }
        }
    )


# Union type for any variable request
AddVariableRequest = Union[
    AddRealVariableRequest,
    AddIntegerVariableRequest,
    AddCategoricalVariableRequest
]


# ============================================================
# Experiment Models
# ============================================================

class AddExperimentRequest(BaseModel):
    """Request to add a single experiment."""
    inputs: Dict[str, Union[float, int, str]] = Field(..., description="Variable values")
    output: Optional[float] = Field(None, description="Target/output value")
    noise: Optional[float] = Field(None, description="Measurement uncertainty")
    iteration: Optional[int] = Field(None, description="Iteration number (auto-assigned if None)")
    reason: Optional[str] = Field(None, description="Reason for this experiment")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "inputs": {"temperature": 350, "catalyst": "A"},
                "output": 0.85,
                "noise": 0.02,
                "iteration": 1,
                "reason": "Initial Design"
            }
        }
    )


class StageExperimentRequest(BaseModel):
    """Request to stage an experiment for later execution."""
    inputs: Dict[str, Union[float, int, str]] = Field(..., description="Variable values")
    reason: Optional[str] = Field(None, description="Reason for this experiment (e.g., acquisition strategy)")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "inputs": {"temperature": 375.2, "catalyst": "B"},
                "reason": "qEI"
            }
        }
    )


class StageExperimentsBatchRequest(BaseModel):
    """Request to stage multiple experiments at once."""
    experiments: List[Dict[str, Union[float, int, str]]] = Field(..., description="List of experiment inputs")
    reason: Optional[str] = Field(None, description="Reason for these experiments")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "experiments": [
                    {"temperature": 375.2, "catalyst": "B"},
                    {"temperature": 412.8, "catalyst": "A"}
                ],
                "reason": "qEI batch"
            }
        }
    )


class CompleteStagedExperimentsRequest(BaseModel):
    """Request to complete staged experiments with outputs."""
    outputs: List[float] = Field(..., description="Output values for staged experiments (same order)")
    noises: Optional[List[float]] = Field(None, description="Measurement uncertainties (optional)")
    iteration: Optional[int] = Field(None, description="Iteration number (auto-assigned if None)")
    reason: Optional[str] = Field(None, description="Reason (uses staged reason if not provided)")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "outputs": [0.87, 0.92],
                "noises": [0.02, 0.03],
                "iteration": 5,
                "reason": "qEI"
            }
        }
    )


class AddExperimentsBatchRequest(BaseModel):
    """Request to add multiple experiments."""
    experiments: List[AddExperimentRequest] = Field(..., description="List of experiments")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "experiments": [
                    {"inputs": {"temperature": 350, "catalyst": "A"}, "output": 0.85},
                    {"inputs": {"temperature": 400, "catalyst": "B"}, "output": 0.92}
                ]
            }
        }
    )


# ============================================================
# Model Training Models
# ============================================================

class TrainModelRequest(BaseModel):
    """Request to train a surrogate model."""
    backend: Literal["sklearn", "botorch"] = Field(default="sklearn", description="Modeling backend")
    kernel: str = Field(default="Matern", description="Kernel type (RBF, Matern, RationalQuadratic for sklearn; RBF, Matern, IBNN for botorch)")
    kernel_params: Optional[Dict[str, Any]] = Field(None, description="Kernel-specific parameters")
    input_transform: Optional[str] = Field(None, description="Input transformation (Normalize, Standardize, etc.)")
    output_transform: Optional[str] = Field(None, description="Output transformation (Standardize, etc.)")
    calibration_enabled: bool = Field(default=False, description="Enable uncertainty calibration")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "backend": "sklearn",
                "kernel": "Matern",
                "kernel_params": {"nu": 2.5}
            }
        }
    )


# ============================================================
# Acquisition Models
# ============================================================

class AcquisitionRequest(BaseModel):
    """Request to suggest next experiments."""
    strategy: str = Field(default="EI", description="Acquisition strategy (EI, PI, UCB, qEI, qUCB, qNIPV)")
    goal: Literal["maximize", "minimize"] = Field(default="maximize", description="Optimization goal")
    n_suggestions: int = Field(default=1, ge=1, le=10, description="Number of suggestions (batch size)")
    xi: Optional[float] = Field(default=0.01, description="Exploration parameter for EI/PI")
    kappa: Optional[float] = Field(default=2.0, description="Exploration parameter for UCB")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "strategy": "EI",
                "goal": "maximize",
                "n_suggestions": 1,
                "xi": 0.01
            }
        }
    )


class FindOptimumRequest(BaseModel):
    """Request to find model's predicted optimum."""
    goal: Literal["maximize", "minimize"] = Field(default="maximize", description="Optimization goal")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "goal": "maximize"
            }
        }
    )


# ============================================================
# Initial Design (DoE) Models
# ============================================================

class InitialDesignRequest(BaseModel):
    """Request for generating initial experimental design.

    Space-filling methods (random, lhs, sobol, halton, hammersly) require n_points.
    Classical RSM methods (full_factorial, fractional_factorial, ccd, box_behnken)
    determine run count from design structure; n_points is ignored.
    """
    method: Literal[
        "random", "lhs", "sobol", "halton", "hammersly",
        "full_factorial", "fractional_factorial", "ccd", "box_behnken"
    ] = Field(default="lhs", description="Sampling method")
    n_points: Optional[int] = Field(
        None, ge=1, le=1000,
        description="Number of points (required for space-filling, ignored for classical designs)"
    )
    random_seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    lhs_criterion: str = Field(
        default="maximin",
        pattern="^(maximin|correlation|ratio)$",
        description="Criterion for LHS method"
    )
    # Classical design parameters
    n_levels: int = Field(default=2, ge=2, le=5, description="Levels per factor (full factorial)")
    n_center: int = Field(default=1, ge=0, le=10, description="Center point replicates")
    generators: Optional[str] = Field(None, description="Fractional factorial generator string")
    ccd_alpha: Literal["orthogonal", "rotatable"] = Field(
        default="orthogonal", description="CCD alpha type"
    )
    ccd_face: Literal["circumscribed", "inscribed", "faced"] = Field(
        default="circumscribed", description="CCD face type"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "method": "lhs",
                "n_points": 10,
                "random_seed": 42,
                "lhs_criterion": "maximin"
            }
        }
    )


# ============================================================
# Prediction Models
# ============================================================

class PredictionRequest(BaseModel):
    """Request to make predictions at new points."""
    inputs: List[Dict[str, Union[float, int, str]]] = Field(..., description="Input points for prediction")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "inputs": [
                    {"temperature": 375, "catalyst": "A"},
                    {"temperature": 425, "catalyst": "B"}
                ]
            }
        }
    )


# ============================================================
# Audit Log & Session Management Models
# ============================================================

class UpdateMetadataRequest(BaseModel):
    """Request to update session metadata."""
    name: Optional[str] = Field(None, description="Session name")
    description: Optional[str] = Field(None, description="Session description")
    tags: Optional[List[str]] = Field(None, description="Session tags")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Catalyst_Screening_Nov2025",
                "description": "Pt/Pd ratio optimization for CO2 reduction",
                "tags": ["catalyst", "CO2", "electrochemistry"]
            }
        }
    )


class LockDecisionRequest(BaseModel):
    """Request to lock in a decision to the audit log."""
    lock_type: Literal["data", "model", "acquisition"] = Field(..., description="Type of decision to lock")
    notes: Optional[str] = Field(None, description="Optional notes about this decision")
    
    # For acquisition lock
    strategy: Optional[str] = Field(None, description="Acquisition strategy (required for acquisition lock)")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Acquisition parameters (required for acquisition lock)")
    suggestions: Optional[List[Dict[str, Any]]] = Field(None, description="Suggested experiments (required for acquisition lock)")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "lock_type": "model",
                "notes": "Best cross-validation performance: R²=0.93"
            }
        }
    )


# ============================================================
# Session Lock Models
# ============================================================

class SessionLockRequest(BaseModel):
    """Request to lock a session for programmatic control."""
    locked_by: str = Field(..., description="Identifier of the client locking the session")
    client_id: Optional[str] = Field(None, description="Optional unique client identifier")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "locked_by": "Reactor Controller v1.2",
                "client_id": "lab-3-workstation"
            }
        }
    )
