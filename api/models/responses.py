"""
Pydantic response models for API endpoints.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
from datetime import datetime


# ============================================================
# Session Models
# ============================================================

class SessionCreateResponse(BaseModel):
    """Response when creating a new session."""
    session_id: str = Field(..., description="Unique session identifier")
    created_at: str = Field(..., description="Session creation timestamp")
    expires_at: Optional[str] = Field(None, description="Legacy field - no longer used (sessions don't expire)")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "created_at": "2025-10-31T14:30:00",
                "expires_at": "2025-11-01T14:30:00"
            }
        }
    )


class VariableInfo(BaseModel):
    """Information about a variable."""
    name: str
    type: str
    bounds: Optional[List[float]] = None
    categories: Optional[List[str]] = None
    allowed_values: Optional[List[float]] = None


class DataSummary(BaseModel):
    """Summary of experimental data."""
    n_experiments: int
    has_data: bool
    has_noise: bool = False
    target_stats: Optional[Dict[str, float]] = None
    feature_names: Optional[List[str]] = None


class ModelSummary(BaseModel):
    """Summary of trained model."""
    backend: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    is_trained: bool


class SessionInfoResponse(BaseModel):
    """Full session information."""
    session_id: str
    created_at: str
    last_accessed: str
    expires_at: Optional[str] = None  # Legacy field - no longer used
    search_space: Dict[str, Any]
    data: DataSummary
    model: Optional[ModelSummary]
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "created_at": "2025-10-31T14:30:00",
                "last_accessed": "2025-10-31T14:35:00",
                "expires_at": "2025-11-01T14:30:00",
                "search_space": {
                    "n_variables": 2,
                    "variables": [
                        {"name": "temp", "type": "real", "bounds": [300, 500]},
                        {"name": "catalyst", "type": "categorical", "categories": ["A", "B"]}
                    ]
                },
                "data": {
                    "n_experiments": 10,
                    "has_data": True,
                    "has_noise": False
                },
                "model": None
            }
        }
    )


class SessionStateResponse(BaseModel):
    """Current state of an optimization session for monitoring."""
    session_id: str
    n_variables: int
    n_experiments: int
    model_trained: bool
    model_backend: Optional[str] = None
    last_suggestion: Optional[Dict[str, Any]] = None
    last_acquisition_value: Optional[float] = None
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "n_variables": 2,
                "n_experiments": 15,
                "model_trained": True,
                "model_backend": "botorch",
                "last_suggestion": {"temperature": 385.2, "flow_rate": 4.3},
                "last_acquisition_value": 0.025
            }
        }
    )


# ============================================================
# Variable Models
# ============================================================

class VariableResponse(BaseModel):
    """Response when adding a variable."""
    message: str = "Variable added successfully"
    variable: Dict[str, Any]


class VariablesListResponse(BaseModel):
    """List of all variables in search space."""
    variables: List[Dict[str, Any]]
    n_variables: int


# ============================================================
# Experiment Models
# ============================================================

class ExperimentResponse(BaseModel):
    """Response when adding an experiment."""
    message: str = "Experiment added successfully"
    n_experiments: int
    model_trained: bool = Field(default=False, description="Whether model was auto-trained")
    training_metrics: Optional[Dict[str, Any]] = Field(
        None,
        description="Training metrics if auto-train was enabled"
    )


class ExperimentsListResponse(BaseModel):
    """List of all experiments."""
    experiments: List[Dict[str, Any]]
    n_experiments: int


class ExperimentsSummaryResponse(BaseModel):
    """Statistical summary of experimental data."""
    n_experiments: int
    has_data: bool
    has_noise: Optional[bool] = None
    target_stats: Optional[Dict[str, float]] = None
    feature_names: Optional[List[str]] = None
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "n_experiments": 10,
                "has_data": True,
                "has_noise": False,
                "target_stats": {
                    "min": 0.5,
                    "max": 0.95,
                    "mean": 0.75,
                    "std": 0.12
                },
                "feature_names": ["temperature", "pressure"]
            }
        }
    )


# ============================================================
# Initial Design (DoE) Models
# ============================================================

class InitialDesignResponse(BaseModel):
    """Response containing generated initial design points."""
    points: List[Dict[str, Any]] = Field(..., description="Generated experimental points")
    method: str = Field(..., description="Sampling method used")
    n_points: int = Field(..., description="Number of points generated")
    design_info: Optional[Dict[str, Any]] = Field(
        None,
        description="Design structure metadata for classical methods (runs breakdown, etc.)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "points": [
                    {"temperature": 350.2, "flow_rate": 2.47},
                    {"temperature": 421.8, "flow_rate": 7.92}
                ],
                "method": "lhs",
                "n_points": 2,
                "design_info": None
            }
        }
    )


# ============================================================
# Optimal Design (OED) Models
# ============================================================

class OptimalDesignInfoResponse(BaseModel):
    """Response from optimal design preview (dry-run)."""
    model_terms: List[str] = Field(..., description="Human-readable model term names")
    p_columns: int = Field(..., description="Number of model columns (parameters)")
    n_points_minimum: int = Field(..., description="Minimum runs to fit model (= p)")
    n_points_recommended: int = Field(..., description="Recommended runs (= 2p)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_terms": [
                    "Intercept", "Temperature", "Pressure",
                    "Temperature*Pressure", "Temperature²"
                ],
                "p_columns": 5,
                "n_points_minimum": 5,
                "n_points_recommended": 10
            }
        }
    )


class OptimalDesignResponse(BaseModel):
    """Response containing optimal design points and quality metrics."""
    points: List[Dict[str, Any]] = Field(..., description="Generated experimental points")
    n_points: int = Field(..., description="Number of points generated")
    design_info: Dict[str, Any] = Field(
        ...,
        description="Design quality metrics (criterion, score, D_eff, A_eff, model_terms, etc.)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "points": [
                    {"temperature": 300.0, "pressure": 1.0},
                    {"temperature": 500.0, "pressure": 10.0}
                ],
                "n_points": 12,
                "design_info": {
                    "criterion": "D",
                    "algorithm": "fedorov",
                    "score": 0.042,
                    "D_eff": 89.3,
                    "A_eff": 76.1,
                    "p_columns": 6,
                    "n_runs": 12,
                    "model_terms": ["Intercept", "Temperature", "Pressure"]
                }
            }
        }
    )


# ============================================================
# Staged Experiments Models
# ============================================================

class StagedExperimentResponse(BaseModel):
    """Response when staging an experiment."""
    message: str = "Experiment staged successfully"
    n_staged: int = Field(..., description="Total number of staged experiments")
    staged_inputs: Dict[str, Any] = Field(..., description="The staged experiment inputs")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Experiment staged successfully",
                "n_staged": 3,
                "staged_inputs": {"temperature": 375.2, "catalyst": "B"}
            }
        }
    )


class StagedExperimentsListResponse(BaseModel):
    """Response containing all staged experiments."""
    experiments: List[Dict[str, Any]] = Field(..., description="List of staged experiment inputs (variable values only)")
    n_staged: int = Field(..., description="Number of staged experiments")
    reason: Optional[str] = Field(None, description="Reason/strategy for these staged experiments")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "experiments": [
                    {"temperature": 375.2, "catalyst": "B"},
                    {"temperature": 412.8, "catalyst": "A"}
                ],
                "n_staged": 2,
                "reason": "qEI"
            }
        }
    )


class StagedExperimentsClearResponse(BaseModel):
    """Response when clearing staged experiments."""
    message: str = "Staged experiments cleared"
    n_cleared: int = Field(..., description="Number of experiments cleared")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Staged experiments cleared",
                "n_cleared": 3
            }
        }
    )


class StagedExperimentsCompletedResponse(BaseModel):
    """Response when completing staged experiments with outputs."""
    message: str = "Staged experiments completed and added to dataset"
    n_added: int = Field(..., description="Number of experiments added")
    n_experiments: int = Field(..., description="Total experiments in dataset")
    model_trained: bool = Field(default=False, description="Whether model was auto-trained")
    training_metrics: Optional[Dict[str, Any]] = Field(None, description="Training metrics if auto-trained")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Staged experiments completed and added to dataset",
                "n_added": 2,
                "n_experiments": 15,
                "model_trained": True,
                "training_metrics": {"rmse": 0.05, "r2": 0.92}
            }
        }
    )


# ============================================================
# Model Training Models
# ============================================================

class TrainModelResponse(BaseModel):
    """Response from model training."""
    success: bool
    backend: str
    kernel: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    message: str = "Model trained successfully"
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "backend": "sklearn",
                "kernel": "Matern",
                "hyperparameters": {
                    "lengthscales": [50.2, 0.8],
                    "noise_variance": 0.01
                },
                "metrics": {
                    "rmse": 0.05,
                    "mae": 0.03,
                    "r2": 0.95
                },
                "message": "Model trained successfully"
            }
        }
    )


class ModelInfoResponse(BaseModel):
    """Model information response."""
    backend: Optional[str]
    hyperparameters: Optional[Dict[str, Any]]
    metrics: Optional[Dict[str, float]]
    is_trained: bool
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "backend": "sklearn",
                "hyperparameters": {"lengthscales": [50.2, 0.8]},
                "metrics": {"rmse": 0.05, "r2": 0.95},
                "is_trained": True
            }
        }
    )


# ============================================================
# Acquisition Models
# ============================================================

class AcquisitionResponse(BaseModel):
    """Response from acquisition function."""
    suggestions: List[Dict[str, Any]]
    n_suggestions: int
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "suggestions": [
                    {"temperature": 375.2, "catalyst": "A"}
                ],
                "n_suggestions": 1
            }
        }
    )


class FindOptimumResponse(BaseModel):
    """Response from find model optimum."""
    optimum: Dict[str, Any] = Field(..., description="Optimal point found by model")
    predicted_value: float = Field(..., description="Predicted value at optimum")
    predicted_std: Optional[float] = Field(None, description="Standard deviation at optimum")
    goal: str = Field(..., description="Optimization goal (maximize/minimize)")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "optimum": {"temperature": 425.7, "catalyst": "B"},
                "predicted_value": 0.956,
                "predicted_std": 0.023,
                "goal": "maximize"
            }
        }
    )


# ============================================================
# Prediction Models
# ============================================================

class PredictionResult(BaseModel):
    """Single prediction result."""
    inputs: Dict[str, Any]
    prediction: float
    uncertainty: float


class PredictionResponse(BaseModel):
    """Response from prediction endpoint."""
    predictions: List[PredictionResult]
    n_predictions: int
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predictions": [
                    {
                        "inputs": {"temperature": 375, "catalyst": "A"},
                        "prediction": 0.87,
                        "uncertainty": 0.04
                    }
                ],
                "n_predictions": 1
            }
        }
    )


# ============================================================
# Error Models
# ============================================================

class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str
    error_type: str
    status_code: int
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "detail": "Session not found",
                "error_type": "SessionNotFoundError",
                "status_code": 404
            }
        }
    )


# ============================================================
# Audit Log & Session Management Responses
# ============================================================

class SessionMetadataResponse(BaseModel):
    """Response containing session metadata."""
    session_id: str
    name: str
    created_at: str
    last_modified: str
    description: str
    tags: List[str]
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "Catalyst_Screening_Nov2025",
                "created_at": "2025-11-19T09:00:00",
                "last_modified": "2025-11-19T14:30:00",
                "description": "Pt/Pd ratio optimization",
                "tags": ["catalyst", "CO2"]
            }
        }
    )


class AuditEntryResponse(BaseModel):
    """Response containing a single audit log entry."""
    timestamp: str
    entry_type: str
    parameters: Dict[str, Any]
    hash: str
    notes: str
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "timestamp": "2025-11-19T09:15:00",
                "entry_type": "data_locked",
                "parameters": {
                    "n_experiments": 15,
                    "variables": [],
                    "data_hash": "abc123"
                },
                "hash": "a1b2c3d4",
                "notes": "Initial screening dataset"
            }
        }
    )


class AuditLogResponse(BaseModel):
    """Response containing complete audit log."""
    entries: List[AuditEntryResponse]
    n_entries: int
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "entries": [],
                "n_entries": 0
            }
        }
    )


class LockDecisionResponse(BaseModel):
    """Response after locking a decision."""
    success: bool
    entry: AuditEntryResponse
    message: str
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "entry": {
                    "timestamp": "2025-11-19T09:15:00",
                    "entry_type": "data_locked",
                    "parameters": {},
                    "hash": "a1b2c3d4",
                    "notes": ""
                },
                "message": "Data decision locked successfully"
            }
        }
    )


# ============================================================
# Session Lock Models
# ============================================================

class SessionLockResponse(BaseModel):
    """Response for session lock operations."""
    locked: bool = Field(..., description="Whether the session is locked")
    locked_by: Optional[str] = Field(None, description="Identifier of who locked the session")
    locked_at: Optional[str] = Field(None, description="When the session was locked")
    lock_token: Optional[str] = Field(None, description="Token for unlocking (only on lock)")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "locked": True,
                "locked_by": "Reactor Controller v1.2",
                "locked_at": "2025-12-04T16:30:00",
                "lock_token": "550e8400-e29b-41d4-a716-446655440000"
            }
        }
    )
