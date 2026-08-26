"""
Pydantic request models for API endpoints.
"""

from pydantic import BaseModel, Field, ConfigDict, field_validator
from pydantic_core import PydanticCustomError
from typing import ClassVar, List, Dict, Any, Optional, Literal, Union


# ============================================================
# LLM / AI-assisted design models
# ============================================================

class LLMProviderConfig(BaseModel):
    """Configuration for a structuring LLM provider (OpenAI or Ollama)."""
    provider: Literal["openai", "ollama"] = Field(
        ..., description="Provider identifier"
    )
    model: str = Field(
        ..., description="Model name, e.g. 'gpt-4o', 'gpt-4.1', 'llama3.2'"
    )
    api_key: Optional[str] = Field(
        None, description="API key (required for openai; omit for ollama)"
    )
    base_url: Optional[str] = Field(
        None, description="Base URL override (for custom Ollama installs)"
    )


class EdisonConfig(BaseModel):
    """Optional Edison Scientific literature search configuration."""
    api_key: Optional[str] = Field(
        None,
        description=(
            "Edison platform API key. If None, the SDK uses the EDISON_API_KEY env var."
        ),
    )
    job_type: Literal["literature", "literature_high", "precedent"] = Field(
        "literature",
        description=(
            "'literature' — standard PaperQA3 search; "
            "'literature_high' — high-reasoning mode; "
            "'precedent' — HasAnyone-style precedent search"
        ),
    )
    timeout_secs: Optional[int] = Field(
        None,
        ge=60,
        le=3600,
        description=(
            "Maximum seconds to wait for the Edison response (default 1200 = 20 min). "
            "If the task is not complete by this deadline, the search is abandoned and "
            "the structuring model proceeds without literature context."
        ),
    )
    force_refresh: bool = Field(
        False,
        description=(
            "If True, ignore any cached result and re-submit a fresh search to Edison, "
            "even if a completed result is already stored for this query."
        ),
    )


class SuggestEffectsRequest(BaseModel):
    """Request body for POST /api/v1/llm/suggest-effects/{session_id}."""
    structuring_provider: LLMProviderConfig = Field(
        ..., description="LLM used to extract the structured effects list"
    )
    edison_config: Optional[EdisonConfig] = Field(
        None,
        description=(
            "If provided, Edison Scientific is queried first for grounded "
            "literature context; its cited answer is then fed to the structuring model."
        ),
    )
    system_context: str = Field(
        ...,
        description=(
            "Free-text description of the experimental system and optimization target, "
            "e.g. 'Fischer-Tropsch synthesis over supported metal catalysts, "
            "maximizing C5+ selectivity at 250 °C and 20 bar.'"
        ),
    )


# ============================================================
# Shared field validation
# ============================================================

class AddressableNameRequest(BaseModel):
    """Mixin for request models whose ``name`` becomes a URL path segment.

    Variables and constraints are both addressed by name --
    ``DELETE /sessions/{session_id}/variables/{name}`` and
    ``.../constraints/{name}`` -- so a name that cannot survive one URL path
    segment yields a resource that can be created but never removed. Three
    forms do not survive:

    - ``""`` collapses the path to ``.../variables/``, which is a different
      route (405/404).
    - anything containing ``/`` splits into two segments. Percent-encoding it
      does not help: routing matches on the decoded path (404).
    - ``.`` and ``..`` are dot segments. RFC 3986 section 5.2.4 removal is
      performed by clients and proxies *before the request is sent*, so ``..``
      does not merely fail to match. ``DELETE .../variables/..`` is rewritten
      to ``DELETE .../sessions/{session_id}`` in transit, lands on
      session-delete, returns 204, and destroys the entire session -- every
      variable, every experiment and the trained model -- while reporting
      success. Reproduced end to end against a live uvicorn server, and
      directly: ``httpx.URL(".../sessions/abc/variables/..").path`` is
      ``"/api/v1/sessions/abc"``.

    The rule is deliberately narrow: only what provably breaks addressing is
    rejected. Spaces, unicode, ``%``, ``...``, ``.hidden`` and operator-bearing
    punctuation are all legitimate in a human-readable name and all round-trip
    correctly, so all are accepted.

    PydanticCustomError rather than ValueError, on purpose. The app's
    RequestValidationError handler JSON-encodes ``exc.errors()``
    (api/middleware/error_handlers.py), and a plain ValueError raised from a
    field_validator is placed in the error ``ctx`` as a live exception object,
    which is not JSON serializable -- that would turn *every* 422 in the
    application into a 500. PydanticCustomError carries a plain dict instead.

    Subclasses set ``_name_resource`` and ``_name_collection`` so the error
    code and message name the resource the caller actually posted to; the rule
    itself is defined once, here.
    """

    # Singular noun and URL collection segment for the concrete resource.
    # Used only to build the error code and message.
    _name_resource: ClassVar[str] = "resource"
    _name_collection: ClassVar[str] = "resources"

    # check_fields=False because this mixin declares no fields of its own;
    # every model that inherits it declares ``name``.
    @field_validator("name", check_fields=False)
    @classmethod
    def _name_must_be_addressable(cls, value: Optional[str]) -> Optional[str]:
        """Reject names the DELETE route could never address.

        See the class docstring for why each form is rejected and why the rule
        stops where it does.
        """
        if value is None:
            return value
        reason = None
        if value == "":
            reason = "an empty name has no URL to address"
        elif "/" in value:
            reason = "'/' would split the name across two URL path segments"
        elif value in (".", ".."):
            reason = f"{value!r} is a URL dot segment and is resolved away before routing"
        if reason is not None:
            raise PydanticCustomError(
                f"{cls._name_resource}_name_not_addressable",
                # PydanticCustomError substitutes {key} from the context dict
                # with a plain scan, not str.format, so "{{name}}" does not
                # escape to a literal "{name}" -- it renders as the value in
                # braces. The route placeholder is written as <name> instead.
                "The {resource} name {name} cannot be addressed by "
                "DELETE .../{collection}/<name>: {reason}. The name is the "
                "identity used to delete the {resource}, so such a {resource} "
                "could never be removed.",
                {
                    "resource": cls._name_resource,
                    "collection": cls._name_collection,
                    "name": repr(value),
                    "reason": reason,
                },
            )
        return value


# ============================================================
# Variable Models
# ============================================================

class VariableRequest(AddressableNameRequest):
    """Common base for the four add/update-variable request bodies.

    Carries no fields; it exists so the addressable-name rule and its resource
    labels are attached to every variable type exactly once. A fifth variable
    type added without this base would silently reintroduce the ``..`` session
    deletion.
    """

    _name_resource: ClassVar[str] = "variable"
    _name_collection: ClassVar[str] = "variables"


class AddRealVariableRequest(VariableRequest):
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


class AddIntegerVariableRequest(VariableRequest):
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


class AddCategoricalVariableRequest(VariableRequest):
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


class AddDiscreteVariableRequest(VariableRequest):
    """Request to add a discrete numerical variable."""
    name: str = Field(..., description="Variable name")
    type: Literal["discrete"] = Field(default="discrete", description="Variable type")
    allowed_values: List[float] = Field(..., min_length=2, description="List of allowed numeric values (at least 2)")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    description: Optional[str] = Field(None, description="Variable description")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "SAR",
                "type": "discrete",
                "allowed_values": [80, 280],
                "unit": "-",
                "description": "Silicon-to-aluminium ratio (only synthesizable at specific values)"
            }
        }
    )


# Union type for any variable request
AddVariableRequest = Union[
    AddRealVariableRequest,
    AddIntegerVariableRequest,
    AddCategoricalVariableRequest,
    AddDiscreteVariableRequest
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
        "full_factorial", "fractional_factorial", "ccd", "box_behnken",
        "plackett_burman", "gsd"
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
    gsd_reduction: int = Field(
        default=2, ge=2, le=10,
        description="GSD reduction factor (larger = fewer runs)"
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
# Optimal Design Models
# ============================================================

class OptimalDesignInfoRequest(BaseModel):
    """Request to preview optimal design model terms and recommended run count.

    Performs a dry-run inspection without running the exchange algorithm.
    Specify either model_type (shortcut) or effects (explicit), not both.
    """
    model_type: Optional[Literal["linear", "interaction", "quadratic"]] = Field(
        None, description="Shortcut model type"
    )
    effects: Optional[List[str]] = Field(
        None,
        description=(
            "Explicit effect strings using variable names. "
            "Main effects: 'Temperature'; interactions: 'Temperature*Pressure'; "
            "quadratic: 'Temperature**2'. Intercept is added automatically."
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_type": "quadratic"
            }
        }
    )


class OptimalDesignRequest(BaseModel):
    """Request to generate a statistically optimal experimental design.

    Specify either model_type (shortcut) or effects (explicit), not both.
    Specify either n_points (absolute) or p_multiplier (relative to model columns), not both.
    """
    model_type: Optional[Literal["linear", "interaction", "quadratic"]] = Field(
        None, description="Shortcut model type"
    )
    effects: Optional[List[str]] = Field(
        None,
        description=(
            "Explicit effect strings using variable names. "
            "Main effects: 'Temperature'; interactions: 'Temperature*Pressure'; "
            "quadratic: 'Temperature**2'. Intercept is added automatically."
        ),
    )
    n_points: Optional[int] = Field(
        None, ge=1, le=10000,
        description="Absolute number of experimental runs"
    )
    p_multiplier: Optional[float] = Field(
        None, ge=1.0, le=10.0,
        description="Run count as multiple of model columns p (e.g. 2.0 → 2p runs)"
    )
    criterion: Literal["D", "A", "I"] = Field(
        default="D",
        description="Optimality criterion: D (parameter estimation), A (min avg variance), I (min prediction variance)"
    )
    algorithm: Literal[
        "sequential", "simple_exchange", "fedorov", "modified_fedorov", "detmax"
    ] = Field(default="fedorov", description="Exchange algorithm")
    n_levels: int = Field(
        default=5, ge=2, le=20,
        description="Candidate grid levels per continuous variable"
    )
    max_iter: int = Field(
        default=200, ge=10, le=10000,
        description="Maximum exchange iterations"
    )
    random_seed: Optional[int] = Field(None, description="Random seed for reproducibility")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_type": "quadratic",
                "p_multiplier": 2.0,
                "criterion": "D",
                "algorithm": "fedorov"
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


class QueueStageItem(BaseModel):
    inputs: Dict[str, Union[float, int, str]] = Field(..., description="Variable values")
    reason: Optional[str] = Field(None, description="Per-item reason/strategy")


class QueueStageRequest(BaseModel):
    items: List[QueueStageItem] = Field(..., description="Items to stage")


class QueueCompleteRequest(BaseModel):
    outputs: List[float] = Field(..., description="Objective value(s); one per objective")
    noise: Optional[List[float]] = Field(None, description="Per-objective measurement uncertainty")
    iteration: Optional[int] = Field(None, description="Iteration number (auto-assigned if None)")
    actual_inputs: Optional[Dict[str, Union[float, int, str]]] = Field(
        None,
        description="Actual conditions run (for provenance). Defaults to the "
                    "staged suggested inputs when omitted.",
    )
    expected_objective_label: Optional[Dict[str, str]] = Field(
        None, description="{objective_name: label} guard; 409 on mismatch unless force")
    force: bool = Field(False, description="Override objective-label mismatch")


class QueueFailRequest(BaseModel):
    error: str = Field(..., description="Failure reason")


class SetObjectiveMetadataRequest(BaseModel):
    metadata: Dict[str, Dict[str, Optional[str]]] = Field(
        ..., description="{objective_name: {label, unit?}} opaque display strings")


class AddConstraintRequest(AddressableNameRequest):
    """Request to register a linear input constraint on the search space.

    'inequality' means sum(coeff_i * x_i) <= rhs.
    'equality'   means sum(coeff_i * x_i) == rhs.

    Coefficient variables must be numeric (real, integer, or discrete).
    """
    constraint_type: Literal["inequality", "equality"] = Field(
        ..., description="'inequality' (<= rhs) or 'equality' (== rhs)"
    )
    # Finiteness of rhs/coefficients is enforced in SearchSpace.add_constraint,
    # not here. Pydantic's allow_inf_nan=False does reject these, but the
    # resulting 422 carries the offending nan/inf in the error's ``input``
    # field, and the app's RequestValidationError handler cannot JSON-encode
    # that (Starlette renders with allow_nan=False). The request still fails
    # closed, but as an opaque "Out of range float values are not JSON
    # compliant: nan" rather than a usable message. The core check produces
    # "Constraint rhs must be finite, got nan" and covers the Python and
    # desktop-GUI callers too. See test_non_finite_* in the router tests.
    coefficients: Dict[str, float] = Field(
        ..., min_length=1,
        description="Mapping of variable name to coefficient (must be finite)"
    )
    rhs: float = Field(
        ..., description="Right-hand side value (must be finite)"
    )
    name: Optional[str] = Field(
        None, description="Optional name; auto-generated as constraint_N if omitted"
    )

    _name_resource: ClassVar[str] = "constraint"
    _name_collection: ClassVar[str] = "constraints"

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "constraint_type": "inequality",
                "coefficients": {"x1": 0.5, "x2": -1.0},
                "rhs": -10.0,
                "name": "half_plane_1",
            }
        }
    )
