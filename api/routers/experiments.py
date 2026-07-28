"""
Experiments router - Experimental data management.
"""

from fastapi import APIRouter, Depends, UploadFile, File, Query, HTTPException
from ..models.requests import (
    AddExperimentRequest, 
    AddExperimentsBatchRequest, 
    InitialDesignRequest,
    OptimalDesignInfoRequest,
    OptimalDesignRequest,
    StageExperimentRequest,
    StageExperimentsBatchRequest,
    CompleteStagedExperimentsRequest,
    QueueStageRequest,
    QueueCompleteRequest,
    QueueFailRequest,
    SetObjectiveMetadataRequest,
)
from ..models.responses import (
    ExperimentResponse, 
    ExperimentsListResponse, 
    ExperimentsSummaryResponse,
    InitialDesignResponse,
    OptimalDesignInfoResponse,
    OptimalDesignResponse,
    StagedExperimentResponse,
    StagedExperimentsListResponse,
    StagedExperimentsClearResponse,
    StagedExperimentsCompletedResponse,
    QueueItemResponse,
    QueueListResponse,
    QueuePurgeResponse,
    ObjectiveMetadataResponse,
    ConfigChangesResponse,
    ConfigChangeEntry,
    ProvenanceRecordResponse,
    ProvenanceListResponse,
)
from ..dependencies import get_session
from ..middleware.error_handlers import NoVariablesError
from .websocket import broadcast_to_session
from alchemist_core.session import OptimizationSession
import logging
import pandas as pd
import tempfile
import os
from typing import Optional

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/{session_id}/experiments", response_model=ExperimentResponse)
async def add_experiment(
    session_id: str,
    experiment: AddExperimentRequest,
    auto_train: bool = Query(False, description="Auto-train model after adding data"),
    training_backend: Optional[str] = Query(None, description="Model backend (sklearn/botorch)"),
    training_kernel: Optional[str] = Query(None, description="Kernel type (rbf/matern)"),
    session: OptimizationSession = Depends(get_session)
):
    """
    Add a single experiment to the dataset.
    
    The experiment must include values for all defined variables.
    Output value is optional for candidate experiments.
    
    Args:
        auto_train: If True, retrain model after adding data
        training_backend: Model backend (uses last if None)
        training_kernel: Kernel type (uses last or 'rbf' if None)
    """
    # Check if variables are defined
    if len(session.search_space.variables) == 0:
        raise NoVariablesError("No variables defined. Add variables to search space first.")
    
    session.add_experiment(
        inputs=experiment.inputs,
        output=experiment.output,
        noise=experiment.noise,
        iteration=experiment.iteration,
        reason=experiment.reason
    )
    
    # Manual entries (no staged suggestion) still get a provenance record so
    # "what was suggested?" is uniformly answerable (answer: nothing).
    import uuid as _uuid
    from types import SimpleNamespace
    from alchemist_core.data.experiment_manager import PROVENANCE_COL
    manual_id = str(_uuid.uuid4())
    try:
        # Record provenance BEFORE stamping the column, so a row can never carry
        # a ProvenanceId that has no matching record.
        session._record_provenance(
            SimpleNamespace(id=manual_id, inputs=None, reason="Manual"),
            dict(experiment.inputs),
            experiment.output,
            experiment.noise,
        )
        session.experiment_manager.df.loc[
            session.experiment_manager.df.index[-1], PROVENANCE_COL
        ] = manual_id
    except Exception as e:
        logger.warning(f"Failed to record manual provenance: {e}")

    n_experiments = len(session.experiment_manager.df)
    logger.info(f"Added experiment to session {session_id}. Total: {n_experiments}")
    
    # Auto-train if requested (need at least 5 points to train)
    model_trained = False
    training_metrics = None
    
    if auto_train and n_experiments >= 5:
        try:
            # Use previous config or provided config
            backend = training_backend or (session.model_backend if session.model else "sklearn")
            kernel = training_kernel or "rbf"
            
            # Note: Input/output transforms are now automatically applied by core Session.train_model()
            # for BoTorch models. No need to specify them here unless overriding defaults.
            result = session.train_model(backend=backend, kernel=kernel)
            model_trained = True
            metrics = result.get("metrics", {})
            hyperparameters = result.get("hyperparameters", {})
            training_metrics = {
                "rmse": metrics.get("rmse"),
                "r2": metrics.get("r2"),
                "backend": backend
            }
            logger.info(f"Auto-trained model for session {session_id}: {training_metrics}")
            
            # Record in audit log if this is an optimization iteration
            if experiment.iteration is not None and experiment.iteration > 0:
                session.audit_log.lock_model(
                    backend=backend,
                    kernel=kernel,
                    hyperparameters=hyperparameters,
                    cv_metrics=metrics,
                    iteration=experiment.iteration,
                    notes=f"Auto-trained after iteration {experiment.iteration}"
                )
        except Exception as e:
            logger.error(f"Auto-train failed for session {session_id}: {e}")
            # Don't fail the whole request, just log it
    
    # Broadcast experiment update to WebSocket clients
    await broadcast_to_session(session_id, {
        "event": "experiments_updated",
        "n_experiments": n_experiments
    })
    if model_trained:
        await broadcast_to_session(session_id, {
            "event": "model_trained",
            "metrics": training_metrics
        })

    return ExperimentResponse(
        message="Experiment added successfully",
        n_experiments=n_experiments,
        model_trained=model_trained,
        training_metrics=training_metrics
    )


@router.post("/{session_id}/experiments/batch", response_model=ExperimentResponse)
async def add_experiments_batch(
    session_id: str,
    batch: AddExperimentsBatchRequest,
    auto_train: bool = Query(False, description="Auto-train model after adding data"),
    training_backend: Optional[str] = Query(None, description="Model backend (sklearn/botorch)"),
    training_kernel: Optional[str] = Query(None, description="Kernel type (rbf/matern)"),
    session: OptimizationSession = Depends(get_session)
):
    """
    Add multiple experiments at once.
    
    Useful for bulk data import or initialization.
    
    Args:
        auto_train: If True, retrain model after adding data
        training_backend: Model backend (uses last if None)
        training_kernel: Kernel type (uses last or 'rbf' if None)
    """
    # Check if variables are defined
    if len(session.search_space.variables) == 0:
        raise NoVariablesError("No variables defined. Add variables to search space first.")
    
    for exp in batch.experiments:
        session.add_experiment(
            inputs=exp.inputs,
            output=exp.output,
            noise=exp.noise
        )
    
    n_experiments = len(session.experiment_manager.df)
    logger.info(f"Added {len(batch.experiments)} experiments to session {session_id}. Total: {n_experiments}")
    
    # Auto-train if requested
    model_trained = False
    training_metrics = None
    
    if auto_train and n_experiments >= 5:  # Minimum data for training
        try:
            backend = training_backend or (session.model_backend if session.model else "sklearn")
            kernel = training_kernel or "rbf"
            
            result = session.train_model(backend=backend, kernel=kernel)
            model_trained = True
            metrics = result.get("metrics", {})
            training_metrics = {
                "rmse": metrics.get("rmse"),
                "r2": metrics.get("r2"),
                "backend": backend
            }
            logger.info(f"Auto-trained model for session {session_id}: {training_metrics}")
        except Exception as e:
            logger.error(f"Auto-train failed for session {session_id}: {e}")
    
    # Broadcast experiment update to WebSocket clients
    await broadcast_to_session(session_id, {
        "event": "experiments_updated",
        "n_experiments": n_experiments
    })
    if model_trained:
        await broadcast_to_session(session_id, {
            "event": "model_trained",
            "metrics": training_metrics
        })

    return ExperimentResponse(
        message=f"Added {len(batch.experiments)} experiments successfully",
        n_experiments=n_experiments,
        model_trained=model_trained,
        training_metrics=training_metrics
    )


@router.post("/{session_id}/initial-design", response_model=InitialDesignResponse)
async def generate_initial_design(
    session_id: str,
    request: InitialDesignRequest,
    session: OptimizationSession = Depends(get_session)
):
    """
    Generate initial experimental design (DoE) for autonomous operation.

    **Space-filling methods** (require n_points):
    - random, lhs, sobol, halton, hammersly

    **Classical RSM methods** (run count from design structure):
    - full_factorial, fractional_factorial, ccd, box_behnken

    **Screening methods** (run count from design structure):
    - plackett_burman (2-level main-effect screening, continuous only)
    - gsd (Generalized Subset Design, supports mixed categorical/continuous)

    Returns list of experiments (input combinations) to evaluate.
    """
    # Check if variables are defined
    if len(session.search_space.variables) == 0:
        raise NoVariablesError("No variables defined. Add variables to search space first.")

    # Build kwargs, only passing n_points if provided
    kwargs = dict(
        method=request.method,
        random_seed=request.random_seed,
        lhs_criterion=request.lhs_criterion,
        n_levels=request.n_levels,
        n_center=request.n_center,
        generators=request.generators,
        ccd_alpha=request.ccd_alpha,
        ccd_face=request.ccd_face,
        gsd_reduction=request.gsd_reduction,
    )
    if request.n_points is not None:
        kwargs['n_points'] = request.n_points

    design_points = session.generate_initial_design(**kwargs)

    # Get design metadata for classical methods
    from alchemist_core.utils.doe import get_design_info
    design_info = get_design_info(
        method=request.method,
        search_space=session.search_space,
        n_levels=request.n_levels,
        n_center=request.n_center,
        generators=request.generators,
        ccd_alpha=request.ccd_alpha,
        ccd_face=request.ccd_face,
        gsd_reduction=request.gsd_reduction,
    )

    logger.info(f"Generated {len(design_points)} initial design points using {request.method} for session {session_id}")

    return InitialDesignResponse(
        points=design_points,
        method=request.method,
        n_points=len(design_points),
        design_info=design_info
    )


@router.post("/{session_id}/optimal-design/info", response_model=OptimalDesignInfoResponse)
async def get_optimal_design_info(
    session_id: str,
    request: OptimalDesignInfoRequest,
    session: OptimizationSession = Depends(get_session)
):
    """
    Preview optimal design model terms and recommended run count.

    Dry-run inspection without running the exchange algorithm.
    Use this to verify your model specification and choose n_points
    before calling the generate endpoint.

    Specify either **model_type** (shortcut) or **effects** (explicit list),
    not both.
    """
    if len(session.search_space.variables) == 0:
        raise NoVariablesError("No variables defined. Add variables to search space first.")

    info = session.get_optimal_design_info(
        model_type=request.model_type,
        effects=request.effects,
    )

    return OptimalDesignInfoResponse(
        model_terms=info["model_terms"],
        p_columns=info["p_columns"],
        n_points_minimum=info["n_points_minimum"],
        n_points_recommended=info["n_points_recommended"],
    )


@router.post("/{session_id}/optimal-design", response_model=OptimalDesignResponse)
async def generate_optimal_design(
    session_id: str,
    request: OptimalDesignRequest,
    session: OptimizationSession = Depends(get_session)
):
    """
    Generate a statistically optimal experimental design (D/A/I-optimal).

    Specify either **model_type** (shortcut) or **effects** (explicit list),
    not both.  Specify either **n_points** (absolute) or **p_multiplier**
    (relative to model columns), not both.

    Returns the generated design points along with design quality metrics
    (D_eff, A_eff, score, model_terms, etc.).
    """
    if len(session.search_space.variables) == 0:
        raise NoVariablesError("No variables defined. Add variables to search space first.")

    try:
        points, info = session.generate_optimal_design(
            model_type=request.model_type,
            effects=request.effects,
            n_points=request.n_points,
            p_multiplier=request.p_multiplier,
            criterion=request.criterion,
            algorithm=request.algorithm,
            n_levels=request.n_levels,
            max_iter=request.max_iter,
            random_seed=request.random_seed,
        )

        logger.info(
            f"Generated optimal design: {len(points)} runs, "
            f"D_eff={info.get('D_eff', 0):.1f}%, criterion={request.criterion} "
            f"for session {session_id}"
        )

        return OptimalDesignResponse(
            points=points,
            n_points=len(points),
            design_info=info,
        )
    except (ValueError, RuntimeError, ImportError):
        raise
    except Exception as e:
        logger.error(f"Optimal design generation failed for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Optimal design generation failed. Check server logs for details.")


@router.get("/{session_id}/experiments", response_model=ExperimentsListResponse)
async def list_experiments(
    session_id: str,
    session: OptimizationSession = Depends(get_session)
):
    """
    Get all experiments in the dataset.
    
    Returns complete experimental data including inputs, outputs, and noise values.
    """
    df = session.experiment_manager.get_data()
    experiments = df.to_dict('records')
    
    return ExperimentsListResponse(
        experiments=experiments,
        n_experiments=len(experiments)
    )


@router.post("/{session_id}/experiments/preview")
async def preview_csv_columns(
    session_id: str,
    file: UploadFile = File(...),
    session: OptimizationSession = Depends(get_session)
):
    """
    Preview CSV file columns before uploading to check for target columns.
    
    Returns:
        - available_columns: List of all columns in CSV
        - has_output: Whether 'Output' column exists
        - recommended_target: Suggested target column if 'Output' missing
    """
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Read CSV to get column names
        df = pd.read_csv(tmp_path)
        columns = df.columns.tolist()
        
        # Check for 'Output' column
        has_output = 'Output' in columns
        
        # Filter out metadata columns
        metadata_cols = {'Iteration', 'Reason', 'Noise'}
        available_targets = [col for col in columns if col not in metadata_cols]
        
        # Recommend target column
        recommended = None
        if not has_output:
            # Look for common target column names
            common_names = ['output', 'y', 'target', 'yield', 'response']
            for name in common_names:
                if name in [col.lower() for col in available_targets]:
                    recommended = [col for col in available_targets if col.lower() == name][0]
                    break
            
            # If no common name found, use first numeric column
            if not recommended and available_targets:
                # Check if first available column is numeric
                if pd.api.types.is_numeric_dtype(df[available_targets[0]]):
                    recommended = available_targets[0]
        
        return {
            "columns": columns,
            "available_targets": available_targets,
            "has_output": has_output,
            "recommended_target": recommended,
            "n_rows": len(df)
        }
        
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.post("/{session_id}/experiments/upload")
async def upload_experiments(
    session_id: str,
    file: UploadFile = File(...),
    target_columns: str = "Output",  # Note: API accepts string, will be normalized by Session API
    session: OptimizationSession = Depends(get_session)
):
    """
    Upload experimental data from CSV file.
    
    The CSV should have columns matching the variable names,
    plus target column(s) (default: "Output") and optional noise column ("Noise").
    
    Args:
        target_columns: Target column name (single-objective) or comma-separated names (multi-objective).
                       Examples: "Output", "yield", "yield,selectivity"
    """
    # Check if variables are defined
    if len(session.search_space.variables) == 0:
        raise NoVariablesError("No variables defined. Add variables to search space first.")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Parse target_columns (handle comma-separated for future multi-objective support)
        target_cols_parsed = target_columns.split(',') if ',' in target_columns else target_columns
        
        # Load data using session's load_data method
        session.load_data(tmp_path, target_columns=target_cols_parsed)
        
        n_experiments = len(session.experiment_manager.df)
        logger.info(f"Loaded {n_experiments} experiments from CSV for session {session_id}")

        # Broadcast experiment update to WebSocket clients
        await broadcast_to_session(session_id, {
            "event": "experiments_updated",
            "n_experiments": n_experiments
        })

        return {
            "message": f"Loaded {n_experiments} experiments successfully",
            "n_experiments": n_experiments
        }

    except (ValueError, RuntimeError):
        raise
    except Exception as e:
        logger.error(f"Experiment upload failed for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Experiment upload failed. Check server logs for details.")
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.get("/{session_id}/experiments/summary", response_model=ExperimentsSummaryResponse)
async def get_experiments_summary(
    session_id: str,
    session: OptimizationSession = Depends(get_session)
):
    """
    Get statistical summary of experimental data.
    
    Returns sample size, target variable statistics, and feature information.
    """
    return session.get_data_summary()


# ============================================================
# Staged Experiments Endpoints
# ============================================================

@router.post("/{session_id}/experiments/staged", response_model=StagedExperimentResponse,
             deprecated=True)
async def stage_experiment(
    session_id: str,
    request: StageExperimentRequest,
    session: OptimizationSession = Depends(get_session)
):
    """
    Stage an experiment for later execution.
    
    Staged experiments are stored in a queue awaiting evaluation.
    This is useful for autonomous workflows where the controller
    needs to track which experiments are pending execution.
    
    Use GET /experiments/staged to retrieve staged experiments,
    and POST /experiments/staged/complete to finalize them with outputs.
    """
    # Check if variables are defined
    if len(session.search_space.variables) == 0:
        raise NoVariablesError("No variables defined. Add variables to search space first.")
    
    # Add reason metadata if provided
    inputs_with_meta = dict(request.inputs)
    if request.reason:
        inputs_with_meta['_reason'] = request.reason
    
    session.add_staged_experiment(inputs_with_meta)
    
    n_staged = len(session.get_staged_experiments())
    logger.info(f"Staged experiment for session {session_id}. Total staged: {n_staged}")
    
    return StagedExperimentResponse(
        message="Experiment staged successfully",
        n_staged=n_staged,
        staged_inputs=request.inputs
    )


@router.post("/{session_id}/experiments/staged/batch", response_model=StagedExperimentsListResponse,
             deprecated=True)
async def stage_experiments_batch(
    session_id: str,
    request: StageExperimentsBatchRequest,
    session: OptimizationSession = Depends(get_session)
):
    """
    Stage multiple experiments at once.
    
    Useful after acquisition functions suggest multiple points for parallel execution.
    The `reason` parameter is stored as metadata and will be used when completing
    the experiments (recorded in the 'Reason' column of the experiment data).
    """
    # Check if variables are defined
    if len(session.search_space.variables) == 0:
        raise NoVariablesError("No variables defined. Add variables to search space first.")
    
    for inputs in request.experiments:
        inputs_with_meta = dict(inputs)
        if request.reason:
            inputs_with_meta['_reason'] = request.reason
        session.add_staged_experiment(inputs_with_meta)
    
    logger.info(f"Staged {len(request.experiments)} experiments for session {session_id}. Total staged: {len(session.get_staged_experiments())}")
    
    # Return clean experiments (without metadata) for client use
    return StagedExperimentsListResponse(
        experiments=request.experiments,  # Return the original clean inputs
        n_staged=len(session.get_staged_experiments()),
        reason=request.reason
    )


@router.get("/{session_id}/experiments/staged", response_model=StagedExperimentsListResponse,
            deprecated=True)
async def get_staged_experiments(
    session_id: str,
    session: OptimizationSession = Depends(get_session)
):
    """DEPRECATED: use GET /experiments/queue for full per-item state.

    Returns pending staged experiments. Per-item reasons are now available in
    the `reasons` list (aligned with `experiments`); the scalar `reason` field
    remains the first item's value for backward compatibility.
    """
    pending = session.queue.pending_items()
    clean_experiments = [dict(i.inputs) for i in pending]
    reasons = [i.reason for i in pending]
    ids = [i.id for i in pending]
    first_reason = reasons[0] if reasons else None
    return StagedExperimentsListResponse(
        experiments=clean_experiments,
        n_staged=len(pending),
        reason=first_reason,
        reasons=reasons,
        ids=ids,
    )


@router.delete("/{session_id}/experiments/staged", response_model=StagedExperimentsClearResponse,
               deprecated=True)
async def clear_staged_experiments(
    session_id: str,
    session: OptimizationSession = Depends(get_session)
):
    """
    Clear all staged experiments.
    
    Use this to reset the staging queue if experiments were cancelled
    or need to be regenerated.
    """
    n_cleared = session.clear_staged_experiments()
    logger.info(f"Cleared {n_cleared} staged experiments for session {session_id}")
    
    return StagedExperimentsClearResponse(
        message="Staged experiments cleared",
        n_cleared=n_cleared
    )


@router.post("/{session_id}/experiments/staged/complete", response_model=StagedExperimentsCompletedResponse,
             deprecated=True)
async def complete_staged_experiments(
    session_id: str,
    request: CompleteStagedExperimentsRequest,
    auto_train: bool = Query(False, description="Auto-train model after adding data"),
    training_backend: Optional[str] = Query(None, description="Model backend (sklearn/botorch)"),
    training_kernel: Optional[str] = Query(None, description="Kernel type (rbf/matern)"),
    session: OptimizationSession = Depends(get_session)
):
    """
    Complete staged experiments by providing output values.
    
    This pairs the staged experiment inputs with the provided outputs,
    adds them to the experiment dataset, and clears the staging queue.
    
    The number of outputs must match the number of staged experiments.
    Outputs should be in the same order as the staged experiments were added.
    
    Args:
        auto_train: If True, retrain model after adding data
        training_backend: Model backend (uses last if None)
        training_kernel: Kernel type (uses last or 'rbf' if None)
    """
    # New-model guard: block only on RUNNING items. The batch path completes
    # exactly the pending items in order, so a running (in-flight) item would be
    # silently skipped and break the 1:1 output-count contract -- that's the one
    # genuinely ambiguous case. Terminal (done/failed) items are inert to this
    # path (they aren't in pending_items()), so they must NOT block: a pure
    # legacy consumer doing repeated stage->complete cycles leaves 'done' items
    # in the queue and would otherwise be permanently 409'd with no legacy way
    # to purge them.
    running = [i for i in session.queue.list() if i.status == "running"]
    if running:
        raise HTTPException(
            status_code=409,
            detail=("Batch complete is unavailable while items are running. "
                    "Use POST /experiments/queue/{id}/complete instead."),
        )
    staged = session.get_staged_experiments()
    
    if len(staged) == 0:
        return StagedExperimentsCompletedResponse(
            message="No staged experiments to complete",
            n_added=0,
            n_experiments=len(session.experiment_manager.df),
            model_trained=False
        )
    
    if len(request.outputs) != len(staged):
        raise ValueError(
            f"Number of outputs ({len(request.outputs)}) must match "
            f"number of staged experiments ({len(staged)})"
        )
    
    # Use the core Session method to move staged experiments to dataset
    n_added = session.move_staged_to_experiments(
        outputs=request.outputs,
        noises=request.noises,
        iteration=request.iteration,
        reason=request.reason
    )
    
    n_experiments = len(session.experiment_manager.df)
    logger.info(f"Completed {n_added} staged experiments for session {session_id}. Total: {n_experiments}")
    
    # Auto-train if requested
    model_trained = False
    training_metrics = None
    
    if auto_train and n_experiments >= 5:
        try:
            backend = training_backend or (session.model_backend if session.model else "sklearn")
            kernel = training_kernel or "rbf"
            
            result = session.train_model(backend=backend, kernel=kernel)
            model_trained = True
            metrics = result.get("metrics", {})
            training_metrics = {
                "rmse": metrics.get("rmse"),
                "r2": metrics.get("r2"),
                "backend": backend
            }
            logger.info(f"Auto-trained model for session {session_id}: {training_metrics}")
        except Exception as e:
            logger.error(f"Auto-train failed for session {session_id}: {e}")
    
    # Broadcast experiment update to WebSocket clients
    await broadcast_to_session(session_id, {
        "event": "experiments_updated",
        "n_experiments": n_experiments
    })
    if model_trained:
        await broadcast_to_session(session_id, {
            "event": "model_trained",
            "metrics": training_metrics
        })

    return StagedExperimentsCompletedResponse(
        message="Staged experiments completed and added to dataset",
        n_added=n_added,
        n_experiments=n_experiments,
        model_trained=model_trained,
        training_metrics=training_metrics
    )


def _item_response(item) -> QueueItemResponse:
    return QueueItemResponse(**item.to_dict())


def _item_event(item) -> dict:
    """WebSocket payload for a per-item transition.

    Mirrors the core EventEmitter's ``queue_item_updated`` contract (item_id +
    status + reason + output + error) so a subscribed client can update a row
    without a follow-up GET.
    """
    return {
        "event": "queue_item_updated",
        "item_id": item.id,
        "status": item.status,
        "reason": item.reason,
        "output": item.output,
        "error": item.error,
    }


def _list_response(session) -> QueueListResponse:
    items = session.queue.list()
    counts = {"pending": 0, "running": 0, "done": 0, "failed": 0}
    for i in items:
        counts[i.status] = counts.get(i.status, 0) + 1
    return QueueListResponse(
        items=[_item_response(i) for i in items],
        n_pending=counts["pending"], n_running=counts["running"],
        n_done=counts["done"], n_failed=counts["failed"],
    )


@router.post("/{session_id}/experiments/queue", response_model=QueueListResponse)
async def stage_queue_items(session_id: str, request: QueueStageRequest,
                            session: OptimizationSession = Depends(get_session)):
    if len(session.search_space.variables) == 0:
        raise NoVariablesError("No variables defined. Add variables first.")
    for it in request.items:
        session.queue.stage(dict(it.inputs), reason=it.reason)
    await broadcast_to_session(session_id, {"event": "queue_updated"})
    return _list_response(session)


@router.get("/{session_id}/experiments/queue", response_model=QueueListResponse)
async def list_queue(session_id: str, status: Optional[str] = Query(None),
                     session: OptimizationSession = Depends(get_session)):
    resp = _list_response(session)
    if status:
        resp.items = [i for i in resp.items if i.status == status]
    return resp


@router.post("/{session_id}/experiments/queue/purge", response_model=QueuePurgeResponse)
async def purge_queue(session_id: str, session: OptimizationSession = Depends(get_session)):
    n = session.queue.purge()
    await broadcast_to_session(session_id, {"event": "queue_updated"})
    return QueuePurgeResponse(n_purged=n)


@router.get("/{session_id}/experiments/queue/{item_id}", response_model=QueueItemResponse)
async def get_queue_item(session_id: str, item_id: str,
                         session: OptimizationSession = Depends(get_session)):
    item = session.queue.get(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Unknown queue item: {item_id}")
    return _item_response(item)


@router.post("/{session_id}/experiments/queue/{item_id}/start", response_model=QueueItemResponse)
async def start_queue_item(session_id: str, item_id: str,
                           session: OptimizationSession = Depends(get_session)):
    # No pre-check: the queue distinguishes unknown (KeyError -> 404) from
    # illegal transition (ValueError -> 409). Catching both here is race-free
    # against a concurrent delete/purge by another consumer.
    try:
        item = session.queue.start(item_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown queue item: {item_id}")
    except ValueError as e:
        # 409: user-facing domain message (illegal transition), safe to surface.
        raise HTTPException(status_code=409, detail=str(e))
    await broadcast_to_session(session_id, _item_event(item))
    return _item_response(item)


@router.post("/{session_id}/experiments/queue/{item_id}/complete", response_model=QueueItemResponse)
async def complete_queue_item(session_id: str, item_id: str, request: QueueCompleteRequest,
                              auto_train: bool = False,
                              session: OptimizationSession = Depends(get_session)):
    # Completion writes a single scalar objective into the dataset. Multi-target
    # completion is not yet supported end-to-end (ExperimentManager.add_experiment
    # records one target column), so reject a multi-output completion explicitly
    # rather than silently corrupting the dataset.
    if len(request.outputs) != 1:
        raise HTTPException(
            status_code=400,
            detail=(
                f"complete expects exactly one output value, got "
                f"{len(request.outputs)}. Multi-objective completion via the "
                f"work queue is not supported yet."
            ),
        )
    if request.noise is not None and len(request.noise) != 1:
        raise HTTPException(
            status_code=400,
            detail=(
                f"complete expects at most one noise value, got "
                f"{len(request.noise)}."
            ),
        )
    if request.expected_objective_label and not request.force:
        try:
            session.check_objective_label(request.expected_objective_label)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))
    output = request.outputs[0]
    noise = request.noise[0] if request.noise is not None else None
    try:
        item = session.queue.complete(
            item_id, output=output, noise=noise,
            actual_inputs=request.actual_inputs,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown queue item: {item_id}")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    # Auto-train if requested (mirrors the direct add-experiment path so the
    # "retrain model" control works when recording results via the work queue).
    if auto_train and len(session.experiment_manager.df) >= 5:
        try:
            backend = session.model_backend if session.model else "sklearn"
            session.train_model(backend=backend, kernel="rbf")
            logger.info(f"Auto-trained model after completing queue item {item_id}")
        except Exception as e:
            logger.error(f"Auto-train failed after queue completion for session {session_id}: {e}")

    await broadcast_to_session(session_id, _item_event(item))
    await broadcast_to_session(session_id, {"event": "experiments_updated",
                                            "n_experiments": len(session.experiment_manager.df)})
    return _item_response(item)


@router.post("/{session_id}/experiments/queue/{item_id}/fail", response_model=QueueItemResponse)
async def fail_queue_item(session_id: str, item_id: str, request: QueueFailRequest,
                          session: OptimizationSession = Depends(get_session)):
    try:
        item = session.queue.fail(item_id, request.error)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown queue item: {item_id}")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    await broadcast_to_session(session_id, _item_event(item))
    return _item_response(item)


@router.delete("/{session_id}/experiments/queue/{item_id}", response_model=QueueListResponse)
async def delete_queue_item(session_id: str, item_id: str,
                            session: OptimizationSession = Depends(get_session)):
    try:
        session.queue.delete(item_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown queue item: {item_id}")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    await broadcast_to_session(session_id, {"event": "queue_updated"})
    return _list_response(session)


@router.get("/{session_id}/experiments/provenance", response_model=ProvenanceListResponse)
async def list_provenance(session_id: str,
                          session: OptimizationSession = Depends(get_session)):
    records = session.get_provenance()
    return ProvenanceListResponse(records=records, n_records=len(records))


@router.get("/{session_id}/experiments/provenance/{provenance_id}",
            response_model=ProvenanceRecordResponse)
async def get_provenance_record(session_id: str, provenance_id: str,
                                session: OptimizationSession = Depends(get_session)):
    for r in session.get_provenance():
        if r["id"] == provenance_id:
            return r
    raise HTTPException(status_code=404, detail=f"Unknown provenance id: {provenance_id}")


@router.get("/{session_id}/objective-metadata", response_model=ObjectiveMetadataResponse)
async def get_objective_metadata(session_id: str,
                                 session: OptimizationSession = Depends(get_session)):
    return ObjectiveMetadataResponse(metadata=session.get_objective_metadata())


@router.put("/{session_id}/objective-metadata", response_model=ObjectiveMetadataResponse)
async def set_objective_metadata(session_id: str, request: SetObjectiveMetadataRequest,
                                 session: OptimizationSession = Depends(get_session)):
    session.set_objective_metadata(request.metadata)
    return ObjectiveMetadataResponse(metadata=session.get_objective_metadata())


@router.get("/{session_id}/audit/config-changes", response_model=ConfigChangesResponse)
async def get_config_changes(session_id: str,
                             session: OptimizationSession = Depends(get_session)):
    """Return timestamped mid-campaign optimizer-config changes (provenance)."""
    entries = session.audit_log.get_entries("config_changed")
    changes = [
        ConfigChangeEntry(
            timestamp=e.timestamp,
            component=e.parameters.get("component", ""),
            old=e.parameters.get("old", {}),
            new=e.parameters.get("new", {}),
            iteration=e.parameters.get("iteration"),
        )
        for e in entries
    ]
    return ConfigChangesResponse(changes=changes)
