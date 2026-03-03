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
    CompleteStagedExperimentsRequest
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
    StagedExperimentsCompletedResponse
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

@router.post("/{session_id}/experiments/staged", response_model=StagedExperimentResponse)
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


@router.post("/{session_id}/experiments/staged/batch", response_model=StagedExperimentsListResponse)
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


@router.get("/{session_id}/experiments/staged", response_model=StagedExperimentsListResponse)
async def get_staged_experiments(
    session_id: str,
    session: OptimizationSession = Depends(get_session)
):
    """
    Get all staged experiments awaiting execution.
    
    Returns the list of experiments that have been queued but not yet
    completed with output values. The response includes:
    - experiments: Clean variable inputs only (no metadata)
    - reason: The strategy/reason for these experiments (if provided when staging)
    """
    staged = session.get_staged_experiments()
    
    # Extract reason from first experiment (if present) and clean all experiments
    reason = None
    clean_experiments = []
    for exp in staged:
        if '_reason' in exp and reason is None:
            reason = exp['_reason']
        # Return only variable values, not metadata
        clean_exp = {k: v for k, v in exp.items() if not k.startswith('_')}
        clean_experiments.append(clean_exp)
    
    return StagedExperimentsListResponse(
        experiments=clean_experiments,
        n_staged=len(staged),
        reason=reason
    )


@router.delete("/{session_id}/experiments/staged", response_model=StagedExperimentsClearResponse)
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


@router.post("/{session_id}/experiments/staged/complete", response_model=StagedExperimentsCompletedResponse)
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
