"""
Visualization Router - API endpoints for model visualizations

Provides data for:
1. Contour plots (2D model predictions)
2. Parity plots (actual vs predicted)
3. Metrics plots (CV performance over training size)
4. Q-Q plots (residual calibration)
5. Calibration curves (reliability diagrams)
6. Hyperparameter info
"""
from fastapi import APIRouter, HTTPException, status, Query, Depends
from fastapi.concurrency import run_in_threadpool
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
import numpy as np
from scipy import stats
from alchemist_core.session import OptimizationSession

from api.dependencies import get_session
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["visualizations"])


# ============================================================================
# Request/Response Models
# ============================================================================

class ContourDataRequest(BaseModel):
    """Request for contour plot data"""
    x_var: str = Field(..., description="Variable name for X axis")
    y_var: str = Field(..., description="Variable name for Y axis")
    fixed_values: Dict[str, Any] = Field(default_factory=dict, description="Fixed values for other variables")
    grid_resolution: int = Field(default=50, ge=10, le=200, description="Grid resolution (NxN)")
    include_experiments: bool = Field(default=True, description="Include experimental data points")
    include_suggestions: bool = Field(default=False, description="Include next suggested points")


class ContourDataResponse(BaseModel):
    """Response with contour plot data"""
    x_var: str
    y_var: str
    x_grid: List[List[float]] = Field(..., description="X coordinate grid")
    y_grid: List[List[float]] = Field(..., description="Y coordinate grid")
    predictions: List[List[float]] = Field(..., description="Predicted values (mean)")
    uncertainties: List[List[float]] = Field(..., description="Prediction uncertainties (std)")
    experiments: Optional[Dict[str, List[float]]] = Field(None, description="Experimental data {x: [...], y: [...], output: [...]}")
    suggestions: Optional[Dict[str, List[float]]] = Field(None, description="Suggested points {x: [...], y: [...]}") 
    x_bounds: List[float] = Field(..., description="[min, max] bounds for X axis")
    y_bounds: List[float] = Field(..., description="[min, max] bounds for Y axis")
    colorbar_bounds: List[float] = Field(..., description="[min, max] bounds for predictions")


class ParityDataResponse(BaseModel):
    """Response with parity plot data"""
    y_true: List[float] = Field(..., description="Actual values from CV")
    y_pred: List[float] = Field(..., description="Predicted values from CV")
    y_std: List[float] = Field(..., description="Prediction uncertainties from CV")
    metrics: Dict[str, float] = Field(..., description="Performance metrics (rmse, mae, r2, mape)")
    bounds: List[float] = Field(..., description="[min, max] for both axes (square plot)")
    calibrated: bool = Field(..., description="Whether uncertainty is calibrated")


class MetricsDataResponse(BaseModel):
    """Response with CV metrics over training size"""
    training_sizes: List[int] = Field(..., description="Number of training samples")
    rmse: List[Optional[float]] = Field(..., description="RMSE values (null for NaN/Inf)")
    mae: List[Optional[float]] = Field(..., description="MAE values (null for NaN/Inf)")
    r2: List[Optional[float]] = Field(..., description="R² values (null for NaN/Inf)")
    mape: List[Optional[float]] = Field(..., description="MAPE values (%) (null for NaN/Inf)")


class QQPlotDataResponse(BaseModel):
    """Response with Q-Q plot data"""
    theoretical_quantiles: List[float] = Field(..., description="Standard normal quantiles")
    sample_quantiles: List[float] = Field(..., description="Standardized residual quantiles")
    z_mean: float = Field(..., description="Mean of z-scores")
    z_std: float = Field(..., description="Std dev of z-scores")
    n_samples: int = Field(..., description="Number of samples")
    bounds: List[float] = Field(..., description="[min, max] for both axes")
    calibrated: bool = Field(..., description="Whether using calibrated results")


class CalibrationCurveDataResponse(BaseModel):
    """Response with calibration curve data"""
    nominal_coverage: List[float] = Field(..., description="Nominal coverage probabilities")
    empirical_coverage: List[float] = Field(..., description="Empirical coverage fractions")
    confidence_levels: List[str] = Field(..., description="Confidence level labels (e.g., '±1.96σ (95%)')")
    nominal_probabilities: List[float] = Field(..., description="Same as nominal_coverage")
    empirical_probabilities: List[float] = Field(..., description="Same as empirical_coverage")
    n_samples: int = Field(..., description="Number of samples")
    calibrated: bool = Field(..., description="Whether using calibrated results")
    results_type: str = Field(..., description="'calibrated' or 'uncalibrated'")


class HyperparametersResponse(BaseModel):
    """Response with model hyperparameters"""
    hyperparameters: Dict[str, Any] = Field(..., description="Model hyperparameters")
    backend: str = Field(..., description="Model backend")
    kernel: str = Field(..., description="Kernel type")
    input_transform: Optional[str] = Field(None, description="Input scaling method")
    output_transform: Optional[str] = Field(None, description="Output scaling method")
    calibration_enabled: bool = Field(False, description="Whether uncertainty calibration is enabled")
    calibration_factor: Optional[float] = Field(None, description="Calibration factor if enabled")


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/{session_id}/visualizations/contour", response_model=ContourDataResponse)
async def get_contour_data(
    session_id: str,
    request: ContourDataRequest,
    session: OptimizationSession = Depends(get_session)
):
    """
    Generate 2D contour plot data for model predictions.
    
    Creates a grid over the selected X and Y variables while holding
    other variables at fixed values. Returns predictions and uncertainties
    for each grid point.
    """
    if not session.model or not session.model.is_trained:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model must be trained before generating visualizations"
        )
    
    # Validate that x_var and y_var exist in search space
    var_names = session.search_space.get_variable_names()
    if request.x_var not in var_names:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Variable '{request.x_var}' not found in search space"
        )
    if request.y_var not in var_names:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Variable '{request.y_var}' not found in search space"
        )
    if request.x_var == request.y_var:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X and Y variables must be different"
        )
    
    # Get variable bounds
    x_var_info = next(v for v in session.search_space.variables if v["name"] == request.x_var)
    y_var_info = next(v for v in session.search_space.variables if v["name"] == request.y_var)
    
    for _axis_var, _axis_info in [("x", x_var_info), ("y", y_var_info)]:
        if _axis_info["type"] not in ["real", "integer"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Contour plots only support continuous (real/integer) variables on axes. "
                       f"'{_axis_info['name']}' is type '{_axis_info['type']}'."
            )

    x_bounds = (x_var_info["min"], x_var_info["max"])
    y_bounds = (y_var_info["min"], y_var_info["max"])
    
    # Create meshgrid
    x_range = np.linspace(x_bounds[0], x_bounds[1], request.grid_resolution)
    y_range = np.linspace(y_bounds[0], y_bounds[1], request.grid_resolution)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    
    # Build prediction DataFrame with fixed values
    import pandas as pd
    grid_points = []
    
    # CRITICAL: Filter out x and y variables from fixed_values in case frontend sent them
    filtered_fixed_values = {
        k: v for k, v in request.fixed_values.items() 
        if k != request.x_var and k != request.y_var
    }
    
    for i in range(request.grid_resolution):
        for j in range(request.grid_resolution):
            point = filtered_fixed_values.copy()
            point[request.x_var] = X_grid[i, j]
            point[request.y_var] = Y_grid[i, j]
            
            # Fill in any missing variables with their midpoint
            for var in session.search_space.variables:
                if var["name"] not in point:
                    if var["type"] in ["real", "integer"]:
                        point[var["name"]] = (var["min"] + var["max"]) / 2.0
                    elif var["type"] == "categorical":
                        # Use first category as default
                        point[var["name"]] = var["values"][0]
                    elif var["type"] == "discrete":
                        _av = var["allowed_values"]
                        point[var["name"]] = _av[len(_av) // 2]
            
            grid_points.append(point)
    
    grid_df = pd.DataFrame(grid_points)
    
    # CRITICAL FIX: Reorder columns to match training data
    # The model was trained with a specific column order, we must match it.
    # Exclude metadata columns that are part of the experiments table but
    # are not model input features (e.g., Iteration, Reason, target columns, Noise).
    train_data = session.experiment_manager.get_data()
    target_cols = set(session.experiment_manager.target_columns)
    metadata_cols = {'Iteration', 'Reason', 'Noise'} | target_cols
    feature_cols = [col for col in train_data.columns if col not in metadata_cols]

    # Safely align the prediction grid to the model feature order.
    # Use reindex so missing columns (shouldn't happen) are filled with the
    # midpoint/defaults the grid already supplies; this avoids KeyError.
    grid_df = grid_df.reindex(columns=feature_cols)
    
    # IMPORTANT: The model's predict() method handles preprocessing internally
    # (including categorical encoding), so we can pass the raw DataFrame directly.
    # Run in a thread so the event loop stays free for health checks and WebSocket pings.
    try:
        predictions, uncertainties = await run_in_threadpool(
            session.model.predict, grid_df, return_std=True
        )
    except Exception as e:
        logger.error(f"Contour prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model prediction failed: {str(e)}"
        )
    
    # Reshape to grid
    pred_grid = predictions.reshape((request.grid_resolution, request.grid_resolution))
    unc_grid = uncertainties.reshape((request.grid_resolution, request.grid_resolution))
    
    # Get experimental data if requested
    experiments_data = None
    if request.include_experiments and len(session.experiment_manager) > 0:
        exp_df = session.experiment_manager.get_data()
        target_col = session.experiment_manager.target_columns[0]  # Use first target for visualization
        if request.x_var in exp_df.columns and request.y_var in exp_df.columns and target_col in exp_df.columns:
            experiments_data = {
                "x": exp_df[request.x_var].tolist(),
                "y": exp_df[request.y_var].tolist(),
                "output": exp_df[target_col].tolist()
            }
    
    # Get suggestion data if requested (would need to be stored in session)
    suggestions_data = None
    if request.include_suggestions and hasattr(session, 'last_suggestions') and session.last_suggestions:
        suggestions_df = pd.DataFrame(session.last_suggestions)
        if request.x_var in suggestions_df.columns and request.y_var in suggestions_df.columns:
            suggestions_data = {
                "x": suggestions_df[request.x_var].tolist(),
                "y": suggestions_df[request.y_var].tolist()
            }
    
    return ContourDataResponse(
        x_var=request.x_var,
        y_var=request.y_var,
        x_grid=X_grid.tolist(),
        y_grid=Y_grid.tolist(),
        predictions=pred_grid.tolist(),
        uncertainties=unc_grid.tolist(),
        experiments=experiments_data,
        suggestions=suggestions_data,
        x_bounds=[float(x_bounds[0]), float(x_bounds[1])],
        y_bounds=[float(y_bounds[0]), float(y_bounds[1])],
        colorbar_bounds=[float(pred_grid.min()), float(pred_grid.max())]
    )


@router.post("/{session_id}/visualizations/surface", response_model=ContourDataResponse)
async def get_surface_data(
    session_id: str,
    request: ContourDataRequest,
    session: OptimizationSession = Depends(get_session)
):
    """
    Generate 3D surface plot data for model predictions.
    
    Returns the same grid data as the contour endpoint (X grid, Y grid,
    predictions, uncertainties) for rendering as a 3D surface where
    the Z axis represents the predicted output.
    """
    # Delegate to the contour endpoint — the data is identical;
    # the rendering difference is client-side.
    return await get_contour_data(session_id, request, session)


@router.get("/{session_id}/visualizations/parity", response_model=ParityDataResponse)
async def get_parity_data(
    session_id: str,
    use_calibrated: bool = Query(default=False, description="Use calibrated uncertainty estimates"),
    session: OptimizationSession = Depends(get_session)
):
    """
    Get parity plot data (actual vs predicted from cross-validation).
    
    Returns the cached CV results with performance metrics.
    """
    if not session.model or not session.model.is_trained:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model must be trained before generating visualizations"
        )
    
    # Check if model has CV results
    if not hasattr(session.model, 'cv_cached_results') or session.model.cv_cached_results is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model does not have cached cross-validation results"
        )
    
    # Select calibrated or uncalibrated results
    if use_calibrated and hasattr(session.model, 'cv_cached_results_calibrated') and session.model.cv_cached_results_calibrated:
        cv_results = session.model.cv_cached_results_calibrated
        is_calibrated = True
    else:
        cv_results = session.model.cv_cached_results
        is_calibrated = False
    
    y_true = np.array(cv_results['y_true'])
    y_pred = np.array(cv_results['y_pred'])
    y_std = np.array(cv_results['y_std'])
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    
    # Calculate MAPE (handle division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = float(np.nanmean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100)
    
    # Determine plot bounds (square aspect ratio)
    min_val = float(min(y_true.min(), y_pred.min()))
    max_val = float(max(y_true.max(), y_pred.max()))
    
    return ParityDataResponse(
        y_true=y_true.tolist(),
        y_pred=y_pred.tolist(),
        y_std=y_std.tolist(),
        metrics={
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape
        },
        bounds=[min_val, max_val],
        calibrated=is_calibrated
    )


@router.get("/{session_id}/visualizations/metrics", response_model=MetricsDataResponse)
async def get_metrics_data(
    session_id: str,
    cv_splits: int = Query(default=5, ge=2, le=10, description="Number of cross-validation splits"),
    session: OptimizationSession = Depends(get_session)
):
    """
    Get cross-validation metrics over increasing training size.
    
    Evaluates model performance with 5, 6, 7, ... N training samples
    to show learning curve.
    """
    if not session.model or not session.model.is_trained:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model must be trained before generating visualizations"
        )
    
    if len(session.experiment_manager) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No experimental data available"
        )
    
    # Run evaluation to get metrics over training size
    # This calls the model's evaluate method which returns metrics for different training sizes
    try:
        metrics_dict = await run_in_threadpool(
            session.model.evaluate,
            session.experiment_manager,
            cv_splits=cv_splits,
            debug=False
        )
        
        # Get training sizes (5 to N)
        n_experiments = len(session.experiment_manager.get_data())
        training_sizes = list(range(5, n_experiments + 1))
        
        # Replace NaN/Inf values with None for JSON serialization
        def sanitize_metrics(metric_list):
            """Replace NaN and Inf values with None"""
            result = []
            for val in metric_list:
                if isinstance(val, (int, float)) and (np.isnan(val) or np.isinf(val)):
                    result.append(None)
                else:
                    result.append(float(val) if val is not None else None)
            return result
        
        return MetricsDataResponse(
            training_sizes=training_sizes,
            rmse=sanitize_metrics(metrics_dict["RMSE"]),
            mae=sanitize_metrics(metrics_dict["MAE"]),
            r2=sanitize_metrics(metrics_dict["R²"]),
            mape=sanitize_metrics(metrics_dict["MAPE"])
        )
    except Exception as e:
        logger.error(f"Error computing metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error computing metrics. Check server logs for details."
        )


@router.get("/{session_id}/visualizations/qq-plot", response_model=QQPlotDataResponse)
async def get_qq_plot_data(
    session_id: str,
    use_calibrated: bool = Query(default=False, description="Use calibrated uncertainty estimates"),
    session: OptimizationSession = Depends(get_session)
):
    """
    Get Q-Q plot data for standardized residuals.
    
    Compares distribution of standardized residuals (z-scores) to
    standard normal distribution to assess calibration.
    """
    if not session.model or not session.model.is_trained:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model must be trained before generating visualizations"
        )
    
    # Check if model has CV results
    if not hasattr(session.model, 'cv_cached_results') or session.model.cv_cached_results is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model does not have cached cross-validation results"
        )
    
    # Select calibrated or uncalibrated results
    if use_calibrated and hasattr(session.model, 'cv_cached_results_calibrated') and session.model.cv_cached_results_calibrated:
        cv_results = session.model.cv_cached_results_calibrated
        is_calibrated = True
    else:
        cv_results = session.model.cv_cached_results
        is_calibrated = False
    
    y_true = np.array(cv_results['y_true'])
    y_pred = np.array(cv_results['y_pred'])
    y_std = np.array(cv_results['y_std'])
    
    # Compute standardized residuals (z-scores)
    z_scores = (y_true - y_pred) / y_std
    z_mean = float(np.mean(z_scores))
    z_std = float(np.std(z_scores, ddof=1))
    n_samples = int(len(z_scores))
    
    # Sort z-scores
    z_sorted = np.sort(z_scores)
    
    # Compute theoretical quantiles from standard normal
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(z_scores)))
    
    # Determine bounds
    min_val = float(min(theoretical_quantiles.min(), z_sorted.min()))
    max_val = float(max(theoretical_quantiles.max(), z_sorted.max()))
    
    return QQPlotDataResponse(
        theoretical_quantiles=theoretical_quantiles.tolist(),
        sample_quantiles=z_sorted.tolist(),
        z_mean=z_mean,
        z_std=z_std,
        n_samples=n_samples,
        bounds=[min_val, max_val],
        calibrated=is_calibrated
    )


@router.get("/{session_id}/visualizations/calibration-curve", response_model=CalibrationCurveDataResponse)
async def get_calibration_curve_data(
    session_id: str,
    use_calibrated: bool = Query(default=False, description="Use calibrated uncertainty estimates"),
    session: OptimizationSession = Depends(get_session)
):
    """
    Get calibration curve data (reliability diagram).
    
    Compares nominal coverage probabilities to empirical coverage
    to assess uncertainty calibration quality.
    """
    if not session.model or not session.model.is_trained:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model must be trained before generating visualizations"
        )
    
    # Check if model has CV results
    if not hasattr(session.model, 'cv_cached_results') or session.model.cv_cached_results is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model does not have cached cross-validation results"
        )
    
    # Select calibrated or uncalibrated results
    if use_calibrated and hasattr(session.model, 'cv_cached_results_calibrated') and session.model.cv_cached_results_calibrated:
        cv_results = session.model.cv_cached_results_calibrated
        is_calibrated = True
    else:
        cv_results = session.model.cv_cached_results
        is_calibrated = False
    
    y_true = np.array(cv_results['y_true'])
    y_pred = np.array(cv_results['y_pred'])
    y_std = np.array(cv_results['y_std'])
    n_samples = int(len(y_true))
    
    # Compute empirical coverage for range of nominal probabilities
    nominal_probs = np.arange(0.10, 1.00, 0.05)
    empirical_coverage = []
    confidence_labels = []
    
    for prob in nominal_probs:
        # Convert probability to sigma multiplier (inverse CDF of standard normal)
        sigma_multiplier = stats.norm.ppf((1 + prob) / 2)
        
        # Create label (e.g., "±1.96σ (95%)")
        percentage = int(prob * 100)
        confidence_labels.append(f"±{sigma_multiplier:.2f}σ ({percentage}%)")
        
        # Check coverage
        lower_bound = y_pred - sigma_multiplier * y_std
        upper_bound = y_pred + sigma_multiplier * y_std
        within_interval = np.logical_and(y_true >= lower_bound, y_true <= upper_bound)
        empirical_coverage.append(float(np.mean(within_interval)))
    
    return CalibrationCurveDataResponse(
        nominal_coverage=nominal_probs.tolist(),
        empirical_coverage=empirical_coverage,
        confidence_levels=confidence_labels,
        nominal_probabilities=nominal_probs.tolist(),  # Same as nominal_coverage
        empirical_probabilities=empirical_coverage,  # Same as empirical_coverage
        n_samples=n_samples,
        calibrated=is_calibrated,
        results_type='calibrated' if is_calibrated else 'uncalibrated'
    )


@router.get("/{session_id}/visualizations/hyperparameters", response_model=HyperparametersResponse)
async def get_hyperparameters(
    session_id: str,
    session: OptimizationSession = Depends(get_session)
):
    """
    Get trained model hyperparameters and configuration.
    """
    if not session.model or not session.model.is_trained:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model must be trained before accessing hyperparameters"
        )
    
    # Get hyperparameters - convert to serializable format
    hyperparams_raw = session.model.get_hyperparameters()
    
    # Convert any non-serializable objects to strings
    hyperparams = {}
    for key, value in hyperparams_raw.items():
        try:
            # Try to serialize to check if it's JSON-compatible
            import json
            json.dumps(value)
            hyperparams[key] = value
        except (TypeError, ValueError):
            # If not serializable, convert to string representation
            hyperparams[key] = str(value)
    
    # Get model configuration
    backend = "sklearn" if hasattr(session.model, 'model') else "botorch"
    kernel = session.model.kernel_type if hasattr(session.model, 'kernel_type') else "unknown"
    
    # Get transform info
    input_transform = getattr(session.model, 'input_transform_type', None)
    output_transform = getattr(session.model, 'output_transform_type', None)
    
    # Get calibration info
    calibration_enabled = getattr(session.model, 'calibration_enabled', False)
    calibration_factor = getattr(session.model, 'calibration_factor', None)
    
    return HyperparametersResponse(
        hyperparameters=hyperparams,
        backend=backend,
        kernel=kernel,
        input_transform=input_transform,
        output_transform=output_transform,
        calibration_enabled=calibration_enabled,
        calibration_factor=float(calibration_factor) if calibration_factor is not None else None
    )
    
