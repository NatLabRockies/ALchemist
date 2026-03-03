"""
Acquisition router - Next experiment suggestions.
"""

from fastapi import APIRouter, Depends, HTTPException
from ..models.requests import AcquisitionRequest, FindOptimumRequest
from ..models.responses import AcquisitionResponse, FindOptimumResponse
from ..dependencies import get_session
from ..middleware.error_handlers import NoModelError
from alchemist_core.session import OptimizationSession
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/{session_id}/acquisition/suggest", response_model=AcquisitionResponse)
async def suggest_next_experiments(
    session_id: str,
    request: AcquisitionRequest,
    session: OptimizationSession = Depends(get_session)
):
    """
    Suggest next experiments using acquisition function.
    
    Requires a trained model. Returns one or more suggested experiments
    based on the acquisition strategy and batch size.
    
    Common strategies:
    - EI (Expected Improvement): Balances exploration and exploitation
    - PI (Probability of Improvement): More conservative than EI
    - UCB (Upper Confidence Bound): More exploratory
    - qEI, qUCB: Batch versions for parallel experiments
    - qNIPV: Pure exploration for model improvement
    """
    # Check if model exists
    if session.model is None:
        raise NoModelError("No trained model available. Train a model first.")
    
    try:
        # Build kwargs for acquisition function
        acq_kwargs = {}
        if request.xi is not None:
            acq_kwargs['xi'] = request.xi
        if request.kappa is not None:
            acq_kwargs['kappa'] = request.kappa
        
        # Get suggestions
        suggestions_df = session.suggest_next(
            strategy=request.strategy,
            goal=request.goal,
            n_suggestions=request.n_suggestions,
            **acq_kwargs
        )
        
        # Store suggestions in session for visualization access
        session.last_suggestions = suggestions_df.to_dict('records')
        
        # Convert to list of dicts
        suggestions = suggestions_df.to_dict('records')
        
        # Record acquisition in audit log
        if suggestions:
            # Get current max iteration from experiments
            iteration = None
            if not session.experiment_manager.df.empty and 'Iteration' in session.experiment_manager.df.columns:
                iteration = int(session.experiment_manager.df['Iteration'].max()) + 1
            
            # Build parameters dict with only fields that exist
            acq_params = {
                "goal": request.goal,
                "n_suggestions": request.n_suggestions
            }
            if request.xi is not None:
                acq_params["xi"] = request.xi
            if request.kappa is not None:
                acq_params["kappa"] = request.kappa
            
            session.audit_log.lock_acquisition(
                strategy=request.strategy,
                parameters=acq_params,
                suggestions=suggestions,
                iteration=iteration,
                notes=f"Suggested {len(suggestions)} point(s) using {request.strategy}"
            )
        
        logger.info(f"Generated {len(suggestions)} suggestions for session {session_id} using {request.strategy}")
        
        return AcquisitionResponse(
            suggestions=suggestions,
            n_suggestions=len(suggestions)
        )
    except (ValueError, RuntimeError, ImportError):
        raise
    except Exception as e:
        logger.error(f"Acquisition failed for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Acquisition computation failed. Check server logs for details.")


@router.post("/{session_id}/acquisition/find-optimum", response_model=FindOptimumResponse)
async def find_model_optimum(
    session_id: str,
    request: FindOptimumRequest,
    session: OptimizationSession = Depends(get_session)
):
    """
    Find the point where the model predicts the optimal value.
    
    This searches the input space to find where the model's predicted
    mean is highest (for maximize) or lowest (for minimize), without
    considering exploration. This is pure exploitation of the model.
    
    Warning: This relies entirely on the model's predictions and does
    not use acquisition functions that balance exploration/exploitation.
    Best used when you're confident in the model's accuracy.
    """
    # Check if model exists
    if session.model is None:
        raise NoModelError("No trained model available. Train a model first.")
    
    try:
        # Determine backend
        backend = session.model_backend
        
        # Create appropriate acquisition function instance
        if backend == 'sklearn':
            from alchemist_core.acquisition.skopt_acquisition import SkoptAcquisition
            
            acquisition = SkoptAcquisition(
                search_space=session.search_space,
                model=session.model,
                maximize=(request.goal == 'maximize'),
                random_state=42
            )
            
            result = acquisition.find_optimum(
                model=session.model,
                maximize=(request.goal == 'maximize'),
                random_state=42
            )
            
        elif backend == 'botorch':
            from alchemist_core.acquisition.botorch_acquisition import BoTorchAcquisition
            
            acquisition = BoTorchAcquisition(
                search_space=session.search_space,
                model=session.model,
                acq_func="ucb",  # Doesn't matter for find_optimum
                maximize=(request.goal == 'maximize'),
                random_state=42
            )
            
            result = acquisition.find_optimum(
                model=session.model,
                maximize=(request.goal == 'maximize'),
                random_state=42
            )
        else:
            raise ValueError(f"Find optimum not supported for backend: {backend}")
        
        # Extract results
        opt_point_df = result['x_opt']
        opt_value = float(result['value'])
        opt_std = float(result['std']) if result.get('std') is not None else None
        
        # Handle NaN/Inf values - convert to None for JSON serialization
        import math
        if opt_std is not None and (math.isnan(opt_std) or math.isinf(opt_std)):
            opt_std = None
        if math.isnan(opt_value) or math.isinf(opt_value):
            logger.warning(f"Invalid optimum value (NaN/Inf) found, setting to 0.0")
            opt_value = 0.0
        
        # Convert to dict and clean any NaN values in the optimum point
        optimum_point = opt_point_df.to_dict('records')[0]
        
        # Clean NaN/Inf values in the optimum point dictionary
        cleaned_optimum = {}
        for key, value in optimum_point.items():
            if isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    cleaned_optimum[key] = None
                else:
                    cleaned_optimum[key] = value
            else:
                cleaned_optimum[key] = value
        
        logger.info(f"Found model optimum for session {session_id}: value={opt_value}")
        
        return FindOptimumResponse(
            optimum=cleaned_optimum,
            predicted_value=opt_value,
            predicted_std=opt_std,
            goal=request.goal
        )
    except (ValueError, RuntimeError, ImportError):
        raise
    except Exception as e:
        logger.error(f"Find optimum failed for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Optimum search failed. Check server logs for details.")
