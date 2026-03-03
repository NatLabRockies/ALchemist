"""
Custom exception handlers for API errors.
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging

logger = logging.getLogger(__name__)


# Custom Exceptions
class SessionNotFoundError(Exception):
    """Session not found or expired."""
    pass


class NoDataError(Exception):
    """No experimental data available."""
    pass


class NoModelError(Exception):
    """No trained model available."""
    pass


class NoVariablesError(Exception):
    """No variables defined in search space."""
    pass


def add_exception_handlers(app: FastAPI):
    """Add custom exception handlers to FastAPI app."""
    
    @app.exception_handler(SessionNotFoundError)
    async def session_not_found_handler(request: Request, exc: SessionNotFoundError):
        """Handle session not found errors."""
        logger.warning(f"Session not found: {exc}")
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "detail": str(exc) if str(exc) else "Session not found or expired",
                "error_type": "SessionNotFoundError",
                "status_code": status.HTTP_404_NOT_FOUND
            }
        )
    
    @app.exception_handler(NoDataError)
    async def no_data_handler(request: Request, exc: NoDataError):
        """Handle no data errors."""
        logger.warning(f"No data error: {exc}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "detail": str(exc) if str(exc) else "No experimental data available. Add experiments first.",
                "error_type": "NoDataError",
                "status_code": status.HTTP_400_BAD_REQUEST
            }
        )
    
    @app.exception_handler(NoModelError)
    async def no_model_handler(request: Request, exc: NoModelError):
        """Handle no model errors."""
        logger.warning(f"No model error: {exc}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "detail": str(exc) if str(exc) else "No trained model available. Train a model first.",
                "error_type": "NoModelError",
                "status_code": status.HTTP_400_BAD_REQUEST
            }
        )
    
    @app.exception_handler(NoVariablesError)
    async def no_variables_handler(request: Request, exc: NoVariablesError):
        """Handle no variables errors."""
        logger.warning(f"No variables error: {exc}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "detail": str(exc) if str(exc) else "No variables defined. Add variables to search space first.",
                "error_type": "NoVariablesError",
                "status_code": status.HTTP_400_BAD_REQUEST
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors."""
        logger.warning(f"Validation error: {exc}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "detail": str(exc),
                "error_type": "ValidationError",
                "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY,
                "errors": exc.errors()
            }
        )
    
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        """Handle value errors from core library."""
        logger.error(f"ValueError: {exc}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "detail": str(exc),
                "error_type": "ValueError",
                "status_code": status.HTTP_400_BAD_REQUEST
            }
        )
    
    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, exc: RuntimeError):
        """Handle runtime errors from core library (model evaluation, acquisition failures)."""
        logger.error(f"RuntimeError: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "detail": str(exc),
                "error_type": "RuntimeError",
                "status_code": status.HTTP_400_BAD_REQUEST
            }
        )
    
    @app.exception_handler(ImportError)
    async def import_error_handler(request: Request, exc: ImportError):
        """Handle missing optional dependency errors (e.g., BoTorch, OpenAI)."""
        logger.error(f"ImportError: {exc}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "detail": str(exc),
                "error_type": "ImportError",
                "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle all other exceptions."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "Internal server error. Please check logs for details.",
                "error_type": type(exc).__name__,
                "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR
            }
        )
