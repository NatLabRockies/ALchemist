"""
ALchemist FastAPI Application

RESTful API wrapper for alchemist_core Session API.
Designed for React frontend but framework-agnostic.
"""

import sys
import os
import re
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from .routers import sessions, variables, experiments, models, acquisition, visualizations, websocket, llm
from .middleware.error_handlers import add_exception_handlers
import logging
import alchemist_core
from .models.responses import ErrorResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Error responses declared on every router so the OpenAPI schema documents the
# shared error body produced by api/middleware/error_handlers.py.
COMMON_ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Invalid request for the session's current state"},
    404: {"model": ErrorResponse, "description": "Session or resource not found"},
    500: {"model": ErrorResponse, "description": "Unexpected server error"},
    503: {"model": ErrorResponse, "description": "The session's file exists but could not be read"},
}

# Create FastAPI app
app = FastAPI(
    title="ALchemist API",
    description="REST API for Bayesian optimization and active learning",
    version=alchemist_core.__version__,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# CORS configuration - allows frontend in both dev and production
# Default origins include dev servers and common production patterns
# Override with ALLOWED_ORIGINS environment variable for specific deployments
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://localhost:5174,http://localhost:8000,http://127.0.0.1:8000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Not CORS-safelisted, so browsers hide it unless it is exposed (autosave
    # reports a failed write there).
    expose_headers=["X-ALchemist-Saved"],
)

# Add custom exception handlers
add_exception_handlers(app)

# Autosave: after a request changes a session, write the session to disk, so a
# restart or crash loses nothing a client was told succeeded. Routes that only
# read a session are skipped; /save writes on its own. A failed write does not
# turn the response into an error -- the change was applied in memory and a
# retry would apply it twice -- it is logged and reported in X-ALchemist-Saved.
_SESSION_ROUTE = re.compile(r"^/api/v1/sessions/([^/]+)/(.+)$")
_READ_ONLY_ROUTES = (
    "save",
    "model/predict",
    "visualizations/",
    "experiments/preview",
    "acquisition/find-optimum",
    "optimal-design/info",
    # The web app posts a backup every 30 s; it copies the session to the
    # recovery directory and changes nothing in it. The lock routes save the
    # session themselves.
    "backup",
    "lock",
)


@app.middleware("http")
async def autosave_sessions(request: Request, call_next):
    response = await call_next(request)
    if request.method not in ("POST", "PUT", "PATCH", "DELETE") or response.status_code >= 400:
        return response
    match = _SESSION_ROUTE.match(request.url.path)
    if not match or match.group(2).startswith(_READ_ONLY_ROUTES):
        return response
    from .services.session_store import session_store
    session_id = match.group(1)
    if session_id not in session_store._sessions:
        return response
    saved = await run_in_threadpool(session_store.persist_session_to_disk, session_id)
    if not saved:
        logger.error(f"Autosave failed for session {session_id} after {request.method} {request.url.path}")
    response.headers["X-ALchemist-Saved"] = "true" if saved else "false"
    return response

# Include routers
app.include_router(sessions.router, prefix="/api/v1", tags=["Sessions"], responses=COMMON_ERROR_RESPONSES)
app.include_router(variables.router, prefix="/api/v1/sessions", tags=["Variables"], responses=COMMON_ERROR_RESPONSES)
app.include_router(experiments.router, prefix="/api/v1/sessions", tags=["Experiments"], responses=COMMON_ERROR_RESPONSES)
app.include_router(models.router, prefix="/api/v1/sessions", tags=["Models"], responses=COMMON_ERROR_RESPONSES)
app.include_router(acquisition.router, prefix="/api/v1/sessions", tags=["Acquisition"], responses=COMMON_ERROR_RESPONSES)
app.include_router(visualizations.router, prefix="/api/v1/sessions", tags=["Visualizations"], responses=COMMON_ERROR_RESPONSES)
app.include_router(websocket.router, prefix="/api/v1", tags=["WebSocket"], responses=COMMON_ERROR_RESPONSES)
app.include_router(llm.router, prefix="/api/v1", tags=["LLM"], responses=COMMON_ERROR_RESPONSES)


def _api_info() -> dict:
    return {
        "name": "ALchemist API",
        "version": alchemist_core.__version__,
        "docs": "/api/docs",
        "health": "/health"
    }


@app.get("/")
async def root(request: Request):
    """
    The web UI for a browser, API information for everything else.

    A browser (Accept: text/html) gets the built frontend, so the address
    `alchemist-web` prints opens the app. Scripts, curl and clients that don't
    ask for HTML keep getting the API-information JSON, as does a browser when
    no frontend is built.
    """
    index = static_dir / "index.html"
    if "text/html" in request.headers.get("accept", "") and index.is_file():
        return FileResponse(index)
    return _api_info()


@app.get("/api", include_in_schema=False)
async def api_info():
    """API information (name, version, docs and health paths)."""
    return _api_info()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "alchemist-api"
    }


# Mount static files for production (if they exist)
# Priority order:
# 1. api/static/ - Production (pip installed or built package)
# 2. alchemist-web/dist/ - Development (after manual npm run build)
api_static_dir = Path(__file__).parent / "static"
dev_static_dir = Path(__file__).parent.parent / "alchemist-web" / "dist"

# Use api/static if it exists (production), otherwise fall back to dev build
static_dir = api_static_dir if api_static_dir.exists() else dev_static_dir

if static_dir.exists():
    logger.info(f"Serving static files from: {static_dir}")
    app.mount("/assets", StaticFiles(directory=str(static_dir / "assets")), name="assets")
    
    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        """Serve React SPA for all non-API routes."""
        # Don't intercept API routes
        if full_path.startswith("api/"):
            return {"detail": "Not Found"}
        
        # Try to serve the requested file
        file_path = static_dir / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        
        # For all other routes, serve index.html (SPA routing)
        index_path = static_dir / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        
        return {"detail": "Not Found"}
else:
    logger.warning("Static files not found. Web UI will not be available. Run 'npm run build' in alchemist-web/ or install from built wheel.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
