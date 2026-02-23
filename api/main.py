"""
ALchemist FastAPI Application

RESTful API wrapper for alchemist_core Session API.
Designed for React frontend but framework-agnostic.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from .routers import sessions, variables, experiments, models, acquisition, visualizations, websocket, llm
from .middleware.error_handlers import add_exception_handlers
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="ALchemist API",
    description="REST API for Bayesian optimization and active learning",
    version="0.3.2",
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
)

# Add custom exception handlers
add_exception_handlers(app)

# Include routers
app.include_router(sessions.router, prefix="/api/v1", tags=["Sessions"])
app.include_router(variables.router, prefix="/api/v1/sessions", tags=["Variables"])
app.include_router(experiments.router, prefix="/api/v1/sessions", tags=["Experiments"])
app.include_router(models.router, prefix="/api/v1/sessions", tags=["Models"])
app.include_router(acquisition.router, prefix="/api/v1/sessions", tags=["Acquisition"])
app.include_router(visualizations.router, prefix="/api/v1/sessions", tags=["Visualizations"])
app.include_router(websocket.router, prefix="/api/v1", tags=["WebSocket"])
app.include_router(llm.router, prefix="/api/v1", tags=["LLM"])


@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "name": "ALchemist API",
        "version": "0.1.0",
        "docs": "/api/docs",
        "health": "/health"
    }


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
    
    @app.get("/{full_path:path}")
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
