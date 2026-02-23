"""
LLM / AI-assisted design endpoints.

Routes (all under /api/v1 prefix set in main.py):
  POST /llm/suggest-effects/{session_id}  — SSE streaming suggest-effects call
  GET  /llm/config                        — read saved provider config
  POST /llm/config                        — save provider config
  GET  /llm/ollama/models                 — list locally available Ollama models
"""

import logging
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.models.requests import SuggestEffectsRequest
from api.models.responses import OllamaModelsResponse, LLMConfigResponse
from api.services import session_store
from api.services.llm_config import get_llm_config, save_llm_config
from api.services.llm_service import suggest_effects_stream

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm", tags=["LLM"])


@router.post("/suggest-effects/{session_id}")
async def suggest_effects(session_id: str, request: SuggestEffectsRequest):
    """
    Stream SSE events for AI-suggested effects for an optimal design.

    Sequential workflow:
      1. [Optional] Edison Scientific literature search (if edison_config provided)
      2. OpenAI / Ollama structuring call → JSON effects list

    Returns a text/event-stream where each event is:
      data: <json>\\n\\n

    Terminal events have status "complete" (with result) or "error" (with message).
    """
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    variables = session.search_space.variables
    if not variables:
        raise HTTPException(
            status_code=400,
            detail="No variables defined in search space. Add variables before suggesting effects.",
        )

    logger.info(
        "suggest-effects: session=%s provider=%s/%s edison=%s",
        session_id,
        request.structuring_provider.provider,
        request.structuring_provider.model,
        bool(request.edison_config),
    )

    return StreamingResponse(
        suggest_effects_stream(variables, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


@router.get("/config", response_model=LLMConfigResponse)
async def get_config():
    """Return saved LLM provider configuration (API keys are returned as stored)."""
    return get_llm_config()


@router.post("/config")
async def update_config(config: dict[str, Any]):
    """
    Save LLM provider configuration to ~/.alchemist/config.json.
    Merges into the existing 'llm' section (does not overwrite other config keys).
    """
    save_llm_config(config)
    return {"status": "saved"}


@router.get("/ollama/models", response_model=OllamaModelsResponse)
async def list_ollama_models(base_url: str = "http://localhost:11434"):
    """
    Query a running Ollama instance for available models.
    Returns model names or an empty list with an error message if unreachable.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{base_url}/api/tags")
            r.raise_for_status()
            data = r.json()
            names = [m["name"] for m in data.get("models", [])]
            return OllamaModelsResponse(models=names)
    except Exception as exc:
        return OllamaModelsResponse(models=[], error=str(exc))
