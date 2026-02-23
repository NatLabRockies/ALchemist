"""
OpenAI provider using the Responses API with strict JSON schema output.
Requires: pip install openai>=1.50.0
"""

import json

from .base import StructuringProvider

try:
    from openai import AsyncOpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


class OpenAIProvider(StructuringProvider):
    """Uses OpenAI Responses API (recommended over Chat Completions)."""

    def __init__(self, api_key: str, model: str):
        if not _OPENAI_AVAILABLE:
            raise ImportError(
                "openai package not installed. Run: pip install 'alchemist-nrel[llm]'"
            )
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def suggest_effects(
        self,
        system_instructions: str,
        user_prompt: str,
        schema: dict,
    ) -> dict:
        response = await self.client.responses.create(
            model=self.model,
            instructions=system_instructions,
            input=user_prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "effect_suggestions",
                    "strict": True,
                    "schema": schema,
                }
            },
            store=False,
        )
        return json.loads(response.output_text)
