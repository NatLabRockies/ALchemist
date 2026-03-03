"""
Ollama provider via the OpenAI-compatible Chat Completions API.
Embeds the JSON schema in the system prompt for maximum model compatibility,
and uses response_format=json_object to enforce JSON output mode.
Requires: pip install openai>=1.50.0  (covers Ollama via base_url)
"""

import json

from .base import StructuringProvider

try:
    from openai import AsyncOpenAI
    import httpx
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


class OllamaProvider(StructuringProvider):
    """Uses Ollama's OpenAI-compatible Chat Completions endpoint."""

    DEFAULT_BASE_URL = "http://localhost:11434/v1"

    def __init__(self, model: str, base_url: str = DEFAULT_BASE_URL):
        if not _OPENAI_AVAILABLE:
            raise ImportError(
                "openai package not installed. Run: pip install 'alchemist-nrel[llm]'"
            )
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key="ollama",
            timeout=httpx.Timeout(120.0, connect=10.0),
        )
        self.model = model

    async def suggest_effects(
        self,
        system_instructions: str,
        user_prompt: str,
        schema: dict,
    ) -> dict:
        # Embed the schema in the system prompt — most compatible approach across
        # all Ollama model versions (avoids json_schema response_format version issues).
        schema_str = json.dumps(schema, indent=2)
        system_with_schema = (
            f"{system_instructions}\n\n"
            f"You MUST respond with valid JSON that exactly matches this schema "
            f"(no extra keys, no markdown):\n{schema_str}"
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_with_schema},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
