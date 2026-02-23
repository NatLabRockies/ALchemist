"""Abstract base class for LLM structuring providers."""

from abc import ABC, abstractmethod


class StructuringProvider(ABC):
    """Calls an LLM and returns a parsed JSON dict matching the provided schema."""

    @abstractmethod
    async def suggest_effects(
        self,
        system_instructions: str,
        user_prompt: str,
        schema: dict,
    ) -> dict:
        """
        Args:
            system_instructions: Fixed system-level guidance.
            user_prompt: User-facing prompt assembled from session + context.
            schema: JSON Schema dict the response must conform to.

        Returns:
            Parsed dict matching the schema.
        """
        ...
