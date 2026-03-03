"""
Unit tests for api/services/llm_service.py

Tests prompt building, sanitization, provider factory, SSE helper,
and the streaming orchestrator (with mocked providers).
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from api.services.llm_service import (
    _sanitize_text,
    _sanitize_variable_name,
    _format_variable,
    _build_effects_lists,
    build_user_prompt,
    build_edison_query,
    _get_structuring_provider,
    _sse,
    suggest_effects_stream,
    _MAX_CONTEXT_LENGTH,
    _MAX_VARIABLE_NAME_LENGTH,
    EFFECTS_SCHEMA,
    _DEFAULT_DISCLAIMER,
)


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

REAL_VAR = {"name": "Temperature", "type": "real", "lower_bound": 100, "upper_bound": 500, "unit": "°C"}
INT_VAR = {"name": "Pressure", "type": "integer", "lower_bound": 1, "upper_bound": 10}
CAT_VAR = {"name": "Catalyst", "type": "categorical", "categories": ["Pt", "Pd", "Rh"]}
DISC_VAR = {"name": "Speed", "type": "discrete", "lower_bound": 10, "upper_bound": 50}

SAMPLE_VARS = [REAL_VAR, INT_VAR, CAT_VAR]


# ===================================================================
# Sanitization
# ===================================================================

class TestSanitizeText:
    def test_clean_text_unchanged(self):
        assert _sanitize_text("hello world", 100) == "hello world"

    def test_strips_control_characters(self):
        assert _sanitize_text("he\x00ll\x07o", 100) == "hello"

    def test_preserves_newline_and_tab(self):
        # \n (0x0a) and \t (0x09) are NOT in the stripped range
        assert _sanitize_text("line1\nline2\tend", 100) == "line1\nline2\tend"

    def test_truncates_with_ellipsis(self):
        result = _sanitize_text("abcdef", 3)
        assert result == "abc…"
        assert len(result) == 4  # 3 chars + ellipsis

    def test_exact_max_length_not_truncated(self):
        assert _sanitize_text("abc", 3) == "abc"

    def test_empty_string(self):
        assert _sanitize_text("", 100) == ""


class TestSanitizeVariableName:
    def test_normal_name(self):
        assert _sanitize_variable_name("Temperature") == "Temperature"

    def test_strips_control_chars(self):
        assert _sanitize_variable_name("Temp\x00erature") == "Temperature"

    def test_truncates_long_name(self):
        long_name = "x" * 200
        result = _sanitize_variable_name(long_name)
        assert len(result) == _MAX_VARIABLE_NAME_LENGTH + 1  # +1 for ellipsis char


# ===================================================================
# Variable formatting
# ===================================================================

class TestFormatVariable:
    def test_real_variable_with_unit(self):
        result = _format_variable(REAL_VAR)
        assert "Temperature" in result
        assert "real" in result
        assert "100" in result
        assert "500" in result
        assert "°C" in result

    def test_categorical_variable(self):
        result = _format_variable(CAT_VAR)
        assert "Catalyst" in result
        assert "categorical" in result
        assert "Pt" in result

    def test_integer_variable_no_unit(self):
        result = _format_variable(INT_VAR)
        assert "Pressure" in result
        assert "integer" in result
        # No trailing unit
        assert result.endswith("10")

    def test_variable_without_type_defaults_to_real(self):
        result = _format_variable({"name": "X", "lower_bound": 0, "upper_bound": 1})
        assert "real" in result


# ===================================================================
# Effects list building
# ===================================================================

class TestBuildEffectsLists:
    def test_main_effects(self):
        main, _, _ = _build_effects_lists(SAMPLE_VARS)
        assert main == ["Temperature", "Pressure", "Catalyst"]

    def test_interactions_are_pairwise(self):
        _, interactions, _ = _build_effects_lists(SAMPLE_VARS)
        assert "Temperature*Pressure" in interactions
        assert "Temperature*Catalyst" in interactions
        assert "Pressure*Catalyst" in interactions
        assert len(interactions) == 3  # C(3,2) = 3

    def test_quadratics_exclude_categorical(self):
        _, _, quadratics = _build_effects_lists(SAMPLE_VARS)
        assert "Temperature**2" in quadratics
        assert "Pressure**2" in quadratics
        # Categorical should NOT have quadratic
        assert "Catalyst**2" not in quadratics

    def test_single_variable_no_interactions(self):
        _, interactions, _ = _build_effects_lists([REAL_VAR])
        assert interactions == []

    def test_discrete_included_in_quadratics(self):
        _, _, quadratics = _build_effects_lists([DISC_VAR])
        assert "Speed**2" in quadratics

    def test_sanitizes_variable_names(self):
        dirty_var = {"name": "Temp\x00erature", "type": "real", "lower_bound": 0, "upper_bound": 1}
        main, _, _ = _build_effects_lists([dirty_var])
        assert main == ["Temperature"]


# ===================================================================
# Prompt building
# ===================================================================

class TestBuildUserPrompt:
    def test_contains_system_context(self):
        prompt = build_user_prompt(SAMPLE_VARS, "Fischer-Tropsch synthesis")
        assert "System description: Fischer-Tropsch synthesis" in prompt

    def test_contains_variable_space(self):
        prompt = build_user_prompt(SAMPLE_VARS, "test system")
        assert "Variable space:" in prompt
        assert "Temperature" in prompt
        assert "Catalyst" in prompt

    def test_contains_available_effects(self):
        prompt = build_user_prompt(SAMPLE_VARS, "test system")
        assert "Main effects:" in prompt
        assert "Interactions:" in prompt
        assert "Quadratic:" in prompt

    def test_without_literature_context(self):
        prompt = build_user_prompt(SAMPLE_VARS, "test")
        assert "Literature context" not in prompt

    def test_with_literature_context(self):
        prompt = build_user_prompt(SAMPLE_VARS, "test", literature_context="Some cited paper.")
        assert "Literature context" in prompt
        assert "Some cited paper." in prompt
        assert "based on the literature context above" in prompt

    def test_system_context_is_sanitized(self):
        long_context = "x" * (_MAX_CONTEXT_LENGTH + 500)
        prompt = build_user_prompt([REAL_VAR], long_context)
        # The sanitized context should be truncated
        assert "x" * _MAX_CONTEXT_LENGTH in prompt
        assert "…" in prompt

    def test_single_variable_shows_no_interactions(self):
        prompt = build_user_prompt([REAL_VAR], "test")
        assert "(none — fewer than 2 variables)" in prompt

    def test_only_categorical_shows_no_quadratics(self):
        prompt = build_user_prompt([CAT_VAR], "test")
        assert "(none — no continuous variables)" in prompt


class TestBuildEdisonQuery:
    def test_contains_system_context(self):
        query = build_edison_query(SAMPLE_VARS, "Catalyst optimization")
        assert "System: Catalyst optimization" in query

    def test_contains_variable_descriptions(self):
        query = build_edison_query(SAMPLE_VARS, "test")
        assert "Temperature (real)" in query
        assert "Pressure (integer)" in query
        assert "Catalyst (categorical)" in query

    def test_system_context_sanitized(self):
        long_context = "y" * (_MAX_CONTEXT_LENGTH + 100)
        query = build_edison_query([REAL_VAR], long_context)
        assert "…" in query


# ===================================================================
# Provider factory
# ===================================================================

class TestGetStructuringProvider:
    @patch("api.services.llm_config.resolve_api_key", return_value="test-key")
    @patch("api.services.providers.openai_provider.OpenAIProvider")
    def test_openai_provider(self, MockProvider, mock_resolve):
        config = MagicMock()
        config.provider = "openai"
        config.api_key = "sk-test"
        config.model = "gpt-4o"

        result = _get_structuring_provider(config)
        MockProvider.assert_called_once_with(api_key="test-key", model="gpt-4o")

    @patch("api.services.llm_config.resolve_api_key", return_value=None)
    @patch("api.services.providers.ollama_provider.OllamaProvider")
    def test_ollama_provider(self, MockProvider, mock_resolve):
        config = MagicMock()
        config.provider = "ollama"
        config.api_key = None
        config.model = "llama3.2"
        config.base_url = "http://localhost:11434"

        result = _get_structuring_provider(config)
        MockProvider.assert_called_once_with(
            model="llama3.2",
            base_url="http://localhost:11434/v1"
        )

    @patch("api.services.llm_config.resolve_api_key", return_value=None)
    @patch("api.services.providers.ollama_provider.OllamaProvider")
    def test_ollama_base_url_already_has_v1(self, MockProvider, mock_resolve):
        config = MagicMock()
        config.provider = "ollama"
        config.api_key = None
        config.model = "llama3.2"
        config.base_url = "http://myserver:11434/v1"

        _get_structuring_provider(config)
        MockProvider.assert_called_once_with(
            model="llama3.2",
            base_url="http://myserver:11434/v1"
        )

    def test_unknown_provider_raises(self):
        config = MagicMock()
        config.provider = "anthropic"
        with pytest.raises(ValueError, match="Unknown structuring provider"):
            _get_structuring_provider(config)


# ===================================================================
# SSE helper
# ===================================================================

class TestSse:
    def test_format(self):
        result = _sse({"status": "complete", "result": {}})
        assert result.startswith("data: ")
        assert result.endswith("\n\n")

    def test_valid_json_payload(self):
        result = _sse({"key": "value"})
        payload = result[len("data: "):-2]  # strip "data: " prefix and "\n\n" suffix
        parsed = json.loads(payload)
        assert parsed == {"key": "value"}


# ===================================================================
# Streaming orchestrator
# ===================================================================

class TestSuggestEffectsStream:
    """Tests for the main streaming orchestrator with mocked providers."""

    @pytest.fixture
    def mock_request(self):
        """Create a minimal SuggestEffectsRequest-like mock (no Edison)."""
        request = MagicMock()
        request.edison_config = None
        request.system_context = "Catalyst optimization for CO2 reduction"
        request.structuring_provider = MagicMock()
        request.structuring_provider.provider = "openai"
        request.structuring_provider.model = "gpt-4o"
        return request

    def _collect_events(self, variables, request):
        """Helper to run the async generator and collect parsed events."""
        async def _run():
            events = []
            async for event in suggest_effects_stream(variables, request):
                events.append(json.loads(event[len("data: "):-2]))
            return events
        return asyncio.run(_run())

    def test_happy_path_no_edison(self, mock_request):
        """Without Edison, should emit 'structuring' then 'complete' events."""
        mock_result = {
            "effects": ["Temperature", "Pressure"],
            "reasoning": [{"effect": "Temperature", "reason": "Primary driver"}],
            "confidence": [{"effect": "Temperature", "level": "high"}],
            "sources": [],
            "disclaimer": "Test disclaimer",
        }

        mock_provider = AsyncMock()
        mock_provider.suggest_effects.return_value = mock_result

        with patch("api.services.llm_service._get_structuring_provider", return_value=mock_provider):
            events = self._collect_events(SAMPLE_VARS, mock_request)

        assert len(events) == 2
        assert events[0]["status"] == "structuring"
        assert events[1]["status"] == "complete"
        assert events[1]["result"]["effects"] == ["Temperature", "Pressure"]

    def test_provider_error_emits_error_event(self, mock_request):
        """If the structuring provider raises, should emit 'error' event."""
        mock_provider = AsyncMock()
        mock_provider.suggest_effects.side_effect = RuntimeError("API timeout")

        with patch("api.services.llm_service._get_structuring_provider", return_value=mock_provider):
            events = self._collect_events(SAMPLE_VARS, mock_request)

        assert len(events) == 2
        assert events[0]["status"] == "structuring"
        assert events[1]["status"] == "error"
        assert "API timeout" in events[1]["message"]

    def test_result_has_default_keys(self, mock_request):
        """Result should have all required keys even if provider returns partial."""
        mock_provider = AsyncMock()
        mock_provider.suggest_effects.return_value = {"effects": ["X"]}

        with patch("api.services.llm_service._get_structuring_provider", return_value=mock_provider):
            events = self._collect_events(SAMPLE_VARS, mock_request)

        result = events[-1]["result"]
        assert "effects" in result
        assert "reasoning" in result
        assert "confidence" in result
        assert "sources" in result
        assert "disclaimer" in result
        assert "literature_context" in result

    def test_default_disclaimer_applied(self, mock_request):
        """If provider doesn't return a disclaimer, the default should be used."""
        mock_provider = AsyncMock()
        mock_provider.suggest_effects.return_value = {"effects": []}

        with patch("api.services.llm_service._get_structuring_provider", return_value=mock_provider):
            events = self._collect_events(SAMPLE_VARS, mock_request)

        result = events[-1]["result"]
        assert result["disclaimer"] == _DEFAULT_DISCLAIMER

    def test_literature_context_none_without_edison(self, mock_request):
        """Without Edison, literature_context should be None."""
        mock_provider = AsyncMock()
        mock_provider.suggest_effects.return_value = {"effects": []}

        with patch("api.services.llm_service._get_structuring_provider", return_value=mock_provider):
            events = self._collect_events(SAMPLE_VARS, mock_request)

        result = events[-1]["result"]
        assert result["literature_context"] is None


# ===================================================================
# Schema validation
# ===================================================================

class TestEffectsSchema:
    def test_schema_has_required_fields(self):
        assert "effects" in EFFECTS_SCHEMA["properties"]
        assert "reasoning" in EFFECTS_SCHEMA["properties"]
        assert "confidence" in EFFECTS_SCHEMA["properties"]
        assert "sources" in EFFECTS_SCHEMA["properties"]
        assert "disclaimer" in EFFECTS_SCHEMA["properties"]

    def test_schema_is_strict(self):
        assert EFFECTS_SCHEMA["additionalProperties"] is False
