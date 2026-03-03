"""
Unit tests for api/services/llm_config.py

Tests config masking, persistence (with temp directory), and key resolution.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from api.services.llm_config import (
    _mask_value,
    _mask_dict,
    get_llm_config,
    get_llm_config_masked,
    resolve_api_key,
    save_llm_config,
    _load_full,
    _save_full,
)


# ===================================================================
# Masking
# ===================================================================

class TestMaskValue:
    def test_short_key_fully_masked(self):
        assert _mask_value("sk-1234") == "••••••••"

    def test_exactly_eight_chars_fully_masked(self):
        assert _mask_value("12345678") == "••••••••"

    def test_long_key_shows_prefix_and_suffix(self):
        result = _mask_value("sk-abc123def456ghi")
        assert result.startswith("sk-")
        assert result.endswith("6ghi")
        assert "•••" in result

    def test_nine_char_key_partially_masked(self):
        result = _mask_value("123456789")
        assert result == "123•••6789"


class TestMaskDict:
    def test_masks_api_key(self):
        d = {"api_key": "sk-very-long-secret-key-1234", "model": "gpt-4o"}
        result = _mask_dict(d)
        assert result["api_key"] != "sk-very-long-secret-key-1234"
        assert "•••" in result["api_key"]
        assert result["has_api_key"] is True
        assert result["model"] == "gpt-4o"

    def test_non_secret_keys_unchanged(self):
        d = {"provider": "openai", "model": "gpt-4o", "base_url": "http://localhost"}
        result = _mask_dict(d)
        assert result == d

    def test_nested_dict_masked(self):
        d = {
            "openai": {"api_key": "sk-test-long-key-12345", "model": "gpt-4o"},
            "ollama": {"model": "llama3.2"}
        }
        result = _mask_dict(d)
        assert "•••" in result["openai"]["api_key"]
        assert result["openai"]["has_api_key"] is True
        assert result["ollama"]["model"] == "llama3.2"

    def test_empty_api_key_not_masked(self):
        d = {"api_key": "", "model": "gpt-4o"}
        result = _mask_dict(d)
        assert result["api_key"] == ""
        assert "has_api_key" not in result

    def test_non_string_api_key_not_masked(self):
        d = {"api_key": 12345, "model": "gpt-4o"}
        result = _mask_dict(d)
        assert result["api_key"] == 12345
        assert "has_api_key" not in result


# ===================================================================
# Config I/O (with temp directory)
# ===================================================================

class TestConfigIO:
    @pytest.fixture(autouse=True)
    def temp_config(self, tmp_path):
        """Redirect CONFIG_PATH to a temp directory for each test."""
        self.config_path = tmp_path / "config.json"
        with patch("api.services.llm_config.CONFIG_PATH", self.config_path):
            yield

    def test_missing_file_returns_empty(self):
        with patch("api.services.llm_config.CONFIG_PATH", self.config_path):
            assert get_llm_config() == {}

    def test_save_and_load_roundtrip(self):
        config = {"openai": {"api_key": "sk-test", "model": "gpt-4o"}}
        with patch("api.services.llm_config.CONFIG_PATH", self.config_path):
            save_llm_config(config)
            result = get_llm_config()
        assert result == config

    def test_get_masked_hides_keys(self):
        config = {"openai": {"api_key": "sk-very-long-secret-key-1234", "model": "gpt-4o"}}
        with patch("api.services.llm_config.CONFIG_PATH", self.config_path):
            save_llm_config(config)
            result = get_llm_config_masked()
        assert "•••" in result["openai"]["api_key"]
        assert result["openai"]["model"] == "gpt-4o"

    def test_save_creates_parent_directory(self, tmp_path):
        nested_path = tmp_path / "deep" / "nested" / "config.json"
        with patch("api.services.llm_config.CONFIG_PATH", nested_path):
            save_llm_config({"test": True})
        assert nested_path.exists()

    def test_save_preserves_file_format(self):
        with patch("api.services.llm_config.CONFIG_PATH", self.config_path):
            save_llm_config({"openai": {"model": "gpt-4o"}})
        # File should be valid JSON with indentation
        raw = self.config_path.read_text()
        parsed = json.loads(raw)
        assert parsed["llm"]["openai"]["model"] == "gpt-4o"

    def test_corrupt_file_returns_empty(self):
        self.config_path.write_text("not valid json {{{")
        with patch("api.services.llm_config.CONFIG_PATH", self.config_path):
            assert get_llm_config() == {}


# ===================================================================
# Key resolution
# ===================================================================

class TestResolveApiKey:
    @pytest.fixture(autouse=True)
    def temp_config(self, tmp_path):
        """Redirect CONFIG_PATH to a temp directory for each test."""
        self.config_path = tmp_path / "config.json"
        with patch("api.services.llm_config.CONFIG_PATH", self.config_path):
            yield

    def test_request_key_takes_precedence(self):
        config = {"openai": {"api_key": "saved-key"}}
        with patch("api.services.llm_config.CONFIG_PATH", self.config_path):
            save_llm_config(config)
            result = resolve_api_key("openai", "request-key")
        assert result == "request-key"

    def test_falls_back_to_saved_config(self):
        config = {"openai": {"api_key": "saved-key-long-enough"}}
        with patch("api.services.llm_config.CONFIG_PATH", self.config_path):
            save_llm_config(config)
            result = resolve_api_key("openai", None)
        assert result == "saved-key-long-enough"

    def test_returns_none_when_no_key_available(self):
        with patch("api.services.llm_config.CONFIG_PATH", self.config_path):
            result = resolve_api_key("openai", None)
        assert result is None

    def test_returns_none_for_unknown_provider(self):
        config = {"openai": {"api_key": "sk-test"}}
        with patch("api.services.llm_config.CONFIG_PATH", self.config_path):
            save_llm_config(config)
            result = resolve_api_key("anthropic", None)
        assert result is None
