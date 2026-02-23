"""
Persistent LLM provider configuration stored in ~/.alchemist/config.json.
Handles read/write of API keys and provider preferences across sessions.
"""

import json
from pathlib import Path
from typing import Any

CONFIG_PATH = Path.home() / ".alchemist" / "config.json"


def _load_full() -> dict[str, Any]:
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_full(config: dict[str, Any]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def get_llm_config() -> dict[str, Any]:
    """Return the 'llm' section of ~/.alchemist/config.json."""
    return _load_full().get("llm", {})


def save_llm_config(llm_config: dict[str, Any]) -> None:
    """Merge-update the 'llm' section of ~/.alchemist/config.json."""
    full = _load_full()
    full["llm"] = llm_config
    _save_full(full)
