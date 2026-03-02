"""
Persistent LLM provider configuration stored in ~/.alchemist/config.json.
Handles read/write of API keys and provider preferences across sessions.
"""

import json
import os
import stat
from pathlib import Path
from typing import Any

CONFIG_PATH = Path.home() / ".alchemist" / "config.json"

# Keys that contain secrets and must be masked in GET responses.
_SECRET_KEYS = {"api_key"}


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
    # Restrict to owner-only read/write (0o600).
    try:
        os.chmod(CONFIG_PATH, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass  # Best-effort on platforms that don't support chmod


def _mask_value(val: str) -> str:
    """Return a masked version of a secret string (e.g. 'sk-...Ab12')."""
    if len(val) <= 8:
        return "••••••••"
    return val[:3] + "•••" + val[-4:]


def _mask_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy a config dict, masking any secret values."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if k in _SECRET_KEYS and isinstance(v, str) and v:
            out[k] = _mask_value(v)
            out[f"has_{k}"] = True
        elif isinstance(v, dict):
            out[k] = _mask_dict(v)
        else:
            out[k] = v
    return out


def get_llm_config() -> dict[str, Any]:
    """Return the 'llm' section of ~/.alchemist/config.json (raw, with secrets)."""
    return _load_full().get("llm", {})


def get_llm_config_masked() -> dict[str, Any]:
    """Return the 'llm' section with API keys masked for safe client display."""
    return _mask_dict(get_llm_config())


def resolve_api_key(provider: str, request_key: str | None) -> str | None:
    """
    Resolve an API key: use the per-request key if provided, otherwise
    fall back to the saved config. Returns None if neither is available.
    """
    if request_key:
        return request_key
    cfg = get_llm_config()
    provider_cfg = cfg.get(provider, {})
    return provider_cfg.get("api_key") if isinstance(provider_cfg, dict) else None


def save_llm_config(llm_config: dict[str, Any]) -> None:
    """Merge-update the 'llm' section of ~/.alchemist/config.json."""
    full = _load_full()
    full["llm"] = llm_config
    _save_full(full)
