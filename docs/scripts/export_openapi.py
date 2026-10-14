"""Write the ALchemist FastAPI OpenAPI schema to docs/reference/openapi.json.

Run from the repository root before building the docs:

    python docs/scripts/export_openapi.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "docs" / "reference" / "openapi.json"


def main(out: Path = DEFAULT_OUT) -> Path:
    sys.path.insert(0, str(ROOT))
    from api.main import app  # imported here so the path is set first

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n")
    return out


if __name__ == "__main__":
    print(main())
