"""Find the zensical binary and build the docs site for the tests that need it.

Without the docs extra (`pip install -e ".[docs]"`) zensical is missing and the
build-dependent tests skip, so plain `pytest tests/` passes. CI's docs job sets
ALCHEMIST_DOCS_REQUIRE_ZENSICAL=1, which turns that skip into a failure.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REQUIRE_ENV = "ALCHEMIST_DOCS_REQUIRE_ZENSICAL"


def find_zensical() -> Path | None:
    """The zensical next to this interpreter, else the one on PATH, else None."""
    beside = Path(sys.executable).parent / "zensical"
    if beside.exists():
        return beside
    on_path = shutil.which("zensical")
    return Path(on_path) if on_path else None


def require_zensical(finder=find_zensical, environ=os.environ) -> Path:
    """Return the zensical path; skip the test if it is missing, or fail when REQUIRE_ENV=1."""
    path = finder()
    if path is not None:
        return path
    reason = 'zensical not installed (pip install -e ".[docs]")'
    if environ.get(REQUIRE_ENV) == "1":
        pytest.fail(f"{reason}, and {REQUIRE_ENV}=1 requires it", pytrace=False)
    pytest.skip(reason)


def build_site() -> Path:
    """Export the OpenAPI schema and run a strict, clean build; return the site directory."""
    zensical = require_zensical()
    subprocess.run([sys.executable, "docs/scripts/export_openapi.py"], cwd=ROOT, check=True)
    result = subprocess.run(
        [str(zensical), "build", "--strict", "--clean"], cwd=ROOT, capture_output=True, text=True
    )
    if result.returncode != 0:
        pytest.fail(f"zensical build failed:\n{result.stdout}\n{result.stderr}", pytrace=False)
    return ROOT / "site"
