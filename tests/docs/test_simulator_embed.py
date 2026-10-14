"""The pinned simulator is copied verbatim and embedded on the tutorial page."""
from pathlib import Path
import hashlib
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / "docs" / "tutorial" / "reactor" / "index.html"
PIN = "2ac5123"
SIM_REPO = Path.home() / "Library/CloudStorage/OneDrive-NLR/Active learning code development/alchemist-reactor-sim"


def test_simulator_copy_matches_pinned_commit():
    if not SIM_REPO.exists():
        import pytest; pytest.skip("simulator repo not on this machine (CI)")
    pinned = subprocess.run(["git", "-C", str(SIM_REPO), "show", f"{PIN}:dist/index.html"], check=True, capture_output=True).stdout
    assert hashlib.sha256(SIM.read_bytes()).hexdigest() == hashlib.sha256(pinned).hexdigest()


def test_tutorial_embeds_simulator():
    # Source-relative path: zensical rewrites it to ../../tutorial/... in the built page.
    text = (ROOT / "docs" / "get-started" / "first-loop.md").read_text()
    assert '<iframe class="alc-sim" src="../tutorial/reactor/index.html"' in text
    assert "2ac5123" in text
