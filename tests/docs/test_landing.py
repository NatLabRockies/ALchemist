"""Built landing page has the hero, both calls to action and three cards."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
ZENSICAL = Path(sys.executable).parent / "zensical"


def _built_index() -> str:
    subprocess.run([str(ZENSICAL), "build", "--strict", "--clean"], cwd=ROOT, check=True, capture_output=True)
    return (ROOT / "site" / "index.html").read_text()


def test_landing_hero_and_ctas():
    html = _built_index()
    assert 'class="alc-hero"' in html
    assert 'href="get-started/first-loop/"' in html
    assert "pip install alchemist-nrel" in html
    assert html.count('class="alc-card"') == 3


def test_landing_hides_sidebars():
    html = _built_index()
    assert 'data-md-type="navigation" hidden' in html
    assert 'data-md-type="toc" hidden' in html


def test_wordmark_has_light_and_dark_variants():
    html = _built_index()
    assert "NEW_LOGO_LIGHT.png" in html and "NEW_LOGO_DARK.png" in html
