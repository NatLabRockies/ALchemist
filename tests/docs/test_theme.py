"""Theme tokens exist and are wired into the built site."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
CSS = ROOT / "docs" / "stylesheets" / "extra.css"
TOKENS = {
    "--alc-blue": "#0079c1", "--alc-blue-deep": "#005e96", "--alc-navy": "#0b2f4f",
    "--alc-ink": "#13283b", "--alc-ground": "#f2f6f9", "--alc-hairline": "#d5dfe8",
}


def test_tokens_defined_with_spec_values():
    css = CSS.read_text()
    for name, value in TOKENS.items():
        assert re.search(rf"{name}:\s*{value}\b", css), name


def test_slate_scheme_overrides_present():
    assert '[data-md-color-scheme="slate"]' in CSS.read_text()


def test_toc_integrate_not_enabled():
    assert "toc.integrate" not in (ROOT / "zensical.toml").read_text()


def test_external_scripts_pinned_with_sri():
    html = (ROOT / "overrides" / "main.html").read_text()
    assert "mathjax@3.2.2" in html and 'integrity="sha384-' in html
