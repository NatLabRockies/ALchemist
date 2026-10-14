"""Checks on the built site: sub-path safe links."""
from pathlib import Path
import re
import tomllib
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]


def site_base_path() -> str:
    """Path part of site_url in zensical.toml, e.g. '/ALchemist/'."""
    config = tomllib.loads((ROOT / "zensical.toml").read_text())
    return urlparse(config["project"]["site_url"]).path


def is_unsafe_link(value: str, base_path: str) -> bool:
    """True for a root-absolute URL outside the site's own sub-path."""
    if not value.startswith("/") or value.startswith("//"):
        return False
    return not value.startswith(base_path)


def test_unsafe_link_predicate():
    base = "/ALchemist/"
    assert not is_unsafe_link("/ALchemist/assets/x.css", base)
    assert is_unsafe_link("/assets/x.css", base)
    assert not is_unsafe_link("//cdn.example.com/x.js", base)
    assert not is_unsafe_link("../assets/x.css", base)
    assert not is_unsafe_link("https://example.com/x.js", base)


def test_no_root_absolute_links(built_site):
    base = site_base_path()
    assert base.startswith("/") and base.endswith("/")
    offenders = []
    for html in built_site.rglob("*.html"):
        for m in re.finditer(r'(?:href|src)="(/[^"]*)"', html.read_text()):
            if is_unsafe_link(m.group(1), base):
                offenders.append(f"{html.relative_to(built_site)}: {m.group(1)}")
    assert offenders == []
