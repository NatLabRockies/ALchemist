"""Checks on zensical.toml: every nav entry exists, legacy is excluded."""
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"


def _config():
    with open(ROOT / "zensical.toml", "rb") as fh:
        return tomllib.load(fh)["project"]


def _nav_paths(nav):
    for item in nav:
        if isinstance(item, str):  # bare section-index entry, e.g. "reference/python/index.md"
            yield item
            continue
        for _title, target in item.items():
            if isinstance(target, list):
                yield from _nav_paths(target)
            else:
                yield target


def test_every_nav_entry_exists():
    missing = [p for p in _nav_paths(_config()["nav"]) if not (DOCS / p).exists()]
    assert missing == []


def test_nav_walker_accepts_bare_entries():
    assert list(_nav_paths(["a.md", {"B": ["b.md", {"C": "c.md"}]}])) == ["a.md", "b.md", "c.md"]


def test_six_top_level_tabs_in_order():
    titles = [next(iter(item)) for item in _config()["nav"]]
    assert titles == ["Get started", "Guide", "Python", "Lab automation", "Concepts", "Reference"]


def test_legacy_and_dev_material_excluded():
    excluded = _config()["exclude_docs"]
    for pattern in ("/_legacy/", "superpowers/", "/plans/", "/specs/", "/notes/"):
        assert pattern in excluded


def test_mkdocs_yml_removed():
    assert not (ROOT / "mkdocs.yml").exists()
