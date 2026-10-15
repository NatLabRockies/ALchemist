"""Every app screenshot a docs page or the landing page shows comes as a light/dark pair.

The markup is checked here: the light image carries .only-light and is followed by its -dark.png sibling
carrying .only-dark. That the -dark.png files exist is checked by `node shoot.mjs --check-pages` (and they
are written by the full screenshot run), not here.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
SHOTS = json.loads((DOCS / "scripts/screenshots/shots.json").read_text())["shots"]
SHOT_FILES = {s.get("file", f"guide/{s['name']}.png") for s in SHOTS}
EXCLUDED = {"_legacy", "scripts", "superpowers"}

MD_IMAGE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)(\{[^}]*\})?")
IMG_TAG = re.compile(r"<img\b[^>]*>")


def dark_of(name: str) -> str:
    return re.sub(r"\.png$", "-dark.png", name)


def image_refs(path: Path) -> list[tuple[str, str]]:
    """(file under docs/images, attribute text) for each image reference in a Markdown or HTML file."""
    text = path.read_text()
    refs = [(m.group(1), m.group(2) or "") for m in MD_IMAGE.finditer(text)]
    for m in IMG_TAG.finditer(text):
        src = re.search(r'\bsrc="([^"]+)"', m.group(0))
        if src:
            refs.append((src.group(1), m.group(0)))
    return [(src.split("images/", 1)[1], attrs) for src, attrs in refs if "images/" in src]


def pages() -> list[Path]:
    md = [p for p in DOCS.rglob("*.md") if not EXCLUDED & set(p.relative_to(DOCS).parts)]
    return md + [ROOT / "overrides/home.html"]


def test_every_app_screenshot_has_a_dark_pair_in_the_markup():
    problems = []
    for page in pages():
        refs = image_refs(page)
        for name, attrs in refs:
            if name in SHOT_FILES:
                if "only-light" not in attrs:
                    problems.append(f"{page.relative_to(ROOT)}: {name} is not .only-light")
                if not any(n == dark_of(name) and "only-dark" in a for n, a in refs):
                    problems.append(f"{page.relative_to(ROOT)}: {name} has no {dark_of(name)} with .only-dark")
            elif name in {dark_of(f) for f in SHOT_FILES} and "only-dark" not in attrs:
                problems.append(f"{page.relative_to(ROOT)}: {name} is not .only-dark")
    assert problems == []


def test_the_hero_and_the_tutorial_screenshots_are_among_the_pages_checked():
    # guards the test itself: it must see the landing hero and a page with several screenshots
    assert "alchemist-web.png" in {n for n, _ in image_refs(ROOT / "overrides/home.html")}
    assert len(image_refs(DOCS / "get-started/first-loop.md")) >= 8
