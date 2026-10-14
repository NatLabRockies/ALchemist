"""List docs pages that are still placeholders, and fail if any remain.

Placeholder pages carry the note "Being rewritten for ALchemist 0.4.0".
They must all be written before a release is tagged and published; the
release workflows run this script and stop when it fails.

    python docs/scripts/check_unwritten_pages.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MARKER = "Being rewritten for ALchemist"
SKIP_DIRS = {"_legacy", "superpowers"}


def unwritten_pages(docs: Path) -> list[Path]:
    return sorted(
        page for page in docs.rglob("*.md")
        if not SKIP_DIRS.intersection(page.relative_to(docs).parts)
        and MARKER in page.read_text(encoding="utf-8")
    )


def main(docs: Path = ROOT / "docs") -> int:
    pages = unwritten_pages(docs)
    if not pages:
        print("All docs pages are written.")
        return 0
    print(f"{len(pages)} docs pages are still placeholders and must be written before release:")
    for page in pages:
        print(f"  {page.relative_to(docs)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
