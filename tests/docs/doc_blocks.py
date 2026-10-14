"""Extract fenced code blocks from the docs Markdown (top level and inside content tabs)."""
from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
EXCLUDED_DIRS = {"_legacy", "superpowers", "scripts"}
_OPEN = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,})\s*(?P<info>.*?)\s*$")


@dataclass(frozen=True)
class Block:
    lang: str
    classes: tuple[str, ...]
    code: str
    line: int  # 1-based line of the opening fence


def _parse_info(info: str) -> tuple[str, tuple[str, ...]]:
    info = info.strip()
    if info.startswith("{") and info.endswith("}"):
        classes = tuple(tok[1:] for tok in info[1:-1].split() if tok.startswith("."))
        return (classes[0] if classes else ""), classes[1:]
    lang = info.split()[0] if info else ""
    return lang, ()


def extract_blocks(text: str) -> list[Block]:
    lines = text.splitlines(keepends=True)
    blocks: list[Block] = []
    i = 0
    while i < len(lines):
        m = _OPEN.match(lines[i])
        if not m:
            i += 1
            continue
        indent, fence = m.group("indent"), m.group("fence")
        lang, classes = _parse_info(m.group("info"))
        close = re.compile(rf"^{re.escape(indent)}{fence}\s*$")
        body: list[str] = []
        j = i + 1
        while j < len(lines) and not close.match(lines[j]):
            body.append(lines[j])
            j += 1
        if j == len(lines):
            raise ValueError(f"unclosed code fence opened at line {i + 1}")
        blocks.append(Block(lang, classes, textwrap.dedent("".join(body)), i + 1))
        i = j + 1
    return blocks


def doc_pages() -> list[Path]:
    return sorted(
        p for p in DOCS.rglob("*.md")
        if not EXCLUDED_DIRS.intersection(p.relative_to(DOCS).parts)
    )
