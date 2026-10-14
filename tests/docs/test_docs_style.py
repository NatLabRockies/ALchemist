"""House style: no emoji, no old NREL addresses, no root-absolute links in Markdown."""
import re

import pytest

from doc_blocks import DOCS, doc_pages

# Pictographic emoji only: plain marks (check, cross, arrows, math) stay allowed.
EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u26FF\u2705\u274C\u2753\u2757\u2B50\u26A1\u2728\uFE0F]")
PAGES = doc_pages()


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(DOCS)))
def test_no_emoji(page):
    assert not EMOJI.search(page.read_text())


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(DOCS)))
def test_no_nrel_addresses(page):
    assert "@nrel.gov" not in page.read_text()


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(DOCS)))
def test_no_root_absolute_markdown_links(page):
    assert not re.search(r"\]\(/(?!/)", page.read_text())


@pytest.mark.parametrize("text", ["\u2713", "\u2714", "\u2717", "\u2718", "\u00b1", "\u00b0", "\u2248", "\u2192", "\u00d7"])
def test_emoji_regex_allows_plain_symbols(text):
    assert not EMOJI.search(text)


@pytest.mark.parametrize("text", ["\u2705", "\U0001F3AF", "\u26a0\ufe0f", "\u2b50", "\u274c", "\u2728"])
def test_emoji_regex_flags_pictographs(text):
    assert EMOJI.search(text)
