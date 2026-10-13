"""House style: no emoji, no old NREL addresses, no root-absolute links in Markdown."""
import re

import pytest

from doc_blocks import DOCS, doc_pages

EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿⭐✅❌]")
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
