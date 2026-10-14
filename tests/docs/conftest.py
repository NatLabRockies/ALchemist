"""Shared fixtures for the docs tests, and a count of docs Python blocks excluded from execution."""
import pytest

from doc_blocks import doc_pages, extract_blocks
from site_build import build_site


@pytest.fixture(scope="session")
def built_site():
    """The site built once per test session (skips without zensical; see site_build.py)."""
    return build_site()


def pytest_terminal_summary(terminalreporter):
    n = sum(
        1
        for page in doc_pages()
        for block in extract_blocks(page.read_text())
        if block.lang == "python" and "no-run" in block.classes
    )
    terminalreporter.write_line(f"docs python blocks marked no-run: {n}")
