"""Report how many docs Python blocks are excluded from execution."""
from doc_blocks import doc_pages, extract_blocks


def pytest_terminal_summary(terminalreporter):
    n = sum(
        1
        for page in doc_pages()
        for block in extract_blocks(page.read_text())
        if block.lang == "python" and "no-run" in block.classes
    )
    terminalreporter.write_line(f"docs python blocks marked no-run: {n}")
