"""Run every Python block on every docs page, page by page, top to bottom."""
import os
from pathlib import Path

import pytest

from doc_blocks import DOCS, doc_pages, extract_blocks


def _runnable(page: Path):
    return [b for b in extract_blocks(page.read_text()) if b.lang == "python" and "no-run" not in b.classes]


def _label(page: Path) -> str:
    try:
        return str(page.relative_to(DOCS))
    except ValueError:
        return page.name


def run_page(page: Path, workdir: Path) -> None:
    """Execute a page's runnable Python blocks in one namespace, with cwd = workdir."""
    namespace: dict = {"__name__": f"docs_{page.stem}"}
    old = os.getcwd()
    os.chdir(workdir)
    try:
        for block in _runnable(page):
            try:
                exec(compile(block.code, f"{_label(page)}:{block.line}", "exec"), namespace)
            except Exception as exc:
                pytest.fail(f"{_label(page)}:{block.line}: {type(exc).__name__}: {exc}")
    finally:
        os.chdir(old)


PAGES = [p for p in doc_pages() if _runnable(p)]


@pytest.mark.parametrize("page", PAGES, ids=_label)
def test_python_examples_run(page, tmp_path):
    run_page(page, tmp_path)


def test_python_page_runs_in_tmp_dir(tmp_path):
    page = tmp_path / "page.md"
    page.write_text('```python\nopen("written.txt", "w").write("x")\n```\n')
    work = tmp_path / "work"
    work.mkdir()
    run_page(page, work)
    assert (work / "written.txt").exists()
    assert not (DOCS / "written.txt").exists()


def test_failure_reports_page_and_line(tmp_path):
    page = tmp_path / "bad.md"
    page.write_text("# t\n\n```python\nraise ValueError('boom')\n```\n")
    with pytest.raises(pytest.fail.Exception, match=r"bad.md:3: ValueError: boom"):
        run_page(page, tmp_path)
