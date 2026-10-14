"""The release gate finds placeholder pages and passes once they are written."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "scripts"))
import check_unwritten_pages as gate  # noqa: E402


def test_placeholder_page_fails_the_gate(tmp_path):
    (tmp_path / "guide").mkdir()
    (tmp_path / "guide" / "train.md").write_text('# Train\n\n!!! note "Being rewritten for ALchemist 0.4.0"\n')
    (tmp_path / "done.md").write_text("# Done\n\nReal content.\n")
    assert gate.unwritten_pages(tmp_path) == [tmp_path / "guide" / "train.md"]
    assert gate.main(tmp_path) == 1


def test_legacy_pages_are_ignored_and_written_site_passes(tmp_path):
    (tmp_path / "_legacy").mkdir()
    (tmp_path / "_legacy" / "old.md").write_text("Being rewritten for ALchemist 0.4.0\n")
    (tmp_path / "index.md").write_text("# Home\n")
    assert gate.main(tmp_path) == 0
