"""Every public OptimizationSession method appears in the built reference."""
from pathlib import Path
import inspect
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
ZENSICAL = Path(sys.executable).parent / "zensical"


def test_every_public_session_method_is_documented():
    from alchemist_core.session import OptimizationSession
    public = [n for n, m in inspect.getmembers(OptimizationSession) if not n.startswith("_") and callable(m)]
    subprocess.run([str(ZENSICAL), "build", "--strict", "--clean"], cwd=ROOT, check=True, capture_output=True)
    html = (ROOT / "site" / "reference" / "python" / "session" / "index.html").read_text()
    missing = [n for n in public if f'id="alchemist_core.session.OptimizationSession.{n}"' not in html]
    assert missing == []


def test_no_blockquote_examples_in_session_docstring():
    from alchemist_core.session import OptimizationSession
    # Python 3.13 dedents __doc__, so match a leading ">" at any indentation.
    assert not re.search(r"^\s*> ", OptimizationSession.__doc__ or "", re.M)
