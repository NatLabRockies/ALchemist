"""Every public OptimizationSession method appears in the built reference."""
import inspect
import re


def _public_members():
    from alchemist_core.session import OptimizationSession
    return [(n, m) for n, m in inspect.getmembers(OptimizationSession) if not n.startswith("_")]


def test_every_public_session_method_is_documented(built_site):
    public = [n for n, m in _public_members() if callable(m)]
    html = (built_site / "reference" / "python" / "session" / "index.html").read_text()
    missing = [n for n in public if f'id="alchemist_core.session.OptimizationSession.{n}"' not in html]
    assert missing == []


def test_no_blockquote_examples_in_session_docstrings():
    """A '> ' example renders as a blockquote; examples belong in fenced python blocks."""
    from alchemist_core.session import OptimizationSession
    docs = {"OptimizationSession": OptimizationSession.__doc__}
    docs.update({n: getattr(m, "__doc__", None) for n, m in _public_members()})
    # Python 3.13 dedents __doc__, so match a leading ">" at any indentation.
    offenders = [n for n, d in docs.items() if d and re.search(r"^\s*> ", d, re.M)]
    assert offenders == []
