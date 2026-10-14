"""The desktop entry point warns it is deprecated and explains a missing extra."""
import importlib
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_pyproject():
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


def _fresh_main(monkeypatch):
    """Import ``main`` fresh so this test never reuses a cached module."""
    monkeypatch.delitem(sys.modules, "main", raising=False)
    return importlib.import_module("main")


def test_missing_desktop_extra_exits_with_install_hint(monkeypatch, capsys):
    # A None entry in sys.modules makes `import <name>` raise ImportError, which is
    # what a missing optional package does. Dropping `ui.ui` forces a real import
    # attempt so the failure comes from customtkinter, not from a cached module.
    monkeypatch.setitem(sys.modules, "customtkinter", None)
    monkeypatch.delitem(sys.modules, "ui.ui", raising=False)
    main = _fresh_main(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert 'pip install "alchemist-nrel[desktop]"' in err
    assert "deprecated" in err.lower()


def test_main_module_imports_without_desktop_packages(monkeypatch):
    """Importing main must not pull in the GUI; only main() may."""
    monkeypatch.setitem(sys.modules, "customtkinter", None)
    monkeypatch.delitem(sys.modules, "ui.ui", raising=False)
    main = _fresh_main(monkeypatch)
    assert callable(main.main)


def test_deprecation_notice_printed_before_launch(monkeypatch, capsys):
    """With the extra present, the notice appears on stderr and the app starts."""
    launched = []

    class FakeApp:
        def mainloop(self):
            launched.append(True)

    import types

    fake_ui = types.ModuleType("ui.ui")
    fake_ui.ALchemistApp = FakeApp
    monkeypatch.setitem(sys.modules, "ui.ui", fake_ui)
    main = _fresh_main(monkeypatch)
    main.main()
    assert launched == [True]
    err = capsys.readouterr().err
    assert "deprecated" in err.lower()
    assert "alchemist-web" in err


def test_pyproject_has_desktop_extra():
    data = _load_pyproject()
    deps = " ".join(data["project"]["dependencies"])
    desktop = " ".join(data["project"]["optional-dependencies"]["desktop"])
    for pkg in ("customtkinter", "tksheet", "ctkmessagebox", "mplcursors", "tabulate"):
        assert pkg not in deps, f"{pkg} must not be a core dependency"
        assert pkg in desktop, f"{pkg} must be in the desktop extra"


def test_ci_test_job_installs_desktop_extra():
    """tests/unit/ui needs the GUI packages, so the CI test job must install them."""
    workflow = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    assert '.[test,desktop]' in workflow


def test_dev_extra_includes_desktop():
    extras = _load_pyproject()["project"]["optional-dependencies"]
    assert any("desktop" in d for d in extras["dev"])


def test_ui_package_still_ships_in_the_wheel():
    data = _load_pyproject()
    assert "ui*" in data["tool"]["setuptools"]["packages"]["find"]["include"]
    assert "main" in data["tool"]["setuptools"]["py-modules"]


def test_core_never_imports_desktop_packages():
    """A dependency may leave core only if nothing outside ui/ and main.py imports it."""
    import re

    pattern = re.compile(
        r"^\s*(?:import|from)\s+(customtkinter|tksheet|CTkMessagebox|ctkmessagebox|mplcursors|tabulate)\b",
        re.M,
    )
    offenders = []
    for pkg in ("alchemist_core", "api"):
        for path in (ROOT / pkg).rglob("*.py"):
            if pattern.search(path.read_text(encoding="utf-8")):
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
