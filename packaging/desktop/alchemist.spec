# PyInstaller spec for alchemist-server, the headless server the Electron app runs (folder mode).
#
# Build from the repo root after `npm run build` in alchemist-web/:
#   pyinstaller packaging/desktop/alchemist.spec --noconfirm
# or use packaging/desktop/build_macos.sh. Output: dist/alchemist-server/.
#
# Folder mode, not one-file: one-file unpacks torch to a temp dir on every launch.

from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules, copy_metadata

ROOT = Path(SPECPATH).resolve().parents[1]
WEB_DIST = ROOT / "alchemist-web" / "dist"

if not (WEB_DIST / "index.html").is_file():
    raise SystemExit(f"No frontend build at {WEB_DIST}. Run `npm ci && npm run build` in alchemist-web/ first.")

datas = [
    # api/main.py serves api/static/ relative to its own __file__.
    (str(WEB_DIST), "api/static"),
]
for dist in ["botorch", "gpytorch", "linear_operator", "torch", "alchemist-nrel", "fastapi", "uvicorn", "pydantic", "scikit-learn"]:
    try:
        datas += copy_metadata(dist)
    except Exception:
        pass

hiddenimports = (
    collect_submodules("api")
    + collect_submodules("alchemist_core")
    + collect_submodules("uvicorn")
)

excludes = [
    # Desktop (Tk) app and its deps: not part of the web app.
    "ui", "main", "customtkinter", "tksheet", "CTkMessagebox", "mplcursors",
    # Optional LLM extras, left out of the desktop build for now.
    "openai", "edison_client",
    # Dev and notebook tooling that may be importable from the build env.
    "IPython", "jupyter", "notebook", "pytest", "PyInstaller", "zmq", "sphinx", "alabaster",
    # Optional backends other packages pull in through hooks; ALchemist uses none.
    # pyarrow alone is ~110 MB (pandas only uses it when installed).
    "pyarrow", "PySide6", "shiboken6", "PyQt5", "PyQt6", "cryptography",
]
# The server has no UI of its own (Electron shows the window).
excludes.append("tkinter")

a = Analysis(
    [str(Path(SPECPATH) / "alchemist_server.py")],
    pathex=[str(ROOT)],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="alchemist-server",
    console=False,
    # PyInstaller ad-hoc signs on macOS (Apple Silicon won't run unsigned code).
)

coll = COLLECT(exe, a.binaries, a.datas, name="alchemist-server")
