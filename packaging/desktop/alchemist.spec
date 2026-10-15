# PyInstaller spec for the ALchemist desktop app (folder mode).
#
# Build from the repo root after `npm run build` in alchemist-web/:
#   pyinstaller packaging/desktop/alchemist.spec --noconfirm
# or use packaging/desktop/build_macos.sh, which also makes the .dmg.
#
# Folder mode, not one-file: one-file unpacks torch to a temp dir on every launch.

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules, copy_metadata

ROOT = Path(SPECPATH).resolve().parents[1]
WEB_DIST = ROOT / "alchemist-web" / "dist"
PUBLIC = ROOT / "alchemist-web" / "public"

if not (WEB_DIST / "index.html").is_file():
    raise SystemExit(f"No frontend build at {WEB_DIST}. Run `npm ci && npm run build` in alchemist-web/ first.")

sys.path.insert(0, str(ROOT))
import alchemist_core  # noqa: E402

VERSION = alchemist_core.__version__

datas = [
    # api/main.py serves api/static/ relative to its own __file__.
    (str(WEB_DIST), "api/static"),
    (str(PUBLIC / "NEW_ICON.png"), "."),
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
    + ["pystray._darwin" if sys.platform == "darwin" else "pystray._win32"]
)

excludes = [
    # Desktop (Tk) app and its deps: not part of the web app.
    "ui", "main", "customtkinter", "tksheet", "CTkMessagebox", "mplcursors", "tkinter",
    # Optional LLM extras, left out of the desktop build for now.
    "openai", "edison_client",
    # Dev and notebook tooling that may be importable from the build env.
    "IPython", "jupyter", "notebook", "pytest", "PyInstaller", "zmq", "sphinx", "alabaster",
    # Optional backends other packages pull in through hooks; ALchemist uses none.
    # pyarrow alone is ~110 MB (pandas only uses it when installed).
    "pyarrow", "PySide6", "shiboken6", "PyQt5", "PyQt6", "cryptography",
]

a = Analysis(
    [str(Path(SPECPATH) / "alchemist_launcher.py")],
    pathex=[str(ROOT)],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

icon = str(PUBLIC / ("NEW_ICON.ico" if sys.platform == "win32" else "NEW_ICON.png"))

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ALchemist",
    console=False,
    icon=icon,
    # PyInstaller ad-hoc signs on macOS (Apple Silicon won't run unsigned code).
)

coll = COLLECT(exe, a.binaries, a.datas, name="ALchemist")

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="ALchemist.app",
        icon=icon,
        bundle_identifier="gov.nlr.alchemist",
        version=VERSION,
        info_plist={
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            # Menu-bar app: no Dock icon, no main window.
            "LSUIElement": True,
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "12.0",
        },
    )
