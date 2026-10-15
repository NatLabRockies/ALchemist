"""
Desktop launcher for the ALchemist web app (PyInstaller entry point).

Starts the FastAPI server on 127.0.0.1, opens the user's default browser and
puts an icon in the system tray (macOS menu bar) with Open and Quit. A second
launch while one is running only reopens the browser.

Choices that matter (see packaging/desktop/README.md):
- A fixed preferred port. The browser keys localStorage and IndexedDB by
  origin, port included, so a new port each launch would lose the theme, the
  last session and every linked Save file.
- 127.0.0.1 only: no firewall prompt, nothing exposed to the network.
- A per-user working directory, set before ``api.main`` is imported, because
  the session store creates ``cache/`` relative to the current directory and a
  .app opened from Finder starts in ``/``.
- The default browser rather than an embedded web view: linked-file Save needs
  the File System Access API, which macOS's WebKit view lacks.
"""

import json
import logging
import multiprocessing
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

APP_NAME = "ALchemist"
PREFERRED_PORTS = [8765, 8766, 8767, 8768, 8769]
HOST = "127.0.0.1"

log = logging.getLogger("alchemist.launcher")

if not getattr(sys, "frozen", False):
    # Run from a source checkout: make the repo root (api/, alchemist_core/) importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def app_data_dir() -> Path:
    """Per-user directory for sessions, recovery files and logs."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    path = base / APP_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def icon_path() -> Path:
    """The app icon: bundled at the top level by PyInstaller, in public/ in a source checkout."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "NEW_ICON.png"
    return Path(__file__).resolve().parents[2] / "alchemist-web" / "public" / "NEW_ICON.png"


def configure_logging(data_dir: Path) -> None:
    """Log to logs/alchemist.log. A windowed app has no console to print to.

    alchemist_core reconfigures the root logger with force=True when it is
    imported, replacing any handler set here, and its handler writes to stdout.
    So in a frozen build stdout and stderr themselves go to the log file
    (they are None on Windows, which would break anything that prints).
    """
    log_dir = data_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "alchemist.log"
    if log_file.exists() and log_file.stat().st_size > 5_000_000:
        log_file.replace(log_dir / "alchemist.log.1")

    if getattr(sys, "frozen", False):
        stream = open(log_file, "a", buffering=1, encoding="utf-8")
        sys.stdout = stream
        sys.stderr = stream
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )


def health(port: int, timeout: float = 1.0) -> bool:
    """True if an ALchemist server answers on this port."""
    try:
        with urllib.request.urlopen(f"http://{HOST}:{port}/health", timeout=timeout) as resp:
            return json.load(resp).get("service") == "alchemist-api"
    except Exception:
        return False


def port_is_free(port: int) -> bool:
    """Whether uvicorn could listen here.

    Uvicorn sets SO_REUSEADDR, so connections in TIME_WAIT from a server that
    just quit don't block it. Probing without the option would call the port
    taken for a minute after every quit and move a quick relaunch to another
    port (and origin). Windows' SO_REUSEADDR means something else (sharing a
    live port), so it is left off there.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if sys.platform != "win32":
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((HOST, port))
            return True
        except OSError:
            return False


def choose_port() -> int:
    """The first free preferred port, else any free port (origin data won't carry over)."""
    for port in PREFERRED_PORTS:
        if port_is_free(port):
            return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        port = s.getsockname()[1]
    log.warning("Preferred ports %s are all taken; using %d. Browser-stored "
                "settings and linked Save files from other launches won't be available.",
                PREFERRED_PORTS, port)
    return port


def url_for(port: int) -> str:
    return f"http://localhost:{port}/"


def running_instance_port(instance_file: Path):
    """Port of an ALchemist already started by this launcher, if it's still up."""
    candidates = []
    try:
        candidates.append(int(json.loads(instance_file.read_text())["port"]))
    except Exception:
        pass
    candidates += [p for p in PREFERRED_PORTS if p not in candidates]
    for port in candidates:
        if not port_is_free(port) and health(port):
            return port
    return None


def install_macos_reopen_handler(on_reopen):
    """Reopen the browser when the running .app is launched again from Finder or the Dock.

    macOS doesn't start a second process for a running app; it sends the
    running one a reopen event, which NSApplication passes to its delegate.
    """
    try:
        import AppKit
        from Foundation import NSObject
    except ImportError:
        return None

    class ReopenDelegate(NSObject):
        def applicationShouldHandleReopen_hasVisibleWindows_(self, app, has_windows):
            on_reopen()
            return False

    delegate = ReopenDelegate.alloc().init()
    AppKit.NSApplication.sharedApplication().setDelegate_(delegate)
    return delegate  # the caller keeps a reference; NSApp's delegate is weak


def main() -> int:
    multiprocessing.freeze_support()

    data_dir = app_data_dir()
    configure_logging(data_dir)
    instance_file = data_dir / "instance.json"

    existing = running_instance_port(instance_file)
    if existing is not None:
        log.info("ALchemist is already running on port %d; opening the browser.", existing)
        webbrowser.open(url_for(existing))
        return 0

    port = choose_port()
    url = url_for(port)

    # Before importing api.main: the session store creates cache/ in the current
    # directory at import time, and matplotlib must not pick a GUI backend.
    os.chdir(data_dir)
    os.environ.setdefault("MPLBACKEND", "Agg")
    # PyInstaller's matplotlib runtime hook points MPLCONFIGDIR at a fresh temp
    # dir, which rebuilds the font cache (~3 s) on every launch. Keep it instead.
    os.environ["MPLCONFIGDIR"] = str(data_dir / "matplotlib")
    os.environ.setdefault("ALLOWED_ORIGINS", f"http://localhost:{port},http://{HOST}:{port}")

    t0 = time.perf_counter()
    import uvicorn
    from api.main import app
    log.info("Imported the API in %.1f s; serving %s from %s", time.perf_counter() - t0, url, data_dir)

    server = uvicorn.Server(uvicorn.Config(
        app, host=HOST, port=port, log_level="warning", access_log=False, workers=1,
    ))
    server_thread = threading.Thread(target=server.run, name="uvicorn", daemon=True)
    server_thread.start()

    def open_browser(*_):
        webbrowser.open(url)

    def wait_then_open():
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and server_thread.is_alive():
            if health(port, timeout=0.5):
                log.info("Server ready %.1f s after launch.", time.perf_counter() - t0)
                instance_file.write_text(json.dumps({"port": port, "pid": os.getpid()}))
                open_browser()
                return
            time.sleep(0.2)
        log.error("Server did not become ready; see the errors above.")

    import pystray
    from PIL import Image

    def quit_app(icon, _item=None):
        log.info("Quit requested.")
        server.should_exit = True
        server_thread.join(timeout=10)
        try:
            instance_file.unlink()
        except FileNotFoundError:
            pass
        icon.stop()

    icon = pystray.Icon(
        APP_NAME,
        Image.open(icon_path()),
        f"{APP_NAME} ({url})",
        menu=pystray.Menu(
            pystray.MenuItem("Open ALchemist", open_browser, default=True),
            pystray.MenuItem(f"Running at {url}", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit ALchemist", quit_app),
        ),
    )

    _delegate = install_macos_reopen_handler(open_browser) if sys.platform == "darwin" else None

    def setup(icon):
        icon.visible = True
        wait_then_open()

    icon.run(setup=setup)  # blocks on the main thread (required on macOS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
