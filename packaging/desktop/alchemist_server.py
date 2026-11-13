"""
Headless ALchemist server for the Electron desktop app (PyInstaller entry point).

The Electron app (packaging/electron) picks the port, starts this program as a
child process and shows the web app in its own window:

    alchemist-server --port 38765 --watch-stdin

It serves the API and the built web app on 127.0.0.1 only. With --watch-stdin
it shuts down when its stdin closes, which happens when Electron quits or dies,
so the server can never outlive the app.

Choices that matter (see packaging/desktop/README.md):
- A per-user working directory, set before ``api.main`` is imported, because
  the session store creates ``cache/`` relative to the current directory and an
  app opened from Finder starts in ``/``.
- 127.0.0.1 only: no firewall prompt, nothing exposed to the network.
"""

import argparse
import logging
import multiprocessing
import os
import sys
import threading
import time
from pathlib import Path
from typing import IO, Callable

APP_NAME = "ALchemist"
HOST = "127.0.0.1"

log = logging.getLogger("alchemist.server")

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


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="alchemist-server", description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, required=True, help="port to serve on (127.0.0.1)")
    parser.add_argument("--watch-stdin", action="store_true",
                        help="shut down when stdin closes (the parent app quit or died)")
    return parser.parse_args(argv)


def watch_stdin(stream: IO[bytes], on_eof: Callable[[], None]) -> threading.Thread:
    """Call ``on_eof`` once when ``stream`` reaches end of file, from a daemon thread."""
    def run():
        try:
            while stream.read(4096):
                pass
        except (OSError, ValueError):
            pass  # a closed or broken pipe means the parent is gone too
        on_eof()

    thread = threading.Thread(target=run, name="watch-stdin", daemon=True)
    thread.start()
    return thread


def _stdin_stream() -> IO[bytes] | None:
    """Binary stdin. A windowed (no-console) build can start with sys.stdin = None
    even when the parent passed a pipe, so fall back to file descriptor 0."""
    if sys.stdin is not None:
        return sys.stdin.buffer
    try:
        return os.fdopen(0, "rb")
    except OSError:
        return None


def main(argv: list[str] | None = None) -> int:
    multiprocessing.freeze_support()
    args = parse_args(sys.argv[1:] if argv is None else argv)

    data_dir = app_data_dir()
    configure_logging(data_dir)

    # Before importing api.main: the session store creates cache/ in the current
    # directory at import time, and matplotlib must not pick a GUI backend.
    os.chdir(data_dir)
    os.environ.setdefault("MPLBACKEND", "Agg")
    # PyInstaller's matplotlib runtime hook points MPLCONFIGDIR at a fresh temp
    # dir, which rebuilds the font cache (~3 s) on every launch. Keep it instead.
    os.environ["MPLCONFIGDIR"] = str(data_dir / "matplotlib")
    os.environ.setdefault("ALLOWED_ORIGINS", f"http://localhost:{args.port},http://{HOST}:{args.port}")

    t0 = time.perf_counter()
    import uvicorn
    from api.main import app
    log.info("Imported the API in %.1f s; serving 127.0.0.1:%d from %s",
             time.perf_counter() - t0, args.port, data_dir)

    server = uvicorn.Server(uvicorn.Config(
        app, host=HOST, port=args.port, log_level="warning", access_log=False, workers=1,
    ))

    if args.watch_stdin:
        def parent_gone():
            log.info("stdin closed; shutting down.")
            server.should_exit = True
        stdin = _stdin_stream()
        if stdin is None:
            # The parent still kills the process on quit; only the crash case is lost.
            log.error("--watch-stdin given but stdin is unavailable; the server "
                      "will not notice if the app dies.")
        else:
            watch_stdin(stdin, parent_gone)

    server.run()
    if not server.started:
        log.error("The server did not start on port %d; see the errors above.", args.port)
        return 1
    log.info("Server stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
