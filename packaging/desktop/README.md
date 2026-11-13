# alchemist-server (the Electron app's Python server)

PyInstaller bundles Python, ALchemist's dependencies and the built web frontend
into `alchemist-server`, a headless server. The Electron app in
`packaging/electron` ships it and runs it; see that folder's README for the app
itself.

## What the server does

`alchemist_server.py`:

```
alchemist-server --port 38765 --watch-stdin
```

- Serves the API and the web app on `127.0.0.1` only, at `--port`. The Electron
  app chooses the port.
- With `--watch-stdin`, shuts down when its stdin closes. The Electron app keeps
  stdin open, so the server exits when the app quits or crashes and can never
  outlive it.
- Runs from a per-user data directory, which holds `cache/` (sessions, recovery
  files), `logs/` and `matplotlib/` (a kept font cache):
  - macOS: `~/Library/Application Support/ALchemist`
  - Windows: `%LOCALAPPDATA%\ALchemist`
- In a frozen build, writes everything it logs or prints to
  `logs/alchemist.log`; `alchemist_core` reconfigures logging on import, so
  stdout and stderr themselves go to the file.

The optional LLM extras (`openai`, `edison-client`) are left out of the bundle,
as are pyarrow, PySide6 and other large packages ALchemist doesn't use
(`alchemist.spec`, `excludes`).

## Building

Builds can't be cross-compiled: build macOS on a Mac, Windows on Windows (or CI).

Prerequisites: a Python environment with ALchemist's dependencies
(`pip install -e .` with `SKIP_FRONTEND_BUILD=1`) plus `pip install pyinstaller`,
and Node.js for the frontend.

**macOS** (writes `packaging/desktop/dist/alchemist-server/`):

```bash
PYTHON=/path/to/env/bin/python packaging/desktop/build_macos.sh
```

Pass `--skip-frontend` to reuse an existing `alchemist-web/dist/`.

**Windows**:

```powershell
cd alchemist-web; npm ci; npm run build; cd ..
pyinstaller packaging/desktop/alchemist.spec --noconfirm --distpath packaging/desktop/dist --workpath packaging/desktop/build
```

Run it by hand to check it: `packaging/desktop/dist/alchemist-server/alchemist-server --port 38790`,
then open `http://localhost:38790/`. The first start of a new build on macOS
can take over a minute while the system scans the new binaries; later starts
take a few seconds.

## Tests

`tests/unit/test_desktop_server.py` and
`tests/integration/test_desktop_server_process.py` (runs the server from source
as a real process and checks it exits when stdin closes).
