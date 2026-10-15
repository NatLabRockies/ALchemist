# Desktop app (experimental, unsigned)

A double-clickable ALchemist for people without Python: PyInstaller bundles
Python, the dependencies and the built web frontend; a small launcher starts the
server and opens the default browser.

These builds are **not signed**. macOS and Windows warn before the first launch
(see "Opening an unsigned build"). Nothing here is published.

## What the launcher does

`alchemist_launcher.py`:

- Serves on `127.0.0.1` only, on port 8765 (falling back to 8766–8769, then any
  free port). The port stays fixed because the browser keys saved settings, the
  last session and linked Save files by origin, port included.
- Runs from a per-user data directory, which holds `cache/` (sessions, recovery
  files) and `logs/`:
  - macOS: `~/Library/Application Support/ALchemist`
  - Windows: `%LOCALAPPDATA%\ALchemist`
- Opens `http://localhost:8765/` in the default browser. Linked-file Save needs
  Chrome or Edge.
- Shows a tray icon (menu bar on macOS) with **Open ALchemist** and **Quit
  ALchemist**. Launching the app again while it runs only reopens the browser.

The optional LLM extras (`openai`, `edison-client`) are left out of the bundle.

## Building

Builds can't be cross-compiled: build macOS on a Mac, Windows on Windows (or CI).

Prerequisites: a Python environment with ALchemist's dependencies
(`pip install -e .` with `SKIP_FRONTEND_BUILD=1`), plus `pip install pyinstaller pystray`,
and Node.js for the frontend.

**macOS** (writes `packaging/desktop/dist/ALchemist.app` and a `.dmg`):

```bash
PYTHON=/path/to/env/bin/python packaging/desktop/build_macos.sh
```

Pass `--skip-frontend` to reuse an existing `alchemist-web/dist/`.

**Windows**: the `Desktop build (unsigned)` workflow
(`.github/workflows/desktop-build.yml`) runs on pushes to `packaging/**` branches,
smoke-tests the app and uploads the `ALchemist` folder as the
`ALchemist-windows-x64` artifact. Locally, the same steps are:

```powershell
cd alchemist-web; npm ci; npm run build; cd ..
pyinstaller packaging/desktop/alchemist.spec --noconfirm --distpath packaging/desktop/dist --workpath packaging/desktop/build
```

## Opening an unsigned build

- **macOS 15+**: open the `.dmg`, drag ALchemist to Applications, and open it.
  When macOS refuses, go to **System Settings → Privacy & Security**, find the
  ALchemist message and click **Open Anyway**. (Right-click → Open no longer
  bypasses Gatekeeper.)
- **Windows**: unzip and run `ALchemist.exe`. On the SmartScreen prompt click
  **More info → Run anyway**.

## Troubleshooting

The log is `logs/alchemist.log` in the data directory (launcher, server and
anything printed). If the app doesn't open, check there first.
