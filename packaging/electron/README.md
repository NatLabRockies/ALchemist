# ALchemist desktop app (Electron)

The ALchemist web app in its own window. The app starts the headless Python
server from `packaging/desktop` (`alchemist-server`) as a child process, waits
for it, and shows `http://localhost:<port>/`. Closing the window quits the app
and stops the server, on macOS and Windows alike. Users need no Python.

Design: `.superpowers/specs/2026-11-12-electron-desktop-app-design.md` (local,
not in the repository).

## How it works

- `src/main.js` wires everything: single instance (a second launch focuses the
  window), the Electron profile in `<data dir>/electron`, the menus, startup,
  quit and updates.
- `src/ports.js` picks the port: 38765, then 38766–38769, then any free port.
  The port stays fixed because the window's storage (theme, last session,
  linked Save files) is keyed by origin. If an ALchemist server already answers
  on a preferred port, typically the old browser-launcher build, the app says
  so and starts nothing.
- `src/server.js` starts `alchemist-server --port N --watch-stdin`, waits for
  `/health`, and on quit closes the server's stdin, then kills it after 5 s
  (the whole process tree on Windows).
- `src/window.js` keeps the window on the local server: other links open in the
  default browser. `contextIsolation`, `sandbox`, no `nodeIntegration`, no
  preload. The web app knows nothing about Electron.
- `src/updates.js` checks GitHub Releases at startup and daily (packaged builds
  only). Windows downloads in the background and installs on quit; unsigned
  macOS builds can't install updates, so they show a dialog that opens the
  release page.
- `src/pages/` holds the loading and error pages; the error page shows the log
  path and opens the logs folder.
- Logs: `<data dir>/logs/electron.log` (this app) and `alchemist.log` (server).

Linked-file Save works in the window with Electron's default permissions; see
`SPIKE-NOTES.md`.

## Develop

```bash
cd alchemist-web && npm ci && npm run build && cd ..
cd packaging/electron
npm ci
npm test
ALCHEMIST_PYTHON=/path/to/env/bin/python npm start   # runs the server from source
```

## Build

macOS (Apple Silicon), from anywhere:

```bash
PYTHON=/path/to/env/bin/python packaging/electron/build_macos.sh
```

It builds the frontend, the server (`packaging/desktop/build_macos.sh`) and the
app: `dist/ALchemist-<version>-mac-arm64.dmg` and `.zip`, block maps and
`latest-mac.yml`. Ad-hoc signed. The version comes from `pyproject.toml`
(`scripts/set-version.mjs`, PEP 440 to semver).

Windows: CI only (`.github/workflows/desktop-release.yml`), producing
`ALchemist-Setup-<version>.exe` (NSIS, per-user, no admin) and `latest.yml`.

## Release

Publishing a GitHub Release runs `desktop-release.yml`: it builds the tag on
macOS and Windows, smoke-tests the packaged apps (`ALCHEMIST_SMOKE_OUT`, see
`src/smoke.js`) and attaches the installers and update metadata to the
Release. Run it by hand on any branch with
`gh workflow run desktop-release.yml -f ref=<branch>`; the installers are then
kept as run artifacts for 14 days.

## Signing (not yet)

macOS: set a Developer ID identity, hardened runtime and notarization in
`electron-builder.yml`, with the certificate in CI secrets; electron-updater
then installs updates on macOS too, using the Windows code path in
`src/updates.js`. Windows: add the certificate (or Azure signing) to the `win`
section and CI secrets.
