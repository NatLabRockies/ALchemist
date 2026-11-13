# Spike: linked-file Save inside Electron (2026-11-12)

Electron 45.1.0 (Chromium 156), macOS 26 on Apple Silicon. A plain `BrowserWindow`
with `contextIsolation: true`, `nodeIntegration: false`, `sandbox: true`, no preload,
loading the ALchemist web app from a source server at `http://localhost:38790/`.

- `showSaveFilePicker`, `showOpenFilePicker` and IndexedDB exist in the window.
- Caleb created a session, saved it as a linked file through the native Save
  dialog, quit, relaunched, and saved again: it worked, with no permission prompt.
- **No permission handler is needed.** Electron's defaults (no
  `setPermissionRequestHandler` / `setPermissionCheckHandler`) were in place, and
  no permission requests were logged.
- In the spike the Python server kept running across the relaunch, so the app
  reopened the session directly. In the real app the server restarts with the
  window, so the expected relaunch is today's: the "restore previous session"
  banner. Checked in Task 4.

Gate: passed. Continue with the plan as written.
