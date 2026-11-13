// Pure helpers for the main process (no Electron imports, so they can be unit tested).
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const PAGES_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), 'pages')

/** True for the local server's origin and the app's own bundled pages; everything else opens in the browser. */
export function isLocalUrl (url, port) {
  let u
  try {
    u = new URL(url)
  } catch {
    return false
  }
  if (u.protocol === 'file:') {
    try {
      return path.dirname(fileURLToPath(u)) === PAGES_DIR
    } catch {
      return false // e.g. file://host/share/..., which isn't one of our pages
    }
  }
  return u.protocol === 'http:' && (u.hostname === 'localhost' || u.hostname === '127.0.0.1') && u.port === String(port)
}

const overlaps = (b, d) =>
  b.x < d.x + d.width && b.x + b.width > d.x && b.y < d.y + d.height && b.y + b.height > d.y

/** The saved window bounds, or null if missing, unreadable or off every display. */
export function loadBounds (file, displays) {
  try {
    const b = JSON.parse(fs.readFileSync(file, 'utf8'))
    const ok = ['x', 'y', 'width', 'height'].every(k => Number.isFinite(b[k])) && b.width > 0 && b.height > 0
    if (!ok || !displays.some(d => overlaps(b, d))) return null
    return { x: b.x, y: b.y, width: b.width, height: b.height }
  } catch {
    return null
  }
}

export function saveBounds (file, bounds) {
  fs.mkdirSync(path.dirname(file), { recursive: true })
  fs.writeFileSync(file, JSON.stringify(bounds))
}

/** The server's per-user data directory (same rules as app_data_dir in alchemist_server.py). */
export function dataDir (platform, env, homedir) {
  if (platform === 'darwin') return path.posix.join(homedir, 'Library', 'Application Support', 'ALchemist')
  if (platform === 'win32') return path.win32.join(env.LOCALAPPDATA || path.win32.join(homedir, 'AppData', 'Local'), 'ALchemist')
  return path.posix.join(env.XDG_DATA_HOME || path.posix.join(homedir, '.local', 'share'), 'ALchemist')
}

/** What the window shows when the server process exits; null while the app is quitting. */
export function pageForServerExit ({ phase, code, signal, logFile }) {
  if (phase === 'quitting') return null
  const how = signal ? `signal ${signal}` : `exit code ${code}`
  const headline = phase === 'starting' ? "ALchemist couldn't start." : "ALchemist's server stopped."
  return { page: 'error', message: `${headline} (${how}) See the log: ${logFile}` }
}

/** Shown when an ALchemist server already answers on the port; it may be an orphan with no window. */
export function conflictMessage (port, platform) {
  const tool = platform === 'win32' ? 'Task Manager' : platform === 'darwin' ? 'Activity Monitor' : 'your system monitor'
  return `Another ALchemist is running on port ${port}. Quit it and reopen this app. ` +
    `If no other ALchemist window is open, end "alchemist-server" (or an older "ALchemist") in ${tool}, then reopen.`
}
