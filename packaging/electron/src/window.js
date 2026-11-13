// The main window: secure defaults, remembered bounds, and outside links sent to the browser.
import { BrowserWindow, screen, shell } from 'electron'
import { isLocalUrl, loadBounds, saveBounds } from './logic.js'

export function createMainWindow ({ boundsFile, getPort, icon, onAppLink }) {
  const displays = screen.getAllDisplays().map(d => d.workArea)
  const bounds = loadBounds(boundsFile, displays) ?? { width: 1400, height: 900 }
  const win = new BrowserWindow({
    ...bounds,
    minWidth: 900,
    minHeight: 600,
    title: 'ALchemist',
    show: false,
    ...(icon ? { icon } : {}),
    webPreferences: { contextIsolation: true, nodeIntegration: false, sandbox: true },
  })
  win.once('ready-to-show', () => win.show())
  win.on('close', () => {
    try {
      saveBounds(boundsFile, win.getNormalBounds())
    } catch {
      // Not worth an error dialog: the window opens at its default size next time.
    }
  })

  // The app never leaves the local server or its own pages; other links open in the default browser.
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (!isLocalUrl(url, getPort())) openOutside(url)
    return { action: 'deny' }
  })
  win.webContents.on('will-navigate', (event, url) => {
    if (url.startsWith('alchemist:')) {
      event.preventDefault()
      onAppLink(url.slice('alchemist:'.length))
    } else if (!isLocalUrl(url, getPort())) {
      event.preventDefault()
      openOutside(url)
    }
  })
  return win
}

function openOutside (url) {
  if (/^https?:|^mailto:/.test(url)) shell.openExternal(url)
}
