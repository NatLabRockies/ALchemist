// The main window: secure defaults, remembered bounds, and outside links sent to the browser.
import { BrowserWindow, dialog, screen, shell } from 'electron'
import { isLocalUrl, loadBounds, saveBounds } from './logic.js'
import { confirmUnload } from './quit.js'

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

  // The web app guards unsaved changes with beforeunload. Ask the way a browser would,
  // so the X button and Quit don't silently do nothing.
  win.webContents.on('will-prevent-unload', event => confirmUnload(event, () =>
    dialog.showMessageBoxSync(win, {
      type: 'warning',
      buttons: ['Quit Without Saving', 'Cancel'],
      defaultId: 1,
      cancelId: 1,
      message: 'This session has changes that are not saved to a file.',
      detail: 'Cancel and save the session, or quit anyway.',
    }) === 0))

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
