// ALchemist desktop app: the web app in its own window, backed by a local alchemist-server.
import { app, dialog, Menu, shell } from 'electron'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { createLogger } from './log.js'
import { conflictMessage, dataDir, pageForServerExit, saveBounds } from './logic.js'
import { buildMenu } from './menu.js'
import { choosePort } from './ports.js'
import { createQuitFlow } from './quit.js'
import { serverCommand, startServer, stopServer, waitForHealth } from './server.js'
import { runSmoke } from './smoke.js'
import { setupUpdates } from './updates.js'
import { createMainWindow } from './window.js'

const here = path.dirname(fileURLToPath(import.meta.url))
const DOCS_URL = 'https://natlabrockies.github.io/ALchemist/'
const ICON = path.join(here, '..', 'resources', 'icon.png')

// Electron's own profile lives beside the server's data, not mixed into it.
const DATA_DIR = dataDir(process.platform, process.env, os.homedir())
const LOGS_DIR = path.join(DATA_DIR, 'logs')
app.setPath('userData', path.join(DATA_DIR, 'electron'))
const log = createLogger(path.join(LOGS_DIR, 'electron.log'))

let win = null
let server = null
let port = null
let phase = 'starting' // starting | running | quitting
const BOUNDS_FILE = path.join(app.getPath('userData'), 'window-state.json')

const quitFlow = createQuitFlow({
  getServer: () => server,
  saveWindowBounds: () => { if (win) saveBounds(BOUNDS_FILE, win.getNormalBounds()) },
  stop: child => stopServer(child, { log }),
  exit: () => app.exit(0),
  setPhase: p => { phase = p },
  log,
})

const page = (name, params = {}) => {
  const url = pathToFileURL(path.join(here, 'pages', `${name}.html`))
  for (const [k, v] of Object.entries(params)) url.searchParams.set(k, v)
  return url.href
}

function showError (message, title) {
  log.error(message)
  win?.loadURL(page('error', title ? { message, title } : { message }))
}

async function start () {
  const choice = await choosePort({ log })
  if ('conflict' in choice) {
    showError(conflictMessage(choice.conflict, process.platform), 'ALchemist is already running')
    return
  }
  port = choice.port
  const { command, args } = serverCommand({
    isPackaged: app.isPackaged, resourcesPath: process.resourcesPath, platform: process.platform, env: process.env,
  })
  log.info(`Starting ${command} ${args.join(' ')} on port ${port}`)
  const t0 = Date.now()
  try {
    server = startServer({ command, args, port })
  } catch (err) {
    showError(`ALchemist couldn't start its server: ${err.message}`)
    return
  }
  server.on('error', err => showError(`ALchemist couldn't start its server: ${err.message}`))
  server.on('exit', (code, signal) => {
    log.info(`Server exited (code ${code}, signal ${signal}) during ${phase}`)
    const shown = pageForServerExit({ phase, code, signal, logFile: path.join(LOGS_DIR, 'alchemist.log') })
    if (shown) showError(shown.message)
  })
  const alive = () => server.exitCode === null && server.signalCode === null
  if (!await waitForHealth(port, { isAlive: alive })) {
    if (alive()) showError(`ALchemist's server didn't answer within 2 minutes. See the log: ${path.join(LOGS_DIR, 'alchemist.log')}`)
    return
  }
  log.info(`Server ready in ${((Date.now() - t0) / 1000).toFixed(1)} s`)
  phase = 'running'
  await win.loadURL(`http://localhost:${port}/`)
  if (process.env.ALCHEMIST_SMOKE_OUT) {
    // CI: give React a moment to render, record the result, quit.
    await new Promise(resolve => setTimeout(resolve, 5000))
    await runSmoke({ port, window: win, outFile: process.env.ALCHEMIST_SMOKE_OUT, quit: () => app.quit(), log })
    return
  }
  if (app.isPackaged) {
    // electron-updater is CommonJS; its autoUpdater is a property of the default export.
    const { autoUpdater } = (await import('electron-updater')).default
    setupUpdates({ updater: autoUpdater, platform: process.platform, dialog, shell, log, beforeInstall: quitFlow.stopForInstall })
  }
}

if (!app.requestSingleInstanceLock()) {
  app.quit()
} else {
  app.on('second-instance', () => {
    if (!win) return
    if (win.isMinimized()) win.restore()
    win.focus()
  })

  app.whenReady().then(async () => {
    if (!app.isPackaged && process.platform === 'darwin') app.dock.setIcon(ICON)
    Menu.setApplicationMenu(buildMenu({
      platform: process.platform, version: app.getVersion(), docsUrl: DOCS_URL, logsDir: LOGS_DIR, isPackaged: app.isPackaged,
    }))
    win = createMainWindow({
      boundsFile: BOUNDS_FILE,
      getPort: () => port,
      icon: app.isPackaged ? undefined : ICON,
      onAppLink: link => { if (link === 'open-logs') shell.openPath(LOGS_DIR) },
    })
    win.on('closed', () => { win = null })
    await win.loadURL(page('loading'))
    await start()
  })

  // Closing the window quits, on every platform: nothing runs without a window.
  app.on('window-all-closed', () => app.quit())

  app.on('before-quit', event => quitFlow.onBeforeQuit(event))
}
