// Quitting: save the window's bounds, stop the server, then exit. Electron's app.exit()
// closes windows without their 'close' event, so bounds are saved here, not on close.
export function createQuitFlow ({ getServer, saveWindowBounds, stop, exit, setPhase, log }) {
  let stopped = false
  let pending = Promise.resolve()

  const save = () => {
    try {
      saveWindowBounds()
    } catch (err) {
      log.error('Could not save the window size:', err)
    }
  }

  async function stopServerOnce () {
    setPhase('quitting')
    await stop(getServer())
    stopped = true
  }

  return {
    onBeforeQuit (event) {
      if (stopped) return
      save()
      if (!getServer()) return
      event.preventDefault()
      pending = stopServerOnce().finally(() => {
        log.info('Quit.')
        exit()
      })
    },
    // Restart to update: the installer replaces the server's files, so stop it first.
    async stopForInstall () {
      save()
      if (getServer()) await stopServerOnce()
      stopped = true
    },
    idle: () => pending,
  }
}
