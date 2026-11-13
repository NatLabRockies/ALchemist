// Quitting. Electron closes the window first (the page's unsaved-changes guard can
// cancel that, see confirmUnload), then emits will-quit, where the server is stopped
// before the app exits.
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
    onWillQuit (event) {
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

/**
 * The page's beforeunload guard (unsaved changes) wants to keep the window open.
 * A browser would ask; Electron silently cancels the close unless we ask ourselves.
 * Must stay synchronous: Electron only honours preventDefault() made inside the event.
 * @param {(() => boolean)} askToLeave  true to close anyway
 */
export function confirmUnload (event, askToLeave) {
  if (askToLeave()) event.preventDefault()
}
