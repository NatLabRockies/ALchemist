// Update checks against GitHub Releases (electron-updater).
//
// Windows: download in the background, install on quit, offer a restart.
// macOS: unsigned builds can't install updates, so only say a new version exists
// and open its release page. Once the app is signed, macOS can use the Windows path.
const RELEASES = 'https://github.com/NatLabRockies/ALchemist/releases'
const DAY_MS = 86_400_000

export const releasePageUrl = version => `${RELEASES}/tag/v${version}`

export function setupUpdates ({ updater, platform, dialog, shell, log, intervalMs = DAY_MS, setInterval = globalThis.setInterval }) {
  updater.logger = log
  const canInstall = platform === 'win32'
  updater.autoDownload = canInstall
  updater.autoInstallOnAppQuit = canInstall

  // Offline, rate-limited or no release yet: log it and carry on; the app works without updates.
  updater.on('error', err => log.warn('Update check failed:', err))

  if (canInstall) {
    updater.on('update-downloaded', async info => {
      const { response } = await dialog.showMessageBox({
        type: 'info', buttons: ['Restart to update', 'Later'], defaultId: 0, cancelId: 1,
        message: `ALchemist ${info.version} is ready to install.`,
        detail: 'It will also install the next time you quit ALchemist.',
      })
      if (response === 0) updater.quitAndInstall()
    })
  } else {
    updater.on('update-available', async info => {
      const { response } = await dialog.showMessageBox({
        type: 'info', buttons: ['Download', 'Later'], defaultId: 0, cancelId: 1,
        message: `ALchemist ${info.version} is available.`,
        detail: `You have ${updater.currentVersion?.version ?? 'an older version'}. Download the new version from GitHub.`,
      })
      if (response === 0) shell.openExternal(releasePageUrl(info.version))
    })
  }

  const check = () => Promise.resolve()
    .then(() => updater.checkForUpdates())
    .catch(err => log.warn('Update check failed:', err))
  check()
  setInterval(check, intervalMs)
}
