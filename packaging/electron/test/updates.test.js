import { test, expect, vi, beforeEach } from 'vitest'
import { EventEmitter } from 'node:events'
import { setupUpdates, releasePageUrl } from '../src/updates.js'

let updater, dialog, shell, log, timers
const flush = () => new Promise(resolve => setTimeout(resolve, 0))

beforeEach(() => {
  updater = Object.assign(new EventEmitter(), { checkForUpdates: vi.fn(async () => {}), quitAndInstall: vi.fn() })
  dialog = { showMessageBox: vi.fn(async () => ({ response: 0 })) }
  shell = { openExternal: vi.fn() }
  log = { info: vi.fn(), warn: vi.fn(), error: vi.fn() }
  timers = []
})
const setup = platform => setupUpdates({ updater, platform, dialog, shell, log, setInterval: (fn, ms) => timers.push({ fn, ms }) })

test('release page URL', () => {
  expect(releasePageUrl('0.5.0')).toBe('https://github.com/NatLabRockies/ALchemist/releases/tag/v0.5.0')
})

test('offline: an update error logs and shows no dialog', async () => {
  setup('darwin')
  updater.emit('error', new Error('getaddrinfo ENOTFOUND github.com'))
  await flush()
  expect(dialog.showMessageBox).not.toHaveBeenCalled()
  expect(log.warn).toHaveBeenCalled()
})

test('mac: an available update opens the release page, nothing is downloaded', async () => {
  setup('darwin')
  expect(updater.autoDownload).toBe(false)
  updater.emit('update-available', { version: '0.5.0' })
  await flush()
  expect(shell.openExternal).toHaveBeenCalledWith('https://github.com/NatLabRockies/ALchemist/releases/tag/v0.5.0')
})

test('mac: choosing Later opens nothing', async () => {
  dialog.showMessageBox = vi.fn(async () => ({ response: 1 }))
  setup('darwin')
  updater.emit('update-available', { version: '0.5.0' })
  await flush()
  expect(shell.openExternal).not.toHaveBeenCalled()
})

test('windows: downloads in the background and offers a restart', async () => {
  setup('win32')
  expect(updater.autoDownload).toBe(true)
  expect(updater.autoInstallOnAppQuit).toBe(true)
  updater.emit('update-available', { version: '0.5.0' })
  await flush()
  expect(dialog.showMessageBox).not.toHaveBeenCalled()
  updater.emit('update-downloaded', { version: '0.5.0' })
  await flush()
  expect(updater.quitAndInstall).toHaveBeenCalled()
})

test('checks at start and once a day, swallowing rejections', async () => {
  updater.checkForUpdates = vi.fn(async () => { throw new Error('offline') })
  setup('win32')
  await flush()
  expect(updater.checkForUpdates).toHaveBeenCalledTimes(1)
  expect(timers).toHaveLength(1)
  expect(timers[0].ms).toBe(86_400_000)
  timers[0].fn()
  await flush()
  expect(updater.checkForUpdates).toHaveBeenCalledTimes(2)
  expect(log.warn).toHaveBeenCalledTimes(2)
})

test('windows: Restart to update stops the server before installing', async () => {
  const order = []
  updater.quitAndInstall = vi.fn(() => order.push('install'))
  setupUpdates({ updater, platform: 'win32', dialog, shell, log, setInterval: () => {},
    beforeInstall: async () => { order.push('stop') } })
  updater.emit('update-downloaded', { version: '0.5.0' })
  await flush(); await flush()
  expect(order).toEqual(['stop', 'install'])
})
