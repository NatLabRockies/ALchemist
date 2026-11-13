import { test, expect, vi } from 'vitest'
import { createQuitFlow } from '../src/quit.js'

const setup = (overrides = {}) => {
  const calls = []
  const flow = createQuitFlow({
    getServer: () => ({ pid: 1 }),
    saveWindowBounds: () => calls.push('save'),
    stop: async () => { calls.push('stop') },
    exit: () => calls.push('exit'),
    setPhase: p => calls.push(`phase:${p}`),
    log: { info () {}, error () {} },
    ...overrides,
  })
  return { flow, calls }
}

test('quit saves the window bounds before stopping the server, then exits', async () => {
  const { flow, calls } = setup()
  const event = { preventDefault: vi.fn() }
  flow.onBeforeQuit(event)
  await flow.idle()
  expect(event.preventDefault).toHaveBeenCalledTimes(1)
  expect(calls).toEqual(['save', 'phase:quitting', 'stop', 'exit'])
})

test('quit with no server yet still saves bounds and lets the quit through', () => {
  const { flow, calls } = setup({ getServer: () => null })
  const event = { preventDefault: vi.fn() }
  flow.onBeforeQuit(event)
  expect(event.preventDefault).not.toHaveBeenCalled()
  expect(calls).toEqual(['save'])
})

test('a second before-quit while stopping is not blocked again', async () => {
  const { flow } = setup()
  flow.onBeforeQuit({ preventDefault () {} })
  await flow.idle()
  const again = { preventDefault: vi.fn() }
  flow.onBeforeQuit(again)
  expect(again.preventDefault).not.toHaveBeenCalled()
})

test('installing an update stops the server first, then lets the installer quit the app', async () => {
  const { flow, calls } = setup()
  await flow.stopForInstall()
  calls.push('install')
  expect(calls).toEqual(['save', 'phase:quitting', 'stop', 'install'])
  const event = { preventDefault: vi.fn() }
  flow.onBeforeQuit(event)
  expect(event.preventDefault).not.toHaveBeenCalled()
})

test('a failed bounds save does not stop the quit', async () => {
  const { flow, calls } = setup({ saveWindowBounds: () => { throw new Error('disk full') } })
  flow.onBeforeQuit({ preventDefault () {} })
  await flow.idle()
  expect(calls).toEqual(['phase:quitting', 'stop', 'exit'])
})
