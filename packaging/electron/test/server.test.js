import { test, expect, vi } from 'vitest'
import path from 'node:path'
import { spawn } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { serverCommand, startServer, stopServer, waitForHealth } from '../src/server.js'

const here = path.dirname(fileURLToPath(import.meta.url))
const fixture = name => path.join(here, 'fixtures', name)

test('packaged command points inside resources (Windows)', () => {
  const c = serverCommand({ isPackaged: true, resourcesPath: '/R', platform: 'win32', env: {} })
  expect(c.command).toBe(path.join('/R', 'alchemist-server', 'alchemist-server.exe'))
  expect(c.args).toEqual([])
})

test('packaged command points inside resources (macOS)', () => {
  const c = serverCommand({ isPackaged: true, resourcesPath: '/R', platform: 'darwin', env: {} })
  expect(c.command).toBe(path.join('/R', 'alchemist-server', 'alchemist-server'))
})

test('dev command runs the Python source with ALCHEMIST_PYTHON', () => {
  const c = serverCommand({ isPackaged: false, resourcesPath: '/R', platform: 'darwin', env: { ALCHEMIST_PYTHON: '/py' } })
  expect(c.command).toBe('/py')
  expect(c.args).toEqual([path.resolve(here, '..', '..', 'desktop', 'alchemist_server.py')])
})

test('startServer passes the port and --watch-stdin', async () => {
  const child = startServer({ command: process.execPath, args: [fixture('echo-args.mjs')], port: 38799, stdio: ['pipe', 'pipe', 'ignore'] })
  let out = ''
  child.stdout.on('data', d => { out += d })
  await new Promise(res => child.on('exit', res))
  expect(JSON.parse(out)).toEqual(['--port', '38799', '--watch-stdin'])
})

test('stopServer kills a server that ignores stdin close', async () => {
  const child = spawn(process.execPath, [fixture('stubborn-server.mjs')], { stdio: ['pipe', 'ignore', 'ignore'] })
  const t0 = Date.now()
  await stopServer(child, { graceMs: 500 })
  expect(child.exitCode !== null || child.signalCode !== null).toBe(true)
  expect(Date.now() - t0).toBeLessThan(3000)
})

test('stopServer is graceful when the server exits on stdin close', async () => {
  const child = spawn(process.execPath, [fixture('polite-server.mjs')], { stdio: ['pipe', 'ignore', 'ignore'] })
  await stopServer(child, { graceMs: 5000 })
  expect(child.exitCode).toBe(0)
})

test('stopServer resolves at once for a server that already exited', async () => {
  const child = spawn(process.execPath, ['-e', ''], { stdio: ['pipe', 'ignore', 'ignore'] })
  await new Promise(res => child.on('exit', res))
  await stopServer(child, { graceMs: 5000 })
})

test('waitForHealth gives up when the process dies', async () => {
  const t0 = Date.now()
  expect(await waitForHealth(1, { timeoutMs: 5000, isAlive: () => false })).toBe(false)
  expect(Date.now() - t0).toBeLessThan(2000)
})

test('stopServer gives up and logs when a killed server never reports exit', async () => {
  const { EventEmitter } = await import('node:events')
  const child = Object.assign(new EventEmitter(), {
    pid: 999999, exitCode: null, signalCode: null, stdin: { end () {} }, kill () {},
  })
  const log = { error: vi.fn() }
  const t0 = Date.now()
  await stopServer(child, { graceMs: 100, killWaitMs: 200, log })
  expect(Date.now() - t0).toBeLessThan(2000)
  expect(log.error).toHaveBeenCalled()
})
