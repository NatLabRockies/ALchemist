import { test, expect } from 'vitest'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { isLocalUrl, loadBounds, saveBounds, dataDir, pageForServerExit, conflictMessage } from '../src/logic.js'

test('only the local server origin is local', () => {
  expect(isLocalUrl('http://localhost:38765/session/1', 38765)).toBe(true)
  expect(isLocalUrl('http://127.0.0.1:38765/', 38765)).toBe(true)
  expect(isLocalUrl('http://localhost:38766/', 38765)).toBe(false)
  expect(isLocalUrl('https://natlabrockies.github.io/ALchemist/', 38765)).toBe(false)
  expect(isLocalUrl('not a url', 38765)).toBe(false)
})

test('the bundled loading and error pages count as local', () => {
  const page = 'file://' + path.join(path.dirname(new URL(import.meta.url).pathname), '..', 'src', 'pages', 'error.html')
  expect(isLocalUrl(page, 38765)).toBe(true)
  expect(isLocalUrl('file:///etc/passwd', 38765)).toBe(false)
})

const display = { x: 0, y: 0, width: 1440, height: 900 }

test('saved bounds round-trip', () => {
  const file = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'bounds-')), 'window-state.json')
  saveBounds(file, { x: 10, y: 20, width: 1200, height: 800 })
  expect(loadBounds(file, [display])).toEqual({ x: 10, y: 20, width: 1200, height: 800 })
})

test('saved bounds off every display are ignored', () => {
  const file = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'bounds-')), 'window-state.json')
  saveBounds(file, { x: 5000, y: 20, width: 1200, height: 800 })
  expect(loadBounds(file, [display])).toBeNull()
})

test('missing or corrupt bounds are ignored', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'bounds-'))
  expect(loadBounds(path.join(dir, 'none.json'), [display])).toBeNull()
  fs.writeFileSync(path.join(dir, 'bad.json'), '{not json')
  expect(loadBounds(path.join(dir, 'bad.json'), [display])).toBeNull()
})

test('dataDir matches the Python server', () => {
  expect(dataDir('darwin', {}, '/Users/a')).toBe('/Users/a/Library/Application Support/ALchemist')
  expect(dataDir('win32', { LOCALAPPDATA: 'C:\\Users\\a\\AppData\\Local' }, 'C:\\Users\\a'))
    .toBe(path.win32.join('C:\\Users\\a\\AppData\\Local', 'ALchemist'))
  expect(dataDir('win32', {}, 'C:\\Users\\a')).toBe(path.win32.join('C:\\Users\\a', 'AppData', 'Local', 'ALchemist'))
})

test('a server exit after load shows the error page with the log path', () => {
  const r = pageForServerExit({ phase: 'running', code: 1, signal: null, logFile: '/L/alchemist.log' })
  expect(r.page).toBe('error')
  expect(r.message).toContain("ALchemist's server stopped")
  expect(r.message).toContain('/L/alchemist.log')
})

test('a server exit while starting says it could not start', () => {
  expect(pageForServerExit({ phase: 'starting', code: 1, signal: null, logFile: '/L' }).message).toContain("ALchemist couldn't start")
})

test('a server exit while quitting shows nothing', () => {
  expect(pageForServerExit({ phase: 'quitting', code: 0, signal: null, logFile: '/L' })).toBeNull()
})

test('the conflict message says how to end a stray server', () => {
  expect(conflictMessage(38765, 'darwin')).toMatch(/38765/)
  expect(conflictMessage(38765, 'darwin')).toMatch(/Activity Monitor/)
  expect(conflictMessage(38765, 'win32')).toMatch(/Task Manager/)
  expect(conflictMessage(38765, 'win32')).toMatch(/alchemist-server/)
})
