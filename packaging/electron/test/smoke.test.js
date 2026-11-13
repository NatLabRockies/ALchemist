import { test, expect, vi } from 'vitest'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { runSmoke } from '../src/smoke.js'

const fakeWindow = loaded => ({
  webContents: {
    executeJavaScript: vi.fn(async () => loaded),
    capturePage: vi.fn(async () => ({ toPNG: () => Buffer.from('png') })),
  },
})

test('writes health and page state, saves a screenshot, then quits once', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'smoke-'))
  const outFile = path.join(dir, 'result.json')
  const quit = vi.fn()
  await runSmoke({ port: 38765, window: fakeWindow(true), outFile, quit, isAlchemist: async () => true })
  expect(JSON.parse(fs.readFileSync(outFile, 'utf8'))).toEqual({ health: true, loaded: true })
  expect(fs.existsSync(path.join(dir, 'window.png'))).toBe(true)
  expect(quit).toHaveBeenCalledTimes(1)
})

test('records a page that did not load', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'smoke-'))
  const outFile = path.join(dir, 'result.json')
  await runSmoke({ port: 38765, window: fakeWindow(false), outFile, quit: () => {}, isAlchemist: async () => true })
  expect(JSON.parse(fs.readFileSync(outFile, 'utf8')).loaded).toBe(false)
})

test('a failed screenshot does not stop the result or the quit', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'smoke-'))
  const outFile = path.join(dir, 'result.json')
  const win = fakeWindow(true)
  win.webContents.capturePage = vi.fn(async () => { throw new Error('no display') })
  const quit = vi.fn()
  await runSmoke({ port: 38765, window: win, outFile, quit, isAlchemist: async () => true })
  expect(JSON.parse(fs.readFileSync(outFile, 'utf8'))).toEqual({ health: true, loaded: true })
  expect(quit).toHaveBeenCalledTimes(1)
})
