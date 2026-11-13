import { test, expect } from 'vitest'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createLogger } from '../src/log.js'

test('appends timestamped lines with the level, creating the folder', () => {
  const file = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'log-')), 'logs', 'electron.log')
  const log = createLogger(file)
  log.info('hello')
  log.warn('careful', new Error('boom'))
  const lines = fs.readFileSync(file, 'utf8').trim().split('\n')
  expect(lines).toHaveLength(2)
  expect(lines[0]).toMatch(/^\d{4}-\d\d-\d\dT.* INFO hello$/)
  expect(lines[1]).toContain('WARN careful Error: boom')
})
