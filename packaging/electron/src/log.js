// A small file logger for the main process (also handed to electron-updater).
import fs from 'node:fs'
import path from 'node:path'

const text = a => (a instanceof Error ? `${a.name}: ${a.message}` : typeof a === 'string' ? a : JSON.stringify(a))

export function createLogger (file) {
  fs.mkdirSync(path.dirname(file), { recursive: true })
  const write = level => (...args) => {
    try {
      fs.appendFileSync(file, `${new Date().toISOString()} ${level} ${args.map(text).join(' ')}\n`)
    } catch {
      // Logging must never take the app down.
    }
  }
  return { info: write('INFO'), warn: write('WARN'), error: write('ERROR'), debug: () => {} }
}
