// Start, watch and stop the alchemist-server child process.
import { spawn, execFile } from 'node:child_process'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { isAlchemist } from './ports.js'

const here = path.dirname(fileURLToPath(import.meta.url))

/** How to run the server: the bundled build when packaged, the Python source in development. */
export function serverCommand ({ isPackaged, resourcesPath, platform, env }) {
  if (isPackaged) {
    const exe = platform === 'win32' ? 'alchemist-server.exe' : 'alchemist-server'
    return { command: path.join(resourcesPath, 'alchemist-server', exe), args: [] }
  }
  return {
    command: env.ALCHEMIST_PYTHON ?? 'python3',
    args: [path.resolve(here, '..', '..', 'desktop', 'alchemist_server.py')],
  }
}

/**
 * Start the server. Its stdin stays open as a lifeline: the server exits when it
 * closes, so it can't outlive this app even if the app crashes.
 */
export function startServer ({ command, args, port, stdio = ['pipe', 'ignore', 'ignore'] }) {
  return spawn(command, [...args, '--port', String(port), '--watch-stdin'], { stdio, windowsHide: true })
}

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms))

/** Poll /health until the server answers, it dies, or the time runs out. */
export async function waitForHealth (port, { timeoutMs = 120000, isAlive = () => true, intervalMs = 250 } = {}) {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline && isAlive()) {
    if (await isAlchemist(port, 1000)) return true
    await sleep(intervalMs)
  }
  return false
}

const exited = child => child.exitCode !== null || child.signalCode !== null

/** Close stdin (graceful), then kill after graceMs; on Windows, the whole process tree. */
export async function stopServer (child, { graceMs = 5000 } = {}) {
  if (exited(child)) return
  const done = new Promise(resolve => child.once('exit', resolve))
  child.stdin?.end()
  const timer = sleep(graceMs).then(() => 'timeout')
  if (await Promise.race([done, timer]) !== 'timeout') return
  if (process.platform === 'win32') {
    await new Promise(resolve => execFile('taskkill', ['/pid', String(child.pid), '/T', '/F'], () => resolve()))
  } else {
    child.kill('SIGKILL')
  }
  await done
}
