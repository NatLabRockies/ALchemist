// CI smoke test hook: with ALCHEMIST_SMOKE_OUT set, record whether the server answers
// and the web app rendered, keep a screenshot for the log, then quit.
import fs from 'node:fs'
import path from 'node:path'
import { isAlchemist as defaultIsAlchemist } from './ports.js'

export async function runSmoke ({ port, window, outFile, quit, isAlchemist = defaultIsAlchemist, log }) {
  fs.mkdirSync(path.dirname(outFile), { recursive: true })
  const health = await isAlchemist(port)
  const loaded = Boolean(await window.webContents.executeJavaScript(
    "document.querySelector('#root') !== null && document.querySelector('#root').childElementCount > 0"))
  fs.writeFileSync(outFile, JSON.stringify({ health, loaded }))
  try {
    const image = await window.webContents.capturePage()
    fs.writeFileSync(path.join(path.dirname(outFile), 'window.png'), image.toPNG())
  } catch (err) {
    // The screenshot is only for the CI log; it never decides the result.
    log?.warn('Smoke screenshot failed:', err)
  }
  quit()
}
