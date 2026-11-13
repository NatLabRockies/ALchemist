// Port choice for the local ALchemist server.
//
// The port stays fixed between launches because the window's storage (theme,
// last session, linked Save files) is keyed by origin, port included.
import http from 'node:http'
import net from 'node:net'

export const HOST = '127.0.0.1'
export const PREFERRED_PORTS = [38765, 38766, 38767, 38768, 38769]

/** Whether a server could listen on this port now. */
export function isPortFree (port) {
  return new Promise(resolve => {
    const srv = net.createServer()
    srv.once('error', () => resolve(false))
    srv.listen(port, HOST, () => srv.close(() => resolve(true)))
  })
}

/** Whether an ALchemist server answers /health on this port. */
export function isAlchemist (port, timeoutMs = 1000) {
  return new Promise(resolve => {
    const req = http.get({ host: HOST, port, path: '/health', timeout: timeoutMs }, res => {
      let body = ''
      res.setEncoding('utf8')
      res.on('data', d => { body += d })
      res.on('end', () => {
        try {
          resolve(JSON.parse(body).service === 'alchemist-api')
        } catch {
          resolve(false)
        }
      })
    })
    req.on('timeout', () => req.destroy())
    req.on('error', () => resolve(false))
  })
}

function osAssignedPort () {
  return new Promise((resolve, reject) => {
    const srv = net.createServer()
    srv.once('error', reject)
    srv.listen(0, HOST, () => {
      const { port } = srv.address()
      srv.close(() => resolve(port))
    })
  })
}

/**
 * The port to start the server on, or the port of an ALchemist already running.
 * @returns {Promise<{port: number} | {conflict: number}>}
 */
export async function choosePort ({ isPortFree: free = isPortFree, isAlchemist: alchemist = isAlchemist, log } = {}) {
  for (const port of PREFERRED_PORTS) {
    if (await free(port)) return { port }
    if (await alchemist(port)) return { conflict: port }
  }
  const port = await osAssignedPort()
  log?.warn(`Ports ${PREFERRED_PORTS.join(', ')} are all taken; using ${port}. ` +
    'The origin changes, so settings and linked Save files from other launches will not be available.')
  return { port }
}
