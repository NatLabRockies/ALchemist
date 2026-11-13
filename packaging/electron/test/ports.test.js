import { test, expect } from 'vitest'
import net from 'node:net'
import { choosePort, isPortFree, PREFERRED_PORTS } from '../src/ports.js'

test('preferred ports are 38765 to 38769', () => {
  expect(PREFERRED_PORTS).toEqual([38765, 38766, 38767, 38768, 38769])
})

test('takes the first preferred port when it is free', async () => {
  const r = await choosePort({ isPortFree: async () => true, isAlchemist: async () => false })
  expect(r).toEqual({ port: 38765 })
})

test('falls back when the preferred port is held by something else', async () => {
  const r = await choosePort({ isPortFree: async p => p !== 38765, isAlchemist: async () => false })
  expect(r).toEqual({ port: 38766 })
})

test('reports a running ALchemist instead of starting another', async () => {
  const r = await choosePort({ isPortFree: async () => false, isAlchemist: async p => p === 38765 })
  expect(r).toEqual({ conflict: 38765 })
})

test('uses an OS-assigned port when every preferred port is taken', async () => {
  const r = await choosePort({ isPortFree: async () => false, isAlchemist: async () => false })
  expect(PREFERRED_PORTS).not.toContain(r.port)
  expect(r.port).toBeGreaterThan(0)
})

test('isPortFree is false while a real listener holds the port', async () => {
  const srv = net.createServer()
  await new Promise(res => srv.listen(0, '127.0.0.1', res))
  const { port } = srv.address()
  try {
    expect(await isPortFree(port)).toBe(false)
  } finally {
    await new Promise(res => srv.close(res))
  }
  expect(await isPortFree(port)).toBe(true)
})
