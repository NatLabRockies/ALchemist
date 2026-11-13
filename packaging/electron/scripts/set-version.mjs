// Set packaging/electron/package.json's version from pyproject.toml, so the
// desktop app always carries the release's version (PEP 440 -> semver).
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const PRE = { a: 'alpha', b: 'beta', rc: 'rc' }

export function pep440ToSemver (v) {
  const m = /^(\d+)\.(\d+)\.(\d+)(?:(a|b|rc)(\d+))?$/.exec(v)
  if (!m) throw new Error(`Can't map version ${JSON.stringify(v)} to semver; expected X.Y.Z, X.Y.ZaN, X.Y.ZbN or X.Y.ZrcN.`)
  const [, major, minor, patch, pre, n] = m
  return `${major}.${minor}.${patch}` + (pre ? `-${PRE[pre]}.${n}` : '')
}

export function readPyprojectVersion (toml) {
  const project = /^\[project\]\s*$([\s\S]*?)(?=^\[|(?![\s\S]))/m.exec(toml)
  const m = project && /^version\s*=\s*"([^"]+)"/m.exec(project[1])
  if (!m) throw new Error('No version in the [project] table of pyproject.toml')
  return m[1]
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const here = path.dirname(fileURLToPath(import.meta.url))
  const pyproject = fs.readFileSync(path.join(here, '..', '..', '..', 'pyproject.toml'), 'utf8')
  const version = pep440ToSemver(readPyprojectVersion(pyproject))
  const pkgFile = path.join(here, '..', 'package.json')
  const pkg = JSON.parse(fs.readFileSync(pkgFile, 'utf8'))
  pkg.version = version
  fs.writeFileSync(pkgFile, JSON.stringify(pkg, null, 2) + '\n')
  console.log(`package.json version: ${version}`)
}
