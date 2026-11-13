import { test, expect } from 'vitest'
import { pep440ToSemver, readPyprojectVersion } from '../scripts/set-version.mjs'

test.each([
  ['0.4.0', '0.4.0'],
  ['0.5.0rc1', '0.5.0-rc.1'],
  ['0.5.0a2', '0.5.0-alpha.2'],
  ['0.5.0b1', '0.5.0-beta.1'],
])('%s -> %s', (pep, semver) => {
  expect(pep440ToSemver(pep)).toBe(semver)
})

test.each(['0.5.0.post1', '0.5', '0.5.0.dev3', 'v0.5.0'])('refuses %s', v => {
  expect(() => pep440ToSemver(v)).toThrow()
})

test('reads the project version from pyproject.toml', () => {
  expect(readPyprojectVersion('[build-system]\nrequires = []\n\n[project]\nname = "x"\nversion = "0.4.0"\n')).toBe('0.4.0')
  expect(() => readPyprojectVersion('[project]\nname = "x"\n')).toThrow()
})
