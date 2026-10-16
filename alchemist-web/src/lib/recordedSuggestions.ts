/**
 * Suggestions restored from the audit log may already have been recorded.
 * A suggestion counts as recorded when some experiment row has the same
 * value for every input key (W26). A real is compared to REAL_DECIMALS places
 * (the Add Point dialog records the full suggested value since W33, which
 * matches too; a point recorded by an older version is rounded), an integer
 * to a whole number, a discrete value exactly, a categorical level as text.
 * With no variable information a numeric pair is compared as a real.
 */
import { inputKeysOf } from '../features/experiments/addPointEntries';
import type { VariableDetail } from '../api/types';
import { REAL_DECIMALS } from './rounding';

function missing(v: unknown): boolean {
  return v === null || v === undefined || (typeof v === 'string' && v.trim() === '');
}

function asNumber(v: unknown): number | null {
  if (typeof v === 'number') return Number.isFinite(v) ? v : null;
  if (typeof v === 'string' && v.trim() !== '') {
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

function same(a: unknown, b: unknown, variable: VariableDetail | undefined): boolean {
  if (missing(a) || missing(b)) return false;
  if (variable?.type === 'categorical') return String(a) === String(b);
  const x = asNumber(a);
  const y = asNumber(b);
  if (x === null || y === null) return String(a) === String(b);
  switch (variable?.type) {
    case 'integer':
      return Math.round(x) === Math.round(y);
    case 'discrete':
      return x === y;
    default:
      // Same rounding as roundSuggested() applies to a real.
      return Number(x.toFixed(REAL_DECIMALS)) === Number(y.toFixed(REAL_DECIMALS));
  }
}

export function dropRecorded(
  suggestions: Record<string, unknown>[],
  experiments: Record<string, unknown>[],
  inputKeys?: string[],
  variables: VariableDetail[] = [],
): Record<string, unknown>[] {
  if (experiments.length === 0) return suggestions;
  const varByName = new Map(variables.map((v) => [v.name, v]));
  return suggestions.filter((s) => {
    const keys = inputKeys ?? inputKeysOf(s);
    if (keys.length === 0) return true; // nothing to compare: do not guess it is recorded
    return !experiments.some((row) =>
      keys.every((k) => k in row && same(s[k], row[k], varByName.get(k))),
    );
  });
}
