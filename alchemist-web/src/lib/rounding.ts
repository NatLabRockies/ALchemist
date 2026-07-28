import type { VariableDetail } from '../api/types';

/** Default decimal places for real (continuous) variables. */
export const REAL_DECIMALS = 3;

/**
 * Round a suggested value for display / actual-input pre-fill based on the
 * variable's type. Integers -> whole numbers, reals -> REAL_DECIMALS places,
 * categorical/discrete and non-numeric -> unchanged.
 */
export function roundSuggested(
  value: unknown,
  variable: VariableDetail | undefined,
): unknown {
  if (!variable) return value;
  if (typeof value !== 'number' || !Number.isFinite(value)) return value;

  switch (variable.type) {
    case 'integer':
      return Math.round(value);
    case 'real':
      return Number(value.toFixed(REAL_DECIMALS));
    default:
      // categorical, discrete (allowed_values) — pass through
      return value;
  }
}

/**
 * String form for display, preserving trailing zeros for real vars
 * (e.g. 901.410 rather than 901.41).
 */
export function formatSuggested(
  value: unknown,
  variable: VariableDetail | undefined,
): string {
  if (variable && typeof value === 'number' && Number.isFinite(value) && variable.type === 'real') {
    return value.toFixed(REAL_DECIMALS);
  }
  return String(roundSuggested(value, variable) ?? '');
}
