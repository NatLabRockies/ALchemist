import { describe, it, expect } from 'vitest';
import { roundSuggested, formatSuggested, REAL_DECIMALS } from './rounding';
import type { VariableDetail } from '../api/types';

const realVar = (name: string): VariableDetail => ({ name, type: 'real', bounds: [0, 1000] });
const intVar = (name: string): VariableDetail => ({ name, type: 'integer', bounds: [0, 10] });
const catVar = (name: string): VariableDetail =>
  ({ name, type: 'categorical', categories: ['A', 'B'] });

describe('REAL_DECIMALS', () => {
  it('is 3 (per spec)', () => {
    expect(REAL_DECIMALS).toBe(3);
  });
});

describe('roundSuggested', () => {
  it('rounds real variables to 3 places (numeric collapses trailing zero)', () => {
    // 901.4096982394 -> toFixed(3) = "901.410" -> Number(...) = 901.41
    expect(roundSuggested(901.4096982394, realVar('temp'))).toBe(901.41);
  });

  it('rounds integer variables to whole numbers', () => {
    expect(roundSuggested(4.9997, intVar('count'))).toBe(5);
    expect(roundSuggested(3.2, intVar('count'))).toBe(3);
  });

  it('passes categorical values through unchanged', () => {
    expect(roundSuggested('A', catVar('mode'))).toBe('A');
  });

  it('passes value through unchanged when no matching variable is provided', () => {
    expect(roundSuggested(1.23456, undefined)).toBe(1.23456);
  });

  it('leaves non-numeric values unchanged even for real vars', () => {
    expect(roundSuggested('n/a', realVar('temp'))).toBe('n/a');
  });
});

describe('formatSuggested', () => {
  it('preserves trailing zeros for real display strings', () => {
    expect(formatSuggested(901.4096982394, realVar('temp'))).toBe('901.410');
  });

  it('formats integers as whole-number strings', () => {
    expect(formatSuggested(4.9997, intVar('count'))).toBe('5');
  });

  it('formats categorical values as their string form', () => {
    expect(formatSuggested('A', catVar('mode'))).toBe('A');
  });
});
