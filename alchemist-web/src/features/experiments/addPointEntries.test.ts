import { describe, it, expect } from 'vitest';
import type { VariableDetail } from '../../api/types';
import { blankPoint, entryInputKeys, entryProblems, entryWarnings, initialEntry, inputKeysOf, planBatch, setupBannerFor, suggestionKey } from './addPointEntries';

const VARS: VariableDetail[] = [
  { name: 'x1', type: 'real', bounds: [0, 10], role: 'controllable', changes: 'easy' },
  { name: 'c', type: 'categorical', categories: ['a', 'b'], role: 'controllable', changes: 'hard' },
  { name: 'z', type: 'real', role: 'observed' },
];
const S = [
  { x1: 1, c: 'a', WholePlot: 1, RunOrder: 1, _queueItemId: 'q1' },
  { x1: 2, c: 'a', WholePlot: 1, RunOrder: 2, _queueItemId: 'q2' },
  { x1: 3, c: 'b', WholePlot: 2, RunOrder: 3, _queueItemId: 'q3' },
];

describe('plot keys and observed inputs', () => {
  it('plot keys are not inputs; observed variables are', () => {
    expect(inputKeysOf(S[0])).toEqual(['x1', 'c']);
    expect(entryInputKeys(S[0], VARS)).toEqual(['x1', 'c', 'z']);
    expect(initialEntry(S[0], VARS).actual).toEqual({ x1: '1', c: 'a', z: '' });
  });

  it('pre-fills the actual with the full-precision suggestion, so recording it shows no deviation (W33)', () => {
    const s = { x1: 274.123456789, c: 'a', _queueItemId: 'q9' };
    expect(initialEntry(s, VARS).actual.x1).toBe('274.123456789');
    const n: VariableDetail = { name: 'n', type: 'integer', bounds: [1, 9], role: 'controllable', changes: 'easy' };
    expect(initialEntry({ n: 3 }, [n]).actual.n).toBe('3');
  });

  it('a point with an output but no observed value is invalid', () => {
    const entries = { 'q:q1': { ...initialEntry(S[0], VARS), output: '0.5' } };
    const plan = planBatch([S[0]], entries, VARS);
    expect(plan.payload).toHaveLength(0);
    expect(plan.invalid[0].problems.join()).toMatch(/z is empty/);
  });

  it('observed values are sent with the inputs', () => {
    const entries = { 'q:q1': { ...initialEntry(S[0], VARS), output: '0.5', actual: { x1: '1', c: 'a', z: '25' } } };
    expect(planBatch([S[0]], entries, VARS).payload[0].inputs).toEqual({ x1: 1, c: 'a', z: 25 });
  });
});

describe('setupBannerFor', () => {
  it('shows at the first point and at each whole-plot change, listing hard variables', () => {
    expect(setupBannerFor(S, 0, VARS)).toEqual({ title: 'Whole plot 1 of 2', settings: [{ name: 'c', value: 'a' }] });
    expect(setupBannerFor(S, 1, VARS)).toBeNull();
    expect(setupBannerFor(S, 2, VARS)?.title).toBe('Whole plot 2 of 2');
  });

  it('depends on run order, not on how the user navigated', () => {
    // Visiting index 2 then going back to 1: index 1 still has no banner.
    setupBannerFor(S, 2, VARS);
    expect(setupBannerFor(S, 1, VARS)).toBeNull();
  });

  it('is null for points without plot metadata', () => {
    expect(setupBannerFor([{ x1: 1, c: 'a' }], 0, VARS)).toBeNull();
  });
});

describe('entryProblems: categorical values', () => {
  it('flags a value that is not a declared category', () => {
    const entry = { ...initialEntry(S[0], VARS), output: '0.5', actual: { x1: '1', c: 'B', z: '25' } };
    expect(entryProblems(entry, VARS)).toEqual(['c "B" is not one of a, b']);
  });

  it('accepts a declared category', () => {
    const entry = { ...initialEntry(S[0], VARS), output: '0.5', actual: { x1: '1', c: 'b', z: '25' } };
    expect(entryProblems(entry, VARS)).toEqual([]);
  });
});

describe('blankPoint (W28)', () => {
  it('starts empty for every non-observed variable, with a unique stable key', () => {
    const a = blankPoint(VARS, 'id-a');
    const b = blankPoint(VARS, 'id-b');
    expect(a).toEqual({ _manualId: 'id-a', _reason: 'Manual', x1: '', c: '' });
    expect(suggestionKey(a)).toBe('m:id-a');
    expect(suggestionKey(a)).not.toBe(suggestionKey(b));
    expect(initialEntry(a, VARS)).toEqual({ actual: { x1: '', c: '', z: '' }, output: '', noise: '', reason: 'Manual' });
  });

  it('plans no item_id for a completed manual point', () => {
    const a = blankPoint(VARS, 'id-a');
    const key = suggestionKey(a);
    const entries = { [key]: { actual: { x1: '2', c: 'b', z: '1' }, output: '0.5', noise: '', reason: 'Manual' } };
    const plan = planBatch([a], entries, VARS);
    expect(plan.payload).toEqual([{ inputs: { x1: 2, c: 'b', z: 1 }, output: 0.5, reason: 'Manual' }]);
  });

  it('a categorical left unchosen blocks saving', () => {
    const a = blankPoint(VARS, 'id-a');
    const entries = { [suggestionKey(a)]: { actual: { x1: '2', c: '', z: '1' }, output: '0.5', noise: '', reason: 'Manual' } };
    const plan = planBatch([a], entries, VARS);
    expect(plan.payload).toEqual([]);
    expect(plan.invalid[0].problems).toContain('c is empty');
  });
});

describe('entryWarnings', () => {
  it('names an actual value outside its variable bounds, but does not block saving', () => {
    const entry = { ...initialEntry(S[0], VARS), output: '0.5', actual: { x1: '12', c: 'a', z: '3' } };
    expect(entryWarnings(entry, VARS)).toEqual([{ name: 'x1', message: 'outside 0–10' }]);
    expect(entryProblems(entry, VARS)).toEqual([]);
  });

  it('is quiet for values inside the bounds, non-numbers, and variables without bounds', () => {
    expect(entryWarnings({ ...initialEntry(S[0], VARS), actual: { x1: '10', c: 'a', z: '99' } }, VARS)).toEqual([]);
    expect(entryWarnings({ ...initialEntry(S[0], VARS), actual: { x1: 'abc', c: 'a', z: '' } }, VARS)).toEqual([]);
  });
});
