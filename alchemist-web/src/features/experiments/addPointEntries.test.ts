import { describe, it, expect } from 'vitest';
import type { VariableDetail } from '../../api/types';
import { entryInputKeys, entryProblems, initialEntry, inputKeysOf, planBatch, setupBannerFor } from './addPointEntries';

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
