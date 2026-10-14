import { describe, it, expect } from 'vitest';
import { cvMetrics } from './modelMetrics';

describe('cvMetrics', () => {
  it('returns complete metrics', () => {
    const m = { rmse: 1, mae: 0.5, r2: 0.9 };
    expect(cvMetrics(m)).toBe(m);
  });

  it('returns null for the empty metrics a multi-objective model reports', () => {
    expect(cvMetrics({})).toBeNull();
    expect(cvMetrics(null)).toBeNull();
    expect(cvMetrics(undefined)).toBeNull();
  });

  it('returns null when any core metric is missing', () => {
    expect(cvMetrics({ rmse: 1, mae: 0.5 })).toBeNull();
  });
});
