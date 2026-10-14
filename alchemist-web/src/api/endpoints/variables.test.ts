import { describe, it, expect } from 'vitest';
import { toAPIVariable } from './variables';

describe('toAPIVariable role and changes', () => {
  it('sends changes only when not easy', () => {
    expect(toAPIVariable({ name: 'x1', type: 'continuous', bounds: [0, 1] })).not.toHaveProperty('changes');
    expect(toAPIVariable({ name: 'c', type: 'categorical', categories: ['a'], changes: 'hard' }).changes).toBe('hard');
  });
  it('sends role observed and omits missing bounds', () => {
    const api = toAPIVariable({ name: 'z', type: 'continuous', role: 'observed' });
    expect(api.role).toBe('observed');
    expect(api).not.toHaveProperty('min');
    expect(api).not.toHaveProperty('changes');
  });
});
