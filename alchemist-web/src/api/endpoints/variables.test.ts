import { describe, it, expect, vi } from 'vitest';
import { apiClient } from '../client';
import { toAPIVariable, exportVariables } from './variables';

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

describe('exportVariables keeps constraints (W35)', () => {
  const variables = [{ name: 'x1', type: 'real', min: 0, max: 10 }];

  it('asks the server for constraints', async () => {
    const get = vi.spyOn(apiClient, 'get').mockResolvedValue({ data: { variables, constraints: [] } });
    await exportVariables('sess1');
    expect(get).toHaveBeenCalledWith('/sessions/sess1/variables/export', {
      params: { include_constraints: true },
    });
    get.mockRestore();
  });

  it('writes the {variables, constraints} file when the session has constraints', async () => {
    const constraints = [{ type: 'inequality', coefficients: { x1: 3 }, rhs: 8, name: 'c_a' }];
    const get = vi.spyOn(apiClient, 'get').mockResolvedValue({ data: { variables, constraints } });
    const blob = await exportVariables('sess1');
    expect(JSON.parse(await blob.text())).toEqual({ variables, constraints });
    get.mockRestore();
  });

  it('keeps the bare variable list when there are none', async () => {
    const get = vi.spyOn(apiClient, 'get').mockResolvedValue({ data: { variables, constraints: [] } });
    const blob = await exportVariables('sess1');
    expect(JSON.parse(await blob.text())).toEqual(variables);
    get.mockRestore();
  });
});
