import { describe, it, expect, vi, beforeEach } from 'vitest';
import apiClient from '../client';
import { getQueue, getConfigChanges } from './queue';

vi.mock('../client');

describe('queue endpoints', () => {
  beforeEach(() => vi.clearAllMocks());

  it('getQueue calls the queue path and returns data', async () => {
    (apiClient.get as any) = vi.fn().mockResolvedValue({
      data: { items: [], n_pending: 0, n_running: 0, n_done: 0, n_failed: 0 },
    });
    const res = await getQueue('sess1');
    expect(apiClient.get).toHaveBeenCalledWith('/sessions/sess1/experiments/queue');
    expect(res.n_pending).toBe(0);
  });

  it('getConfigChanges calls the config-changes path', async () => {
    (apiClient.get as any) = vi.fn().mockResolvedValue({ data: { changes: [] } });
    const res = await getConfigChanges('sess1');
    expect(apiClient.get).toHaveBeenCalledWith('/sessions/sess1/audit/config-changes');
    expect(res.changes).toEqual([]);
  });
});
