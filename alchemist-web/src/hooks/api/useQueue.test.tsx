import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import * as queueApi from '../../api/endpoints/queue';
import { useExperimentQueue } from './useQueue';

function wrapper() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
}

describe('useExperimentQueue', () => {
  beforeEach(() => vi.restoreAllMocks());

  it('fetches the queue for a session', async () => {
    vi.spyOn(queueApi, 'getQueue').mockResolvedValue({
      items: [], n_pending: 2, n_running: 0, n_done: 0, n_failed: 0,
    });
    const { result } = renderHook(() => useExperimentQueue('sess1'), { wrapper: wrapper() });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.n_pending).toBe(2);
    expect(queueApi.getQueue).toHaveBeenCalledWith('sess1');
  });

  it('is disabled when sessionId is null', () => {
    const { result } = renderHook(() => useExperimentQueue(null), { wrapper: wrapper() });
    expect(result.current.fetchStatus).toBe('idle');
  });
});
