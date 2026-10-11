import { describe, it, expect, vi, afterEach } from 'vitest';
import { renderHook, waitFor, act } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { apiClient } from '../../api/client';
import { useTrainModel } from './useModels';

describe('useTrainModel', () => {
  afterEach(() => vi.restoreAllMocks());

  it('invalidates parity and every model plot after a Model-tab train (W9)', async () => {
    vi.spyOn(apiClient, 'post').mockResolvedValue({ data: { metrics: { r2: 0.9 } } } as any);
    const client = new QueryClient();
    const spy = vi.spyOn(client, 'invalidateQueries');
    const { result } = renderHook(() => useTrainModel('s1'), {
      wrapper: ({ children }: { children: React.ReactNode }) => (
        <QueryClientProvider client={client}>{children}</QueryClientProvider>
      ),
    });
    await act(async () => {
      await result.current.mutateAsync({ backend: 'sklearn', kernel: 'RBF' } as any);
    });
    await waitFor(() => expect(spy).toHaveBeenCalledWith({ queryKey: ['parity-data', 's1'] }));
    for (const key of ['model-info', 'session', 'metrics-data', 'qq-plot-data', 'contour-data']) {
      expect(spy).toHaveBeenCalledWith({ queryKey: [key, 's1'] });
    }
  });
});
