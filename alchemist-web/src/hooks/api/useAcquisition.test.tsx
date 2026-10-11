import { describe, it, expect, vi, afterEach } from 'vitest';
import { renderHook, waitFor, act } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { apiClient } from '../../api/client';
import { useSuggestNext } from './useAcquisition';

describe('useSuggestNext', () => {
  afterEach(() => vi.restoreAllMocks());

  it('invalidates the contour so "Show Next Point" moves to the new suggestion', async () => {
    vi.spyOn(apiClient, 'post').mockResolvedValue({ data: { suggestions: [{ x: 1 }], n_suggestions: 1 } } as any);
    const client = new QueryClient();
    const spy = vi.spyOn(client, 'invalidateQueries');
    const { result } = renderHook(() => useSuggestNext('s1'), {
      wrapper: ({ children }: { children: React.ReactNode }) => (
        <QueryClientProvider client={client}>{children}</QueryClientProvider>
      ),
    });
    await act(async () => {
      await result.current.mutateAsync({ strategy: 'EI', goal: 'maximize' } as any);
    });
    await waitFor(() => expect(spy).toHaveBeenCalledWith({ queryKey: ['contour-data', 's1'] }));
  });
});
