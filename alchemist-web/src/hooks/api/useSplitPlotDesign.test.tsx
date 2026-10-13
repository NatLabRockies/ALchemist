import { describe, it, expect, vi } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { apiClient } from '../../api/client';
import { useGenerateSplitPlotDesign } from './useExperiments';

describe('useGenerateSplitPlotDesign', () => {
  it('posts to the split-plot endpoint', async () => {
    const spy = vi.spyOn(apiClient, 'post').mockResolvedValue({
      data: { points: [{ x1: 1, WholePlot: 1, RunOrder: 1 }], n_points: 1, info: { warnings: [] } },
    } as any);
    const client = new QueryClient();
    const { result } = renderHook(() => useGenerateSplitPlotDesign('s1'), {
      wrapper: ({ children }: { children: React.ReactNode }) => (
        <QueryClientProvider client={client}>{children}</QueryClientProvider>
      ),
    });
    result.current.mutate({ designs: { hard: { method: 'full_factorial' }, easy: { method: 'lhs', n_points: 3 } } });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(spy).toHaveBeenCalledWith('/sessions/s1/split-plot-design', expect.objectContaining({ designs: expect.any(Object) }));
  });
});
