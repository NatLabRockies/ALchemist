import { describe, it, expect, vi, afterEach } from 'vitest';
import { renderHook, waitFor, act } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import * as experimentsAPI from '../../api/endpoints/experiments';
import type { Experiment } from '../../api/types';
import { useCreateExperiment, useUploadExperiments } from './useExperiments';

function setup<T>(hook: () => T) {
  const client = new QueryClient();
  const spy = vi.spyOn(client, 'invalidateQueries');
  const { result } = renderHook(hook, {
    wrapper: ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    ),
  });
  return { result, spy };
}

describe('experiment mutations invalidate the contour (FE-11)', () => {
  afterEach(() => vi.restoreAllMocks());

  it('after a CSV upload, so an open contour picks up the new points', async () => {
    type Upload = Awaited<ReturnType<typeof experimentsAPI.uploadExperimentsCSV>>;
    vi.spyOn(experimentsAPI, 'uploadExperimentsCSV').mockResolvedValue({ n_experiments: 3 } as Upload);
    const { result, spy } = setup(() => useUploadExperiments('s1'));
    await act(async () => {
      await result.current.mutateAsync({ file: new File(['a'], 'a.csv') });
    });
    await waitFor(() => expect(spy).toHaveBeenCalledWith({ queryKey: ['contour-data', 's1'] }));
    expect(spy).toHaveBeenCalledWith({ queryKey: ['experiments', 's1'] });
  });

  it('after a single experiment is added', async () => {
    vi.spyOn(experimentsAPI, 'createExperiment').mockResolvedValue({ message: 'ok' });
    const { result, spy } = setup(() => useCreateExperiment('s1'));
    await act(async () => {
      await result.current.mutateAsync({ inputs: { x: 1 }, output: 2 } as Experiment);
    });
    await waitFor(() => expect(spy).toHaveBeenCalledWith({ queryKey: ['contour-data', 's1'] }));
  });
});
