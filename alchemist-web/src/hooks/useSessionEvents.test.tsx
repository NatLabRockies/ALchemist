import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useSessionEvents } from './useSessionEvents';
import { useContourData } from './api/useVisualizations';
import * as visualizationsApi from '../api/endpoints/visualizations';

class MockWS {
  static instances: MockWS[] = [];
  url: string;
  onopen: any; onmessage: any; onerror: any; onclose: any;
  close = vi.fn();
  constructor(url: string) { this.url = url; MockWS.instances.push(this); }
  emit(data: any) { this.onmessage?.({ data: JSON.stringify(data) }); }
}

describe('useSessionEvents queue handling', () => {
  let client: QueryClient;
  beforeEach(() => {
    (globalThis as any).WebSocket = MockWS as any;
    MockWS.instances = [];
    client = new QueryClient();
  });
  afterEach(() => vi.restoreAllMocks());

  function wrapper() {
    return ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
  }

  it('invalidates the queue query on queue_item_updated', () => {
    const spy = vi.spyOn(client, 'invalidateQueries');
    renderHook(() => useSessionEvents('sess1'), { wrapper: wrapper() });
    MockWS.instances[0].emit({ event: 'queue_item_updated', item_id: 'a', status: 'running' });
    expect(spy).toHaveBeenCalledWith({ queryKey: ['experiments-queue', 'sess1'] });
  });

  it('invalidates the queue query on queue_updated', () => {
    const spy = vi.spyOn(client, 'invalidateQueries');
    renderHook(() => useSessionEvents('sess1'), { wrapper: wrapper() });
    MockWS.instances[0].emit({ event: 'queue_updated' });
    expect(spy).toHaveBeenCalledWith({ queryKey: ['experiments-queue', 'sess1'] });
  });

  it('resyncs the queue after a reconnect (onclose then reopen invalidates)', () => {
    vi.useFakeTimers();
    const spy = vi.spyOn(client, 'invalidateQueries');
    renderHook(() => useSessionEvents('sess1'), { wrapper: wrapper() });
    // simulate a drop: onclose schedules a 5s reconnect that opens a new socket
    MockWS.instances[0].onclose?.();
    vi.advanceTimersByTime(5000);
    // new socket exists; a queue event on it must still invalidate (resync path)
    const latest = MockWS.instances[MockWS.instances.length - 1];
    latest.onopen?.();
    latest.emit({ event: 'queue_updated' });
    expect(spy).toHaveBeenCalledWith({ queryKey: ['experiments-queue', 'sess1'] });
    vi.useRealTimers();
  });
});

describe('useSessionEvents keeps an open contour plot live', () => {
  let client: QueryClient;
  beforeEach(() => {
    (globalThis as any).WebSocket = MockWS as any;
    MockWS.instances = [];
    client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  });
  afterEach(() => vi.restoreAllMocks());

  function wrapper() {
    return ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
  }

  it.each([
    ['model_trained', { event: 'model_trained', metrics: {} }],
    ['experiments_updated', { event: 'experiments_updated', n_experiments: 11 }],
    ['queue_updated', { event: 'queue_updated' }],
  ])('invalidates contour-data on %s', (_name, message) => {
    const spy = vi.spyOn(client, 'invalidateQueries');
    renderHook(() => useSessionEvents('sess1'), { wrapper: wrapper() });
    MockWS.instances[0].emit(message);
    expect(spy).toHaveBeenCalledWith({ queryKey: ['contour-data', 'sess1'] });
  });

  it('still invalidates every model plot on model_trained (parity included)', () => {
    const spy = vi.spyOn(client, 'invalidateQueries');
    renderHook(() => useSessionEvents('sess1'), { wrapper: wrapper() });
    MockWS.instances[0].emit({ event: 'model_trained' });
    for (const key of ['model-info', 'parity-data', 'metrics-data', 'calibration-curve-data', 'qq-plot-data', 'hyperparameters']) {
      expect(spy).toHaveBeenCalledWith({ queryKey: [key, 'sess1'] });
    }
  });

  it('an open contour (staleTime: Infinity) refetches after a retrain without touching its controls', async () => {
    const getContour = vi
      .spyOn(visualizationsApi, 'getContourData')
      .mockResolvedValueOnce({ n_experiments: 10 } as any)
      .mockResolvedValueOnce({ n_experiments: 20 } as any);
    const request = { x_var: 'T', y_var: 'WHSV', fixed_values: {}, grid_resolution: 50, include_experiments: true } as any;
    const { result } = renderHook(
      () => {
        useSessionEvents('sess1');
        return useContourData('sess1', request, true);
      },
      { wrapper: wrapper() },
    );
    await waitFor(() => expect((result.current.data as any)?.n_experiments).toBe(10));
    MockWS.instances[0].emit({ event: 'model_trained' });
    await waitFor(() => expect((result.current.data as any)?.n_experiments).toBe(20));
    expect(getContour).toHaveBeenCalledTimes(2);
  });

  it('dev builds open the socket on the backend port 8000 (Vite does not proxy WebSockets)', () => {
    renderHook(() => useSessionEvents('sess1'), { wrapper: wrapper() });
    expect(MockWS.instances[0].url).toBe(`ws://${window.location.hostname}:8000/api/v1/ws/sessions/sess1`);
  });
});
