import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useSessionEvents } from './useSessionEvents';

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
