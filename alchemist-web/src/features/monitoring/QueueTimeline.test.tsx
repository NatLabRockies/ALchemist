import { describe, it, expect, vi } from 'vitest';
import { screen } from '@testing-library/react';
import { renderWithQuery } from '../../test/queryWrapper';
import * as useQueue from '../../hooks/api/useQueue';
import { QueueTimeline } from './QueueTimeline';

function mockQueue(items: any[]) {
  vi.spyOn(useQueue, 'useExperimentQueue').mockReturnValue({
    data: { items, n_pending: 0, n_running: 0, n_done: 0, n_failed: 0 },
    isLoading: false, isSuccess: true,
  } as any);
}

describe('QueueTimeline', () => {
  it('shows an empty/awaiting state when there are no items', () => {
    mockQueue([]);
    renderWithQuery(<QueueTimeline sessionId="s1" />);
    expect(screen.getByText(/awaiting controller/i)).toBeInTheDocument();
  });

  it('renders each item status and a failed item error', () => {
    mockQueue([
      { id: 'a', inputs: { x: 1 }, reason: 'seed', status: 'done', output: 2, error: null,
        noise: null, dataset_ref: 0, staged_at: null, started_at: null, completed_at: null },
      { id: 'b', inputs: { x: 3 }, reason: null, status: 'failed', output: null,
        error: 'sensor timeout', noise: null, dataset_ref: null,
        staged_at: null, started_at: null, completed_at: null },
    ]);
    renderWithQuery(<QueueTimeline sessionId="s1" />);
    expect(screen.getByText(/done/i)).toBeInTheDocument();
    expect(screen.getByText(/failed/i)).toBeInTheDocument();
    expect(screen.getByText(/sensor timeout/i)).toBeInTheDocument();
  });
});
