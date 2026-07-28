import { describe, it, expect } from 'vitest';
import { computeTrace } from './ObjectiveTrace';
import type { QueueItem } from '../../api/endpoints/queue';

function item(id: string, status: any, output: number | null): QueueItem {
  return { id, inputs: {}, reason: null, status, output, error: null, noise: null,
    dataset_ref: null, staged_at: null, started_at: null,
    completed_at: status === 'done' ? `2026-07-28T00:00:0${id}` : null };
}

describe('computeTrace', () => {
  it('includes only done items, ordered by completion, with cumulative best (maximize)', () => {
    const items = [item('1', 'done', 5), item('2', 'failed', null), item('3', 'done', 8), item('4', 'pending', null)];
    const trace = computeTrace(items, 'maximize');
    expect(trace.map(p => p.value)).toEqual([5, 8]);
    expect(trace.map(p => p.best)).toEqual([5, 8]);
  });

  it('cumulative best for minimize keeps the lowest so far', () => {
    const items = [item('1', 'done', 8), item('2', 'done', 5), item('3', 'done', 9)];
    const trace = computeTrace(items, 'minimize');
    expect(trace.map(p => p.best)).toEqual([8, 5, 5]);
  });
});
