// src/features/experiments/SplitPlotDesignSection.test.tsx
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import type { VariableDetail } from '../../api/types';

const POINTS = [
  { x1: 1, c: 'a', WholePlot: 1, RunOrder: 1, Reason: 'Reference (split-plot)' },
  { x1: 2, c: 'a', WholePlot: 1, RunOrder: 2, Reason: 'Initial DoE (split-plot)' },
  { x1: 3, c: 'b', WholePlot: 2, RunOrder: 3, Reason: 'Initial DoE (split-plot)' },
];
const mutateAsync = vi.fn().mockResolvedValue({
  points: POINTS, n_points: 3, info: { n_plots: { hard: 2 }, n_runs: 3, warnings: ['each whole-plot setting appears once'] },
});
vi.mock('../../hooks/api/useExperiments', () => ({
  useGenerateSplitPlotDesign: () => ({ mutateAsync, isPending: false }),
}));

import { SplitPlotDesignSection, groupByWholePlot } from './SplitPlotDesignSection';

const VARS: VariableDetail[] = [
  { name: 'x1', type: 'real', bounds: [0, 10], role: 'controllable', changes: 'easy' },
  { name: 'c', type: 'categorical', categories: ['a', 'b'], role: 'controllable', changes: 'hard' },
];

describe('SplitPlotDesignSection', () => {
  let fetchMock: ReturnType<typeof vi.fn>;
  beforeEach(() => {
    fetchMock = vi.fn(async () => new Response(JSON.stringify({ ids: ['q1', 'q2', 'q3'] }), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
  });
  afterEach(() => vi.unstubAllGlobals());

  it('groups rows by whole plot', () => {
    expect(groupByWholePlot(POINTS).map((g) => [g.wholePlot, g.rows.length])).toEqual([[1, 2], [2, 1]]);
  });

  it('generates with one design per level and shows the summary and warnings', async () => {
    render(<SplitPlotDesignSection sessionId="s1" variables={VARS} />);
    fireEvent.change(screen.getByLabelText('hard method'), { target: { value: 'full_factorial' } });
    fireEvent.change(screen.getByLabelText('hard replicates'), { target: { value: '2' } });
    fireEvent.click(screen.getByText('Generate split-plot design'));
    await waitFor(() => expect(mutateAsync).toHaveBeenCalled());
    const req = mutateAsync.mock.calls[0][0];
    expect(Object.keys(req.designs).sort()).toEqual(['easy', 'hard']);
    expect(req.replicates).toEqual({ hard: 2 });
    expect(await screen.findByText(/2 whole plots/)).toBeInTheDocument();
    expect(screen.getByText(/appears once/)).toBeInTheDocument();
  });

  it('stages rows with plot metadata and per-row reasons', async () => {
    const onStage = vi.fn();
    render(<SplitPlotDesignSection sessionId="s1" variables={VARS} onStageSuggestions={onStage} />);
    fireEvent.click(screen.getByText('Generate split-plot design'));
    fireEvent.click(await screen.findByText('Stage'));
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    const body = JSON.parse(String(fetchMock.mock.calls[0][1].body));
    expect(body.reason).toBeNull();
    expect(body.experiments[0]).toEqual({ x1: 1, c: 'a', WholePlot: 1, RunOrder: 1, _reason: 'Reference (split-plot)' });
    expect(onStage.mock.calls[0][0][2]).toMatchObject({ WholePlot: 2, _queueItemId: 'q3', _reason: 'Initial DoE (split-plot)' });
  });
});
