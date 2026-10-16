import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { useState } from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { VariableDetail } from '../../api/types';

const variables: VariableDetail[] = [
  { name: 'temp', type: 'real', bounds: [0, 500] },
  { name: 'whsv', type: 'real', bounds: [0, 5] },
];

let targetColumns: string[] | undefined = undefined;
vi.mock('../../hooks/api/useSessions', () => ({
  useSession: () => ({ data: { session_id: 's1', created_at: '', variable_count: 2, experiment_count: 0, model_trained: false, target_columns: targetColumns } }),
}));
vi.mock('../../hooks/api/useVariables', () => ({
  useVariables: () => ({ data: { variables, n_variables: variables.length } }),
}));
// Rows the mocked useExperiments returns; a test sets them before rendering.
let mockRows: Record<string, unknown>[] = [];
vi.mock('../../hooks/api/useExperiments', () => ({
  useExperiments: () => ({ data: { experiments: mockRows, n_experiments: mockRows.length }, isLoading: false }),
  useExperimentsSummary: () => ({
    data: { n_experiments: 1, has_data: true, target_stats: { min: 0.5, max: 0.5, mean: 0.5, std: null } },
  }),
  useUploadExperiments: () => ({ isPending: false, mutateAsync: vi.fn() }),
  usePreviewCSV: () => ({ mutateAsync: vi.fn() }),
}));

import { ExperimentsPanel } from './ExperimentsPanel';

const STAGED = [
  { temp: 101.11111, whsv: 1.11111, _reason: 'Initial DoE (LHS)', _queueItemId: 'q1' },
  { temp: 202.22222, whsv: 2.22222, _reason: 'Initial DoE (LHS)', _queueItemId: 'q2' },
  { temp: 303.33333, whsv: 3.33333, _reason: 'Initial DoE (LHS)', _queueItemId: 'q3' },
];

/** ExperimentsPanel with the app's staged-suggestions state around it. */
function App({ initial }: { initial: any[] }) {
  const [pending, setPending] = useState(initial);
  return <ExperimentsPanel sessionId="s1" pendingSuggestions={pending} onStageSuggestions={setPending} />;
}

function renderApp(initial = STAGED) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <App initial={initial} />
    </QueryClientProvider>,
  );
}

const output = () => screen.getByLabelText('Output') as HTMLInputElement;
const temp = () => screen.getByLabelText('temp actual') as HTMLInputElement;

describe('ExperimentsPanel Add Point', () => {
  let fetchMock: ReturnType<typeof vi.fn>;
  beforeEach(() => {
    fetchMock = vi.fn(async (_url: string, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body ?? '{}'));
      return new Response(
        JSON.stringify({
          message: 'ok',
          n_added: body.entries.length,
          n_experiments: body.entries.length,
          completed_item_ids: body.entries.map((e: any) => e.item_id).filter(Boolean),
          model_trained: true,
          training_metrics: { r2: 0.9 },
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      );
    });
    vi.stubGlobal('fetch', fetchMock);
  });
  afterEach(() => vi.unstubAllGlobals());

  it('shows reals to 3 decimals and bookkeeping columns as integers in the experiments table', () => {
    mockRows = [{ x1: 1.23456, Output: 0.5, Iteration: 1, WholePlot: 2, RunOrder: 7 }];
    try {
      renderApp();
      expect(screen.getByText('1.235')).toBeInTheDocument(); // reals keep 3 decimals
      expect(screen.getByRole('cell', { name: '2' })).toBeInTheDocument(); // WholePlot
      expect(screen.getByRole('cell', { name: '7' })).toBeInTheDocument(); // RunOrder
      expect(screen.queryByText('2.000')).toBeNull();
    } finally {
      mockRows = [];
    }
  });

  it('hides the ProvenanceId column and gives the sticky header a solid background (W14)', () => {
    mockRows = [{ x1: 1.5, Output: 0.5, ProvenanceId: 'prov-abc' }];
    try {
      renderApp();
      expect(screen.queryByRole('columnheader', { name: 'ProvenanceId' })).toBeNull();
      expect(screen.queryByText('prov-abc')).toBeNull();
      expect(screen.getByRole('columnheader', { name: 'x1' })).toBeInTheDocument();
      const thead = screen.getByRole('columnheader', { name: 'x1' }).closest('thead')!;
      expect(thead.className).toContain('sticky');
      expect(thead.className).not.toMatch(/bg-[a-z]+\/\d+/); // no alpha: rows must not show through
    } finally {
      mockRows = [];
    }
  });

  it('labels the Add Point output with the session\'s target name and accepts it as a pasted header (W16, W43)', () => {
    targetColumns = ['Yield'];
    try {
      renderApp();
      fireEvent.click(screen.getByText(/Add Point/));
      expect(screen.getByLabelText('Yield')).toBeInTheDocument();
      expect(screen.queryByLabelText('Output')).toBeNull();
    } finally {
      targetColumns = undefined;
    }
  });

  it('shows "—" for a null std instead of "0.50 ± "', () => {
    renderApp();
    expect(screen.getByText(/0\.50 ±/).textContent).toContain('—');
  });

  it('Save sends ONE request with every entered point and retrains once', async () => {
    renderApp();
    fireEvent.click(screen.getByText(/Add Point/));
    fireEvent.change(output(), { target: { value: '0.1' } });
    fireEvent.click(screen.getByText('Next →'));
    fireEvent.change(temp(), { target: { value: '200' } });
    fireEvent.change(output(), { target: { value: '0.2' } });
    fireEvent.click(screen.getByText('Next →'));
    fireEvent.change(output(), { target: { value: '0.3' } });
    fireEvent.click(screen.getByText('Save 3 & close'));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/v1/sessions/s1/experiments/record-batch?auto_train=true');
    const body = JSON.parse(String(init.body));
    expect(body.entries).toEqual([
      { item_id: 'q1', inputs: { temp: 101.11111, whsv: 1.11111 }, output: 0.1, reason: 'Initial DoE (LHS)' },
      { item_id: 'q2', inputs: { temp: 200, whsv: 2.22222 }, output: 0.2, reason: 'Initial DoE (LHS)' },
      { item_id: 'q3', inputs: { temp: 303.33333, whsv: 3.33333 }, output: 0.3, reason: 'Initial DoE (LHS)' },
    ]);
    // Everything saved: dialog closes and nothing is pending.
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    // Add Point stays enabled with nothing staged (W28: it opens a blank point).
    const button = screen.getByText(/Add Point/).closest('button')!;
    expect(button).toBeEnabled();
    expect(button.textContent?.trim()).toBe('Add Point...');
  });

  it('with nothing staged, Add Point opens one blank point and records it (W28)', async () => {
    renderApp([]);
    const button = screen.getByText(/Add Point/).closest('button')!;
    expect(button).toBeEnabled();
    fireEvent.click(button);
    expect(temp().value).toBe('');
    expect(screen.getByText(/Point 1 of 1/)).toBeInTheDocument();
    // No NaN or 0 anywhere for the empty suggested values.
    expect(screen.getByRole('dialog').textContent).not.toMatch(/NaN|undefined/);
    // Blocked: inputs and output are empty.
    const save = () => screen.getByText(/Save \d+ & close/).closest('button')!;
    expect(save()).toBeDisabled();
    fireEvent.click(save());
    expect(fetchMock).not.toHaveBeenCalled();
    // An output without the conditions is refused too.
    fireEvent.change(output(), { target: { value: '0.42' } });
    expect(save()).toBeDisabled();
    expect(screen.getByRole('dialog').textContent).toContain('temp is empty');
    fireEvent.change(temp(), { target: { value: '150' } });
    fireEvent.change(screen.getByLabelText('whsv actual'), { target: { value: '2.5' } });
    expect(save()).toBeEnabled();
    fireEvent.click(screen.getByText(/Save 1 & close/));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    const body = JSON.parse(String(fetchMock.mock.calls[0][1].body));
    expect(body.entries).toEqual([{ inputs: { temp: 150, whsv: 2.5 }, output: 0.42, reason: 'Manual' }]);
  });

  it('after saving a manual point, Add Point opens a fresh blank point', async () => {
    renderApp([]);
    fireEvent.click(screen.getByText(/Add Point/));
    fireEvent.change(temp(), { target: { value: '150' } });
    fireEvent.change(screen.getByLabelText('whsv actual'), { target: { value: '2.5' } });
    fireEvent.change(output(), { target: { value: '0.42' } });
    fireEvent.click(screen.getByText(/Save 1 & close/));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    fireEvent.click(screen.getByText(/Add Point/));
    expect(temp().value).toBe('');
    expect(output().value).toBe('');
  });

  it('a cancelled manual point is not kept: reopening is blank again', () => {
    renderApp([]);
    fireEvent.click(screen.getByText(/Add Point/));
    fireEvent.change(temp(), { target: { value: '150' } });
    fireEvent.click(screen.getByLabelText('Close'));
    expect(screen.queryByRole('dialog')).toBeNull();
    fireEvent.click(screen.getByText(/Add Point/));
    expect(temp().value).toBe('');
  });

  it('the table view shows a dash, not a blank, for a manual point\'s suggested values', () => {
    renderApp([]);
    fireEvent.click(screen.getByText(/Add Point/));
    fireEvent.click(screen.getByText('Table'));
    expect(screen.getAllByText('—').length).toBeGreaterThanOrEqual(2);
    expect((screen.getByLabelText('Point 1 temp actual') as HTMLInputElement).value).toBe('');
  });

  it('Add Point is disabled until variables are defined', () => {
    const saved = variables.splice(0, variables.length);
    try {
      renderApp([]);
      const button = screen.getByText(/Add Point/).closest('button')!;
      expect(button).toBeDisabled();
      expect(button.title).toBe('Define variables first');
    } finally {
      variables.push(...saved);
    }
  });

  it('adds a hand-typed point alongside staged points; staged keep item_id, manual does not (W28)', async () => {
    renderApp();
    fireEvent.click(screen.getByText(/Add Point/));
    // Staged point 1: output only.
    fireEvent.change(output(), { target: { value: '0.1' } });
    expect(screen.getByText(/Suggested conditions are shown for reference/)).toBeInTheDocument();
    fireEvent.click(screen.getByText('Add blank point'));
    expect(screen.getByTestId('add-point-progress').textContent).toBe('Point 4 of 4, 1 entered');
    expect(temp().value).toBe('');
    expect(screen.getByDisplayValue('Manual')).toBeInTheDocument();
    // The reference help line does not apply to a point with nothing suggested.
    expect(screen.queryByText(/Suggested conditions are shown for reference/)).toBeNull();
    fireEvent.change(temp(), { target: { value: '150' } });
    fireEvent.change(screen.getByLabelText('whsv actual'), { target: { value: '2.5' } });
    fireEvent.change(output(), { target: { value: '0.42' } });
    fireEvent.click(screen.getByText('Save 2 & close'));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    const body = JSON.parse(String(fetchMock.mock.calls[0][1].body));
    expect(body.entries).toEqual([
      { item_id: 'q1', inputs: { temp: 101.11111, whsv: 1.11111 }, output: 0.1, reason: 'Initial DoE (LHS)' },
      { inputs: { temp: 150, whsv: 2.5 }, output: 0.42, reason: 'Manual' },
    ]);
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    // Only the two unsaved staged points remain; the manual point never joined them.
    expect(screen.getByText(/Add Point/).textContent).toContain('(2)');
    fireEvent.click(screen.getByText(/Add Point/));
    expect(screen.getByTestId('add-point-progress').textContent).toBe('Point 1 of 2, 0 entered');
    expect(temp().value).toBe('202.22222');
  });

  it('closing the dialog discards a manual point added next to staged points', () => {
    renderApp();
    fireEvent.click(screen.getByText(/Add Point/));
    fireEvent.click(screen.getByText('Add blank point'));
    fireEvent.click(screen.getByLabelText('Close'));
    fireEvent.click(screen.getByText(/Add Point/));
    expect(screen.getByTestId('add-point-progress').textContent).toBe('Point 1 of 3, 0 entered');
  });

  it('after a save, the next point shows its own values, not the saved one (W19)', async () => {
    renderApp();
    fireEvent.click(screen.getByText(/Add Point/));
    // Point 1: edit the actual temperature and enter an output; point 2 untouched.
    fireEvent.change(temp(), { target: { value: '99.9' } });
    fireEvent.change(output(), { target: { value: '0.1' } });
    fireEvent.click(screen.getByText('Save 1 & close'));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());

    // Two remain pending; reopen: point 1 of 2 is the old point 2.
    fireEvent.click(screen.getByText(/Add Point/));
    expect(screen.getByTestId('add-point-progress').textContent).toBe('Point 1 of 2, 0 entered');
    expect(temp().value).toBe('202.22222');
    expect(output().value).toBe('');
    expect(screen.getByText('202.222')).toBeInTheDocument();
  });

  it('keeps entries for points left pending when the dialog is reopened', async () => {
    renderApp();
    fireEvent.click(screen.getByText(/Add Point/));
    fireEvent.change(output(), { target: { value: '0.1' } });
    fireEvent.click(screen.getByText('Next →'));
    fireEvent.change(temp(), { target: { value: '250' } }); // typed, but no output
    fireEvent.click(screen.getByText('Save 1 & close'));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    fireEvent.click(screen.getByText(/Add Point/));
    expect(temp().value).toBe('250');
  });

  it('a failed save keeps the dialog, the entries and the pending list', async () => {
    fetchMock.mockImplementationOnce(async () =>
      new Response(JSON.stringify({ detail: 'Nothing was recorded. entry 1: unknown queue item q1' }), {
        status: 409,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    renderApp();
    fireEvent.click(screen.getByText(/Add Point/));
    fireEvent.change(output(), { target: { value: '0.1' } });
    fireEvent.click(screen.getByText('Save 1 & close'));
    expect(await screen.findByRole('alert')).toHaveTextContent('unknown queue item q1');
    expect(output().value).toBe('0.1');
    expect(screen.getByTestId('add-point-progress').textContent).toBe('Point 1 of 3, 1 entered');
  });
});
