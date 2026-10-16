import { describe, it, expect, vi } from 'vitest';
import { useState } from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import AddPointDialog from './AddPointDialog';
import type { VariableDetail } from '../api/types';
import type { BatchPlan, EntriesMap } from '../features/experiments/addPointEntries';

const variables: VariableDetail[] = [
  { name: 'temp', type: 'real', bounds: [0, 2000] },
  { name: 'count', type: 'integer', bounds: [0, 10] },
];

const suggestion = { temp: 901.4096982394, count: 4.9997, _reason: 'qEI' };

function makeSuggestions(n: number) {
  return Array.from({ length: n }, (_, i) => ({
    temp: 100 + i * 10.123456,
    count: i % 10,
    _reason: 'Initial DoE (LHS)',
    _queueItemId: `q${i + 1}`,
  }));
}

/** The dialog with its entries held in state, as the panels hold them. */
function Harness({
  suggestions,
  variables: vars = variables,
  onSave = vi.fn(async () => {}),
}: {
  suggestions: any[];
  variables?: VariableDetail[];
  onSave?: (plan: BatchPlan, options: { retrain: boolean }) => Promise<void>;
}) {
  const [entries, setEntries] = useState<EntriesMap>({});
  return (
    <AddPointDialog
      suggestions={suggestions}
      variables={vars}
      entries={entries}
      onEntriesChange={setEntries}
      onCancel={() => {}}
      onSave={onSave}
    />
  );
}

function renderHarness(suggestions: any[], vars?: VariableDetail[]) {
  const onSave = vi.fn<(plan: BatchPlan, options: { retrain: boolean }) => Promise<void>>(async () => {});
  render(<Harness suggestions={suggestions} variables={vars} onSave={onSave} />);
  return { onSave };
}

const output = () => screen.getByLabelText('Output') as HTMLInputElement;
const next = () => fireEvent.click(screen.getByText('Next →'));
const prev = () => fireEvent.click(screen.getByText('← Previous'));

describe('AddPointDialog, one at a time', () => {
  it('shows the suggested value rounded read-only and pre-fills the actual input at full precision (W33)', () => {
    renderHarness([suggestion]);
    expect(screen.getByText('901.410')).toBeInTheDocument();
    expect((screen.getByLabelText('temp actual') as HTMLInputElement).value).toBe('901.4096982394');
    expect((screen.getByLabelText('count actual') as HTMLInputElement).value).toBe('5');
  });

  it('saves the actual (edited) values as typed numbers, not the suggestion', async () => {
    const { onSave } = renderHarness([suggestion]);
    fireEvent.change(screen.getByLabelText('temp actual'), { target: { value: '900' } });
    fireEvent.change(output(), { target: { value: '0.42' } });
    fireEvent.click(screen.getByText('Save 1 & close'));
    await waitFor(() => expect(onSave).toHaveBeenCalledTimes(1));
    const [plan] = onSave.mock.calls[0];
    expect(plan.payload).toEqual([{ inputs: { temp: 900, count: 5 }, output: 0.42, reason: 'qEI' }]);
  });

  it('sends the iteration from the suggestion, and omits it when there is none', async () => {
    const { onSave } = renderHarness([{ ...suggestion, Iteration: 6 }, suggestion]);
    expect(screen.getByText('6')).toBeInTheDocument();
    fireEvent.change(output(), { target: { value: '1.0' } });
    next();
    fireEvent.change(output(), { target: { value: '2.0' } });
    fireEvent.click(screen.getByText('Save 2 & close'));
    await waitFor(() => expect(onSave).toHaveBeenCalled());
    const [plan] = onSave.mock.calls[0];
    expect(plan.payload[0].iteration).toBe(6);
    expect(plan.payload[1].iteration).toBeUndefined();
  });

  it('does not render internal keys like _queueItemId as variable rows', () => {
    renderHarness([{ ...suggestion, _queueItemId: 'abc-123' }]);
    expect(screen.queryByLabelText('_queueItemId actual')).toBeNull();
  });

  it('keeps what was typed for each point across Next and Previous', () => {
    renderHarness(makeSuggestions(3));
    fireEvent.change(screen.getByLabelText('temp actual'), { target: { value: '99.5' } });
    fireEvent.change(output(), { target: { value: '0.11' } });
    next();
    expect(output().value).toBe('');
    expect((screen.getByLabelText('temp actual') as HTMLInputElement).value).toBe('110.123456');
    fireEvent.change(output(), { target: { value: '0.22' } });
    next();
    prev();
    expect(output().value).toBe('0.22');
    prev();
    expect(output().value).toBe('0.11');
    expect((screen.getByLabelText('temp actual') as HTMLInputElement).value).toBe('99.5');
  });

  it('shows progress: point position and how many are entered', () => {
    renderHarness(makeSuggestions(10));
    expect(screen.getByTestId('add-point-progress').textContent).toBe('Point 1 of 10, 0 entered');
    fireEvent.change(output(), { target: { value: '1' } });
    next();
    next();
    fireEvent.change(output(), { target: { value: '3' } });
    expect(screen.getByTestId('add-point-progress').textContent).toBe('Point 3 of 10, 2 entered');
  });

  it('Enter in the output field moves to the next point', () => {
    renderHarness(makeSuggestions(2));
    fireEvent.change(output(), { target: { value: '1' } });
    fireEvent.keyDown(output(), { key: 'Enter' });
    expect(screen.getByTestId('add-point-progress').textContent).toContain('Point 2 of 2');
  });

  it('saves every entered point in one call with one retrain setting, and lists the rest as pending', async () => {
    const { onSave } = renderHarness(makeSuggestions(4));
    fireEvent.change(output(), { target: { value: '1' } });
    next();
    next();
    fireEvent.change(output(), { target: { value: '3' } });
    next();
    fireEvent.change(output(), { target: { value: '4' } });
    expect(screen.getByTestId('add-point-summary').textContent).toContain('1 without an output stays pending (point 2)');
    fireEvent.click(screen.getByRole('checkbox'));  // untick "Retrain model"
    fireEvent.click(screen.getByText('Save 3 & close'));
    await waitFor(() => expect(onSave).toHaveBeenCalledTimes(1));
    const [plan, options] = onSave.mock.calls[0];
    expect(options).toEqual({ retrain: false });
    expect(plan.payload.map((p: any) => [p.item_id, p.output])).toEqual([
      ['q1', 1],
      ['q3', 3],
      ['q4', 4],
    ]);
    expect(plan.pendingPositions).toEqual([2]);
  });

  it('blocks saving an entered point with a non-numeric value', () => {
    renderHarness(makeSuggestions(1));
    fireEvent.change(output(), { target: { value: '1' } });
    fireEvent.change(screen.getByLabelText('temp actual'), { target: { value: 'abc' } });
    expect(screen.getByText(/temp "abc" is not a number/)).toBeInTheDocument();
    expect(screen.getByText('Save 0 & close')).toBeDisabled();
  });

  it('keeps the entries and shows the error when the save fails', async () => {
    const onSave = vi.fn(async () => {
      throw new Error('Save failed (409): Nothing was recorded.');
    });
    render(<Harness suggestions={makeSuggestions(2)} onSave={onSave} />);
    fireEvent.change(output(), { target: { value: '0.5' } });
    fireEvent.click(screen.getByText('Save 1 & close'));
    expect(await screen.findByRole('alert')).toHaveTextContent('Save failed (409)');
    expect(output().value).toBe('0.5');
  });
});

describe('AddPointDialog, table view', () => {
  function openTable(n: number) {
    const result = renderHarness(makeSuggestions(n));
    fireEvent.click(screen.getByText('Table'));
    return result;
  }
  const cell = (row: number, what: string) => screen.getByLabelText(`Point ${row} ${what}`) as HTMLInputElement;

  it('lists every pending point with suggested and editable actual inputs', () => {
    openTable(3);
    expect(cell(3, 'temp actual').value).toBe('120.246912');
    expect(screen.getByText('120.247')).toBeInTheDocument(); // suggested, read-only
    expect(cell(1, 'output').value).toBe('');
    expect(cell(1, 'reason').value).toBe('Initial DoE (LHS)');
  });

  it('pasting a column of numbers into an Output cell fills down from that row', () => {
    openTable(5);
    fireEvent.paste(cell(2, 'output'), {
      clipboardData: { getData: () => '0.12\n0.34\n0.56\n' },
    });
    expect(cell(1, 'output').value).toBe('');
    expect(cell(2, 'output').value).toBe('0.12');
    expect(cell(3, 'output').value).toBe('0.34');
    expect(cell(4, 'output').value).toBe('0.56');
    expect(cell(5, 'output').value).toBe('');
    expect(screen.getByRole('status')).toHaveTextContent('Filled 3 output values from point 2 down');
  });

  it('reports pasted values beyond the last point instead of dropping them silently', () => {
    openTable(2);
    fireEvent.paste(cell(1, 'output'), { clipboardData: { getData: () => '1\n2\n3\n4' } });
    expect(cell(2, 'output').value).toBe('2');
    expect(screen.getByRole('status')).toHaveTextContent('2 extra values ignored');
  });

  it('a CSV block with headers maps columns by name', () => {
    openTable(3);
    const csv = 'Count,Temp,Yield,Noise\n7,150.5,0.9,0.01\n8,160.5,0.8,\n';
    fireEvent.paste(cell(1, 'output'), { clipboardData: { getData: () => csv } });
    expect(cell(1, 'temp actual').value).toBe('150.5');
    expect(cell(1, 'count actual').value).toBe('7');
    expect(cell(1, 'output').value).toBe('0.9');
    expect(cell(2, 'temp actual').value).toBe('160.5');
    expect(cell(2, 'output').value).toBe('0.8');
    expect(cell(3, 'output').value).toBe('');
    expect(cell(3, 'temp actual').value).toBe('120.246912');
  });

  it('table entries and one-at-a-time entries are the same entries', async () => {
    const { onSave } = openTable(3);
    fireEvent.paste(cell(1, 'output'), { clipboardData: { getData: () => '5\n6\n7' } });
    fireEvent.click(screen.getByText('One at a time'));
    expect(output().value).toBe('5');
    fireEvent.click(screen.getByText('Save 3 & close'));
    await waitFor(() => expect(onSave).toHaveBeenCalledTimes(1));
    expect(onSave.mock.calls[0][0].payload.map((p: any) => p.output)).toEqual([5, 6, 7]);
  });
});

describe('AddPointDialog, split-plot setup', () => {
  const SPLIT_VARS: VariableDetail[] = [
    { name: 'x1', type: 'real', bounds: [0, 10], role: 'controllable', changes: 'easy' },
    { name: 'c', type: 'categorical', categories: ['a', 'b'], role: 'controllable', changes: 'hard' },
    { name: 'z', type: 'real', role: 'observed' },
  ];
  const SPLIT = [
    { x1: 1, c: 'a', WholePlot: 1, RunOrder: 1, _queueItemId: 'q1' },
    { x1: 2, c: 'a', WholePlot: 1, RunOrder: 2, _queueItemId: 'q2' },
    { x1: 3, c: 'b', WholePlot: 2, RunOrder: 3, _queueItemId: 'q3' },
  ];

  it('shows the setup banner before a new whole plot and an input for the observed variable', () => {
    renderHarness(SPLIT, SPLIT_VARS);
    expect(screen.getByRole('note', { name: 'Setup change' })).toHaveTextContent('Whole plot 1 of 2');
    expect(screen.getByLabelText('z actual')).toHaveValue('');
    next();
    expect(screen.queryByRole('note', { name: 'Setup change' })).toBeNull();
    next();
    expect(screen.getByRole('note', { name: 'Setup change' })).toHaveTextContent('c = b');
  });

  it('keeps the banner in view: focuses Output without scrolling and scrolls the body to the top', () => {
    const focusSpy = vi.spyOn(HTMLElement.prototype, 'focus');
    renderHarness(SPLIT, SPLIT_VARS);
    const output = screen.getByLabelText('Output');
    expect(document.activeElement).toBe(output);
    expect(focusSpy).toHaveBeenCalledWith({ preventScroll: true });

    const body = screen.getByTestId('add-point-body');
    fireEvent.keyDown(output, { key: 'Enter' }); // point 2: no banner
    body.scrollTop = 120;
    fireEvent.keyDown(output, { key: 'Enter' }); // point 3 starts whole plot 2
    expect(screen.getByRole('note', { name: 'Setup change' })).toHaveTextContent('Whole plot 2 of 2');
    expect(body.scrollTop).toBe(0);
    expect(document.activeElement).toBe(screen.getByLabelText('Output'));
    focusSpy.mockRestore();
  });
});

describe('AddPointDialog out-of-range actuals and the target name', () => {
  it('warns inline for an actual outside the bounds, lists it on Save, and still allows saving', async () => {
    const { onSave } = renderHarness([suggestion]);
    fireEvent.change(screen.getByLabelText('temp actual'), { target: { value: '2500' } });
    fireEvent.change(output(), { target: { value: '0.5' } });
    expect(screen.getByText('outside 0–2000')).toBeInTheDocument();
    const save = screen.getByRole('button', { name: 'Save 1 & close' });
    expect(save).toBeEnabled();
    expect(save).toHaveAttribute('title', expect.stringContaining('temp outside 0–2000'));
    fireEvent.click(save);
    await waitFor(() => expect(onSave).toHaveBeenCalledTimes(1));
    expect(onSave.mock.calls[0][0].payload[0].inputs.temp).toBe(2500);
  });

  it('labels the output with the target name when it is not Output', () => {
    render(
      <AddPointDialog
        suggestions={[suggestion]}
        variables={variables}
        entries={{}}
        onEntriesChange={() => {}}
        onCancel={() => {}}
        onSave={async () => {}}
        outputLabel="Yield"
      />,
    );
    expect(screen.getByLabelText('Yield')).toBeInTheDocument();
    expect(screen.queryByLabelText('Output')).toBeNull();
  });
});
