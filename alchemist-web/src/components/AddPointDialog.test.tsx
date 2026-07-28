import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import AddPointDialog from './AddPointDialog';
import type { VariableDetail } from '../api/types';

const variables: VariableDetail[] = [
  { name: 'temp', type: 'real', bounds: [0, 2000] },
  { name: 'count', type: 'integer', bounds: [0, 10] },
];

const suggestion = { temp: 901.4096982394, count: 4.9997, _reason: 'qEI' };

function renderDialog(props: Partial<React.ComponentProps<typeof AddPointDialog>> = {}) {
  const onConfirm = vi.fn();
  render(
    <AddPointDialog
      suggestion={suggestion}
      variables={variables}
      index={0}
      total={5}
      onCancel={() => {}}
      onConfirm={onConfirm}
      {...props}
    />,
  );
  return { onConfirm };
}

describe('AddPointDialog', () => {
  it('shows the suggested value read-only and pre-fills the actual input rounded', () => {
    renderDialog();
    expect(screen.getByText('901.410')).toBeInTheDocument();
    const tempActual = screen.getByLabelText('temp actual') as HTMLInputElement;
    expect(tempActual.value).toBe('901.41');
    const countActual = screen.getByLabelText('count actual') as HTMLInputElement;
    expect(countActual.value).toBe('5');
  });

  it('submits the actual (edited) values as inputs, not the suggestion', () => {
    const { onConfirm } = renderDialog();
    const tempActual = screen.getByLabelText('temp actual') as HTMLInputElement;
    fireEvent.change(tempActual, { target: { value: '900' } });
    fireEvent.change(screen.getByLabelText('Output'), { target: { value: '0.42' } });
    fireEvent.click(screen.getByText('Save & Close'));
    expect(onConfirm).toHaveBeenCalledTimes(1);
    const payload = onConfirm.mock.calls[0][0];
    expect(payload.inputs.temp).toBe('900');
    expect(payload.inputs.count).toBe('5');
    expect(payload.output).toBe(0.42);
  });

  it('displays the iteration from the suggestion and submits it on save', () => {
    const { onConfirm } = renderDialog({
      suggestion: { ...suggestion, Iteration: 6 },
    });
    // Iteration is shown (not 'N/A')
    expect(screen.getByText('6')).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Output'), { target: { value: '1.0' } });
    fireEvent.click(screen.getByText('Save & Close'));
    const payload = onConfirm.mock.calls[0][0];
    expect(payload.iteration).toBe(6);
  });

  it('omits iteration from the payload when the suggestion has none', () => {
    const { onConfirm } = renderDialog();  // suggestion has no Iteration
    fireEvent.change(screen.getByLabelText('Output'), { target: { value: '1.0' } });
    fireEvent.click(screen.getByText('Save & Close'));
    const payload = onConfirm.mock.calls[0][0];
    expect(payload.iteration).toBeUndefined();
  });
});
