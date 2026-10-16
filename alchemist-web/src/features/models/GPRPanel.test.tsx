import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

const mutateAsync = vi.fn();
vi.mock('../../hooks/api/useModels', () => ({
  useModelInfo: () => ({ data: undefined }),
  useTrainModel: () => ({ mutateAsync, isPending: false, isSuccess: false, data: undefined }),
}));
let summary = { has_data: true, n_experiments: 12 };
vi.mock('../../hooks/api/useExperiments', () => ({
  useExperimentsSummary: () => ({ data: summary }),
}));
vi.mock('../../providers/VisualizationProvider', () => ({
  useVisualization: () => ({ openVisualization: vi.fn() }),
}));

import { GPRPanel } from './GPRPanel';

const selectBoTorch = () => fireEvent.click(screen.getByRole('button', { name: 'BoTorch' }));
const train = () => fireEvent.click(screen.getByRole('button', { name: 'Train Model' }));
const select = (label: string) =>
  screen.getByText(label).parentElement!.querySelector('select') as HTMLSelectElement;

describe('GPRPanel BoTorch scaling (W2)', () => {
  beforeEach(() => {
    mutateAsync.mockReset();
    mutateAsync.mockResolvedValue({});
  });

  it('shows normalize / standardize and sends them when the user changes nothing', async () => {
    render(<GPRPanel sessionId="s1" />);
    selectBoTorch();
    expect(select('Input Scaling').value).toBe('normalize');
    expect(select('Output Scaling').value).toBe('standardize');

    train();
    await waitFor(() => expect(mutateAsync).toHaveBeenCalledTimes(1));
    const body = mutateAsync.mock.calls[0][0];
    expect(body.backend).toBe('botorch');
    expect(body.input_transform).toBe('normalize');
    expect(body.output_transform).toBe('standardize');
  });

  it('sends an explicit none so the server trains unscaled', async () => {
    render(<GPRPanel sessionId="s1" />);
    selectBoTorch();
    fireEvent.click(screen.getByLabelText('Enable Advanced Options'));
    fireEvent.change(select('Input Scaling'), { target: { value: 'none' } });
    fireEvent.change(select('Output Scaling'), { target: { value: 'none' } });

    train();
    await waitFor(() => expect(mutateAsync).toHaveBeenCalledTimes(1));
    const body = mutateAsync.mock.calls[0][0];
    expect(body.input_transform).toBe('none');
    expect(body.output_transform).toBe('none');
  });

  it('leaves the scikit-learn request without transforms by default', async () => {
    render(<GPRPanel sessionId="s1" />);
    train();
    await waitFor(() => expect(mutateAsync).toHaveBeenCalledTimes(1));
    const body = mutateAsync.mock.calls[0][0];
    expect(body.backend).toBe('sklearn');
    expect(body.input_transform).toBeUndefined();
    expect(body.output_transform).toBeUndefined();
  });
});

describe('GPRPanel with too few points', () => {
  it('disables Train Model with a hint instead of a native alert', () => {
    summary = { has_data: true, n_experiments: 3 };
    const alertSpy = vi.spyOn(window, 'alert').mockImplementation(() => {});
    try {
      render(<GPRPanel sessionId="s1" />);
      const button = screen.getByRole('button', { name: 'Train Model' });
      expect(button).toBeDisabled();
      expect(button).toHaveAttribute('title', expect.stringMatching(/at least 5/i));
      expect(screen.getByText(/Need at least 5 experiments \(currently 3\)/)).toBeInTheDocument();
      expect(alertSpy).not.toHaveBeenCalled();
    } finally {
      summary = { has_data: true, n_experiments: 12 };
      alertSpy.mockRestore();
    }
  });
});
