import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest';
import { cleanup, fireEvent, screen, waitFor } from '@testing-library/react';
import { renderWithQuery } from '../../test/queryWrapper';
import { ContourPlot } from './ContourPlot';
import { COLOR_SCALE_STORAGE_KEY } from './contourColorScale';

const getContourData = vi.fn();

vi.mock('../../api/endpoints/visualizations', () => ({
  getContourData: (...args: unknown[]) => getContourData(...args),
}));
vi.mock('../../hooks/useTheme', () => ({ useTheme: () => ({ theme: 'light' }) }));
// A stable object: ContourPlot re-initialises its fixed values whenever `variables` changes identity.
const variablesData = {
  n_variables: 2,
  variables: [
    { name: 'a', type: 'real', bounds: [0, 1] },
    { name: 'b', type: 'real', bounds: [0, 1] },
  ],
};
vi.mock('../../hooks/api/useVariables', () => ({
  useVariables: () => ({ data: variablesData }),
}));
vi.mock('react-plotly.js', () => ({
  default: ({ data }: { data: unknown }) => <div data-testid="plot" data-traces={JSON.stringify(data)} />,
}));

function response(outputRange: [number, number] | null) {
  return {
    x_var: 'a',
    y_var: 'b',
    x_grid: [[0, 1], [0, 1]],
    y_grid: [[0, 0], [1, 1]],
    predictions: [[4, 6], [8, 12]],
    uncertainties: [[0, 0], [0, 0]],
    x_bounds: [0, 1],
    y_bounds: [0, 1],
    colorbar_bounds: [4, 12],
    output_range: outputRange,
  };
}

async function contourTrace() {
  const plot = await screen.findByTestId('plot');
  return JSON.parse(plot.getAttribute('data-traces')!)[0];
}

describe('ContourPlot colour scale', () => {
  beforeEach(() => {
    getContourData.mockReset();
    getContourData.mockResolvedValue(response([0, 18]));
  });

  afterEach(() => {
    cleanup();
    localStorage.clear();
  });

  it('defaults to a fixed scale spanning the observed outputs', async () => {
    renderWithQuery(<ContourPlot sessionId="s1" />);
    const trace = await contourTrace();

    expect(trace.zauto).toBe(false);
    expect(trace.zmin).toBe(0);
    expect(trace.zmax).toBe(18);
    expect(screen.getByRole('button', { name: 'Fixed' })).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('button', { name: 'Per slice' })).toHaveAttribute('aria-pressed', 'false');
    expect(screen.getByText(/0 to 18/)).toBeInTheDocument();
  });

  it('rescales to each slice when switched, without a new request, and remembers it', async () => {
    renderWithQuery(<ContourPlot sessionId="s1" />);
    await contourTrace();
    const requests = getContourData.mock.calls.length;

    fireEvent.click(screen.getByRole('button', { name: 'Per slice' }));

    await waitFor(async () => {
      const trace = await contourTrace();
      expect(trace.zmin).toBeUndefined();
      expect(trace.zmax).toBeUndefined();
      expect(trace.zauto).toBe(true);
    });
    expect(screen.getByRole('button', { name: 'Per slice' })).toHaveAttribute('aria-pressed', 'true');
    expect(localStorage.getItem(COLOR_SCALE_STORAGE_KEY)).toBe('per-slice');
    expect(getContourData).toHaveBeenCalledTimes(requests);
  });

  it('opens in the remembered mode', async () => {
    localStorage.setItem(COLOR_SCALE_STORAGE_KEY, 'per-slice');
    renderWithQuery(<ContourPlot sessionId="s1" />);
    const trace = await contourTrace();

    expect(trace.zmin).toBeUndefined();
    expect(screen.getByRole('button', { name: 'Per slice' })).toHaveAttribute('aria-pressed', 'true');
  });

  it('falls back to the slice range when the session has no usable output range', async () => {
    getContourData.mockResolvedValue(response(null));
    renderWithQuery(<ContourPlot sessionId="s1" />);
    const trace = await contourTrace();

    expect(trace.zmin).toBeUndefined();
    expect(trace.zauto).toBe(true);
    expect(screen.getByRole('button', { name: 'Fixed' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Per slice' })).toHaveAttribute('aria-pressed', 'true');
  });
});
