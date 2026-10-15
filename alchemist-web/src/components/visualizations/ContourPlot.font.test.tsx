import { describe, it, expect, vi, afterEach } from 'vitest';
import { cleanup, screen } from '@testing-library/react';
import { renderWithQuery } from '../../test/queryWrapper';
import { ContourPlot } from './ContourPlot';

vi.mock('../../api/endpoints/visualizations', () => ({
  getContourData: () =>
    Promise.resolve({
      x_var: 'a',
      y_var: 'b',
      x_grid: [[0, 1], [0, 1]],
      y_grid: [[0, 0], [1, 1]],
      predictions: [[0, 1], [2, 3]],
      uncertainties: [[0, 0], [0, 0]],
      x_bounds: [0, 1],
      y_bounds: [0, 1],
      colorbar_bounds: [0, 3],
    }),
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
  default: ({ layout, data }: { layout: unknown; data: unknown }) => (
    <div data-testid="plot" data-layout={JSON.stringify(layout)} data-traces={JSON.stringify(data)} />
  ),
}));

describe('ContourPlot font', () => {
  afterEach(() => {
    cleanup();
    document.head.querySelectorAll('style[data-test]').forEach((n) => n.remove());
  });

  it('sets every Plotly text in the family the page uses', async () => {
    const style = document.createElement('style');
    style.dataset.test = '1';
    style.textContent = 'body{font-family: "Test Sans", system-ui, sans-serif}';
    document.head.appendChild(style);
    const ui = getComputedStyle(document.body).fontFamily;

    renderWithQuery(<ContourPlot sessionId="s1" />);
    const plot = await screen.findByTestId('plot');
    const layout = JSON.parse(plot.getAttribute('data-layout')!);
    const traces = JSON.parse(plot.getAttribute('data-traces')!);

    expect(layout.font.family).toBe(ui);
    expect(layout.title.font.family).toBe(ui);
    expect(layout.xaxis.title.font.family).toBe(ui);
    expect(layout.yaxis.title.font.family).toBe(ui);
    expect(layout.legend.font.family).toBe(ui);
    expect(layout.hoverlabel.font.family).toBe(ui);
    expect(traces[0].colorbar.title.font.family).toBe(ui);
    expect(JSON.stringify(layout)).not.toContain('Arial');
  });
});
