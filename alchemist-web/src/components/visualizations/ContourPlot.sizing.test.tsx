import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import type { ReactNode } from 'react';
import { QueryClientProvider } from '@tanstack/react-query';
import { cleanup, render, screen } from '@testing-library/react';
import { createTestQueryClient } from '../../test/queryWrapper';
import { ContourPlot } from './ContourPlot';
import { contourFigureSize } from './contourSizing';

const getContourData = vi.fn();
vi.mock('../../api/endpoints/visualizations', () => ({
  getContourData: (...args: unknown[]) => getContourData(...args),
}));
vi.mock('../../hooks/useTheme', () => ({ useTheme: () => ({ theme: 'light' }) }));
vi.mock('react-plotly.js', () => ({
  default: ({ layout }: { layout: { width?: number; height?: number } }) => (
    <div data-testid="plot" data-width={layout.width ?? ''} data-height={layout.height ?? ''} />
  ),
}));

const oneReal = {
  n_variables: 1,
  variables: [{ name: 'a', type: 'real', bounds: [0, 1] }],
};
const fourReal = {
  n_variables: 4,
  variables: [
    { name: 'a', type: 'real', bounds: [0, 1] },
    { name: 'b', type: 'real', bounds: [0, 1] },
    { name: 'temp', type: 'real', bounds: [0, 10] },
    { name: 'press', type: 'real', bounds: [100, 200] },
  ],
};
let variablesData: unknown = oneReal;
vi.mock('../../hooks/api/useVariables', () => ({
  useVariables: () => ({ data: variablesData }),
}));

const observed: Element[] = [];
class FakeResizeObserver {
  constructor(public cb: () => void) {}
  observe(el: Element) {
    observed.push(el);
  }
  disconnect() {}
  unobserve() {}
}

const BOX = { width: 508, height: 700 };

describe('ContourPlot figure sizing', () => {
  const proto = HTMLElement.prototype;
  const widthDesc = Object.getOwnPropertyDescriptor(proto, 'clientWidth');
  const heightDesc = Object.getOwnPropertyDescriptor(proto, 'clientHeight');

  beforeEach(() => {
    observed.length = 0;
    variablesData = oneReal;
    getContourData.mockReset();
    getContourData.mockResolvedValue({
      x_var: 'a',
      y_var: 'b',
      x_grid: [[0, 1], [0, 1]],
      y_grid: [[0, 0], [1, 1]],
      predictions: [[0, 1], [2, 3]],
      uncertainties: [[0, 0], [0, 0]],
      x_bounds: [0, 1],
      y_bounds: [0, 1],
      colorbar_bounds: [0, 3],
    });
    vi.stubGlobal('ResizeObserver', FakeResizeObserver);
    Object.defineProperty(proto, 'clientWidth', { configurable: true, get: () => BOX.width });
    Object.defineProperty(proto, 'clientHeight', { configurable: true, get: () => BOX.height });
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    if (widthDesc) Object.defineProperty(proto, 'clientWidth', widthDesc);
    if (heightDesc) Object.defineProperty(proto, 'clientHeight', heightDesc);
  });

  it('sizes the figure after a mount that first took the "need two Real variables" early return', async () => {
    const client = createTestQueryClient();
    const wrapper = ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
    const { rerender } = render(<ContourPlot sessionId="s1" />, { wrapper });
    expect(screen.getByText(/at least two Real/i)).toBeTruthy();
    expect(observed).toHaveLength(0);

    variablesData = fourReal;
    rerender(<ContourPlot sessionId="s1" />);

    const plot = await screen.findByTestId('plot');
    expect(observed.length).toBeGreaterThan(0);
    const expected = contourFigureSize(BOX.width, BOX.height);
    expect(plot.getAttribute('data-width')).toBe(String(expected.width));
    expect(plot.getAttribute('data-height')).toBe(String(expected.height));
  });
});
