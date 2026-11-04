import { describe, it, expect, vi } from 'vitest';
import { cloneElement, type ReactElement } from 'react';
import { render, screen } from '@testing-library/react';
import { QQPlot } from './QQPlot';

vi.mock('../../hooks/api/useVisualizations', () => ({
  useQQPlotData: () => ({
    isLoading: false,
    error: null,
    data: {
      theoretical_quantiles: [-1, 0, 1],
      sample_quantiles: [-0.8, 0.1, 0.9],
      // Far wider than anything 1.96 / sqrt(3) would draw: the axis has to follow the API's band.
      band_lower: [-7, -1, -0.5],
      band_upper: [0.5, 1, 7],
      z_mean: 0,
      z_std: 1,
      n_samples: 3,
      bounds: [-1, 1],
      calibrated: false,
      results_type: 'uncalibrated',
    },
  }),
}));

vi.mock('recharts', async (importOriginal) => {
  const actual = await importOriginal<typeof import('recharts')>();
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: ReactElement<{ width?: number; height?: number }> }) =>
      cloneElement(children, { width: 600, height: 400 }),
  };
});

describe('QQPlot band', () => {
  it('draws the pointwise band the API returns', () => {
    const { container } = render(<QQPlot sessionId="s1" useCalibrated={false} />);
    expect(screen.getByText('Pointwise 95% band')).toBeTruthy();
    const area = container.querySelector('.recharts-area-area');
    expect(area, 'band area').not.toBeNull();
    // The band's lower edge at the first point is the lowest thing on the chart, so it sits at the
    // bottom of the plot area. A band computed from 1.96 / sqrt(n) would not reach it.
    const ys = (area!.getAttribute('d') ?? '').match(/-?\d+(\.\d+)?/g)!.map(Number).filter((_, i) => i % 2 === 1);
    expect(Math.max(...ys) - Math.min(...ys)).toBeGreaterThan(250);
  });
});
