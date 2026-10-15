import { describe, it, expect, vi } from 'vitest';
import { cloneElement, type ReactElement } from 'react';
import { render, screen } from '@testing-library/react';
import { ParityPlot } from './ParityPlot';

vi.mock('../../hooks/api/useVisualizations', () => ({
  useParityData: () => ({
    isLoading: false,
    error: null,
    data: {
      y_true: [1, 2, 3, 4],
      y_pred: [1.1, 1.9, 3.2, 3.8],
      y_std: [0.1, 0.1, 0.1, 0.1],
      calibrated: false,
      metrics: null,
    },
  }),
}));

// jsdom has no layout, so ResponsiveContainer would render nothing. Give the chart a fixed size.
vi.mock('recharts', async (importOriginal) => {
  const actual = await importOriginal<typeof import('recharts')>();
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: ReactElement<{ width?: number; height?: number }> }) =>
      cloneElement(children, { width: 600, height: 400 }),
  };
});

describe('ParityPlot legend', () => {
  it('draws the "Predictions" entry in the theme foreground, not in the white fill of its points', () => {
    const { container } = render(<ParityPlot sessionId="s1" useCalibrated={false} sigmaMultiplier="1.96" />);
    const entry = screen.getByText('Predictions');
    // Recharts colours a legend entry's text from the series fill, which is white for hollow points.
    expect(entry.closest('.text-foreground'), 'legend text must carry the foreground colour').not.toBeNull();
    expect(container.querySelector('.recharts-legend-item')).not.toBeNull();
  });
});
