import { describe, it, expect } from 'vitest';
import { contourFigureSize, growMargins, CONTOUR_MARGINS, MIN_FIGURE_HEIGHT } from './contourSizing';

const hMargin = CONTOUR_MARGINS.l + CONTOUR_MARGINS.r;
const vMargin = CONTOUR_MARGINS.t + CONTOUR_MARGINS.b;

function plotArea(size: { width: number; height: number }) {
  return { w: size.width - hMargin, h: size.height - vMargin };
}

describe('contourFigureSize', () => {
  it('never makes the plot area taller than wide, at any container size', () => {
    for (let width = 300; width <= 2400; width += 37) {
      for (let height = 200; height <= 1400; height += 53) {
        const { w, h } = plotArea(contourFigureSize(width, height));
        expect(h, `container ${width}x${height}`).toBeLessThanOrEqual(w);
      }
    }
  });

  it('is square in the plot area for a narrow, tall column (the 1440x900 case)', () => {
    const size = contourFigureSize(508, 680);
    expect(size.width).toBe(508);
    const { w, h } = plotArea(size);
    expect(w).toBe(508 - hMargin);
    expect(h).toBe(w);
  });

  it('uses the available height when the column is wider than tall', () => {
    const size = contourFigureSize(1600, 700);
    expect(size.width).toBe(1600);
    expect(size.height).toBe(700);
  });

  it('does not shrink below the minimum height unless that would make the plot area tall', () => {
    expect(contourFigureSize(1600, 100).height).toBe(MIN_FIGURE_HEIGHT);
    // Narrow: the square wins over the minimum height.
    const narrow = contourFigureSize(hMargin + 100, 900);
    expect(plotArea(narrow).h).toBeLessThanOrEqual(plotArea(narrow).w);
  });

  it('returns whole pixels and a non-negative size for a degenerate container', () => {
    const size = contourFigureSize(0, 0);
    expect(size.width).toBeGreaterThanOrEqual(0);
    expect(size.height).toBeGreaterThanOrEqual(0);
    const frac = contourFigureSize(500.7, 600.3);
    expect(Number.isInteger(frac.width)).toBe(true);
    expect(Number.isInteger(frac.height)).toBe(true);
  });

  it('keeps the plot area square when Plotly widens a margin (colorbar labels)', () => {
    const margins = { ...CONTOUR_MARGINS, r: 94 };
    const size = contourFigureSize(748, 900, margins);
    expect(size.height - (margins.t + margins.b)).toBe(748 - (margins.l + margins.r));
  });
});

describe('growMargins', () => {
  it('raises a side to the margin Plotly actually used, never lowers one', () => {
    const grown = growMargins(CONTOUR_MARGINS, { l: 50, r: 94, t: 90, b: 52 });
    expect(grown).toEqual({ l: 70, r: 94, t: 90, b: 52 });
  });

  it('returns the same object when nothing grew, so state does not update', () => {
    expect(growMargins(CONTOUR_MARGINS, { l: 70, r: 85, t: 90, b: 50 })).toBe(CONTOUR_MARGINS);
    expect(growMargins(CONTOUR_MARGINS, { l: 70.2, r: 85.4, t: 90, b: 50 })).toBe(CONTOUR_MARGINS);
  });
});
