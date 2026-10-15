/**
 * Contour figure sizing.
 *
 * The visualization column sits between two fixed-width panels, so its width follows the window
 * while its height does not. Filling the column made the plot area taller than wide at common
 * window sizes. The figure is sized from the column's width instead: its plot area (the figure
 * minus the margins) is never taller than it is wide, and a column that is wider than tall
 * keeps the height it has.
 */

export interface Margins {
  l: number;
  r: number;
  t: number;
  b: number;
}

/**
 * Plotly margins in px. The legend sits above the plot (t) and the colorbar in the right margin (r).
 * Plotly widens a margin when its content needs more (long colorbar tick labels); see growMargins.
 */
export const CONTOUR_MARGINS: Margins = { l: 70, r: 85, t: 90, b: 50 };

/** Shortest figure for a wide column, so a short window scrolls instead of flattening the plot. */
export const MIN_FIGURE_HEIGHT = 360;

/**
 * Narrowest plot area. A column too narrow for the margins gets a figure wider than itself, which
 * the column scrolls, rather than a plot area of a few pixels.
 */
export const MIN_PLOT_WIDTH = 120;

export interface FigureSize {
  width: number;
  height: number;
}

/**
 * @param containerWidth  width of the column the figure sits in, px
 * @param containerHeight height of the column that is visible without scrolling, px
 * @param margins         the margins the figure is drawn with
 */
export function contourFigureSize(
  containerWidth: number,
  containerHeight: number,
  margins: Margins = CONTOUR_MARGINS,
): FigureSize {
  const plotWidth = Math.max(MIN_PLOT_WIDTH, Math.floor(containerWidth) - margins.l - margins.r);
  const width = plotWidth + margins.l + margins.r;
  // The tallest figure whose plot area is still no taller than it is wide.
  const squareHeight = plotWidth + margins.t + margins.b;
  const wanted = Math.max(Math.floor(containerHeight), MIN_FIGURE_HEIGHT);
  return { width, height: Math.min(squareHeight, wanted) };
}

/**
 * Raise each requested margin to the one Plotly actually used (it widens a margin to fit the
 * colorbar's labels and title). Never lowers a margin. Returns `requested` itself when nothing
 * grew by more than a pixel, so a state update from it is a no-op.
 */
export function growMargins(requested: Margins, actual: Margins): Margins {
  const grew = (side: keyof Margins) => actual[side] - requested[side] > 1;
  if (!(['l', 'r', 't', 'b'] as const).some(grew)) return requested;
  return {
    l: Math.max(requested.l, Math.ceil(actual.l)),
    r: Math.max(requested.r, Math.ceil(actual.r)),
    t: Math.max(requested.t, Math.ceil(actual.t)),
    b: Math.max(requested.b, Math.ceil(actual.b)),
  };
}
