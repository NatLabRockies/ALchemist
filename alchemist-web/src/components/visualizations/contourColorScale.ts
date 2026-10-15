/**
 * The contour's colour scale: fixed (the default) or rescaled to each slice.
 *
 * Fixed pins the colour bar to the observed output range the API returns with every contour
 * (`output_range`). That range comes from the experiments, not from the slice, so moving a fixed
 * value never moves the colour bar and two slices can be compared by colour. A prediction outside
 * the observed range saturates at the end colour; hover still shows its true value.
 *
 * Per slice is the old behaviour: Plotly spans the colours over the current slice's own min and
 * max, which shows the structure within one slice but not how slices compare.
 *
 * The choice is remembered in localStorage, the way the theme is (useTheme).
 */
export type ContourColorScale = 'fixed' | 'per-slice';

export const COLOR_SCALE_STORAGE_KEY = 'alchemist-contour-color-scale';

const DEFAULT_SCALE: ContourColorScale = 'fixed';

export function readContourColorScale(): ContourColorScale {
  try {
    const stored = localStorage.getItem(COLOR_SCALE_STORAGE_KEY);
    if (stored === 'fixed' || stored === 'per-slice') return stored;
  } catch {
    // localStorage unavailable (private mode, blocked): use the default
  }
  return DEFAULT_SCALE;
}

export function storeContourColorScale(scale: ContourColorScale): void {
  try {
    localStorage.setItem(COLOR_SCALE_STORAGE_KEY, scale);
  } catch {
    // not remembered; the choice still holds while the page is open
  }
}

/** A range the fixed scale can use: two finite numbers, the first below the second. */
export function usableRange(range: readonly number[] | null | undefined): [number, number] | null {
  if (!range || range.length !== 2) return null;
  const [low, high] = range;
  if (!Number.isFinite(low) || !Number.isFinite(high) || !(high > low)) return null;
  return [low, high];
}

/** The colour limits for the contour trace, or null to let Plotly fit them to the slice. */
export function contourZRange(
  scale: ContourColorScale,
  outputRange: readonly number[] | null | undefined,
): { zmin: number; zmax: number } | null {
  if (scale !== 'fixed') return null;
  const range = usableRange(outputRange);
  return range ? { zmin: range[0], zmax: range[1] } : null;
}
