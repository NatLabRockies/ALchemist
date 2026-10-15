/**
 * Shared theming for the Recharts plots.
 */
import type { ReactNode } from 'react';

/**
 * Legend entry text. Recharts colours an entry's text from its series' fill, which for the hollow
 * points (white fill) is white and vanishes on a light background. Colour the text from the theme
 * and let the swatch beside it carry the series colour.
 */
export function legendFormatter(value: ReactNode) {
  return <span className="text-foreground">{value}</span>;
}
