/**
 * Theme for the Recharts plots.
 *
 * Recharts draws SVG with colours passed as props, so the plots read the app's theme tokens (the
 * CSS variables Tailwind uses, defined in index.css for :root and .dark) and re-read them when the
 * theme changes. Series colours (the blue points and lines, the red reference line) are data
 * colours and stay the same in both themes; both clear 3:1 against the card colour in each theme.
 *
 * Text follows the UI too. The family is read once from the page (the body's computed font-family,
 * which is Tailwind's sans stack) so every plot, Recharts and Plotly alike, sets its text in the
 * face of the controls around it. The sizes are the app's own scale: a plot title is a panel
 * heading (text-sm, medium), everything else in a plot is small text (text-xs).
 */
import { useMemo, type CSSProperties, type ReactNode } from 'react';
import { useTheme } from '../../hooks/useTheme';

export interface PlotColors {
  /** Grid lines: quiet, behind the data. */
  grid: string;
  /** Axis lines. */
  axis: string;
  /** Tick numbers and axis titles. */
  tick: string;
  /** Titles, legend entries and other primary text. */
  text: string;
  /** Fill of hollow points: the card colour, so the series ring reads and the grid does not show through. */
  pointFill: string;
  tooltipBg: string;
  tooltipText: string;
  tooltipBorder: string;
  /** Soft tint for the Q-Q confidence band. */
  band: string;
}

/** Light-theme token values, used when a token is not defined (no stylesheet, e.g. in tests). */
const FALLBACK_TOKENS: Record<string, string> = {
  '--foreground': '222.2 84% 4.9%',
  '--card': '0 0% 100%',
  '--popover': '0 0% 100%',
  '--popover-foreground': '222.2 84% 4.9%',
  '--muted-foreground': '215.4 16.3% 40%',
  '--border': '214.3 31.8% 85%',
  '--destructive': '0 84.2% 50%',
};

/** Pure: derive the plot colours from a token reader (`--name` to "H S% L%", or '' if undefined). */
export function plotColorsFrom(readToken: (name: string) => string): PlotColors {
  const token = (name: string) => readToken(name).trim() || FALLBACK_TOKENS[name];
  const hsl = (name: string, alpha?: number) =>
    alpha === undefined ? `hsl(${token(name)})` : `hsl(${token(name)} / ${alpha})`;
  return {
    // The border token is too faint on the dark card, so the grid is the muted text colour at low
    // alpha: the same step as the border on white, a little stronger on the dark surface.
    grid: hsl('--muted-foreground', 0.25),
    axis: hsl('--muted-foreground'),
    tick: hsl('--muted-foreground'),
    text: hsl('--foreground'),
    pointFill: hsl('--card'),
    tooltipBg: hsl('--popover'),
    tooltipText: hsl('--popover-foreground'),
    tooltipBorder: hsl('--border'),
    band: hsl('--destructive', 0.16),
  };
}

/** The app's sans stack (Tailwind's default `font-sans`), used when no stylesheet is loaded (tests). */
export const FALLBACK_FONT_FAMILY =
  'ui-sans-serif, system-ui, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji"';

/** Text sizes in px, from the UI scale: text-sm for a title, text-xs for the rest. */
export const PLOT_TEXT = {
  /** A plot's own title (the heading above a Recharts plot is HTML, text-sm font-medium). */
  title: 14,
  titleWeight: 500,
  /** Axis titles, tick labels, legend entries, colorbar labels and tooltips. */
  label: 12,
} as const;

/** Pure: the plot font family from a reader of the page's computed font-family ('' if unknown). */
export function plotFontFamilyFrom(readFamily: () => string): string {
  return readFamily().trim() || FALLBACK_FONT_FAMILY;
}

function readBodyFontFamily(): string {
  if (typeof document === 'undefined' || !document.body) return '';
  return getComputedStyle(document.body).fontFamily;
}

/**
 * Plotly's text styles, all in one family. `base` goes on the layout (it covers ticks, axis titles,
 * the legend, the colorbar and hover labels); the others state the family and size again where the
 * plot sets its own size, so no part falls back to Plotly's own face (Open Sans).
 */
export function plotlyFonts(family: string, color: string) {
  return {
    base: { family, size: PLOT_TEXT.label, color },
    title: { family, size: PLOT_TEXT.title, weight: PLOT_TEXT.titleWeight },
    axisTitle: { family, size: PLOT_TEXT.label },
    legend: { family, size: PLOT_TEXT.label, color },
    hover: { family, size: PLOT_TEXT.label },
  };
}

function readRootToken(name: string): string {
  if (typeof document === 'undefined') return '';
  return getComputedStyle(document.documentElement).getPropertyValue(name);
}

/**
 * Legend entry text. Recharts colours an entry's text from its series' fill, which for the hollow
 * points (white fill) is white and vanishes on a light background. Colour the text from the theme
 * and let the swatch beside it carry the series colour.
 */
export function legendFormatter(value: ReactNode) {
  return <span className="text-foreground">{value}</span>;
}

/** Props to spread onto the Recharts parts, from the current theme. */
export function usePlotTheme() {
  const { theme } = useTheme();
  return useMemo(() => {
    const colors = plotColorsFrom(readRootToken);
    const fontFamily = plotFontFamilyFrom(readBodyFontFamily);
    const fontSize = PLOT_TEXT.label;
    const tooltipBox: CSSProperties = {
      background: colors.tooltipBg,
      color: colors.tooltipText,
      border: `1px solid ${colors.tooltipBorder}`,
      borderRadius: 6,
      fontFamily,
      fontSize,
      boxShadow: '0 4px 12px hsl(0 0% 0% / 0.12)',
    };
    return {
      colors,
      /** The UI's font family, for text a plot draws itself (Plotly). */
      fontFamily,
      /** <CartesianGrid {...grid} /> */
      grid: { stroke: colors.grid, strokeDasharray: '3 3' },
      /** <XAxis {...axis} /> and <YAxis {...axis} /> */
      axis: { stroke: colors.axis, tick: { fill: colors.tick, fontFamily, fontSize } },
      /** The `label` prop of an axis: its title, in the tick colour. */
      label: (value: string, extra: Record<string, unknown> = {}) => ({
        value,
        fill: colors.tick,
        fontFamily,
        fontSize,
        ...extra,
      }),
      /** <Tooltip {...tooltip} /> */
      tooltip: {
        contentStyle: tooltipBox,
        labelStyle: { color: colors.tooltipText, fontWeight: 500 },
        // An entry is coloured from its series fill, white for hollow points: use the theme text.
        itemStyle: { color: colors.tooltipText },
        cursor: { stroke: colors.tick, strokeOpacity: 0.5, strokeDasharray: '3 3' },
      },
      /** <Legend {...legend} /> */
      legend: {
        verticalAlign: 'top' as const,
        height: 36,
        formatter: legendFormatter,
        wrapperStyle: { fontFamily, fontSize },
      },
    };
    // `theme` is the dependency: the tokens are re-read when it changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [theme]);
}
