/**
 * Theme for the Recharts plots.
 *
 * Recharts draws SVG with colours passed as props, so the plots read the app's theme tokens (the
 * CSS variables Tailwind uses, defined in index.css for :root and .dark) and re-read them when the
 * theme changes. Series colours (the blue points and lines, the red reference line) are data
 * colours and stay the same in both themes; both clear 3:1 against the card colour in each theme.
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
    const tooltipBox: CSSProperties = {
      background: colors.tooltipBg,
      color: colors.tooltipText,
      border: `1px solid ${colors.tooltipBorder}`,
      borderRadius: 6,
      fontSize: 12,
      boxShadow: '0 4px 12px hsl(0 0% 0% / 0.12)',
    };
    return {
      colors,
      /** <CartesianGrid {...grid} /> */
      grid: { stroke: colors.grid, strokeDasharray: '3 3' },
      /** <XAxis {...axis} /> and <YAxis {...axis} /> */
      axis: { stroke: colors.axis, tick: { fill: colors.tick, fontSize: 12 } },
      /** The `label` prop of an axis: its title, in the tick colour. */
      label: (value: string, extra: Record<string, unknown> = {}) => ({
        value,
        fill: colors.tick,
        fontSize: 12,
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
        wrapperStyle: { fontSize: 12 },
      },
    };
    // `theme` is the dependency: the tokens are re-read when it changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [theme]);
}
