import { describe, it, expect, afterEach } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { FALLBACK_FONT_FAMILY, PLOT_TEXT, plotColorsFrom, plotFontFamilyFrom, plotlyFonts, usePlotTheme } from './plotTheme';
import { useTheme } from '../../hooks/useTheme';

describe('plotColorsFrom', () => {
  const tokens: Record<string, string> = {
    '--foreground': '222 84% 5%',
    '--card': '0 0% 100%',
    '--popover': '0 0% 100%',
    '--popover-foreground': '222 84% 5%',
    '--muted-foreground': '215 16% 40%',
    '--border': '214 32% 85%',
    '--destructive': '0 84% 50%',
  };

  it('builds every colour from the theme tokens', () => {
    const c = plotColorsFrom((name) => tokens[name] ?? '');
    expect(c.text).toBe('hsl(222 84% 5%)');
    expect(c.tick).toBe('hsl(215 16% 40%)');
    expect(c.axis).toBe('hsl(215 16% 40%)');
    expect(c.pointFill).toBe('hsl(0 0% 100%)');
    expect(c.tooltipBg).toBe('hsl(0 0% 100%)');
    expect(c.tooltipBorder).toBe('hsl(214 32% 85%)');
    expect(c.grid).toBe('hsl(215 16% 40% / 0.25)');
    expect(c.band).toBe('hsl(0 84% 50% / 0.16)');
  });

  it('falls back to readable colours when a token is missing (no stylesheet loaded)', () => {
    const c = plotColorsFrom(() => '');
    for (const value of Object.values(c)) expect(value).toMatch(/^hsl\(/);
  });
});

describe('usePlotTheme', () => {
  afterEach(() => {
    document.head.querySelectorAll('style[data-test]').forEach((n) => n.remove());
  });

  it('re-reads the tokens when the theme changes', () => {
    const style = document.createElement('style');
    style.dataset.test = '1';
    style.textContent = ':root{--foreground:10 10% 10%} .dark{--foreground:200 20% 90%}';
    document.head.appendChild(style);

    const { result } = renderHook(() => ({ plot: usePlotTheme(), theme: useTheme() }));
    const before = result.current.plot.colors.text;
    act(() => result.current.theme.toggleTheme());
    const after = result.current.plot.colors.text;
    expect(after).not.toBe(before);
    act(() => result.current.theme.toggleTheme());
    expect(result.current.plot.colors.text).toBe(before);
  });
});

describe('plot font', () => {
  afterEach(() => {
    document.head.querySelectorAll('style[data-test]').forEach((n) => n.remove());
  });

  it('uses the family the page sets on the body, in every Recharts part and every Plotly text', () => {
    const style = document.createElement('style');
    style.dataset.test = '1';
    style.textContent = 'body{font-family: "Test Sans", system-ui, sans-serif}';
    document.head.appendChild(style);
    const ui = getComputedStyle(document.body).fontFamily;
    expect(ui).toContain('Test Sans');

    const { result } = renderHook(() => usePlotTheme());
    const plot = result.current;
    expect(plot.fontFamily).toBe(ui);
    expect(plot.axis.tick.fontFamily).toBe(ui);
    expect(plot.label('x').fontFamily).toBe(ui);
    expect(plot.tooltip.contentStyle.fontFamily).toBe(ui);
    expect(plot.legend.wrapperStyle.fontFamily).toBe(ui);

    const fonts = plotlyFonts(plot.fontFamily, 'black');
    for (const font of Object.values(fonts)) expect(font.family).toBe(ui);
  });

  it('falls back to the Tailwind sans stack when the page sets no font', () => {
    expect(plotFontFamilyFrom(() => '')).toBe(FALLBACK_FONT_FAMILY);
    expect(plotFontFamilyFrom(() => 'Inter, sans-serif')).toBe('Inter, sans-serif');
  });

  it('sizes a title as a panel heading and everything else as small text', () => {
    const fonts = plotlyFonts('x', 'black');
    expect(fonts.title.size).toBe(14);
    expect(fonts.base.size).toBe(PLOT_TEXT.label);
    expect(fonts.axisTitle.size).toBe(12);
    expect(fonts.legend.size).toBe(12);
    expect(fonts.hover.size).toBe(12);
  });
});
