import { describe, it, expect, afterEach } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { plotColorsFrom, usePlotTheme } from './plotTheme';
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
