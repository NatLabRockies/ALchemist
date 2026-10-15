import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useTheme } from './useTheme';

function Toggler() {
  const { theme, toggleTheme } = useTheme();
  return (
    <button onClick={toggleTheme} data-testid="toggler">
      {theme}
    </button>
  );
}

function Reader() {
  const { theme } = useTheme();
  return <span data-testid="reader">{theme}</span>;
}

describe('useTheme', () => {
  beforeEach(() => {
    localStorage.clear();
    document.documentElement.classList.remove('dark');
  });

  it('shares one theme across every caller', async () => {
    render(
      <>
        <Toggler />
        <Reader />
      </>
    );
    expect(screen.getByTestId('reader')).toHaveTextContent('dark');

    await userEvent.click(screen.getByTestId('toggler'));

    expect(screen.getByTestId('toggler')).toHaveTextContent('light');
    expect(screen.getByTestId('reader')).toHaveTextContent('light');
    expect(document.documentElement.classList.contains('dark')).toBe(false);
    expect(localStorage.getItem('theme')).toBe('light');

    await userEvent.click(screen.getByTestId('toggler'));

    expect(screen.getByTestId('reader')).toHaveTextContent('dark');
    expect(document.documentElement.classList.contains('dark')).toBe(true);
    expect(localStorage.getItem('theme')).toBe('dark');
  });

  it('defaults to dark and applies the class on mount', () => {
    render(<Reader />);
    expect(screen.getByTestId('reader')).toHaveTextContent('dark');
    expect(document.documentElement.classList.contains('dark')).toBe(true);
  });
});
