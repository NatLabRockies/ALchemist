import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// The theme lives in a module-level store, so each test loads a fresh copy of
// the hook module (after seeding localStorage / <html>) to start from a known state.
async function loadHook() {
  vi.resetModules();
  const { useTheme } = await import('./useTheme');

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

  return { Toggler, Reader };
}

const hasDarkClass = () => document.documentElement.classList.contains('dark');

describe('useTheme', () => {
  beforeEach(() => {
    localStorage.clear();
    document.documentElement.classList.add('dark'); // index.html hardcodes class="dark"
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('shares one theme across every caller', async () => {
    const { Toggler, Reader } = await loadHook();
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
    expect(hasDarkClass()).toBe(false);
    expect(localStorage.getItem('theme')).toBe('light');

    await userEvent.click(screen.getByTestId('toggler'));

    expect(screen.getByTestId('reader')).toHaveTextContent('dark');
    expect(hasDarkClass()).toBe(true);
    expect(localStorage.getItem('theme')).toBe('dark');
  });

  it('defaults to dark, with the class set, and does not persist before a toggle', async () => {
    document.documentElement.classList.remove('dark');
    const { Reader } = await loadHook();
    render(<Reader />);
    expect(screen.getByTestId('reader')).toHaveTextContent('dark');
    expect(hasDarkClass()).toBe(true);
    expect(localStorage.getItem('theme')).toBeNull();
  });

  it('starts light when localStorage holds light, and removes the class at module load', async () => {
    localStorage.setItem('theme', 'light');
    const { Reader } = await loadHook();
    // applied at load, before any component mounts
    expect(hasDarkClass()).toBe(false);
    render(<Reader />);
    expect(screen.getByTestId('reader')).toHaveTextContent('light');
  });

  it('falls back to dark for an invalid stored value', async () => {
    localStorage.setItem('theme', 'purple');
    const { Reader } = await loadHook();
    render(<Reader />);
    expect(screen.getByTestId('reader')).toHaveTextContent('dark');
    expect(hasDarkClass()).toBe(true);
  });

  it('falls back to dark when localStorage.getItem throws, and still toggles', async () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('blocked');
    });
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('blocked');
    });
    const { Toggler, Reader } = await loadHook();
    render(
      <>
        <Toggler />
        <Reader />
      </>
    );
    expect(screen.getByTestId('reader')).toHaveTextContent('dark');
    await userEvent.click(screen.getByTestId('toggler'));
    expect(screen.getByTestId('reader')).toHaveTextContent('light');
    expect(hasDarkClass()).toBe(false);
  });
});
