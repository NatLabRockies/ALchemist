/**
 * Theme Management Hook
 * Handles dark/light mode switching.
 *
 * The theme lives in one module-level store so every caller (the header
 * toggle in App, the Plotly contour, ...) sees the same value and re-renders
 * when it changes.
 */
import { useSyncExternalStore } from 'react';

type Theme = 'light' | 'dark';

const STORAGE_KEY = 'theme';

function readStoredTheme(): Theme {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === 'light' || stored === 'dark') return stored;
  } catch {
    // localStorage unavailable (private mode, blocked): fall through to default
  }
  return 'dark';
}

function applyClass(theme: Theme) {
  const root = document.documentElement;
  if (theme === 'dark') {
    root.classList.add('dark');
  } else {
    root.classList.remove('dark');
  }
}

function persist(theme: Theme) {
  try {
    localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    // ignore: the class on <html> still carries the theme for this session
  }
}

let currentTheme: Theme = readStoredTheme();
const listeners = new Set<() => void>();

// index.html's blocking inline script has already set the stored theme before
// first paint; applying it again at module load keeps the class in step with
// the store (and covers any host page without that script). localStorage is
// written only on a change, never at load.
if (typeof document !== 'undefined') {
  applyClass(currentTheme);
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function getSnapshot(): Theme {
  return currentTheme;
}

function setTheme(next: Theme) {
  if (next === currentTheme) return;
  currentTheme = next;
  applyClass(next);
  persist(next);
  listeners.forEach((listener) => listener());
}

function toggleTheme() {
  setTheme(currentTheme === 'dark' ? 'light' : 'dark');
}

export function useTheme() {
  const theme = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
  return { theme, toggleTheme };
}
