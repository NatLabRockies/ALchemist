import { describe, it, expect } from 'vitest';
import { monitorModeFromUrl, monitoringModeFor } from './monitorMode';

describe('monitor mode', () => {
  it('reads ?mode=monitor from the URL', () => {
    expect(monitorModeFromUrl('?mode=monitor')).toBe(true);
    expect(monitorModeFromUrl('?session=x&mode=monitor')).toBe(true);
    expect(monitorModeFromUrl('?mode=edit')).toBe(false);
    expect(monitorModeFromUrl('')).toBe(false);
  });

  it('follows the lock when the page was opened normally', () => {
    expect(monitoringModeFor(true, false)).toBe(true);
    expect(monitoringModeFor(false, false)).toBe(false);
  });

  it('stays in monitor mode on unlock when the page was opened with ?mode=monitor (W41)', () => {
    expect(monitoringModeFor(false, true)).toBe(true);
    expect(monitoringModeFor(true, true)).toBe(true);
  });
});
