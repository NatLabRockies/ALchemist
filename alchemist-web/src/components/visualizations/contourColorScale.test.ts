import { describe, it, expect, vi, afterEach } from 'vitest';
import {
  COLOR_SCALE_STORAGE_KEY,
  contourZRange,
  readContourColorScale,
  storeContourColorScale,
} from './contourColorScale';

describe('contourZRange', () => {
  it('pins the colour bar to the observed output range in fixed mode', () => {
    expect(contourZRange('fixed', [1.5, 25])).toEqual({ zmin: 1.5, zmax: 25 });
  });

  it('leaves the range to the slice in per-slice mode', () => {
    expect(contourZRange('per-slice', [1.5, 25])).toBeNull();
  });

  it('falls back to the slice when there is no usable range', () => {
    expect(contourZRange('fixed', null)).toBeNull();
    expect(contourZRange('fixed', undefined)).toBeNull();
    // A single observed value (or all equal) gives no span to colour across.
    expect(contourZRange('fixed', [4, 4])).toBeNull();
    expect(contourZRange('fixed', [5, 4])).toBeNull();
    expect(contourZRange('fixed', [Number.NaN, 4])).toBeNull();
    expect(contourZRange('fixed', [0, Number.POSITIVE_INFINITY])).toBeNull();
  });
});

describe('remembered colour scale', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    localStorage.clear();
  });

  it('defaults to fixed', () => {
    expect(readContourColorScale()).toBe('fixed');
  });

  it('reads back the stored choice', () => {
    storeContourColorScale('per-slice');
    expect(localStorage.getItem(COLOR_SCALE_STORAGE_KEY)).toBe('per-slice');
    expect(readContourColorScale()).toBe('per-slice');
    storeContourColorScale('fixed');
    expect(readContourColorScale()).toBe('fixed');
  });

  it('ignores an unknown stored value', () => {
    localStorage.setItem(COLOR_SCALE_STORAGE_KEY, 'logarithmic');
    expect(readContourColorScale()).toBe('fixed');
  });

  it('survives blocked storage', () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('blocked');
    });
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('blocked');
    });
    expect(readContourColorScale()).toBe('fixed');
    expect(() => storeContourColorScale('per-slice')).not.toThrow();
  });
});
