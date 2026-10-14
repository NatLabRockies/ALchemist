import { describe, it, expect } from 'vitest';
import { dropRecorded } from './recordedSuggestions';
import type { VariableDetail } from '../api/types';

const rows = [
  { temp: 101.111, cat: 'Pd', Output: 0.4, Iteration: 3 },
  { temp: 250, cat: 'Pt', Output: 0.7, Iteration: 3 },
];

describe('dropRecorded', () => {
  it('drops a suggestion whose inputs match a recorded row', () => {
    const s = [{ temp: 250, cat: 'Pt', _reason: 'qEI', Iteration: 4 }];
    expect(dropRecorded(s, rows)).toEqual([]);
  });

  it('matches reals after the 3-decimal rounding the Add Point dialog records', () => {
    const s = [{ temp: 101.11111, cat: 'Pd' }];
    expect(dropRecorded(s, rows)).toEqual([]);
  });

  it('matches when the table holds a longer value that rounds to the same 3 decimals', () => {
    const s = [{ temp: 101.1114, cat: 'Pd' }];
    expect(dropRecorded(s, rows)).toEqual([]);
  });

  it('keeps a suggestion that differs in any input', () => {
    const s = [{ temp: 250, cat: 'Pd' }, { temp: 251, cat: 'Pt' }];
    expect(dropRecorded(s, rows)).toEqual(s);
  });

  it('keeps a suggestion one thousandth away: the dialog records them as different points', () => {
    const s = [{ temp: 101.112, cat: 'Pd' }, { temp: 101.11, cat: 'Pd' }];
    expect(dropRecorded(s, rows)).toEqual(s);
  });

  it('ignores bookkeeping keys when comparing', () => {
    const s = [{ temp: 250, cat: 'Pt', _reason: 'x', Iteration: 9, Output: 1 }];
    expect(dropRecorded(s, rows)).toEqual([]);
  });

  it('keeps everything when nothing is recorded', () => {
    const s = [{ temp: 1, cat: 'Pd' }];
    expect(dropRecorded(s, [])).toEqual(s);
  });

  it('matches a discrete numeric returned as a number or a string', () => {
    const table = [{ speed: '2', cat: 'Pd' }, { speed: 4, cat: 'Pd' }];
    const s = [{ speed: 2, cat: 'Pd' }, { speed: '4', cat: 'Pd' }, { speed: 6, cat: 'Pd' }];
    expect(dropRecorded(s, table)).toEqual([{ speed: 6, cat: 'Pd' }]);
  });

  it('does not match a categorical level that merely looks numeric-equal', () => {
    const vars = [{ name: 'lot', type: 'categorical' }] as VariableDetail[];
    const s = [{ lot: '1.0' }];
    expect(dropRecorded(s, [{ lot: '1' }], undefined, vars)).toEqual(s);
    expect(dropRecorded([{ lot: '1' }], [{ lot: '1' }], undefined, vars)).toEqual([]);
  });

  it('matches an integer variable after the whole-number rounding the dialog records', () => {
    const vars = [{ name: 'n', type: 'integer' }] as VariableDetail[];
    expect(dropRecorded([{ n: 3.4 }], [{ n: 3 }], undefined, vars)).toEqual([]);
    expect(dropRecorded([{ n: 3.6 }], [{ n: 3 }], undefined, vars)).toEqual([{ n: 3.6 }]);
  });

  it('keeps a suggestion when the table row lacks one of its inputs', () => {
    const s = [{ temp: 250, cat: 'Pt', extra: 1 }];
    expect(dropRecorded(s, rows)).toEqual(s);
  });

  it('never treats a missing or null value as zero', () => {
    expect(dropRecorded([{ temp: 0 }], [{ temp: null }])).toEqual([{ temp: 0 }]);
    expect(dropRecorded([{ temp: '' }], [{ temp: 0 }])).toEqual([{ temp: '' }]);
  });

  it('honours an explicit input key list', () => {
    const s = [{ temp: 250, cat: 'Zz' }];
    expect(dropRecorded(s, rows, ['temp'])).toEqual([]);
  });

  it('keeps a suggestion that has no input keys to compare', () => {
    const s = [{ _reason: 'qEI', Iteration: 2 }];
    expect(dropRecorded(s, rows)).toEqual(s);
  });

  it('treats extra rows (provenance suggested) as recorded alongside the table', () => {
    const s = [{ temp: 250, cat: 'Pt' }, { temp: 300, cat: 'Pd' }];
    const table = [{ temp: 255, cat: 'Pt', Output: 0.7 }];
    expect(dropRecorded(s, table)).toEqual(s);
    expect(dropRecorded(s, [...table, { temp: 250, cat: 'Pt' }])).toEqual([{ temp: 300, cat: 'Pd' }]);
  });
});
