import { describe, it, expect } from 'vitest';
import { formatCell, INTEGER_COLUMNS } from './formatCell';

describe('formatCell', () => {
  it('lists the bookkeeping columns that render as integers', () => {
    expect([...INTEGER_COLUMNS].sort()).toEqual(['Iteration', 'RunOrder', 'Subplot', 'WholePlot']);
  });

  it.each(['Iteration', 'WholePlot', 'Subplot', 'RunOrder'])(
    'renders %s as an integer',
    (col) => {
      expect(formatCell(col, 2.0)).toBe('2');
      expect(formatCell(col, 7)).toBe('7');
      expect(formatCell(col, 2.6)).toBe('3');
    },
  );

  it('keeps 3 decimals for a real column', () => {
    expect(formatCell('x1', 1.23456)).toBe('1.235');
    expect(formatCell('x1', 2)).toBe('2.000');
  });

  it('renders null and undefined as "-"', () => {
    expect(formatCell('x1', null)).toBe('-');
    expect(formatCell('WholePlot', undefined)).toBe('-');
  });

  it('passes a string through unchanged', () => {
    expect(formatCell('c', 'a')).toBe('a');
    expect(formatCell('WholePlot', 'n/a')).toBe('n/a');
  });
});
