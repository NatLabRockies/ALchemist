/** Columns holding whole-number bookkeeping values, shown without decimals. */
export const INTEGER_COLUMNS: ReadonlySet<string> = new Set([
  'Iteration',
  'WholePlot',
  'Subplot',
  'RunOrder',
]);

/**
 * Display text for one table or card value. Bookkeeping columns are shown as
 * integers; other numbers keep 3 decimals; null and undefined show "-";
 * anything else is shown as its string form.
 */
export function formatCell(col: string, value: unknown): string {
  if (value === null || value === undefined) return '-';
  if (typeof value === 'number') {
    return INTEGER_COLUMNS.has(col) ? String(Math.round(value)) : value.toFixed(3);
  }
  return String(value);
}
