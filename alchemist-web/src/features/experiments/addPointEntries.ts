/**
 * Per-point entries for the Add Point dialog.
 *
 * The dialog steps through every pending (staged) suggestion. What the user
 * types for each point (actual conditions, output, noise, reason) lives in one
 * map keyed by a stable suggestion id, so moving between points never loses
 * an entry and a saved point can never leak its values into the next one.
 */
import Papa from 'papaparse';
import { PLOT_KEYS } from '../../api/types';
import type { VariableDetail } from '../../api/types';
import { formatSuggested, roundSuggested } from '../../lib/rounding';

export interface PointEntry {
  /** Actual conditions, as typed (pre-filled with the rounded suggestion). */
  actual: Record<string, string>;
  output: string;
  noise: string;
  reason: string;
}

export type EntriesMap = Record<string, PointEntry>;

/** Keys of a suggestion that are not variables (UI metadata, bookkeeping, plot position). */
const NON_INPUT_KEYS = new Set<string>(['Output', 'Noise', 'Iteration', 'Reason', ...PLOT_KEYS]);

export function inputKeysOf(suggestion: Record<string, unknown> | null | undefined): string[] {
  return Object.keys(suggestion || {}).filter((k) => !k.startsWith('_') && !NON_INPUT_KEYS.has(k));
}

/** Input keys for a point: the suggestion's variables, then every observed variable. */
export function entryInputKeys(
  suggestion: Record<string, unknown> | null | undefined,
  variables: VariableDetail[] = [],
): string[] {
  const keys = inputKeysOf(suggestion);
  for (const v of variables) if (v.role === 'observed' && !keys.includes(v.name)) keys.push(v.name);
  return keys;
}

export interface SetupBanner {
  title: string;
  settings: Array<{ name: string; value: string }>;
}

/**
 * Setup to establish before point `index`: shown when it starts a new whole
 * plot (or subplot) in run order. Depends only on the previous point in the
 * list, never on navigation history.
 */
export function setupBannerFor(
  suggestions: any[],
  index: number,
  variables: VariableDetail[] = [],
): SetupBanner | null {
  const s = suggestions[index];
  if (!s || s.WholePlot == null) return null;
  const prev = index > 0 ? suggestions[index - 1] : null;
  const changed = !prev || prev.WholePlot !== s.WholePlot || (s.Subplot != null && prev.Subplot !== s.Subplot);
  if (!changed) return null;
  const total = Math.max(...suggestions.map((x) => (typeof x.WholePlot === 'number' ? x.WholePlot : 0)));
  const title = `Whole plot ${s.WholePlot} of ${total}` + (s.Subplot != null ? `, subplot ${s.Subplot}` : '');
  const restricted = variables.filter((v) => v.role !== 'observed' && (v.changes === 'hard' || v.changes === 'very_hard'));
  const settings = restricted
    .filter((v) => v.name in s)
    .map((v) => ({ name: v.name, value: formatSuggested(s[v.name], v) }));
  return { title, settings };
}

// Suggestions without a queue id (an acquisition result that was not staged)
// get a client-side id tied to the object itself. The pending list after a
// save is a filtered copy that holds the same objects, so the id is stable.
const clientIds = new WeakMap<object, string>();
let clientIdCounter = 0;

export function suggestionKey(suggestion: any): string {
  if (suggestion && typeof suggestion._queueItemId === 'string' && suggestion._queueItemId) {
    return `q:${suggestion._queueItemId}`;
  }
  if (suggestion && typeof suggestion === 'object') {
    let id = clientIds.get(suggestion);
    if (!id) {
      clientIdCounter += 1;
      id = `c:${clientIdCounter}`;
      clientIds.set(suggestion, id);
    }
    return id;
  }
  return 'c:none';
}

export function defaultReason(suggestion: any): string {
  return suggestion?._reason || suggestion?.Reason || 'Acquisition';
}

export function initialEntry(suggestion: any, variables: VariableDetail[] = []): PointEntry {
  const varByName = new Map(variables.map((v) => [v.name, v]));
  const actual: Record<string, string> = {};
  for (const k of entryInputKeys(suggestion, variables)) {
    const rounded = k in (suggestion || {}) ? roundSuggested(suggestion[k], varByName.get(k)) : '';
    actual[k] = String(rounded ?? '');
  }
  return {
    actual,
    output: suggestion?.Output != null ? String(suggestion.Output) : '',
    noise: suggestion?.Noise != null ? String(suggestion.Noise) : '',
    reason: defaultReason(suggestion),
  };
}

/** The entry for a suggestion: what the user typed, else the pre-fill. */
export function entryFor(entries: EntriesMap, suggestion: any, variables: VariableDetail[] = []): PointEntry {
  return entries[suggestionKey(suggestion)] ?? initialEntry(suggestion, variables);
}

function parseNumber(text: string): number | null {
  const t = text.trim();
  if (t === '') return null;
  const n = Number(t);
  return Number.isFinite(n) ? n : null;
}

/** True when the point has a usable output and will be saved. */
export function hasOutput(entry: PointEntry): boolean {
  return parseNumber(entry.output) !== null;
}

/** Problems that block saving an entered point (empty list = fine). */
export function entryProblems(entry: PointEntry, variables: VariableDetail[] = []): string[] {
  const problems: string[] = [];
  const varByName = new Map(variables.map((v) => [v.name, v]));
  if (entry.output.trim() !== '' && parseNumber(entry.output) === null) {
    problems.push(`output "${entry.output}" is not a number`);
  }
  if (entry.noise.trim() !== '') {
    const n = parseNumber(entry.noise);
    if (n === null || n < 0) problems.push(`noise "${entry.noise}" must be a non-negative number`);
  }
  for (const [k, raw] of Object.entries(entry.actual)) {
    const v = varByName.get(k);
    if (raw.trim() === '') {
      problems.push(`${k} is empty`);
    } else if (v && v.type !== 'categorical' && parseNumber(raw) === null) {
      problems.push(`${k} "${raw}" is not a number`);
    } else if (v?.type === 'categorical' && v.categories && !v.categories.map(String).includes(raw.trim())) {
      // Same rule as the server: an undeclared level is refused at entry.
      problems.push(`${k} "${raw}" is not one of ${v.categories.join(', ')}`);
    }
  }
  return problems;
}

/** Typed value for the API: numbers for numeric variables, text for categorical. */
function toApiValue(raw: string, variable: VariableDetail | undefined): number | string {
  if (variable?.type === 'categorical') return raw;
  const n = parseNumber(raw);
  return n === null ? raw : n;
}

export interface BatchEntryPayload {
  item_id?: string;
  inputs: Record<string, number | string>;
  output: number;
  noise?: number;
  reason?: string;
  iteration?: number;
}

export interface BatchPlan {
  /** Points with an output, in queue order, ready for one request. */
  payload: BatchEntryPayload[];
  /** Keys of the points in `payload`. */
  savedKeys: string[];
  /** 1-based positions of points without an output (they stay pending). */
  pendingPositions: number[];
  /** Entered points with invalid values, by 1-based position. */
  invalid: { position: number; problems: string[] }[];
}

export function planBatch(
  suggestions: any[],
  entries: EntriesMap,
  variables: VariableDetail[] = [],
): BatchPlan {
  const varByName = new Map(variables.map((v) => [v.name, v]));
  const plan: BatchPlan = { payload: [], savedKeys: [], pendingPositions: [], invalid: [] };
  suggestions.forEach((s, i) => {
    const entry = entryFor(entries, s, variables);
    const problems = entryProblems(entry, variables);
    if (!hasOutput(entry)) {
      // A bad output string counts as "not entered" only if it is blank.
      if (entry.output.trim() !== '') plan.invalid.push({ position: i + 1, problems });
      else plan.pendingPositions.push(i + 1);
      return;
    }
    if (problems.length > 0) {
      plan.invalid.push({ position: i + 1, problems });
      return;
    }
    const inputs: Record<string, number | string> = {};
    for (const [k, raw] of Object.entries(entry.actual)) inputs[k] = toApiValue(raw, varByName.get(k));
    const item: BatchEntryPayload = { inputs, output: parseNumber(entry.output)! };
    const noise = parseNumber(entry.noise);
    if (noise !== null) item.noise = noise;
    if (entry.reason.trim()) item.reason = entry.reason.trim();
    // A whole acquisition round shares one iteration; only a real number is sent.
    if (typeof s?.Iteration === 'number' && Number.isFinite(s.Iteration)) item.iteration = s.Iteration;
    if (typeof s?._queueItemId === 'string' && s._queueItemId) item.item_id = s._queueItemId;
    plan.payload.push(item);
    plan.savedKeys.push(suggestionKey(s));
  });
  return plan;
}

// ---------------------------------------------------------------------------
// Pasting into the table view
// ---------------------------------------------------------------------------

/** A pasted block that is one value per line (no tabs or commas). */
export function isColumnPaste(text: string): boolean {
  const lines = text.replace(/\r/g, '').split('\n').filter((l) => l.trim() !== '');
  return lines.length > 0 && lines.every((l) => !/[\t,;]/.test(l.trim()));
}

/** One value per line, blank lines dropped (trailing newline from a copy). */
export function splitColumn(text: string): string[] {
  return text
    .replace(/\r/g, '')
    .split('\n')
    .map((l) => l.trim())
    .filter((l) => l !== '');
}

export interface CsvMapping {
  /** One object per data row: column name (as matched) → raw value. */
  rows: Record<string, string>[];
  /** Variables matched by header name. */
  variableColumns: string[];
  /** The header used as the output column, if any. */
  outputColumn: string | null;
  noiseColumn: string | null;
  reasonColumn: string | null;
}

const OUTPUT_HEADERS = ['output', 'y', 'yield', 'target', 'result', 'objective'];

/**
 * Parse a pasted CSV/TSV block with a header row. Columns whose header names
 * a variable map to that variable's actual input; an Output column (also
 * accepted: y, yield, target, result, objective, or the session's target
 * name) maps to the output; Noise and Reason map by name. Returns null when
 * the block has no header that names a variable or an output column.
 */
export function parseHeaderedPaste(
  text: string,
  variableNames: string[],
  extraOutputNames: string[] = [],
): CsvMapping | null {
  const parsed = Papa.parse<string[]>(text.replace(/\r/g, '').trim(), {
    delimiter: '', // auto-detect tab / comma / semicolon
    skipEmptyLines: true,
  });
  const data = parsed.data as string[][];
  if (data.length < 2) return null;
  const header = data[0].map((h) => String(h).trim());
  const lower = header.map((h) => h.toLowerCase());
  const varLower = new Map(variableNames.map((n) => [n.toLowerCase(), n]));
  const outputNames = new Set([...OUTPUT_HEADERS, ...extraOutputNames.map((n) => n.toLowerCase())]);

  const columnFor: (string | null)[] = lower.map(() => null);
  const variableColumns: string[] = [];
  let outputColumn: string | null = null;
  let noiseColumn: string | null = null;
  let reasonColumn: string | null = null;
  lower.forEach((h, i) => {
    const varName = varLower.get(h);
    if (varName) {
      columnFor[i] = varName;
      variableColumns.push(varName);
    } else if (!outputColumn && outputNames.has(h)) {
      columnFor[i] = 'Output';
      outputColumn = header[i];
    } else if (h === 'noise') {
      columnFor[i] = 'Noise';
      noiseColumn = header[i];
    } else if (h === 'reason') {
      columnFor[i] = 'Reason';
      reasonColumn = header[i];
    }
  });
  if (variableColumns.length === 0 && !outputColumn) return null;

  const rows = data.slice(1).map((cells) => {
    const row: Record<string, string> = {};
    cells.forEach((cell, i) => {
      const target = columnFor[i];
      if (target) row[target] = String(cell ?? '').trim();
    });
    return row;
  });
  return { rows, variableColumns, outputColumn, noiseColumn, reasonColumn };
}

export type EntryField =
  | { kind: 'output' }
  | { kind: 'noise' }
  | { kind: 'reason' }
  | { kind: 'actual'; name: string };

/**
 * Apply a pasted column (one value per line) to one field, from `startIndex`
 * down. Values beyond the last pending point are ignored and counted.
 */
export function fillColumnDown(
  entries: EntriesMap,
  suggestions: any[],
  startIndex: number,
  field: EntryField,
  values: string[],
  variables: VariableDetail[] = [],
): { entries: EntriesMap; filled: number; overflow: number } {
  const next = { ...entries };
  let filled = 0;
  values.forEach((value, offset) => {
    const s = suggestions[startIndex + offset];
    if (!s) return;
    const key = suggestionKey(s);
    const current = entryFor(next, s, variables);
    next[key] =
      field.kind === 'actual'
        ? { ...current, actual: { ...current.actual, [field.name]: value } }
        : { ...current, [field.kind]: value };
    filled += 1;
  });
  return { entries: next, filled, overflow: values.length - filled };
}

/** Apply a headered CSV block row by row from `startIndex` down. */
export function applyHeaderedRows(
  entries: EntriesMap,
  suggestions: any[],
  startIndex: number,
  mapping: CsvMapping,
  variables: VariableDetail[] = [],
): { entries: EntriesMap; filled: number; overflow: number } {
  const next = { ...entries };
  let filled = 0;
  mapping.rows.forEach((row, offset) => {
    const s = suggestions[startIndex + offset];
    if (!s) return;
    const key = suggestionKey(s);
    const current = entryFor(next, s, variables);
    const actual = { ...current.actual };
    for (const v of mapping.variableColumns) {
      if (v in actual && row[v] !== undefined && row[v] !== '') actual[v] = row[v];
    }
    next[key] = {
      actual,
      output: row.Output !== undefined ? row.Output : current.output,
      noise: row.Noise !== undefined ? row.Noise : current.noise,
      reason: row.Reason !== undefined && row.Reason !== '' ? row.Reason : current.reason,
    };
    filled += 1;
  });
  return { entries: next, filled, overflow: mapping.rows.length - filled };
}
