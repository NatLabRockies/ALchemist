/**
 * AddPointDialog - record results for every pending (staged) point.
 *
 * One dialog for the whole batch: it is not rebuilt per point. Entries live
 * in the parent's map keyed by a stable suggestion id (see addPointEntries),
 * so Previous/Next keep what was typed, and "Save N & close" sends every
 * point that has an output in ONE request with at most one retrain. Points
 * without an output stay pending.
 *
 * Two views over the same entries: "One at a time" (suggested vs actual per
 * variable) and "Table" (all points; pasting a column of numbers into a cell
 * fills down, a CSV block with headers maps by column name).
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import type { ClipboardEvent } from 'react';
import { X } from 'lucide-react';
import type { VariableDetail } from '../api/types';
import { formatSuggested } from '../lib/rounding';
import {
  applyHeaderedRows,
  defaultReason,
  entryFor,
  entryInputKeys,
  entryProblems,
  fillColumnDown,
  hasOutput,
  hasSuggestedValue,
  isColumnPaste,
  parseHeaderedPaste,
  planBatch,
  setupBannerFor,
  splitColumn,
  suggestionKey,
} from '../features/experiments/addPointEntries';
import type { BatchPlan, EntriesMap, EntryField, PointEntry } from '../features/experiments/addPointEntries';

export type AddPointMode = 'single' | 'table';

type Props = {
  suggestions: any[];
  variables?: VariableDetail[];
  entries: EntriesMap;
  onEntriesChange: (update: (prev: EntriesMap) => EntriesMap) => void;
  onCancel: () => void;
  /** Persist the plan's points; reject to keep the dialog open with the error. */
  onSave: (plan: BatchPlan, options: { retrain: boolean }) => Promise<void>;
  initialIndex?: number;
  initialMode?: AddPointMode;
  /** Extra header names accepted as the output column when pasting a CSV. */
  outputColumnNames?: string[];
};

const inputClass =
  'w-full px-3 py-2 text-sm rounded-md border border-border bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50';
const cellInputClass =
  'w-full min-w-[5rem] px-2 py-1 text-xs rounded border border-border bg-background text-foreground tabular-nums focus:outline-none focus:ring-2 focus:ring-primary/50';

export default function AddPointDialog({
  suggestions,
  variables = [],
  entries,
  onEntriesChange,
  onCancel,
  onSave,
  initialIndex = 0,
  initialMode = 'single',
  outputColumnNames = [],
}: Props) {
  const [rawIndex, setRawIndex] = useState(initialIndex);
  const [mode, setMode] = useState<AddPointMode>(initialMode);
  const [retrain, setRetrain] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pasteNote, setPasteNote] = useState<string | null>(null);
  const bodyRef = useRef<HTMLDivElement>(null);
  const outputRef = useRef<HTMLInputElement>(null);

  const total = suggestions.length;
  const index = Math.max(0, Math.min(rawIndex, total - 1));
  const suggestion = suggestions[index];
  const varByName = useMemo(() => new Map(variables.map((v) => [v.name, v])), [variables]);

  const plan = useMemo(() => planBatch(suggestions, entries, variables), [suggestions, entries, variables]);
  const nEntered = suggestions.filter((s) => hasOutput(entryFor(entries, s, variables))).length;
  const nToSave = plan.payload.length;
  const hasBanner = setupBannerFor(suggestions, index, variables) !== null;

  // On open and on every move to another point: focus Output without letting
  // the browser scroll it into view, and when the point has a setup banner
  // bring the body back to the top so the banner is what the user sees.
  useEffect(() => {
    if (mode !== 'single') return;
    outputRef.current?.focus({ preventScroll: true });
    if (hasBanner && bodyRef.current) bodyRef.current.scrollTop = 0;
  }, [index, mode, hasBanner]);

  // Union of variable columns across points, in suggestion order.
  const columns = useMemo(() => {
    const seen: string[] = [];
    for (const s of suggestions) for (const k of entryInputKeys(s, variables)) if (!seen.includes(k)) seen.push(k);
    return seen;
  }, [suggestions, variables]);

  function updateEntry(s: any, change: (e: PointEntry) => PointEntry) {
    const key = suggestionKey(s);
    onEntriesChange((prev) => ({ ...prev, [key]: change(entryFor(prev, s, variables)) }));
    setError(null);
  }

  function setField(s: any, field: EntryField, value: string) {
    updateEntry(s, (e) =>
      field.kind === 'actual'
        ? { ...e, actual: { ...e.actual, [field.name]: value } }
        : { ...e, [field.kind]: value },
    );
  }

  function handleTablePaste(event: ClipboardEvent<HTMLElement>) {
    const target = event.target as HTMLElement;
    const row = Number(target.getAttribute?.('data-row'));
    const fieldAttr = target.getAttribute?.('data-field');
    if (!Number.isInteger(row) || !fieldAttr) return;
    const text = event.clipboardData.getData('text');
    if (!text || !/\n/.test(text.trim())) return; // single value: normal paste

    const mapping = parseHeaderedPaste(text, columns, outputColumnNames);
    if (mapping) {
      event.preventDefault();
      const r = applyHeaderedRows(entries, suggestions, row, mapping, variables);
      onEntriesChange(() => r.entries);
      const mapped = [...mapping.variableColumns, mapping.outputColumn, mapping.noiseColumn, mapping.reasonColumn]
        .filter(Boolean)
        .join(', ');
      setPasteNote(
        `Pasted ${r.filled} row${r.filled === 1 ? '' : 's'} by column name (${mapped})` +
          (r.overflow > 0 ? `; ${r.overflow} extra row${r.overflow === 1 ? '' : 's'} ignored` : ''),
      );
    } else if (isColumnPaste(text)) {
      event.preventDefault();
      const field: EntryField = fieldAttr.startsWith('actual:')
        ? { kind: 'actual', name: fieldAttr.slice('actual:'.length) }
        : ({ kind: fieldAttr } as EntryField);
      const r = fillColumnDown(entries, suggestions, row, field, splitColumn(text), variables);
      onEntriesChange(() => r.entries);
      const label = field.kind === 'actual' ? field.name : field.kind;
      setPasteNote(
        `Filled ${r.filled} ${label} value${r.filled === 1 ? '' : 's'} from point ${row + 1} down` +
          (r.overflow > 0 ? `; ${r.overflow} extra value${r.overflow === 1 ? '' : 's'} ignored` : ''),
      );
    } else {
      return;
    }
    setError(null);
  }

  async function save() {
    if (nToSave === 0 || plan.invalid.length > 0 || saving) return;
    setSaving(true);
    setError(null);
    try {
      await onSave(plan, { retrain });
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setSaving(false);
    }
  }

  if (!suggestion) return null;
  const entry = entryFor(entries, suggestion, variables);
  const iterationValue = suggestion?.Iteration;
  const singleInputKeys = entryInputKeys(suggestion, variables);
  const banner = setupBannerFor(suggestions, index, variables);

  return (
    <div
      role="dialog"
      aria-label="Add experimental results"
      className={`bg-card border border-border rounded-lg shadow-lg w-full ${
        mode === 'table' ? 'max-w-5xl' : 'max-w-2xl'
      } max-h-[85vh] flex flex-col`}
    >
      {/* Header */}
      <div className="border-b border-border p-4 flex items-center gap-3">
        <div className="flex-1 min-w-0">
          <h3 className="text-lg font-semibold">
            {total > 1 ? 'Add Experimental Results' : 'Add Experimental Result'}
          </h3>
          <p className="text-sm text-muted-foreground mt-0.5" data-testid="add-point-progress">
            {mode === 'single' ? `Point ${index + 1} of ${total}, ` : `${total} pending point${total === 1 ? '' : 's'}, `}
            {nEntered} entered
          </p>
        </div>

        <div className="flex rounded-md border border-border overflow-hidden text-sm" role="group" aria-label="Entry view">
          <button
            type="button"
            onClick={() => setMode('single')}
            aria-pressed={mode === 'single'}
            className={`px-3 py-1.5 ${mode === 'single' ? 'bg-primary text-primary-foreground' : 'hover:bg-accent'}`}
          >
            One at a time
          </button>
          <button
            type="button"
            onClick={() => setMode('table')}
            aria-pressed={mode === 'table'}
            className={`px-3 py-1.5 border-l border-border ${mode === 'table' ? 'bg-primary text-primary-foreground' : 'hover:bg-accent'}`}
          >
            Table
          </button>
        </div>

        <button onClick={onCancel} className="p-1.5 rounded hover:bg-accent" title="Close" aria-label="Close">
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Body */}
      <div ref={bodyRef} data-testid="add-point-body" className="p-6 space-y-4 overflow-auto flex-1">
        {mode === 'single' ? (
          <>
            {banner && (
              <div
                role="note"
                aria-label="Setup change"
                className="rounded-md border border-amber-500/50 bg-amber-500/10 p-3 text-sm"
              >
                <p className="font-medium">{banner.title}: change the setup before this point</p>
                <ul className="mt-1 list-disc pl-5">
                  {banner.settings.map((s) => (
                    <li key={s.name}>
                      {s.name} = {s.value}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            <div className="flex items-center justify-between gap-3">
              <p className="text-sm text-green-600 dark:text-green-500">{defaultReason(suggestion)}</p>
              {total > 1 && (
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => setRawIndex(index - 1)}
                    disabled={index === 0}
                    className="px-3 py-1.5 text-sm rounded border border-border hover:bg-accent disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    ← Previous
                  </button>
                  <button
                    type="button"
                    onClick={() => setRawIndex(index + 1)}
                    disabled={index >= total - 1}
                    className="px-3 py-1.5 text-sm rounded border border-border hover:bg-accent disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Next →
                  </button>
                </div>
              )}
            </div>

            <p className="text-xs text-muted-foreground">
              Suggested conditions are shown for reference. Enter the <strong>actual</strong> conditions used.
            </p>

            <div className="space-y-3">
              {singleInputKeys.map((k) => {
                const v = varByName.get(k);
                const inSuggestion = hasSuggestedValue(suggestion, k);
                const measured = !(k in suggestion);
                const rawSuggested = suggestion[k];
                return (
                  <div key={k} className="grid grid-cols-[1fr_1fr] gap-4 items-end">
                    <div className="space-y-1">
                      <label className="block text-sm font-medium text-muted-foreground">
                        {k}
                        {v?.unit ? ` (${v.unit})` : ''} — suggested
                      </label>
                      <div
                        className="px-3 py-2 text-sm rounded-md border border-border bg-muted text-foreground"
                        title={`raw: ${String(rawSuggested)}`}
                      >
                        {inSuggestion ? formatSuggested(rawSuggested, v) : measured ? 'measured' : '—'}
                      </div>
                    </div>
                    <div className="space-y-1">
                      <label htmlFor={`actual-${k}`} className="block text-sm font-medium text-muted-foreground">
                        actual
                      </label>
                      <input
                        id={`actual-${k}`}
                        aria-label={`${k} actual`}
                        type="text"
                        value={entry.actual[k] ?? ''}
                        onChange={(e) => setField(suggestion, { kind: 'actual', name: k }, e.target.value)}
                        className={inputClass}
                      />
                    </div>
                  </div>
                );
              })}
            </div>

            <div className="grid grid-cols-2 gap-4 pt-2 border-t border-border">
              <div className="space-y-1">
                <label htmlFor="add-point-output" className="block text-sm font-medium text-muted-foreground">
                  Output
                </label>
                <input
                  id="add-point-output"
                  aria-label="Output"
                  type="text"
                  inputMode="decimal"
                  value={entry.output}
                  onChange={(e) => setField(suggestion, { kind: 'output' }, e.target.value)}
                  onKeyDown={(e) => {
                    // Enter moves on to the next point, so a run of results
                    // can be typed without reaching for the mouse.
                    if (e.key === 'Enter' && index < total - 1) {
                      e.preventDefault();
                      setRawIndex(index + 1);
                    }
                  }}
                  ref={outputRef}
                  className={inputClass}
                />
              </div>
              <div className="space-y-1">
                <label htmlFor="add-point-noise" className="block text-sm font-medium text-muted-foreground">
                  Noise (optional)
                </label>
                <input
                  id="add-point-noise"
                  aria-label="Noise"
                  type="text"
                  inputMode="decimal"
                  value={entry.noise}
                  onChange={(e) => setField(suggestion, { kind: 'noise' }, e.target.value)}
                  placeholder="1e-6"
                  className={inputClass}
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 pt-2 border-t border-border">
              <div className="space-y-1">
                <label className="block text-sm font-medium text-muted-foreground">Iteration</label>
                <div className="px-3 py-2 text-sm rounded-md border border-border bg-muted text-foreground">
                  {typeof iterationValue === 'number' ? iterationValue : 'Auto'}
                </div>
              </div>
              <div className="space-y-1">
                <label htmlFor="add-point-reason" className="block text-sm font-medium text-muted-foreground">
                  Reason
                </label>
                <input
                  id="add-point-reason"
                  aria-label="Reason"
                  type="text"
                  value={entry.reason}
                  onChange={(e) => setField(suggestion, { kind: 'reason' }, e.target.value)}
                  className={inputClass}
                />
              </div>
            </div>
          </>
        ) : (
          <>
            <p className="text-xs text-muted-foreground">
              Paste a column of numbers into any cell to fill down from that row (one value per line, e.g. the
              simulator's "Copy outputs"). A CSV block with a header row maps by column name.
            </p>
            <div className="border rounded overflow-auto">
              <table className="w-full text-xs" onPaste={handleTablePaste}>
                <thead className="bg-muted/50 border-b sticky top-0">
                  <tr>
                    <th className="px-2 py-1.5 text-left font-medium">#</th>
                    {columns.map((k) => {
                      const v = varByName.get(k);
                      return (
                        <th key={k} className="px-2 py-1.5 text-left font-medium">
                          {k}
                          {v?.unit ? ` (${v.unit})` : ''}
                          <span className="block font-normal text-muted-foreground">suggested / actual</span>
                        </th>
                      );
                    })}
                    <th className="px-2 py-1.5 text-left font-medium">Output</th>
                    <th className="px-2 py-1.5 text-left font-medium">Reason</th>
                  </tr>
                </thead>
                <tbody className="divide-y">
                  {suggestions.map((s, row) => {
                    const e = entryFor(entries, s, variables);
                    const bad = e.output.trim() !== '' && entryProblems(e, variables).length > 0;
                    return (
                      <tr
                        key={suggestionKey(s)}
                        className={[
                          bad ? 'bg-destructive/10' : '',
                          setupBannerFor(suggestions, row, variables) ? 'border-t-2 border-amber-500' : '',
                        ]
                          .filter(Boolean)
                          .join(' ') || undefined}
                      >
                        <td className="px-2 py-1 text-muted-foreground tabular-nums">{row + 1}</td>
                        {columns.map((k) => (
                          <td key={k} className="px-2 py-1 align-top">
                            <div className="text-muted-foreground tabular-nums mb-0.5" title={`raw: ${String(s[k])}`}>
                              {hasSuggestedValue(s, k) ? formatSuggested(s[k], varByName.get(k)) : '—'}
                            </div>
                            {k in e.actual && (
                              <input
                                aria-label={`Point ${row + 1} ${k} actual`}
                                data-row={row}
                                data-field={`actual:${k}`}
                                type="text"
                                value={e.actual[k]}
                                onChange={(ev) => setField(s, { kind: 'actual', name: k }, ev.target.value)}
                                className={cellInputClass}
                              />
                            )}
                          </td>
                        ))}
                        <td className="px-2 py-1 align-bottom">
                          <input
                            aria-label={`Point ${row + 1} output`}
                            data-row={row}
                            data-field="output"
                            type="text"
                            inputMode="decimal"
                            value={e.output}
                            onChange={(ev) => setField(s, { kind: 'output' }, ev.target.value)}
                            className={cellInputClass}
                          />
                        </td>
                        <td className="px-2 py-1 align-bottom">
                          <input
                            aria-label={`Point ${row + 1} reason`}
                            data-row={row}
                            data-field="reason"
                            type="text"
                            value={e.reason}
                            onChange={(ev) => setField(s, { kind: 'reason' }, ev.target.value)}
                            className={`${cellInputClass} min-w-[8rem]`}
                          />
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            {pasteNote && (
              <p className="text-xs text-muted-foreground" role="status">
                {pasteNote}
              </p>
            )}
          </>
        )}
      </div>

      {/* Footer */}
      <div className="border-t border-border p-4 space-y-3">
        <div className="text-xs space-y-1" data-testid="add-point-summary">
          <p>
            <span className="font-medium">{nToSave}</span> point{nToSave === 1 ? '' : 's'} will be saved
            {plan.pendingPositions.length > 0 && (
              <>
                {'; '}
                <span className="text-amber-600 dark:text-amber-400">
                  {plan.pendingPositions.length} without an output stay{plan.pendingPositions.length === 1 ? 's' : ''}{' '}
                  pending (point{plan.pendingPositions.length === 1 ? '' : 's'} {plan.pendingPositions.join(', ')})
                </span>
              </>
            )}
            .
          </p>
          {plan.invalid.map(({ position, problems }) => (
            <p key={position} className="text-destructive">
              Point {position}: {problems.join('; ')}
            </p>
          ))}
          {error && (
            <p className="text-destructive font-medium" role="alert">
              {error} Your entries are kept; nothing was lost.
            </p>
          )}
        </div>

        <div className="flex items-center justify-between gap-3">
          <label className="flex items-center gap-2 text-sm cursor-pointer" title="One retrain after all points are saved">
            <input
              type="checkbox"
              checked={retrain}
              onChange={(e) => setRetrain(e.target.checked)}
              className="w-4 h-4 rounded border-border text-primary focus:ring-2 focus:ring-primary/50"
            />
            <span>Retrain model (once, after saving)</span>
          </label>
          <div className="flex gap-3">
            <button onClick={onCancel} className="px-4 py-2 text-sm rounded-md border border-border hover:bg-accent">
              Cancel
            </button>
            <button
              onClick={save}
              disabled={nToSave === 0 || plan.invalid.length > 0 || saving}
              className="px-4 py-2 text-sm rounded-md bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {saving ? 'Saving…' : `Save ${nToSave} & close`}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
