/**
 * Hold list for the Suggest panel. Any controllable variable can be locked at
 * a value for the next suggestion; observed variables need their measured
 * value. Everything starts unlocked; "Hold current setup" locks every hard and
 * very-hard variable at the last recorded run's values.
 */
import type { VariableDetail } from '../../api/types';

export type HoldState = Record<string, { locked: boolean; value: string }>;

const isRestricted = (v: VariableDetail) => v.role !== 'observed' && (v.changes === 'hard' || v.changes === 'very_hard');

function typed(raw: string, v: VariableDetail | undefined): number | string {
  if (v?.type === 'categorical') return raw;
  const n = Number(raw.trim());
  return raw.trim() !== '' && Number.isFinite(n) ? n : raw;
}

export function holdCurrentSetup(state: HoldState, variables: VariableDetail[], lastRow: Record<string, any> | null): HoldState {
  if (!lastRow) return state;
  const next = { ...state };
  for (const v of variables.filter(isRestricted)) {
    if (lastRow[v.name] !== undefined) next[v.name] = { locked: true, value: String(lastRow[v.name]) };
  }
  return next;
}

export function holdPayload(state: HoldState, observedValues: Record<string, string>, variables: VariableDetail[]) {
  const byName = new Map(variables.map((v) => [v.name, v]));
  const hold: Record<string, number | string> = {};
  for (const [name, s] of Object.entries(state)) {
    if (s.locked && s.value.trim() !== '' && byName.has(name) && byName.get(name)?.role !== 'observed') hold[name] = typed(s.value, byName.get(name));
  }
  const observedVars = variables.filter((v) => v.role === 'observed');
  const observed: Record<string, number | string> = {};
  const missingObserved: string[] = [];
  for (const v of observedVars) {
    const raw = observedValues[v.name] ?? '';
    if (raw.trim() === '') missingObserved.push(v.name);
    else observed[v.name] = typed(raw, v);
  }
  return {
    ...(Object.keys(hold).length ? { hold } : {}),
    ...(observedVars.length ? { observed } : {}),
    missingObserved,
  };
}

interface Props {
  variables: VariableDetail[];
  state: HoldState;
  observedValues: Record<string, string>;
  lastRow: Record<string, any> | null;
  onChange: (next: HoldState) => void;
  onObservedChange: (next: Record<string, string>) => void;
}

export function HoldList({ variables, state, observedValues, lastRow, onChange, onObservedChange }: Props) {
  const controllable = variables.filter((v) => v.role !== 'observed');
  const observed = variables.filter((v) => v.role === 'observed');
  const set = (name: string, patch: Partial<{ locked: boolean; value: string }>) =>
    onChange({ ...state, [name]: { ...(state[name] ?? { locked: false, value: '' }), ...patch } });

  return (
    <div className="space-y-2 text-xs">
      <div className="flex items-center justify-between">
        <span className="font-medium">Hold</span>
        <button type="button" onClick={() => onChange(holdCurrentSetup(state, variables, lastRow))}
          disabled={!lastRow || !variables.some(isRestricted)}
          className="px-2 py-1 border rounded hover:bg-accent disabled:opacity-50">
          Hold current setup
        </button>
      </div>
      {controllable.map((v) => {
        const s = state[v.name] ?? { locked: false, value: '' };
        return (
          <div key={v.name} className="grid grid-cols-[auto_1fr_1fr] items-center gap-2">
            <input type="checkbox" aria-label={`hold ${v.name}`} checked={s.locked}
              onChange={(e) => set(v.name, { locked: e.target.checked })} />
            <span>{v.name}{isRestricted(v) ? ' (hard)' : ''}</span>
            {v.type === 'categorical' ? (
              <select aria-label={`${v.name} held value`} value={s.value} disabled={!s.locked}
                onChange={(e) => set(v.name, { value: e.target.value })}
                className="px-2 py-1 border border-input rounded bg-background">
                <option value="">—</option>
                {(v.categories ?? []).map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            ) : (
              <input aria-label={`${v.name} held value`} value={s.value} disabled={!s.locked}
                onChange={(e) => set(v.name, { value: e.target.value })}
                className="px-2 py-1 border border-input rounded bg-background" />
            )}
          </div>
        );
      })}
      {observed.length > 0 && <p className="font-medium pt-1">Measured now</p>}
      {observed.map((v) => (
        <div key={v.name} className="grid grid-cols-2 items-center gap-2">
          <span>{v.name}{v.unit ? ` (${v.unit})` : ''} *</span>
          {v.type === 'categorical' ? (
            <select aria-label={`${v.name} measured value`} value={observedValues[v.name] ?? ''}
              onChange={(e) => onObservedChange({ ...observedValues, [v.name]: e.target.value })}
              className="px-2 py-1 border border-input rounded bg-background">
              <option value="">—</option>
              {(v.categories ?? []).map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          ) : (
            <input aria-label={`${v.name} measured value`} value={observedValues[v.name] ?? ''}
              onChange={(e) => onObservedChange({ ...observedValues, [v.name]: e.target.value })}
              className="px-2 py-1 border border-input rounded bg-background" />
          )}
        </div>
      ))}
    </div>
  );
}
