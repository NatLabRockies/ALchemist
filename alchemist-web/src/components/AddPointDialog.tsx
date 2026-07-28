/**
 * AddPointDialog - Modal dialog for recording experimental results.
 * Suggested conditions are shown read-only; the user records the ACTUAL
 * conditions per variable (pre-filled with the smart-rounded suggestion).
 */
import { useState } from 'react';
import { X } from 'lucide-react';
import type { VariableDetail } from '../api/types';
import { roundSuggested, formatSuggested } from '../lib/rounding';

type Props = {
  suggestion: any;
  variables?: VariableDetail[];
  index?: number;
  total?: number;
  iteration?: number;
  onCancel: () => void;
  onConfirm: (payload: any, options: { saveToFile: boolean; retrain: boolean }) => void;
  onPrev?: () => void;
  onNext?: () => void;
};

export default function AddPointDialog({
  suggestion,
  variables = [],
  index = 0,
  total = 1,
  iteration,
  onCancel,
  onConfirm,
  onPrev,
  onNext,
}: Props) {
  const varByName = new Map(variables.map((v) => [v.name, v]));

  const inputKeys = Object.keys(suggestion || {}).filter(
    (k) => !k.startsWith('_') && k !== 'Output' && k !== 'Noise' && k !== 'Iteration' && k !== 'Reason',
  );

  const initialActual: Record<string, string> = {};
  inputKeys.forEach((k) => {
    const rounded = roundSuggested(suggestion[k], varByName.get(k));
    initialActual[k] = String(rounded ?? '');
  });

  const [actual, setActual] = useState<Record<string, string>>(initialActual);
  const [output, setOutput] = useState<string>(suggestion?.Output?.toString() ?? '');
  const [noise, setNoise] = useState<string>(suggestion?.Noise?.toString() ?? '');

  const defaultReason = suggestion?._reason || suggestion?.Reason || 'Acquisition';
  const [reason, setReason] = useState<string>(defaultReason);

  const [saveToFile, setSaveToFile] = useState(true);
  const [retrain, setRetrain] = useState(true);

  const displayIteration = suggestion?.Iteration ?? iteration ?? 'N/A';

  function changeActual(field: string, val: string) {
    setActual((prev) => ({ ...prev, [field]: val }));
  }

  function confirm() {
    const payload: any = { inputs: { ...actual } };
    if (output !== '') payload.output = Number(output);
    if (noise !== '') payload.noise = Number(noise);
    if (reason) payload.reason = reason;
    onConfirm(payload, { saveToFile, retrain });
  }

  return (
    <div className="bg-card border border-border rounded-lg shadow-lg w-full max-w-2xl max-h-[85vh] overflow-auto">
      {/* Header */}
      <div className="border-b border-border p-4 flex items-center justify-between">
        <div className="flex-1">
          <h3 className="text-lg font-semibold">
            {total > 1 ? `Pending Suggestion ${index + 1} of ${total}` : 'Add Experimental Result'}
          </h3>
          {total > 1 && (
            <p className="text-sm text-green-600 dark:text-green-500 mt-1">{defaultReason}</p>
          )}
        </div>

        {total > 1 && (
          <div className="flex gap-2 ml-4">
            <button
              onClick={onPrev}
              disabled={!onPrev}
              className="px-3 py-1.5 text-sm rounded border border-border hover:bg-accent disabled:opacity-50 disabled:cursor-not-allowed"
            >
              ← Previous
            </button>
            <button
              onClick={onNext}
              disabled={!onNext}
              className="px-3 py-1.5 text-sm rounded border border-border hover:bg-accent disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Next →
            </button>
          </div>
        )}

        <button onClick={onCancel} className="ml-2 p-1.5 rounded hover:bg-accent" title="Close">
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Form content */}
      <div className="p-6 space-y-4">
        <p className="text-xs text-muted-foreground">
          Suggested conditions are shown for reference. Enter the <strong>actual</strong> conditions used.
        </p>

        <div className="space-y-3">
          {inputKeys.map((k) => {
            const v = varByName.get(k);
            const rawSuggested = suggestion[k];
            return (
              <div key={k} className="grid grid-cols-[1fr_1fr] gap-4 items-end">
                <div className="space-y-1">
                  <label className="block text-sm font-medium text-muted-foreground">
                    {k}{v?.unit ? ` (${v.unit})` : ''} — suggested
                  </label>
                  <div
                    className="px-3 py-2 text-sm rounded-md border border-border bg-muted text-foreground"
                    title={`raw: ${String(rawSuggested)}`}
                  >
                    {formatSuggested(rawSuggested, v)}
                  </div>
                </div>
                <div className="space-y-1">
                  <label
                    htmlFor={`actual-${k}`}
                    className="block text-sm font-medium text-muted-foreground"
                  >
                    actual
                  </label>
                  <input
                    id={`actual-${k}`}
                    aria-label={`${k} actual`}
                    type="text"
                    value={actual[k] ?? ''}
                    onChange={(e) => changeActual(k, e.target.value)}
                    className="w-full px-3 py-2 text-sm rounded-md border border-border bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
                  />
                </div>
              </div>
            );
          })}
        </div>

        {/* Output + Noise */}
        <div className="grid grid-cols-2 gap-4 pt-2 border-t border-border">
          <div className="space-y-1">
            <label htmlFor="add-point-output" className="block text-sm font-medium text-muted-foreground">
              Output
            </label>
            <input
              id="add-point-output"
              aria-label="Output"
              type="number"
              step="any"
              value={output}
              onChange={(e) => setOutput(e.target.value)}
              autoFocus
              className="w-full px-3 py-2 text-sm rounded-md border border-border bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
          </div>
          <div className="space-y-1">
            <label htmlFor="add-point-noise" className="block text-sm font-medium text-muted-foreground">
              Noise (optional)
            </label>
            <input
              id="add-point-noise"
              aria-label="Noise"
              type="number"
              step="any"
              value={noise}
              onChange={(e) => setNoise(e.target.value)}
              placeholder="1e-6"
              className="w-full px-3 py-2 text-sm rounded-md border border-border bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
          </div>
        </div>

        {/* Iteration (read-only) + Reason */}
        <div className="grid grid-cols-2 gap-4 pt-2 border-t border-border">
          <div className="space-y-1">
            <label className="block text-sm font-medium text-muted-foreground">Iteration</label>
            <div className="px-3 py-2 text-sm rounded-md border border-border bg-muted text-foreground">
              {displayIteration}
            </div>
          </div>
          <div className="space-y-1">
            <label htmlFor="add-point-reason" className="block text-sm font-medium text-muted-foreground">
              Reason
            </label>
            <input
              id="add-point-reason"
              type="text"
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              className="w-full px-3 py-2 text-sm rounded-md border border-border bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
          </div>
        </div>

        {/* Options */}
        <div className="flex items-center gap-6 pt-4 border-t border-border">
          <label className="flex items-center gap-2 text-sm cursor-pointer">
            <input
              type="checkbox"
              checked={saveToFile}
              onChange={(e) => setSaveToFile(e.target.checked)}
              className="w-4 h-4 rounded border-border text-primary focus:ring-2 focus:ring-primary/50"
            />
            <span>Save to file</span>
          </label>
          <label className="flex items-center gap-2 text-sm cursor-pointer">
            <input
              type="checkbox"
              checked={retrain}
              onChange={(e) => setRetrain(e.target.checked)}
              className="w-4 h-4 rounded border-border text-primary focus:ring-2 focus:ring-primary/50"
            />
            <span>Retrain model</span>
          </label>
        </div>
      </div>

      {/* Footer */}
      <div className="border-t border-border p-4 flex justify-end gap-3">
        <button
          onClick={onCancel}
          className="px-4 py-2 text-sm rounded-md border border-border hover:bg-accent"
        >
          Cancel
        </button>
        <button
          onClick={confirm}
          className="px-4 py-2 text-sm rounded-md bg-primary text-primary-foreground hover:bg-primary/90"
        >
          Save & Close
        </button>
      </div>
    </div>
  );
}
