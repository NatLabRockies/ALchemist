# Add Point Dialog Redesign + Batch Points Bug Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the batch "all points identical" bug and redesign the Add Point dialog so suggested conditions are read-only while the user records the actual experimental conditions per variable.

**Architecture:** The dialog gains a `variables` prop for per-variable smart rounding. Each variable row shows a read-only suggested value plus an editable "Actual" input pre-filled with the rounded suggestion. The identical-points bug is fixed by adding a React `key` at both call sites so the dialog remounts fresh per point. Actual values are submitted as the experiment inputs; no backend changes.

**Tech Stack:** React 19 + TypeScript, Vite, TanStack Query. Tests via vitest + @testing-library/react (added in Task 1).

**Spec:** `docs/superpowers/specs/2026-07-28-add-point-dialog-redesign-design.md`

---

## File Structure

- `alchemist-web/package.json` — add vitest, @testing-library/react, jsdom devDeps + `test` script (modify).
- `alchemist-web/vitest.config.ts` — new vitest config with jsdom env (create).
- `alchemist-web/src/test/setup.ts` — testing-library jest-dom setup (create).
- `alchemist-web/src/lib/rounding.ts` — pure smart-rounding helper (create).
- `alchemist-web/src/lib/rounding.test.ts` — helper unit tests (create).
- `alchemist-web/src/components/AddPointDialog.tsx` — redesign: read-only suggested + editable actual, `variables` prop (modify).
- `alchemist-web/src/components/AddPointDialog.test.tsx` — component tests (create).
- `alchemist-web/src/features/experiments/ExperimentsPanel.tsx` — pass `variables`, add `key` (modify, ~lines 28, 195).
- `alchemist-web/src/components/PendingSuggestionsPanel.tsx` — pass `variables`, add `key` (modify, ~lines 6, 64).

**Interpreter/commands:** run all web commands from `alchemist-web/`. Node/npm used for the web app (Python env is unrelated here).

---

### Task 1: Set up vitest test infrastructure

**Files:**
- Modify: `alchemist-web/package.json`
- Create: `alchemist-web/vitest.config.ts`
- Create: `alchemist-web/src/test/setup.ts`

- [ ] **Step 1: Install test dependencies**

Run (from `alchemist-web/`):
```bash
npm install -D vitest@^2 jsdom @testing-library/react @testing-library/jest-dom @testing-library/user-event
```
Expected: packages added to `devDependencies`, no errors.

- [ ] **Step 2: Add test script to package.json**

In `alchemist-web/package.json`, change the `scripts` block to include a `test` entry:
```json
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "lint": "eslint .",
    "preview": "vite preview",
    "test": "vitest run",
    "test:watch": "vitest"
  },
```

- [ ] **Step 3: Create vitest config**

Create `alchemist-web/vitest.config.ts`:
```ts
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
  },
});
```

- [ ] **Step 4: Create test setup file**

Create `alchemist-web/src/test/setup.ts`:
```ts
import '@testing-library/jest-dom';
```

- [ ] **Step 5: Verify the runner works**

Run: `npm test`
Expected: vitest runs and reports "No test files found" (exit 0) — infrastructure is wired.

- [ ] **Step 6: Commit**

```bash
git add package.json package-lock.json vitest.config.ts src/test/setup.ts
git commit -m "test: add vitest + testing-library infrastructure to web app"
```

---

### Task 2: Smart-rounding helper (pure, TDD)

Rounds a suggested value for display/pre-fill based on the variable's type.

**Files:**
- Create: `alchemist-web/src/lib/rounding.ts`
- Create: `alchemist-web/src/lib/rounding.test.ts`

- [ ] **Step 1: Write the failing test**

Create `alchemist-web/src/lib/rounding.test.ts`:
```ts
import { describe, it, expect } from 'vitest';
import { roundSuggested, formatSuggested, REAL_DECIMALS } from './rounding';
import type { VariableDetail } from '../api/types';

const realVar = (name: string): VariableDetail => ({ name, type: 'real', bounds: [0, 1000] });
const intVar = (name: string): VariableDetail => ({ name, type: 'integer', bounds: [0, 10] });
const catVar = (name: string): VariableDetail =>
  ({ name, type: 'categorical', categories: ['A', 'B'] });

describe('REAL_DECIMALS', () => {
  it('is 3 (per spec)', () => {
    expect(REAL_DECIMALS).toBe(3);
  });
});

describe('roundSuggested', () => {
  it('rounds real variables to 3 places (numeric collapses trailing zero)', () => {
    // 901.4096982394 -> toFixed(3) = "901.410" -> Number(...) = 901.41
    expect(roundSuggested(901.4096982394, realVar('temp'))).toBe(901.41);
  });

  it('rounds integer variables to whole numbers', () => {
    expect(roundSuggested(4.9997, intVar('count'))).toBe(5);
    expect(roundSuggested(3.2, intVar('count'))).toBe(3);
  });

  it('passes categorical values through unchanged', () => {
    expect(roundSuggested('A', catVar('mode'))).toBe('A');
  });

  it('passes value through unchanged when no matching variable is provided', () => {
    expect(roundSuggested(1.23456, undefined)).toBe(1.23456);
  });

  it('leaves non-numeric values unchanged even for real vars', () => {
    expect(roundSuggested('n/a', realVar('temp'))).toBe('n/a');
  });
});

describe('formatSuggested', () => {
  it('preserves trailing zeros for real display strings', () => {
    expect(formatSuggested(901.4096982394, realVar('temp'))).toBe('901.410');
  });

  it('formats integers as whole-number strings', () => {
    expect(formatSuggested(4.9997, intVar('count'))).toBe('5');
  });

  it('formats categorical values as their string form', () => {
    expect(formatSuggested('A', catVar('mode'))).toBe('A');
  });
});
```

Key distinction the implementer must preserve: `roundSuggested` returns a **number** (so `901.410` collapses to `901.41` — this is the actual-input pre-fill value), while `formatSuggested` returns a **display string** that keeps the trailing zero (`"901.410"` — the read-only suggested display).

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/lib/rounding.test.ts`
Expected: FAIL — `roundSuggested` / `REAL_DECIMALS` not exported.

- [ ] **Step 3: Implement the helper**

Create `alchemist-web/src/lib/rounding.ts`:
```ts
import type { VariableDetail } from '../api/types';

/** Default decimal places for real (continuous) variables. */
export const REAL_DECIMALS = 3;

/**
 * Round a suggested value for display / actual-input pre-fill based on the
 * variable's type. Integers -> whole numbers, reals -> REAL_DECIMALS places,
 * categorical/discrete and non-numeric -> unchanged.
 */
export function roundSuggested(
  value: unknown,
  variable: VariableDetail | undefined,
): unknown {
  if (!variable) return value;
  if (typeof value !== 'number' || !Number.isFinite(value)) return value;

  switch (variable.type) {
    case 'integer':
      return Math.round(value);
    case 'real':
      return Number(value.toFixed(REAL_DECIMALS));
    default:
      // categorical, discrete (allowed_values) — pass through
      return value;
  }
}

/**
 * String form for display, preserving trailing zeros for real vars
 * (e.g. 901.410 rather than 901.41).
 */
export function formatSuggested(
  value: unknown,
  variable: VariableDetail | undefined,
): string {
  if (variable && typeof value === 'number' && Number.isFinite(value) && variable.type === 'real') {
    return value.toFixed(REAL_DECIMALS);
  }
  return String(roundSuggested(value, variable) ?? '');
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/lib/rounding.test.ts`
Expected: PASS (all cases).

- [ ] **Step 5: Commit**

```bash
git add src/lib/rounding.ts src/lib/rounding.test.ts
git commit -m "feat: add smart per-variable rounding helper for suggested values"
```

---

### Task 3: Redesign AddPointDialog (read-only suggested + editable actual)

**Files:**
- Modify: `alchemist-web/src/components/AddPointDialog.tsx`
- Create: `alchemist-web/src/components/AddPointDialog.test.tsx`

- [ ] **Step 1: Write the failing component test**

Create `alchemist-web/src/components/AddPointDialog.test.tsx`:
```tsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import AddPointDialog from './AddPointDialog';
import type { VariableDetail } from '../api/types';

const variables: VariableDetail[] = [
  { name: 'temp', type: 'real', bounds: [0, 2000] },
  { name: 'count', type: 'integer', bounds: [0, 10] },
];

const suggestion = { temp: 901.4096982394, count: 4.9997, _reason: 'qEI' };

function renderDialog(props: Partial<React.ComponentProps<typeof AddPointDialog>> = {}) {
  const onConfirm = vi.fn();
  render(
    <AddPointDialog
      suggestion={suggestion}
      variables={variables}
      index={0}
      total={5}
      onCancel={() => {}}
      onConfirm={onConfirm}
      {...props}
    />,
  );
  return { onConfirm };
}

describe('AddPointDialog', () => {
  it('shows the suggested value read-only and pre-fills the actual input rounded', () => {
    renderDialog();
    // read-only suggested display (real -> 3 decimals with trailing zero)
    expect(screen.getByText('901.410')).toBeInTheDocument();
    // actual input pre-filled with rounded value
    const tempActual = screen.getByLabelText('temp actual') as HTMLInputElement;
    expect(tempActual.value).toBe('901.41');
    const countActual = screen.getByLabelText('count actual') as HTMLInputElement;
    expect(countActual.value).toBe('5');
  });

  it('submits the actual (edited) values as inputs, not the suggestion', () => {
    const { onConfirm } = renderDialog();
    const tempActual = screen.getByLabelText('temp actual') as HTMLInputElement;
    fireEvent.change(tempActual, { target: { value: '900' } });
    fireEvent.change(screen.getByLabelText('Output'), { target: { value: '0.42' } });
    fireEvent.click(screen.getByText('Save & Close'));
    expect(onConfirm).toHaveBeenCalledTimes(1);
    const payload = onConfirm.mock.calls[0][0];
    expect(payload.inputs.temp).toBe('900');
    expect(payload.inputs.count).toBe('5');
    expect(payload.output).toBe(0.42);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/components/AddPointDialog.test.tsx`
Expected: FAIL — `variables` prop unsupported; no read-only suggested display; `getByLabelText('temp actual')` not found.

- [ ] **Step 3: Rewrite AddPointDialog**

Replace the full contents of `alchemist-web/src/components/AddPointDialog.tsx` with:
```tsx
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

  // Suggested input keys (exclude internal + output/metadata keys)
  const inputKeys = Object.keys(suggestion || {}).filter(
    (k) => !k.startsWith('_') && k !== 'Output' && k !== 'Noise' && k !== 'Iteration' && k !== 'Reason',
  );

  // Actual values pre-filled with the smart-rounded suggestion (as strings for inputs).
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

        {/* Variable rows: read-only suggested + editable actual */}
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/components/AddPointDialog.test.tsx`
Expected: PASS (both cases).

- [ ] **Step 5: Typecheck**

Run: `npx tsc -b`
Expected: no type errors.

- [ ] **Step 6: Commit**

```bash
git add src/components/AddPointDialog.tsx src/components/AddPointDialog.test.tsx
git commit -m "feat: read-only suggested + editable actual conditions in AddPointDialog"
```

---

### Task 4: Fix identical-points bug + pass variables in ExperimentsPanel

**Files:**
- Modify: `alchemist-web/src/features/experiments/ExperimentsPanel.tsx`

- [ ] **Step 1: Import and load variables**

At the top of `ExperimentsPanel.tsx`, add to the existing imports:
```tsx
import { useVariables } from '../../hooks/api/useVariables';
```
Inside the component body, after the existing `useExperiments` line (currently line 28), add:
```tsx
  const { data: variablesData } = useVariables(sessionId);
  const variables = variablesData?.variables ?? [];
```

- [ ] **Step 2: Pass `variables` and add `key` to the dialog**

In the modal block (currently line 195), change the `AddPointDialog` opening tag from:
```tsx
            <AddPointDialog
              suggestion={pendingSuggestions[currentIndex]}
              index={currentIndex}
              total={pendingSuggestions.length}
```
to:
```tsx
            <AddPointDialog
              key={currentIndex}
              variables={variables}
              suggestion={pendingSuggestions[currentIndex]}
              index={currentIndex}
              total={pendingSuggestions.length}
```
Leave the rest of the props (`onCancel`, `onConfirm`, `onPrev`, `onNext`) unchanged.

- [ ] **Step 3: Typecheck + lint**

Run: `npx tsc -b && npm run lint`
Expected: no type errors; lint passes.

- [ ] **Step 4: Commit**

```bash
git add src/features/experiments/ExperimentsPanel.tsx
git commit -m "fix: remount AddPointDialog per suggestion and pass variables (batch points bug)"
```

---

### Task 5: Fix identical-points bug + pass variables in PendingSuggestionsPanel

**Files:**
- Modify: `alchemist-web/src/components/PendingSuggestionsPanel.tsx`

- [ ] **Step 1: Accept and load variables**

Change the component signature (currently lines 6-7) from:
```tsx
export default function PendingSuggestionsPanel({ sessionId, pending, onRemove, onAdded }:
  { sessionId: string, pending: Array<any>, onRemove: (idx:number)=>void, onAdded?: (resp:any)=>void }) {
```
to:
```tsx
import { useVariables } from '../hooks/api/useVariables';
// (place this import at the top with the other imports, not inside the function)

export default function PendingSuggestionsPanel({ sessionId, pending, onRemove, onAdded }:
  { sessionId: string, pending: Array<any>, onRemove: (idx:number)=>void, onAdded?: (resp:any)=>void }) {
```
Then, immediately after the existing `useState` hooks (after line 10 `const [currentIndex, ...]`), add:
```tsx
  const { data: variablesData } = useVariables(sessionId);
  const variables = variablesData?.variables ?? [];
```
Note: the early `return` for empty `pending` (line 12) must stay AFTER these hooks to satisfy the rules-of-hooks; move the three `useState` lines and the `useVariables` line above the `if (!pending || pending.length === 0) return (...)` block if not already so. As written in the current file the three `useState` calls are already above the early return — insert `useVariables` in that same block, before the early return.

- [ ] **Step 2: Pass `variables` and add `key` to the dialog**

Change the `AddPointDialog` opening (currently line 64) from:
```tsx
        <AddPointDialog
          suggestion={currentSuggestion}
          index={currentIndex ?? 0}
          total={pending.length}
```
to:
```tsx
        <AddPointDialog
          key={currentIndex ?? 0}
          variables={variables}
          suggestion={currentSuggestion}
          index={currentIndex ?? 0}
          total={pending.length}
```

- [ ] **Step 3: Typecheck + lint**

Run: `npx tsc -b && npm run lint`
Expected: no type errors; lint passes (no rules-of-hooks violation).

- [ ] **Step 4: Commit**

```bash
git add src/components/PendingSuggestionsPanel.tsx
git commit -m "fix: remount AddPointDialog per suggestion and pass variables in PendingSuggestionsPanel"
```

---

### Task 6: Full verification + manual repro

**Files:** none (verification only).

- [ ] **Step 1: Run the full test suite**

Run (from `alchemist-web/`): `npm test`
Expected: all tests pass (rounding + dialog).

- [ ] **Step 2: Typecheck and lint the whole app**

Run: `npx tsc -b && npm run lint`
Expected: clean.

- [ ] **Step 3: Production build sanity**

Run: `npm run build`
Expected: build succeeds.

- [ ] **Step 4: Manual repro with Anna's session**

Start the app (`npm run dev` and the API per project README), load
`examples/Inductive RCC/alchemist_session_89cdcc82.json`, run a q=5 batch
acquisition, open "Add Point...", and click Next through all 5.
Expected:
- Each of the 5 points shows DISTINCT suggested values (bug fixed).
- Suggested values are read-only; "actual" boxes are editable and pre-filled rounded (e.g. a real var `901.4096982394` shows `901.410` suggested, `901.41` prefilled).
- Editing an actual box then Save & Close records the actual value in the experiments table.

- [ ] **Step 5: Commit any doc updates (if needed)**

If manual testing reveals no changes, no commit needed. Otherwise fix + commit.

---

## Self-Review Notes

- **Spec coverage:** read-only suggested (Task 3) ✓; editable actual pre-filled rounded (Task 3) ✓; smart per-variable rounding (Task 2) ✓; store actual as data point (Task 3 `payload.inputs` from `actual`) ✓; `key`-based remount fix at both call sites (Tasks 4, 5) ✓; Save & Close unchanged (Task 3 keeps footer behavior; call-site onConfirm untouched) ✓; regression test for identical points (Task 3 distinct-values + Task 6 manual q=5) ✓.
- **Out of scope confirmed absent:** no queue clear/delete/reorder, no session-load hardening, no auto-advance.
- **Type consistency:** `roundSuggested`/`formatSuggested`/`REAL_DECIMALS` used consistently across Tasks 2–3; `variables: VariableDetail[]` prop name consistent across Tasks 3–5; `actual` state and `payload.inputs` consistent in Task 3.
