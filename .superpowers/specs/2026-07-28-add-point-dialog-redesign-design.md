# Add Point Dialog Redesign + Batch Points Bug Fix

**Date:** 2026-07-28
**Context:** Bugs surfaced while helping Wilson & Anna deploy ALchemist on the inductive RCC project. Data files in `examples/Inductive RCC/` (gitignored).

## Problem

Two coupled issues in `alchemist-web/src/components/AddPointDialog.tsx`:

1. **Identical-batch-points bug.** When a batch acquisition suggests N points and the user steps through them in the Add Point dialog, every point shows the **first** point's field values. Root cause: form state is initialized via `useState(initializer)`, which only runs on mount. Prev/Next swaps the `suggestion` prop, but there is no React `key` and no resync effect, so React reuses the mounted instance and the editable fields keep their original values. Only the "N of M" header (read from props directly) updates.

2. **Free-editing full-precision suggestions.** Suggested values (e.g. `901.4096982394`) render as raw editable text. This conflates "what the model suggested" with "what the experiment actually ran," and real hardware cannot hit arbitrary float precision — conditions get adjusted at the bench.

## Solution

Redesign the dialog so each variable row shows a **read-only suggested value** alongside a separate **editable "Actual" box** pre-filled with the smart-rounded suggestion. Fix the stale-state bug by remounting the dialog per point via a React `key`.

### Component: `AddPointDialog.tsx`

- **New prop:** `variables: VariableDetail[]` — required for per-variable rounding.
- **Per-variable row** (replaces single input at current lines 111–121):
  - Read-only display of the suggested value, smart-rounded, with an affordance (tooltip / small "raw: …") to reveal full precision.
  - Editable **"Actual"** input pre-filled with the smart-rounded suggested value.
- **Smart rounding helper** (in dialog or `lib/utils.ts`):
  - `integer` → `Math.round`, whole number.
  - `categorical` / `discrete` (allowed_values) → exact, no rounding.
  - `real` → fixed default of 3 decimals (schema has no per-variable precision field).
- **Submit:** `payload.inputs` = the **actual** values (recorded data point). Output/Noise/Reason unchanged. Output field keeps `autoFocus`.

### Call sites

Both must pass `variables` and add `key` to force a fresh mount per point:

- **`ExperimentsPanel.tsx`** (dialog ~lines 195–252): `<AddPointDialog key={currentIndex} variables={variables} ... />`. Prev/Next drive `currentIndex`; `key` change remounts with fresh state. Fixes the bug. Verify/add `useVariables(sessionId)` if the panel doesn't already have the list. "Save & Close" behavior preserved.
- **`PendingSuggestionsPanel.tsx`** (dialog ~lines 63–81): same `key` + `variables` treatment.

### Why `key` over `useEffect`

`key={currentIndex}` forces unmount/remount, discarding all internal `useState` — the simplest guarantee no field carries over, and robust against future field additions.

## What does NOT change

- Backend: nothing. Actual inputs flow through the existing `add_experiment` path.
- Storage: actual conditions stored as the data point; suggested values are display-only.
- Navigation: "Save & Close" stays as-is.

## Testing

- **Unit (dialog):** render a batch of 3 distinct suggestions; assert each `currentIndex` shows its own suggested + pre-filled actual values (regression test for the identical-points bug).
- **Rounding:** integer `4.9997` → `5`; real `901.4096982394` → `901.410`; categorical passes through unchanged.
- **Submit:** edit an actual box → confirm → payload `inputs` reflects the edited actual, not the suggestion.
- **Manual repro:** load `examples/Inductive RCC/alchemist_session_89cdcc82.json`, generate q=5 batch, step through — confirm 5 distinct points.

## Out of scope (separate specs)

- Queue clear/delete/reorder UI (backend `/experiments/queue/*` API already exists, unused by web UI).
- Session-load silent per-row drop hardening.
- Auto-advance navigation after save.
