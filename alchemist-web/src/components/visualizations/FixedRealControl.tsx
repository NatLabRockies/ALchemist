/**
 * A fixed Real variable in the contour plot's Fixed Values column: a slider for coarse moves and a
 * compact number box for typing an exact value or scrubbing with the arrow keys.
 *
 * `onScrub` reports every change as it happens (the parent debounces the request); `onCommit`
 * reports a settled value (slider released, Enter, blur) that should be requested at once.
 */
import { useId, useState } from 'react';
import type { ChangeEvent, KeyboardEvent } from 'react';

interface FixedRealControlProps {
  name: string;
  min: number;
  max: number;
  value: number;
  onScrub: (value: number) => void;
  onCommit: (value: number) => void;
}

// The Fixed Values column is narrow, so the label, slider and number box stack as the column's
// other controls do. The number box matches the Integer variable's input; the spinner buttons
// would cover its digits, so they are hidden (the arrow keys step it).
const inputClass =
  'w-full mt-1 px-2 py-1 text-xs tabular-nums bg-background border border-input rounded ' +
  'focus:outline-none focus:ring-2 focus:ring-primary/50 ' +
  '[appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none';

function formatValue(value: number): string {
  return String(parseFloat(value.toFixed(6)));
}

export function FixedRealControl({ name, min, max, value, onScrub, onCommit }: FixedRealControlProps) {
  const id = useId();
  // Text being typed; null while the box just mirrors the value.
  const [draft, setDraft] = useState<string | null>(null);
  const step = (max - min) / 100;

  const clamp = (v: number) => Math.min(max, Math.max(min, v));
  const parse = (text: string): number | null => {
    const n = parseFloat(text);
    return Number.isFinite(n) ? clamp(n) : null;
  };

  const commitDraft = () => {
    if (draft === null) {
      onCommit(value);
      return;
    }
    const parsed = parse(draft);
    setDraft(null);
    if (parsed !== null) {
      onScrub(parsed);
      onCommit(parsed);
    }
  };

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    const text = e.target.value;
    // Typing carries an inputType. A spinner click does not, so it steps like the arrow keys.
    const typed = typeof (e.nativeEvent as InputEvent).inputType === 'string';
    if (typed) {
      setDraft(text);
      return;
    }
    const parsed = parse(text);
    if (parsed === null) return;
    setDraft(null);
    onScrub(parsed);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      commitDraft();
    } else if (e.key === 'Escape') {
      // Cancels typed text only. An arrow-key step already applied stays, and its debounced
      // request still goes out.
      setDraft(null);
    } else if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
      e.preventDefault();
      const base = draft === null ? value : (parse(draft) ?? value);
      const next = clamp(parseFloat((base + (e.key === 'ArrowUp' ? step : -step)).toFixed(10)));
      setDraft(null);
      onScrub(next);
    }
  };

  return (
    <div>
      <label htmlFor={id} className="text-xs font-medium block mb-1 truncate" title={name}>
        {name}:
      </label>
      <input
        type="range"
        aria-label={`${name} (slider)`}
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onScrub(parseFloat(e.target.value))}
        onMouseUp={(e) => onCommit(parseFloat((e.target as HTMLInputElement).value))}
        onTouchEnd={(e) => onCommit(parseFloat((e.target as HTMLInputElement).value))}
        className="w-full"
      />
      <input
        id={id}
        type="number"
        aria-label={`${name} (value)`}
        title={`${formatValue(min)} to ${formatValue(max)}`}
        min={min}
        max={max}
        step={step}
        value={draft ?? formatValue(value)}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        onBlur={() => draft !== null && commitDraft()}
        className={inputClass}
      />
    </div>
  );
}
