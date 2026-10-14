/**
 * Save button with a ▾ menu holding Save and Save As….
 *
 * The menu follows the WAI-ARIA menu-button pattern: arrows move between the
 * items, Enter/Space choose, Escape closes and returns focus to the toggle.
 * Without a menu (browsers that can only download) it is a plain button.
 */
import { useEffect, useId, useRef, useState } from 'react';
import { ChevronDown } from 'lucide-react';
import { SAVE_SHORTCUT, SAVE_AS_SHORTCUT } from '../../lib/shortcuts';

interface SplitSaveButtonProps {
  label: string;
  onSave: () => void;
  onSaveAs: () => void;
  /** False where only downloads work: no menu, Save As would do the same thing. */
  showMenu: boolean;
  disabled?: boolean;
  /** A save is running. Buttons stay enabled (so focus is not lost); callers ignore repeat clicks. */
  busy?: boolean;
  /** Tooltip on the main button. */
  title?: string;
  /** Show keyboard shortcuts in the menu (header Save only). */
  shortcuts?: boolean;
  variant: 'primary' | 'outline';
  className?: string;
}

const STYLES = {
  primary: {
    main: 'bg-primary text-primary-foreground hover:bg-primary/90',
    toggle: 'bg-primary text-primary-foreground hover:bg-primary/90 border-l border-primary-foreground/25',
  },
  outline: {
    main: 'border border-input hover:bg-accent',
    toggle: 'border border-l-0 border-input hover:bg-accent',
  },
};

export function SplitSaveButton({
  label, onSave, onSaveAs, showMenu, disabled, busy, title, shortcuts, variant, className = '',
}: SplitSaveButtonProps) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const toggleRef = useRef<HTMLButtonElement>(null);
  const itemRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const menuId = useId();
  const style = STYLES[variant];

  useEffect(() => {
    if (!open) return;
    itemRefs.current[0]?.focus();
    const onPointerDown = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener('pointerdown', onPointerDown);
    return () => document.removeEventListener('pointerdown', onPointerDown);
  }, [open]);

  const close = (refocus: boolean) => {
    setOpen(false);
    if (refocus) toggleRef.current?.focus();
  };

  const choose = (action: () => void) => {
    close(true);
    action();
  };

  const onMenuKeyDown = (event: React.KeyboardEvent) => {
    const items = itemRefs.current.filter((el): el is HTMLButtonElement => !!el);
    const index = items.indexOf(document.activeElement as HTMLButtonElement);
    const focusAt = (i: number) => items[(i + items.length) % items.length]?.focus();
    switch (event.key) {
      case 'ArrowDown': event.preventDefault(); focusAt(index + 1); break;
      case 'ArrowUp': event.preventDefault(); focusAt(index - 1); break;
      case 'Home': event.preventDefault(); focusAt(0); break;
      case 'End': event.preventDefault(); focusAt(items.length - 1); break;
      case 'Escape': event.preventDefault(); close(true); break;
      case 'Tab': close(false); break;
    }
  };

  const radius = showMenu ? 'rounded-l' : 'rounded';
  const items: { label: string; hint: string; action: () => void }[] = [
    { label: 'Save', hint: SAVE_SHORTCUT, action: onSave },
    { label: 'Save As…', hint: SAVE_AS_SHORTCUT, action: onSaveAs },
  ];

  return (
    <div ref={rootRef} className={`relative inline-flex ${className}`}>
      <button
        type="button"
        onClick={onSave}
        disabled={disabled}
        aria-busy={busy || undefined}
        title={title}
        className={`flex-1 px-3 py-1.5 text-xs transition-colors disabled:opacity-50 ${radius} ${style.main}`}
      >
        {label}
      </button>
      {showMenu && (
        <>
          <button
            ref={toggleRef}
            type="button"
            aria-label={`${label} options`}
            aria-haspopup="menu"
            aria-expanded={open}
            aria-controls={open ? menuId : undefined}
            disabled={disabled}
            onClick={() => setOpen(o => !o)}
            onKeyDown={(event) => {
              if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
                event.preventDefault();
                setOpen(true);
              }
            }}
            className={`px-1.5 rounded-r transition-colors disabled:opacity-50 ${style.toggle}`}
          >
            <ChevronDown className="h-3.5 w-3.5" aria-hidden="true" />
          </button>
          {open && (
            <div
              id={menuId}
              role="menu"
              aria-label={`${label} options`}
              onKeyDown={onMenuKeyDown}
              className="absolute right-0 top-full z-50 mt-1 min-w-[11rem] rounded-md border border-border bg-card p-1 text-card-foreground shadow-lg"
            >
              {items.map((item, i) => (
                <button
                  key={item.label}
                  ref={el => { itemRefs.current[i] = el; }}
                  type="button"
                  role="menuitem"
                  tabIndex={-1}
                  onClick={() => choose(item.action)}
                  className="flex w-full items-center justify-between gap-6 rounded px-2 py-1.5 text-left text-xs hover:bg-accent focus:bg-accent focus:outline-none"
                >
                  <span>{item.label}</span>
                  {shortcuts && <kbd className="font-sans text-[11px] text-muted-foreground">{item.hint}</kbd>}
                </button>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
