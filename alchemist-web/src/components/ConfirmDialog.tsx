/**
 * A small yes/no dialog in the app's own style (the metadata dialog's), for a
 * choice that should not be a native window.confirm.
 */
import { useId } from 'react';
import type { KeyboardEvent } from 'react';

interface ConfirmDialogProps {
  title: string;
  message: string;
  confirmLabel: string;
  onConfirm: () => void;
  onCancel: () => void;
}

export function ConfirmDialog({ title, message, confirmLabel, onConfirm, onCancel }: ConfirmDialogProps) {
  const titleId = useId();
  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'Escape') onCancel();
  };
  return (
    <div
      data-testid="confirm-backdrop"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
      onClick={onCancel}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onKeyDown={onKeyDown}
        className="bg-card border border-border rounded-lg shadow-lg w-full max-w-md mx-4"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="border-b border-border p-4">
          <h3 id={titleId} className="text-lg font-semibold">{title}</h3>
        </div>
        <div className="p-6 text-sm text-muted-foreground">{message}</div>
        <div className="border-t border-border p-4 flex justify-end gap-3">
          <button
            type="button"
            autoFocus
            onClick={onCancel}
            className="px-4 py-2 text-sm rounded-md border border-border hover:bg-accent"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={onConfirm}
            className="px-4 py-2 text-sm rounded-md bg-destructive text-destructive-foreground hover:bg-destructive/90"
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
