/**
 * Target Column Selection Dialog
 * Shows when CSV doesn't contain 'Output' column
 * Supports single and multi-objective optimization
 */
import { useEffect, useState } from 'react';
import { X } from 'lucide-react';

interface TargetColumnDialogProps {
  open: boolean;
  onClose: () => void;
  availableColumns: string[];
  recommendedColumn?: string;
  onConfirm: (selectedColumns: string | string[]) => void;
}

export function TargetColumnDialog({
  open,
  onClose,
  availableColumns,
  recommendedColumn,
  onConfirm,
}: TargetColumnDialogProps) {
  const [mode, setMode] = useState<'single' | 'multi'>('single');
  const [singleColumn, setSingleColumn] = useState<string>(
    recommendedColumn || availableColumns[0] || ''
  );
  const [multiColumns, setMultiColumns] = useState<Set<string>>(new Set());

  // The dialog stays mounted, so its state was seeded once with no columns.
  // Pre-select the server's recommendation (else the first column) each time
  // it opens with the file's columns, so Confirm works without an extra click.
  useEffect(() => {
    if (open) setSingleColumn(recommendedColumn || availableColumns[0] || '');
  }, [open, recommendedColumn, availableColumns]);

  const handleConfirm = () => {
    if (mode === 'single') {
      if (!singleColumn) {
        return; // Don't allow empty selection
      }
      onConfirm(singleColumn);
    } else {
      if (multiColumns.size < 2) {
        return; // Multi-objective requires at least 2 columns
      }
      onConfirm(Array.from(multiColumns));
    }
    onClose();
  };

  const toggleMultiColumn = (column: string) => {
    const newSet = new Set(multiColumns);
    if (newSet.has(column)) {
      newSet.delete(column);
    } else {
      newSet.add(column);
    }
    setMultiColumns(newSet);
  };

  const isConfirmDisabled =
    mode === 'single'
      ? !singleColumn
      : multiColumns.size < 2;

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={onClose}>
      <div 
        className="bg-white dark:bg-gray-800 rounded-lg shadow-xl w-full max-w-md mx-4" 
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
          <div>
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
              Select Target Column(s)
            </h2>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              The 'Output' column was not found. Please select which column(s) to use as optimization target(s).
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
          >
            <X size={20} />
          </button>
        </div>

        {/* Body */}
        <div className="p-4 space-y-4">
          {/* Optimization Type Toggle */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Optimization Type
            </label>
            <div className="flex gap-4">
              <label className="flex items-center space-x-2 cursor-pointer">
                <input
                  type="radio"
                  name="mode"
                  value="single"
                  checked={mode === 'single'}
                  onChange={() => setMode('single')}
                  className="w-4 h-4 text-blue-600"
                />
                <span className="text-sm text-gray-700 dark:text-gray-300">
                  Single-Objective
                </span>
              </label>
              <label className="flex items-center space-x-2 cursor-pointer">
                <input
                  type="radio"
                  name="mode"
                  value="multi"
                  checked={mode === 'multi'}
                  onChange={() => setMode('multi')}
                  className="w-4 h-4 text-blue-600"
                />
                <span className="text-sm text-gray-700 dark:text-gray-300">
                  Multi-Objective
                </span>
              </label>
            </div>
          </div>

          {/* Single-Objective Column Selection */}
          {mode === 'single' && (
            <div className="space-y-2">
              <label 
                htmlFor="target-column" 
                className="text-sm font-medium text-gray-700 dark:text-gray-300"
              >
                Target Column
              </label>
              <select
                id="target-column"
                value={singleColumn}
                onChange={(e) => setSingleColumn(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {availableColumns.map((column) => (
                  <option key={column} value={column}>
                    {column}
                    {column === recommendedColumn ? ' (recommended)' : ''}
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* Multi-Objective Column Selection */}
          {mode === 'multi' && (
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Target Columns{' '}
                <span className="text-xs text-gray-500 dark:text-gray-400">
                  (select at least 2)
                </span>
              </label>
              <div className="border border-gray-300 dark:border-gray-600 rounded-md max-h-48 overflow-y-auto p-3 space-y-2 bg-white dark:bg-gray-700">
                {availableColumns.map((column) => (
                  <label
                    key={column}
                    className="flex items-center space-x-2 cursor-pointer"
                  >
                    <input
                      type="checkbox"
                      checked={multiColumns.has(column)}
                      onChange={() => toggleMultiColumn(column)}
                      className="w-4 h-4 text-blue-600 rounded"
                    />
                    <span className="text-sm text-gray-700 dark:text-gray-300">
                      {column}
                    </span>
                  </label>
                ))}
              </div>
              {mode === 'multi' && multiColumns.size === 1 && (
                <p className="text-xs text-amber-600 dark:text-amber-400">
                  Select at least one more column for multi-objective optimization
                </p>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-3 p-4 border-t border-gray-200 dark:border-gray-700">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md hover:bg-gray-50 dark:hover:bg-gray-600"
          >
            Cancel
          </button>
          <button
            onClick={handleConfirm}
            disabled={isConfirmDisabled}
            className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Confirm
          </button>
        </div>
      </div>
    </div>
  );
}
