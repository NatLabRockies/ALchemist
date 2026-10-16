/**
 * Experiments Panel - Manage experimental data
 * Mimics desktop UI experiment management
 */
import { useMemo, useRef, useState } from 'react';
import { useExperiments, useExperimentsSummary, useUploadExperiments, usePreviewCSV } from '../../hooks/api/useExperiments';
import { useVariables } from '../../hooks/api/useVariables';
import { useSession } from '../../hooks/api/useSessions';
import AddPointDialog from '../../components/AddPointDialog';
import { TargetColumnDialog } from '../../components/TargetColumnDialog';
import { useAddPointBatch } from './useAddPointBatch';
import { blankPoint } from './addPointEntries';
import { formatCell } from '../../lib/formatCell';

interface ExperimentsPanelProps {
  sessionId: string;
  pendingSuggestions?: any[];
  onStageSuggestions?: (pending: any[]) => void;
}

function newManualId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

export function ExperimentsPanel({ sessionId, pendingSuggestions = [], onStageSuggestions }: ExperimentsPanelProps) {
  const [addPointOpen, setAddPointOpen] = useState(false);
  const [targetDialogOpen, setTargetDialogOpen] = useState(false);
  const [availableTargets, setAvailableTargets] = useState<string[]>([]);
  const [recommendedTarget, setRecommendedTarget] = useState<string | undefined>();
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  // Hand-typed points (W28): blank points the user adds in the dialog, with
  // nothing suggested. They live only while the dialog is open and never join
  // the staged list. The dialog shows staged points first, then these.
  const [manual, setManual] = useState<Record<string, unknown>[]>([]);
  const dialogPoints = useMemo(() => [...pendingSuggestions, ...manual], [pendingSuggestions, manual]);
  // Per-point entries for the Add Point dialog, keyed by a stable suggestion
  // id: kept across Next/Previous and across closing/reopening the dialog.
  // After a save the remaining points split back into staged and manual.
  const addPoint = useAddPointBatch(sessionId, dialogPoints, (remaining) => {
    const staged = remaining.filter((s) => !s?._manualId);
    if (staged.length !== pendingSuggestions.length) onStageSuggestions?.(staged);
    setManual(remaining.filter((s) => s?._manualId));
  });

  const { data: experimentsData, isLoading: isLoadingExperiments } = useExperiments(sessionId);
  const { data: variablesData } = useVariables(sessionId);
  const variables = variablesData?.variables ?? [];
  const { data: summaryData } = useExperimentsSummary(sessionId);
  const uploadExperiments = useUploadExperiments(sessionId);
  const previewCSV = usePreviewCSV(sessionId);

  const handleLoadFromFile = () => {
    fileInputRef.current?.click();
  };

  const handleFileSelected = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // First, preview the CSV to check for target columns
    try {
      const preview = await previewCSV.mutateAsync(file);
      
      if (preview.has_output) {
        // Has 'Output' column, upload directly
        await uploadExperiments.mutateAsync({ file });
      } else {
        // Missing 'Output', show target column selection dialog
        setAvailableTargets(preview.available_targets);
        setRecommendedTarget(preview.recommended_target || undefined);
        setPendingFile(file);
        setTargetDialogOpen(true);
      }
    } catch (error) {
      // Preview failed, let upload attempt and show error
      console.error('Preview failed:', error);
    }
    
    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleTargetColumnConfirm = async (selectedColumns: string | string[]) => {
    if (!pendingFile) return;
    
    try {
      await uploadExperiments.mutateAsync({ 
        file: pendingFile, 
        targetColumns: selectedColumns 
      });
      setPendingFile(null);
    } catch (error) {
      console.error('Upload failed:', error);
    }
  };

  const handleTargetDialogClose = () => {
    setTargetDialogOpen(false);
    setPendingFile(null);
  };

  const openAddPoint = () => {
    if (pendingSuggestions.length === 0) setManual([blankPoint(variables, newManualId())]);
    setAddPointOpen(true);
  };
  const addBlankPoint = () => setManual((prev) => [...prev, blankPoint(variables, newManualId())]);
  const closeAddPoint = () => {
    setAddPointOpen(false);
    setManual([]);
  };

  const experiments = experimentsData?.experiments || [];
  const hasExperiments = experiments.length > 0;

  // The session's objective name(s), when the server reports them: a pasted
  // column headed with the target's name is recognised, and the result field
  // is labelled with it instead of "Output" (W16, W43).
  const { data: sessionInfo } = useSession(sessionId);
  const targetColumns = sessionInfo?.target_columns ?? [];
  const outputLabel = targetColumns.length === 1 && targetColumns[0] !== 'Output' ? targetColumns[0] : 'Output';

  // Column headers from the first experiment; the provenance id is bookkeeping
  // (it stays in the data and the export), not a column to read
  const columns = hasExperiments ? Object.keys(experiments[0]).filter((c) => c !== 'ProvenanceId') : [];

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground border-b pb-2">
        Experiment Data
      </h3>
      
      {/* Controls - Compact */}
      <button
        onClick={handleLoadFromFile}
        disabled={uploadExperiments.isPending}
        className="w-full bg-primary text-primary-foreground px-3 py-2 rounded text-sm font-medium hover:bg-primary/90 disabled:opacity-50"
      >
        {uploadExperiments.isPending ? 'Loading...' : 'Load CSV'}
      </button>
      
      <input
        ref={fileInputRef}
        type="file"
        accept=".csv"
        onChange={handleFileSelected}
        className="hidden"
      />

      {/* Summary Stats - Compact */}
      {summaryData && summaryData.has_data && (
        <div className="p-3 bg-muted/30 rounded text-xs space-y-1">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Experiments:</span>
            <span className="font-medium tabular-nums">{summaryData.n_experiments}</span>
          </div>
          {summaryData.target_stats && (
            <>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Range:</span>
                <span className="font-medium tabular-nums">
                  {summaryData.target_stats.min?.toFixed(2) ?? '—'} - {summaryData.target_stats.max?.toFixed(2) ?? '—'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Mean ± Std:</span>
                <span className="font-medium tabular-nums">
                  {summaryData.target_stats.mean?.toFixed(2) ?? '—'} ±{' '}
                  {summaryData.target_stats.std != null ? summaryData.target_stats.std.toFixed(2) : '—'}
                </span>
              </div>
            </>
          )}
        </div>
      )}

      {/* Experiments Table - Compact */}
      {isLoadingExperiments ? (
        <div className="text-xs text-muted-foreground">Loading...</div>
      ) : hasExperiments ? (
        <div className="border rounded overflow-hidden">
          <div className="overflow-x-auto max-h-64">
            <table className="w-full text-xs">
              <thead className="bg-muted border-b sticky top-0">
                <tr>
                  {columns.map((col) => (
                    <th key={col} className="px-2 py-1.5 text-left font-medium text-xs">
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y">
                {experiments.map((exp, idx) => (
                  <tr key={idx} className="hover:bg-accent/50">
                    {columns.map((col) => (
                      <td key={col} className="px-2 py-1.5 tabular-nums">
                        {formatCell(col, exp[col])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          <div className="bg-muted/30 border-t px-2 py-1 text-xs text-muted-foreground">
            {experiments.length} row{experiments.length !== 1 ? 's' : ''}
          </div>
        </div>
      ) : (
        <div className="border border-dashed border-muted-foreground/20 rounded p-6 text-center">
          <p className="text-xs text-muted-foreground">No data loaded</p>
        </div>
      )}

      {/* Add Point button - below table */}
      <button
        onClick={openAddPoint}
        disabled={uploadExperiments.isPending || variables.length === 0}
        title={variables.length === 0 ? 'Define variables first' : undefined}
        className="w-full border border-muted px-3 py-2 rounded text-sm font-medium hover:bg-muted/5 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        Add Point... {pendingSuggestions && pendingSuggestions.length > 0 ? `(${pendingSuggestions.length})` : ''}
      </button>

      {/* Modal overlay for Add Point dialog. One dialog for the whole batch
          (not remounted per point), so entries and position persist. */}
      {addPointOpen && dialogPoints.length > 0 && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4" onClick={closeAddPoint}>
          <div className="w-full flex justify-center" onClick={(e) => e.stopPropagation()}>
            <AddPointDialog
              variables={variables}
              suggestions={dialogPoints}
              outputColumnNames={targetColumns}
              outputLabel={outputLabel}
              onAddBlankPoint={addBlankPoint}
              entries={addPoint.entries}
              onEntriesChange={addPoint.onEntriesChange}
              onCancel={closeAddPoint}
              onSave={async (plan, options) => {
                await addPoint.save(plan, options);
                closeAddPoint();
              }}
            />
          </div>
        </div>
      )}

      {/* Target Column Selection Dialog */}
      <TargetColumnDialog
        open={targetDialogOpen}
        onClose={handleTargetDialogClose}
        availableColumns={availableTargets}
        recommendedColumn={recommendedTarget}
        onConfirm={handleTargetColumnConfirm}
      />
    </div>
  );
}
