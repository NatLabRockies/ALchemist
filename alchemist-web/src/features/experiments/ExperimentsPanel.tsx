/**
 * Experiments Panel - Manage experimental data
 * Mimics desktop UI experiment management
 */
import { useRef, useState } from 'react';
import { useExperiments, useExperimentsSummary, useUploadExperiments, usePreviewCSV } from '../../hooks/api/useExperiments';
import { useVariables } from '../../hooks/api/useVariables';
import { useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';
import AddPointDialog from '../../components/AddPointDialog';
import { TargetColumnDialog } from '../../components/TargetColumnDialog';

interface ExperimentsPanelProps {
  sessionId: string;
  pendingSuggestions?: any[];
  onStageSuggestions?: (pending: any[]) => void;
}

export function ExperimentsPanel({ sessionId, pendingSuggestions = [], onStageSuggestions }: ExperimentsPanelProps) {
  const [addPointOpen, setAddPointOpen] = useState(false);
  const [currentIndex, setCurrentIndex] = useState<number>(0);
  const [targetDialogOpen, setTargetDialogOpen] = useState(false);
  const [availableTargets, setAvailableTargets] = useState<string[]>([]);
  const [recommendedTarget, setRecommendedTarget] = useState<string | undefined>();
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const queryClient = useQueryClient();
  
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

  const experiments = experimentsData?.experiments || [];
  const hasExperiments = experiments.length > 0;

  // Get column headers from first experiment
  const columns = hasExperiments ? Object.keys(experiments[0]) : [];

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
                  {summaryData.target_stats.min?.toFixed(2)} - {summaryData.target_stats.max?.toFixed(2)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Mean ± Std:</span>
                <span className="font-medium tabular-nums">
                  {summaryData.target_stats.mean?.toFixed(2)} ± {summaryData.target_stats.std?.toFixed(2)}
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
              <thead className="bg-muted/50 border-b sticky top-0">
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
                        {typeof exp[col] === 'number' 
                          ? exp[col].toFixed(3) 
                          : exp[col] ?? '-'}
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
        onClick={() => { setAddPointOpen(true); setCurrentIndex(0); }}
        disabled={!pendingSuggestions || pendingSuggestions.length === 0}
        className="w-full border border-muted px-3 py-2 rounded text-sm font-medium hover:bg-muted/5 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        Add Point... {pendingSuggestions && pendingSuggestions.length > 0 ? `(${pendingSuggestions.length})` : ''}
      </button>

      {/* Modal overlay for Add Point dialog */}
      {addPointOpen && pendingSuggestions && pendingSuggestions.length > 0 && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={() => setAddPointOpen(false)}>
          <div onClick={(e) => e.stopPropagation()}>
            <AddPointDialog
              key={currentIndex}
              variables={variables}
              suggestion={pendingSuggestions[currentIndex]}
              index={currentIndex}
              total={pendingSuggestions.length}
              onCancel={() => setAddPointOpen(false)}
              onConfirm={async (payload, options) => {
                // Import API helper
                const { addExperiment } = await import('../../components/api');
                try {
                  await addExperiment(sessionId, payload, options.retrain);
                  
                  // Invalidate queries to refresh experiments table
                  queryClient.invalidateQueries({ queryKey: ['experiments', sessionId] });
                  queryClient.invalidateQueries({ queryKey: ['experiments-summary', sessionId] });
                  queryClient.invalidateQueries({ queryKey: ['session', sessionId] });
                  
                  // Remove current suggestion from staged list (local state)
                  const updated = pendingSuggestions.filter((_, i) => i !== currentIndex);
                  onStageSuggestions && onStageSuggestions(updated);
                  
                  // Sync with staged experiments API - clear all and re-stage remaining
                  // This keeps the API queue in sync with the UI
                  try {
                    await fetch(`/api/v1/sessions/${sessionId}/experiments/staged`, { method: 'DELETE' });
                    if (updated.length > 0) {
                      const cleanExperiments = updated.map(s => {
                        const { _reason, _strategyParams, ...rest } = s;
                        return rest;
                      });
                      await fetch(`/api/v1/sessions/${sessionId}/experiments/staged/batch`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                          experiments: cleanExperiments,
                          reason: updated[0]?._reason || 'Acquisition'
                        })
                      });
                    }
                  } catch (syncError) {
                    console.warn('Failed to sync staged experiments with API:', syncError);
                  }
                  
                  // Show success message
                  toast.success('Experiment added successfully!');
                  
                  // Close modal if no more suggestions, otherwise adjust index
                  if (updated.length === 0) {
                    setAddPointOpen(false);
                  } else if (currentIndex >= updated.length) {
                    setCurrentIndex(updated.length - 1);
                  }
                } catch (e: any) {
                  toast.error('Failed to add point: ' + (e?.message || String(e)));
                }
              }}
              onPrev={currentIndex > 0 ? () => setCurrentIndex((i)=>i-1) : undefined}
              onNext={currentIndex < pendingSuggestions.length - 1 ? () => setCurrentIndex((i)=>i+1) : undefined}
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
