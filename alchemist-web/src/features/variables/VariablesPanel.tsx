/**
 * Variables Panel - Main interface for defining search space
 * Mimics desktop SpaceSetupWindow layout
 */
import { useState, useRef } from 'react';
import { useVariables } from '../../hooks/api/useVariables';
import { VariableList } from './VariableList';
import { VariableForm } from './VariableForm';
import { useLoadVariablesFromFile } from '../../hooks/api/useFileOperations';
import { useSessionMetadata } from '../../hooks/api/useSessions';
import { reportSaveOutcome, useLinkedFile } from '../../hooks/useLinkedFile';
import { exportVariables } from '../../api/endpoints/variables';
import { JSON_FILE, pickOpenFile, type WritableFileHandle } from '../../lib/fileAccess';
import { sessionFileStem } from '../../lib/fileNames';
import { SplitSaveButton } from '../../components/ui';
import { toast } from '../../lib/toast';
import type { VariableDetail } from '../../api/types';

interface VariablesPanelProps {
  sessionId: string;
}

export function VariablesPanel({ sessionId }: VariablesPanelProps) {
  const [isFormOpen, setIsFormOpen] = useState(false);
  const [editingVariable, setEditingVariable] = useState<VariableDetail | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const { data: variablesData, isLoading } = useVariables(sessionId);
  const loadFromFile = useLoadVariablesFromFile(sessionId);
  const variablesFile = useLinkedFile('variables', sessionId);
  const { data: sessionMetadata } = useSessionMetadata(sessionId);
  const [isSaving, setIsSaving] = useState(false);

  const handleAddVariable = () => {
    setEditingVariable(null);
    setIsFormOpen(true);
  };

  const handleEditVariable = (variable: VariableDetail) => {
    setEditingVariable(variable);
    setIsFormOpen(true);
  };

  // Where the browser allows it, open through the file picker so the opened
  // file stays linked and Save File writes back to it; otherwise a file input.
  const handleLoadFromFile = async () => {
    if (!variablesFile.supported) {
      fileInputRef.current?.click();
      return;
    }
    let handle: WritableFileHandle | null;
    let file: File;
    try {
      handle = await pickOpenFile(JSON_FILE);
      if (!handle) return;
      file = await handle.getFile();
    } catch (error) {
      toast.error('Failed to open variables file');
      console.error('Error opening variables file:', error);
      return;
    }
    try {
      await loadFromFile.mutateAsync(file);
    } catch {
      return; // the mutation already reported it
    }
    await variablesFile.adopt(sessionId, handle);
  };

  const handleFileSelected = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      await loadFromFile.mutateAsync(file);
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleSaveToFile = async (saveAs: boolean) => {
    if (isSaving) return;
    setIsSaving(true);
    try {
      const outcome = await variablesFile.save({
        saveAs,
        suggestedName: `${sessionFileStem(sessionMetadata?.name, sessionId)}_variables.json`,
        type: JSON_FILE,
        getBlob: () => exportVariables(sessionId),
      });
      reportSaveOutcome(outcome, () => handleSaveToFile(true));
    } catch (error) {
      toast.error('Failed to save variables');
      console.error('Error saving variables:', error);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground border-b pb-2">
        Variable Management
      </h3>
      
      {/* Variable Display */}
      {isLoading ? (
        <div className="text-sm text-muted-foreground">Loading...</div>
      ) : variablesData && variablesData.n_variables > 0 ? (
        <VariableList 
          variables={variablesData.variables} 
          sessionId={sessionId}
          onEdit={handleEditVariable}
        />
      ) : (
        <div className="border border-dashed border-muted-foreground/20 rounded p-6 text-center">
          <p className="text-xs text-muted-foreground">No variables defined</p>
        </div>
      )}

      {/* Control Buttons - Compact */}
      <div className="flex flex-col gap-1.5">
        <button
          onClick={handleAddVariable}
          className="w-full bg-primary text-primary-foreground px-3 py-2 rounded text-sm font-medium hover:bg-primary/90 transition-colors"
        >
          Add Variable
        </button>
        
        <div className="flex gap-1.5">
          <button
            onClick={handleLoadFromFile}
            disabled={loadFromFile.isPending}
            className="flex-1 border border-input px-3 py-1.5 rounded text-xs hover:bg-accent disabled:opacity-50"
          >
            Load File
          </button>
          
          <SplitSaveButton
            label="Save File"
            variant="outline"
            className="flex-1"
            showMenu={variablesFile.supported}
            busy={isSaving}
            disabled={!variablesData || variablesData.n_variables === 0}
            onSave={() => handleSaveToFile(false)}
            onSaveAs={() => handleSaveToFile(true)}
            title={
              !variablesFile.supported
                ? variablesFile.unavailableReason ?? undefined
                : variablesFile.link
                  ? `Save to ${variablesFile.link.name}`
                  : 'Save the variables to a file'
            }
          />
        </div>
        {variablesFile.supported && variablesFile.link && (
          <p className="truncate text-[11px] text-muted-foreground" title={variablesFile.link.name}>
            Saves to {variablesFile.link.name}
          </p>
        )}
      </div>
      
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".json"
        onChange={handleFileSelected}
        className="hidden"
      />

      {/* Variable Form Modal */}
      {isFormOpen && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-background rounded-lg p-6 max-w-lg w-full mx-4 max-h-[90vh] overflow-y-auto shadow-xl border">
            <VariableForm 
              sessionId={sessionId}
              onClose={() => setIsFormOpen(false)}
              editingVariable={editingVariable}
            />
          </div>
        </div>
      )}
    </div>
  );
}
