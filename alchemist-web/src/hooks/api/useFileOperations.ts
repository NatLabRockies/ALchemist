/**
 * Hooks for variable file operations (load/export)
 */
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { toast } from '../../lib/toast';
import { loadVariablesFromFile, exportVariablesToFile } from '../../api/endpoints/variables';
import { fetchSessionFileStem } from './useSessions';

export function useLoadVariablesFromFile(sessionId: string) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (file: File) => loadVariablesFromFile(sessionId, file),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['variables', sessionId] });
      queryClient.invalidateQueries({ queryKey: ['session', sessionId] });
      toast.success(data.message || `Loaded ${data.n_variables} variables`);
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to load variables from file');
      console.error('Load variables error:', error);
    },
  });
}

export function useExportVariablesToFile(sessionId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async () => exportVariablesToFile(sessionId, await fetchSessionFileStem(queryClient, sessionId)),
    onSuccess: () => {
      toast.success('Variables exported successfully');
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to export variables');
      console.error('Export variables error:', error);
    },
  });
}
