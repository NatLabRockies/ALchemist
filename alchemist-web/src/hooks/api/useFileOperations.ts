/**
 * Hooks for variable file operations (load/export)
 */
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { toast } from '../../lib/toast';
import { loadVariablesFromFile } from '../../api/endpoints/variables';

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
