/**
 * React Query hook for deleting variables
 */
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '../../api/client';
import { toast } from '../../lib/toast';

/**
 * Delete a variable from the search space
 */
export function useDeleteVariable(sessionId: string) {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (variableName: string) => {
      // Backend doesn't have delete endpoint yet, but we'll call it
      await apiClient.delete(`/sessions/${sessionId}/variables/${variableName}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['variables', sessionId] });
      queryClient.invalidateQueries({ queryKey: ['session', sessionId] });
      toast.success('Variable deleted successfully!');
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to delete variable');
    },
  });
}
