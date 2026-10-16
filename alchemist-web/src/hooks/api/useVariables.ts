/**
 * React Query hooks for variables (search space)
 */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import * as variablesAPI from '../../api/endpoints/variables';
import type { Variable } from '../../api/types';
import { toast } from '../../lib/toast';
import { apiErrorMessage } from '../../lib/apiError';

/**
 * Hook to get all variables
 */
export function useVariables(sessionId: string | null) {
  return useQuery({
    queryKey: ['variables', sessionId],
    queryFn: () => variablesAPI.getVariables(sessionId!),
    enabled: !!sessionId,
  });
}

/**
 * Hook to create a variable
 */
export function useCreateVariable(sessionId: string) {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (variable: Variable) => variablesAPI.createVariable(sessionId, variable),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['variables', sessionId] });
      queryClient.invalidateQueries({ queryKey: ['session', sessionId] });
      toast.success('Variable added successfully!');
    },
    onError: (error: unknown) => {
      toast.error(apiErrorMessage(error, 'Failed to add variable'));
    },
  });
}

/**
 * Hook to update a variable
 */
export function useUpdateVariable(sessionId: string) {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ variableName, variable }: { variableName: string; variable: Variable }) => 
      variablesAPI.updateVariable(sessionId, variableName, variable),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['variables', sessionId] });
      queryClient.invalidateQueries({ queryKey: ['session', sessionId] });
      toast.success('Variable updated successfully!');
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to update variable');
    },
  });
}
