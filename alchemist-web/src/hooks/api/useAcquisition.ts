/**
 * API hooks for acquisition functions
 */
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '../../api/client';
import { toast } from '../../lib/toast';
import { invalidateAfterQueueOrSuggestion } from '../../lib/invalidateSessionData';
import type { 
  AcquisitionRequest, 
  AcquisitionResponse,
  FindOptimumRequest,
  FindOptimumResponse
} from '../../api/types';

/**
 * Hook to suggest next experiments using acquisition functions
 */
export function useSuggestNext(sessionId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (request: AcquisitionRequest) => {
      const response = await apiClient.post<AcquisitionResponse>(
        `/sessions/${sessionId}/acquisition/suggest`,
        request
      );
      return response.data;
    },
    onSuccess: (data) => {
      // The contour's "Show Next Point" reads the session's latest
      // suggestions, so an open contour must redraw.
      invalidateAfterQueueOrSuggestion(queryClient, sessionId);
      toast.success(`Generated ${data.n_suggestions} suggestion${data.n_suggestions > 1 ? 's' : ''}!`);
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to generate suggestions');
    },
  });
}

/**
 * Hook to find the model's predicted optimum
 */
export function useFindOptimum(sessionId: string) {
  return useMutation({
    mutationFn: async (request: FindOptimumRequest) => {
      const response = await apiClient.post<FindOptimumResponse>(
        `/sessions/${sessionId}/acquisition/find-optimum`,
        request
      );
      return response.data;
    },
    onSuccess: (data) => {
      toast.success(`Found model optimum: ${data.predicted_value.toFixed(4)}`);
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to find model optimum');
    },
  });
}
