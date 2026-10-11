/**
 * API hooks for model training and management
 */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '../../api/client';
import { toast } from 'sonner';
import type { 
  TrainModelRequest, 
  TrainModelResponse, 
  ModelInfo 
} from '../../api/types';
import { invalidateAfterModelTrained } from '../../lib/invalidateSessionData';

/**
 * Hook to get current model info
 */
export function useModelInfo(sessionId: string) {
  return useQuery({
    queryKey: ['model-info', sessionId],
    queryFn: async () => {
      const response = await apiClient.get<ModelInfo>(`/sessions/${sessionId}/model`);
      return response.data;
    },
    enabled: !!sessionId,
  });
}

/**
 * Hook to train a model
 */
export function useTrainModel(sessionId: string) {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (request: TrainModelRequest) => {
      const response = await apiClient.post<TrainModelResponse>(
        `/sessions/${sessionId}/model/train`,
        request
      );
      return response.data;
    },
    onSuccess: (data) => {
      // Model info, session status and every model plot (parity, metrics,
      // calibration, Q-Q, hyperparameters, contour). POST /model/train sends
      // no model_trained WebSocket event, so without this an open parity plot
      // kept showing the previous model (bug W9).
      invalidateAfterModelTrained(queryClient, sessionId);
      // Show success toast
      toast.success(`Model trained successfully! R² = ${data.metrics.r2.toFixed(4)}`);
    },
    onError: (error: any) => {
      // Show error toast
      toast.error(error.response?.data?.detail || 'Failed to train model');
    },
  });
}
