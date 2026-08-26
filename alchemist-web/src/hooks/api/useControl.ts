import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import type { UseQueryResult } from '@tanstack/react-query';
import apiClient from '../../api/client';
import { toast } from 'sonner';

export interface ControlRecord {
  requested: 'run' | 'pause';
  requested_at: string | null;
  requested_by: string | null;
  reported: 'idle' | 'running' | 'paused' | 'failed';
  reported_at: string | null;
  reported_by: string | null;
  detail: string | null;
}

export function useControl(sessionId: string | null): UseQueryResult<ControlRecord> {
  return useQuery({
    queryKey: ['control', sessionId],
    queryFn: async () =>
      (await apiClient.get<ControlRecord>(`/sessions/${sessionId}/control`)).data,
    enabled: !!sessionId,
    refetchOnWindowFocus: false,
  });
}

/**
 * Write the `requested` half only.
 *
 * This asks the consumer driving the session to hold; it does not stop
 * anything by itself and never can. Whether the request was honored is
 * readable only from `reported`, which this mutation cannot touch.
 */
export function useRequestControl(sessionId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (requested: 'run' | 'pause') =>
      (await apiClient.put<ControlRecord>(`/sessions/${sessionId}/control`, {
        requested,
        requested_by: window.location.hostname,
      })).data,
    onSuccess: (_data, requested) => {
      queryClient.invalidateQueries({ queryKey: ['control', sessionId] });
      toast.info(
        requested === 'pause'
          ? 'Pause requested — waiting for the controller to acknowledge'
          : 'Resume requested',
      );
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to send control request');
    },
  });
}
