import { useQuery, UseQueryResult } from '@tanstack/react-query';
import * as queueApi from '../../api/endpoints/queue';
import type { QueueListResponse, ConfigChangesResponse } from '../../api/endpoints/queue';

export function useExperimentQueue(
  sessionId: string | null,
  enabled = true
): UseQueryResult<QueueListResponse> {
  return useQuery({
    queryKey: ['experiments-queue', sessionId],
    queryFn: () => queueApi.getQueue(sessionId!),
    enabled: enabled && !!sessionId,
    refetchOnWindowFocus: false,
  });
}

export function useConfigChanges(
  sessionId: string | null,
  enabled = true
): UseQueryResult<ConfigChangesResponse> {
  return useQuery({
    queryKey: ['config-changes', sessionId],
    queryFn: () => queueApi.getConfigChanges(sessionId!),
    enabled: enabled && !!sessionId,
    refetchOnWindowFocus: false,
  });
}
