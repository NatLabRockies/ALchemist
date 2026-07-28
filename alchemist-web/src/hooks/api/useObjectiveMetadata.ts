import { useQuery } from '@tanstack/react-query';
import type { UseQueryResult } from '@tanstack/react-query';
import apiClient from '../../api/client';

interface ObjectiveMetadataResponse { metadata: Record<string, { label?: string; unit?: string }>; }

export function useObjectiveMetadata(sessionId: string | null): UseQueryResult<ObjectiveMetadataResponse> {
  return useQuery({
    queryKey: ['objective-metadata', sessionId],
    queryFn: async () => (await apiClient.get<ObjectiveMetadataResponse>(
      `/sessions/${sessionId}/objective-metadata`)).data,
    enabled: !!sessionId,
    refetchOnWindowFocus: false,
  });
}
