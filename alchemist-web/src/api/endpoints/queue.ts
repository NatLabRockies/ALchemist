import apiClient from '../client';

export type QueueStatus = 'pending' | 'running' | 'done' | 'failed';

export interface QueueItem {
  id: string;
  inputs: Record<string, unknown>;
  reason: string | null;
  status: QueueStatus;
  output: number | number[] | null;
  noise: number | number[] | null;
  error: string | null;
  dataset_ref: number | null;
  staged_at: string | null;
  started_at: string | null;
  completed_at: string | null;
}

export interface QueueListResponse {
  items: QueueItem[];
  n_pending: number;
  n_running: number;
  n_done: number;
  n_failed: number;
}

export interface ConfigChangeEntry {
  timestamp: string;
  component: string;
  old: Record<string, unknown>;
  new: Record<string, unknown>;
  iteration: number | null;
}

export interface ConfigChangesResponse {
  changes: ConfigChangeEntry[];
}

export async function getQueue(sessionId: string): Promise<QueueListResponse> {
  const res = await apiClient.get<QueueListResponse>(
    `/sessions/${sessionId}/experiments/queue`
  );
  return res.data;
}

export async function getConfigChanges(sessionId: string): Promise<ConfigChangesResponse> {
  const res = await apiClient.get<ConfigChangesResponse>(
    `/sessions/${sessionId}/audit/config-changes`
  );
  return res.data;
}
