/**
 * Session API endpoints
 */
import { apiClient } from '../client';
import type { 
  CreateSessionRequest, 
  CreateSessionResponse, 
  Session,
  SessionStateResponse,
  UpdateTTLRequest 
} from '../types';

/**
 * Create a new optimization session
 */
export const createSession = async (data?: CreateSessionRequest): Promise<CreateSessionResponse> => {
  const response = await apiClient.post<CreateSessionResponse>('/sessions', data || {});
  return response.data;
};

/**
 * Get session information
 */
export const getSession = async (sessionId: string): Promise<Session> => {
  const response = await apiClient.get(`/sessions/${sessionId}`);
  const raw = response.data;
  // Transform nested API response to flat Session interface
  return {
    ...raw,
    variable_count: raw.search_space?.n_variables ?? raw.variable_count ?? 0,
    experiment_count: raw.data?.n_experiments ?? raw.experiment_count ?? 0,
    model_trained: raw.model?.is_trained ?? raw.model_trained ?? false,
  };
};

/**
 * Get session state for monitoring
 */
export const getSessionState = async (sessionId: string): Promise<SessionStateResponse> => {
  const response = await apiClient.get<SessionStateResponse>(`/sessions/${sessionId}/state`);
  return response.data;
};

/**
 * Update session TTL
 */
export const updateSessionTTL = async (
  sessionId: string, 
  data: UpdateTTLRequest
): Promise<Session> => {
  const response = await apiClient.patch<Session>(`/sessions/${sessionId}/ttl`, data);
  return response.data;
};

/**
 * Delete a session
 */
export const deleteSession = async (sessionId: string): Promise<void> => {
  await apiClient.delete(`/sessions/${sessionId}`);
};

/**
 * Export a session as a downloadable file
 */
export const exportSession = async (sessionId: string): Promise<Blob> => {
  const response = await apiClient.get(`/sessions/${sessionId}/export`, {
    responseType: 'blob'
  });
  return response.data;
};

/**
 * Persist session on the server side (overwrite the stored session file)
 */
export const saveSession = async (sessionId: string): Promise<void> => {
  await apiClient.post(`/sessions/${sessionId}/save`);
};

/**
 * Import a session from a file
 */
export const importSession = async (file: File): Promise<CreateSessionResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await apiClient.post<CreateSessionResponse>('/sessions/import', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};
