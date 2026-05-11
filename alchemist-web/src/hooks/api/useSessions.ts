/**
 * React Query hooks for session management
 */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import * as sessionAPI from '../../api/endpoints/sessions';
import type { CreateSessionRequest } from '../../api/types';

/**
 * Hook to get current session info
 */
export function useSession(sessionId: string | null) {
  return useQuery({
    queryKey: ['session', sessionId],
    queryFn: () => sessionAPI.getSession(sessionId!),
    enabled: !!sessionId, // Only run if sessionId exists
  });
}

/**
 * Hook to get session state for monitoring
 */
export function useSessionState(sessionId: string | null, refetchInterval?: number) {
  return useQuery({
    queryKey: ['session-state', sessionId],
    queryFn: () => sessionAPI.getSessionState(sessionId!),
    enabled: !!sessionId,
    refetchInterval: refetchInterval, // Enable polling for monitoring mode
  });
}

/**
 * Hook to create a new session
 */
export function useCreateSession() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (data?: CreateSessionRequest) => sessionAPI.createSession(data),
    onSuccess: (newSession) => {
      // Store session ID in localStorage using helper
      storeSessionId(newSession.session_id);
      // Invalidate queries to refresh data
      queryClient.invalidateQueries({ queryKey: ['session'] });
    },
  });
}

/**
 * Hook to update session metadata
 */
export function useUpdateSessionMetadata(sessionId: string | null) {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (metadata: { name?: string; description?: string; tags?: string[]; author?: string }) => {
      if (!sessionId) throw new Error('No session ID provided');
      const response = await fetch(`/api/v1/sessions/${sessionId}/metadata`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(metadata)
      });
      if (!response.ok) throw new Error('Failed to update metadata');
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['session', sessionId] });
    },
  });
}

/**
 * Hook to delete a session
 */
export function useDeleteSession() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (sessionId: string) => sessionAPI.deleteSession(sessionId),
    onSuccess: () => {
      // Clear session from localStorage using helper
      clearStoredSessionId();
      // Clear all queries
      queryClient.clear();
    },
  });
}

/**
 * STORAGE KEY - consistent across the app
 */
const SESSION_STORAGE_KEY = 'alchemist_session_id';

/**
 * Get stored session ID from localStorage
 */
export function getStoredSessionId(): string | null {
  return localStorage.getItem(SESSION_STORAGE_KEY);
}

/**
 * Store session ID in localStorage
 */
export function storeSessionId(sessionId: string): void {
  localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
}

/**
 * Clear stored session ID
 */
export function clearStoredSessionId(): void {
  localStorage.removeItem(SESSION_STORAGE_KEY);
}

/**
 * Hook to export a session
 */
export function useExportSession() {
  return useMutation({
    mutationFn: async (opts: { sessionId: string; serverSide?: boolean }) => {
      const { sessionId, serverSide } = opts;
      if (serverSide) {
        // Ask server to persist the session to its storage
        await sessionAPI.saveSession(sessionId);
        return { success: true, serverSide: true };
      }

      const blob = await sessionAPI.exportSession(sessionId);

      // Trigger download
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `alchemist_session_${sessionId.slice(0, 8)}.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);

      return { success: true, serverSide: false };
    },
  });
}

/**
 * Hook to import a session
 */
export function useImportSession() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (file: File) => sessionAPI.importSession(file),
    onSuccess: (newSession) => {
      // Store new session ID in localStorage
      storeSessionId(newSession.session_id);
      // Clear all queries and refetch
      queryClient.clear();
      queryClient.invalidateQueries({ queryKey: ['session'] });
    },
  });
}
