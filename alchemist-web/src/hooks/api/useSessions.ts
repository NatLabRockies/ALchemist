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

/** Query for a session's metadata; under ['session', id] so session invalidations refresh it. */
const sessionMetadataQuery = (sessionId: string) => ({
  queryKey: ['session', sessionId, 'metadata'],
  queryFn: () => sessionAPI.getSessionMetadata(sessionId),
});

/**
 * Hook to get session metadata (name, description, tags)
 */
export function useSessionMetadata(sessionId: string | null) {
  return useQuery({
    ...sessionMetadataQuery(sessionId ?? ''),
    enabled: !!sessionId,
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
 * Get stored session ID from localStorage: the session this browser last
 * worked on, which the app reopens on start (see lib/sessionResume.ts).
 * Storage can throw (blocked site data); then nothing is remembered.
 */
export function getStoredSessionId(): string | null {
  try {
    return localStorage.getItem(SESSION_STORAGE_KEY);
  } catch {
    return null;
  }
}

/**
 * Store session ID in localStorage, and remember it among the sessions this
 * browser has worked on (created, joined, loaded or restored).
 */
export function storeSessionId(sessionId: string): void {
  try { localStorage.setItem(SESSION_STORAGE_KEY, sessionId); } catch { /* not remembered */ }
  rememberSession(sessionId);
}

const KNOWN_SESSIONS_KEY = 'alchemist_known_sessions';
/** How many session IDs this browser remembers, newest first. */
export const KNOWN_SESSIONS_LIMIT = 50;

/**
 * Sessions this browser has worked on, newest first. A recovery backup is
 * offered only for one of these, so a fresh profile never sees another
 * client's backup on a shared server. Clear keeps the list: it forgets the
 * current session, not the history.
 */
export function knownSessionIds(): string[] {
  try {
    const parsed: unknown = JSON.parse(localStorage.getItem(KNOWN_SESSIONS_KEY) ?? '[]');
    return Array.isArray(parsed) ? parsed.filter((id): id is string => typeof id === 'string') : [];
  } catch {
    return [];
  }
}

function rememberSession(sessionId: string): void {
  const ids = [sessionId, ...knownSessionIds().filter((id) => id !== sessionId)].slice(0, KNOWN_SESSIONS_LIMIT);
  try { localStorage.setItem(KNOWN_SESSIONS_KEY, JSON.stringify(ids)); } catch { /* not remembered */ }
}

/**
 * Clear stored session ID
 */
export function clearStoredSessionId(): void {
  try { localStorage.removeItem(SESSION_STORAGE_KEY); } catch { /* nothing stored */ }
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
