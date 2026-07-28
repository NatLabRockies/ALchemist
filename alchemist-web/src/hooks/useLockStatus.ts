import { useEffect, useState, useCallback, useRef } from 'react';
import { toast } from 'sonner';

interface LockStatus {
  locked: boolean;
  locked_by: string | null;
  locked_at: string | null;
}

interface UseLockStatusReturn {
  lockStatus: LockStatus | null;
  isConnected: boolean;
  error: Error | null;
}

/**
 * Hook to monitor session lock status via WebSocket for real-time updates.
 * 
 * When a session is locked by an external controller (e.g., Qt app),
 * this hook receives instant push notifications and can trigger automatic monitor mode.
 * 
 * @param sessionId - The session ID to monitor
 * @param _pollingInterval - Deprecated parameter (kept for backwards compatibility)
 * @param onLockStateChange - Callback when lock state changes
 */
export function useLockStatus(
  sessionId: string | null,
  _pollingInterval?: number, // Kept for backwards compatibility but unused
  onLockStateChange?: (locked: boolean, lockedBy: string | null) => void
): UseLockStatusReturn {
  const [lockStatus, setLockStatus] = useState<LockStatus | null>(null);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [error, setError] = useState<Error | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const previousLockStateRef = useRef<boolean | null>(null);

  const connect = useCallback(() => {
    if (!sessionId) return;

    // Clean up existing connection
    if (wsRef.current) {
      wsRef.current.close();
    }

    try {
      // Create WebSocket connection
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = window.location.hostname;
      const port = '8000'; // FastAPI backend port
      const ws = new WebSocket(`${protocol}//${host}:${port}/api/v1/ws/sessions/${sessionId}`);

      ws.onopen = () => {
        console.log('✓ WebSocket connected for session lock status');
        setIsConnected(true);
        setError(null);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.event === 'connected') {
            console.log('✓ WebSocket connection confirmed');
          } else if (data.event === 'lock_status_changed') {
            const newStatus: LockStatus = {
              locked: data.locked,
              locked_by: data.locked_by,
              locked_at: data.locked_at,
            };
            
            setLockStatus(newStatus);
            
            // Show toast notification only on state changes (not initial connection)
            const previousLockState = previousLockStateRef.current;
            if (previousLockState !== null && previousLockState !== data.locked) {
              if (data.locked) {
                toast.info(`External controller connected: ${data.locked_by}`, {
                  duration: 5000,
                });
              } else {
                toast.info('External controller disconnected - resuming interactive mode', {
                  duration: 3000,
                });
              }
            }
            
            // Update previous state
            previousLockStateRef.current = data.locked;
            
            // Trigger callback
            if (onLockStateChange) {
              onLockStateChange(data.locked, data.locked_by);
            }
          }
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
        }
      };

      ws.onerror = (event) => {
        console.error('WebSocket error:', event);
        setError(new Error('WebSocket connection error'));
      };

      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
        wsRef.current = null;
        
        // Attempt to reconnect after 5 seconds
        if (sessionId) {
          reconnectTimeoutRef.current = setTimeout(() => {
            console.log('Attempting to reconnect WebSocket...');
            connect();
          }, 5000);
        }
      };

      wsRef.current = ws;
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Failed to create WebSocket');
      setError(error);
      console.error('Error creating WebSocket:', error);
    }
  }, [sessionId, onLockStateChange]);

  // Connect on mount or when sessionId changes
  useEffect(() => {
    if (!sessionId) {
      // Clean up if no session
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      setIsConnected(false);
      setLockStatus(null);
      previousLockStateRef.current = null;
      return;
    }

    connect();

    // Cleanup on unmount
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [sessionId, connect]);

  return {
    lockStatus,
    isConnected,
    error,
  };
}
