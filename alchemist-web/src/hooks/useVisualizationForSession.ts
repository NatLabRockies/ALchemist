import { useEffect } from 'react';
import { useVisualization } from '../providers/VisualizationProvider';

/**
 * The visualization context for the app's current session: the panel is bound
 * to the session it was opened in, so it closes when the app moves to another
 * (Clear, New Session, Join) instead of showing the old session's plots there.
 */
export function useVisualizationForSession(sessionId: string | null) {
  const context = useVisualization();
  const { sessionId: vizSessionId, closeVisualization } = context;
  useEffect(() => {
    if (vizSessionId && vizSessionId !== sessionId) closeVisualization();
  }, [sessionId, vizSessionId, closeVisualization]);
  return context;
}
