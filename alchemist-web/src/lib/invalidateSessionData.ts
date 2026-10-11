/**
 * Which cached queries go stale when a session's data or model changes.
 *
 * One place for the WebSocket handler (useSessionEvents) and for callers that
 * changed the data themselves, so a plot that is open stays live. The contour
 * cache never goes stale on its own (staleTime: Infinity), so it must be
 * invalidated here: it shows the model surface, the experimental points and
 * the next suggested point.
 */
import type { QueryClient } from '@tanstack/react-query';

/** New experiment rows (with or without a retrain): tables, summary, points on the contour. */
export function invalidateAfterExperiments(queryClient: QueryClient, sessionId: string) {
  queryClient.invalidateQueries({ queryKey: ['experiments', sessionId] });
  queryClient.invalidateQueries({ queryKey: ['experiments-summary', sessionId] });
  queryClient.invalidateQueries({ queryKey: ['session', sessionId] });
  queryClient.invalidateQueries({ queryKey: ['contour-data', sessionId] });
}

/** The model was (re)trained: every model plot, including the contour surface. */
export function invalidateAfterModelTrained(queryClient: QueryClient, sessionId: string) {
  queryClient.invalidateQueries({ queryKey: ['model-info', sessionId] });
  queryClient.invalidateQueries({ queryKey: ['session', sessionId] });
  queryClient.invalidateQueries({ queryKey: ['parity-data', sessionId] });
  queryClient.invalidateQueries({ queryKey: ['metrics-data', sessionId] });
  queryClient.invalidateQueries({ queryKey: ['calibration-curve-data', sessionId] });
  queryClient.invalidateQueries({ queryKey: ['qq-plot-data', sessionId] });
  queryClient.invalidateQueries({ queryKey: ['hyperparameters', sessionId] });
  queryClient.invalidateQueries({ queryKey: ['contour-data', sessionId] });
}

/** Queue items staged/completed, or a new suggestion: the queue and the contour's next point. */
export function invalidateAfterQueueOrSuggestion(queryClient: QueryClient, sessionId: string) {
  queryClient.invalidateQueries({ queryKey: ['experiments-queue', sessionId] });
  queryClient.invalidateQueries({ queryKey: ['contour-data', sessionId] });
}
