/** Shape of GET /experiments/staged as the app reads it on session load. */
export interface StagedListResponse {
  experiments?: Record<string, any>[];
  reason?: string | null;
  reasons?: (string | null)[] | null;
  ids?: string[];
}

/**
 * Turn a staged-list response into pending-suggestion rows for the Add Point
 * dialog. Each row keeps its plot keys (WholePlot, RunOrder, ...) and gets its
 * own reason (`reasons[i]`), falling back to the scalar `reason`, then "Staged".
 */
export function tagRestoredStaged(data: StagedListResponse): Record<string, any>[] {
  const reasons = data.reasons ?? [];
  const ids = data.ids ?? [];
  return (data.experiments ?? []).map((exp, i) => ({
    ...exp,
    _reason: reasons[i] || data.reason || 'Staged', // UI-only metadata for dialog auto-fill
    _queueItemId: ids[i],
  }));
}
