import { dropRecorded } from './recordedSuggestions';
import type { VariableDetail } from '../api/types';

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

/**
 * Pending suggestions for a session on load: the staged queue when it has
 * rows, else the latest acquisition in the audit log (older sessions). The
 * audit fallback drops any suggestion already in the experiment table (W26);
 * if that table cannot be read it returns nothing rather than risk showing
 * recorded points as pending again. Points recorded with an edited actual value are
 * matched through their provenance record's suggested conditions. Always resolves; never throws.
 */
export async function restorePendingSuggestions(
  sessionId: string,
  fetchFn: typeof fetch = fetch,
): Promise<Record<string, unknown>[]> {
  try {
    // First try the staged experiments API (preferred)
    const stagedResponse = await fetchFn(`/api/v1/sessions/${sessionId}/experiments/staged`);
    if (stagedResponse.ok) {
      const stagedData = await stagedResponse.json();
      if (stagedData.experiments && stagedData.experiments.length > 0) {
        const taggedExperiments = tagRestoredStaged(stagedData);
        console.log(`✓ Restored ${taggedExperiments.length} staged experiments from API`);
        return taggedExperiments;
      }
    }

    // Fallback: check audit log for backward compatibility
    const auditResponse = await fetchFn(`/api/v1/sessions/${sessionId}/audit?entry_type=acquisition_locked`);
    if (auditResponse.ok) {
      const data = await auditResponse.json();
      if (data.entries && data.entries.length > 0) {
        // Get latest acquisition entry
        const latestAcq = data.entries[data.entries.length - 1];
        const suggestions = latestAcq.parameters?.suggestions || [];

        if (suggestions.length > 0) {
          // Tag suggestions with strategy for reason auto-fill
          const strategy = latestAcq.parameters?.strategy || 'Acquisition';
          const taggedSuggestions = suggestions.map((s: Record<string, unknown>) => ({
            ...s,
            _reason: strategy,
            Iteration: latestAcq.parameters?.iteration,
          }));

          // The log keeps a suggestion after it is recorded, so drop the ones
          // already in the experiment table.
          const experimentsResponse = await fetchFn(`/api/v1/sessions/${sessionId}/experiments`);
          if (!experimentsResponse.ok) {
            console.warn(
              `Skipped restoring ${suggestions.length} suggestions from audit log: ` +
                `experiment table unavailable (HTTP ${experimentsResponse.status})`,
            );
            return [];
          }
          const experiments: Record<string, unknown>[] = (await experimentsResponse.json()).experiments ?? [];

          // Variable types decide how a value is rounded; without them a
          // numeric pair is compared as a real.
          let variables: VariableDetail[] = [];
          try {
            const variablesResponse = await fetchFn(`/api/v1/sessions/${sessionId}/variables`);
            if (variablesResponse.ok) variables = (await variablesResponse.json()).variables ?? [];
          } catch {
            /* fall back to generic comparison */
          }

          // A point recorded with an edited actual value no longer matches the
          // table, but its provenance record keeps the conditions as suggested.
          // Treat those as recorded too; if unreadable, the table match stands.
          let suggestedRows: Record<string, unknown>[] = [];
          try {
            const provenanceResponse = await fetchFn(`/api/v1/sessions/${sessionId}/experiments/provenance`);
            if (provenanceResponse.ok) {
              const records: { suggested?: Record<string, unknown> | null }[] =
                (await provenanceResponse.json()).records ?? [];
              suggestedRows = records
                .map((r) => r.suggested)
                .filter((x): x is Record<string, unknown> => !!x && typeof x === 'object');
            }
          } catch {
            /* fall back to the table match only */
          }

          const pending = dropRecorded(
            taggedSuggestions,
            [...experiments, ...suggestedRows],
            undefined,
            variables,
          );
          console.log(
            `✓ Restored ${pending.length} pending suggestions from audit log (fallback), ` +
              `skipped ${taggedSuggestions.length - pending.length} already recorded`,
          );
          return pending;
        }
      }
    }
  } catch (e) {
    console.error('Failed to restore staged experiments:', e);
  }
  // No staged suggestions for this session — resolve to empty so stale
  // suggestions from a prior session don't persist.
  return [];
}
