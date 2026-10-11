/**
 * State and save path for the Add Point dialog, shared by the panels that
 * open it. Entries are keyed by a stable suggestion id and survive closing
 * and reopening the dialog; a save sends every entered point in one request.
 */
import { useCallback, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { toast } from 'sonner';
import { recordResults } from '../../components/api';
import { invalidateAfterExperiments, invalidateAfterModelTrained } from '../../lib/invalidateSessionData';
import { suggestionKey } from './addPointEntries';
import type { BatchPlan, EntriesMap } from './addPointEntries';

export function useAddPointBatch(
  sessionId: string,
  pending: any[],
  onPendingChange?: (remaining: any[]) => void,
) {
  const queryClient = useQueryClient();
  const [entries, setEntries] = useState<EntriesMap>({});

  const onEntriesChange = useCallback((update: (prev: EntriesMap) => EntriesMap) => {
    setEntries(update);
  }, []);

  /**
   * Record the plan's points. Resolves with the points still pending; throws
   * (entries untouched) if the request fails.
   */
  const save = useCallback(
    async (plan: BatchPlan, options: { retrain: boolean }): Promise<any[]> => {
      const resp = await recordResults(sessionId, plan.payload, options.retrain);

      invalidateAfterExperiments(queryClient, sessionId);
      queryClient.invalidateQueries({ queryKey: ['experiments-queue', sessionId] });
      if (resp.model_trained) invalidateAfterModelTrained(queryClient, sessionId);

      const saved = new Set(plan.savedKeys);
      const remaining = pending.filter((s) => !saved.has(suggestionKey(s)));
      onPendingChange?.(remaining);
      setEntries((prev) => {
        const next = { ...prev };
        for (const k of plan.savedKeys) delete next[k];
        return next;
      });

      const n = resp.n_added;
      const parts = [`Recorded ${n} point${n === 1 ? '' : 's'}`];
      if (resp.model_trained) parts.push('model retrained once');
      if (remaining.length > 0) parts.push(`${remaining.length} still pending`);
      toast.success(parts.join(', '));
      if (options.retrain && !resp.model_trained) {
        toast.warning('Model not retrained', {
          description:
            resp.n_experiments < 5
              ? `Retraining needs at least 5 experiments (have ${resp.n_experiments}).`
              : 'Retraining failed on the server; see the API log, then train from the Model tab.',
        });
      }
      return remaining;
    },
    [sessionId, pending, onPendingChange, queryClient],
  );

  return { entries, onEntriesChange, save };
}
