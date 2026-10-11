// api.ts - small client helper for session operations

export async function addExperiment(sessionId: string, experiment: any, auto_train=false) {
  const body = { ...experiment }
  // The API expects a JSON body matching AddExperimentRequest. Ensure keys are correct.
  const res = await fetch(`/api/v1/sessions/${sessionId}/experiments?auto_train=${auto_train}`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
  })
  if (!res.ok) throw new Error(`Add experiment failed: ${res.statusText}`)
  return res.json()
}

export async function createSession() {
  const res = await fetch('/api/v1/sessions', { method: 'POST' })
  if (!res.ok) throw new Error('Create session failed')
  return res.json()
}

export async function downloadSession(sessionId: string) {
  const res = await fetch(`/api/v1/sessions/${sessionId}/download`)
  if (!res.ok) throw new Error('Download failed')
  return res.blob()
}

export async function completeQueueItem(
  sessionId: string,
  itemId: string,
  actualInputs: Record<string, any>,
  output: number,
  noise?: number,
  retrain = false,
) {
  const body: any = { outputs: [output], actual_inputs: actualInputs };
  if (noise !== undefined) body.noise = [noise];
  const res = await fetch(
    `/api/v1/sessions/${sessionId}/experiments/queue/${itemId}/complete?auto_train=${retrain}`,
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) },
  );
  if (!res.ok) throw new Error(`Complete failed: ${res.statusText}`);
  return res.json();
}

export interface RecordResultsResponse {
  message: string;
  n_added: number;
  n_experiments: number;
  completed_item_ids: string[];
  model_trained: boolean;
  training_metrics?: Record<string, any> | null;
}

/**
 * Record several results in ONE request. Entries with an `item_id` complete
 * that staged queue item (with the actual conditions); the rest are added as
 * experiments. The backend validates every entry before writing any, and
 * retrains at most once, after all rows are in, when `retrain` is true.
 */
export async function recordResults(
  sessionId: string,
  entries: Array<Record<string, any>>,
  retrain = false,
): Promise<RecordResultsResponse> {
  const res = await fetch(
    `/api/v1/sessions/${sessionId}/experiments/record-batch?auto_train=${retrain}`,
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ entries }) },
  );
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* keep statusText */
    }
    throw new Error(`Save failed (${res.status}): ${detail}`);
  }
  return res.json();
}
