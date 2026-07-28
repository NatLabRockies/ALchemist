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
) {
  const body: any = { outputs: [output], actual_inputs: actualInputs };
  if (noise !== undefined) body.noise = [noise];
  const res = await fetch(
    `/api/v1/sessions/${sessionId}/experiments/queue/${itemId}/complete`,
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) },
  );
  if (!res.ok) throw new Error(`Complete failed: ${res.statusText}`);
  return res.json();
}
