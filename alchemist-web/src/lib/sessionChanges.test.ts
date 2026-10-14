import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  sessionChangedByRequest,
  markSessionChanged,
  markSessionSaved,
  isSessionChanged,
  subscribeSessionChanges,
  observeFetchForChanges,
  resetSessionChangesForTests,
} from './sessionChanges';

const ID = '77491db8-ce1a-4a91-ad70-34c120a6fea4';
const url = (path: string) => `http://localhost:8000/api/v1/sessions/${ID}${path}`;

describe('sessionChangedByRequest', () => {
  it('counts writes that change the session', () => {
    expect(sessionChangedByRequest('POST', url('/experiments'))).toBe(ID);
    expect(sessionChangedByRequest('post', url('/experiments/record-batch'))).toBe(ID);
    expect(sessionChangedByRequest('POST', url('/model/train'))).toBe(ID);
    expect(sessionChangedByRequest('POST', url('/acquisition/suggest'))).toBe(ID);
    expect(sessionChangedByRequest('PUT', url('/variables/Temperature'))).toBe(ID);
    expect(sessionChangedByRequest('DELETE', url('/variables/Temperature'))).toBe(ID);
    expect(sessionChangedByRequest('PATCH', `/api/v1/sessions/${ID}/metadata`)).toBe(ID);
    expect(sessionChangedByRequest('POST', url('/variables/load'))).toBe(ID);
  });

  it('ignores reads', () => {
    expect(sessionChangedByRequest('GET', url('/experiments'))).toBeNull();
  });

  it('ignores writes that leave the session file unchanged', () => {
    for (const path of [
      '/save', '/backup', '/lock', '/control', '/initial-design', '/optimal-design',
      '/optimal-design/info', '/split-plot-design', '/experiments/preview',
      '/model/predict', '/visualizations/contour', '/visualizations/surface',
      '/acquisition/find-optimum',
    ]) {
      expect(sessionChangedByRequest('POST', url(path)), path).toBeNull();
    }
    expect(sessionChangedByRequest('DELETE', url('/backup'))).toBeNull();
  });

  it('ignores whole-session create, import and delete', () => {
    expect(sessionChangedByRequest('POST', 'http://localhost:8000/api/v1/sessions')).toBeNull();
    expect(sessionChangedByRequest('POST', 'http://localhost:8000/api/v1/sessions/import')).toBeNull();
    expect(sessionChangedByRequest('DELETE', `http://localhost:8000/api/v1/sessions/${ID}`)).toBeNull();
  });
});

describe('changed/saved state', () => {
  beforeEach(() => resetSessionChangesForTests());

  it('tracks changes per session until saved', () => {
    expect(isSessionChanged(ID)).toBe(false);
    markSessionChanged(ID);
    expect(isSessionChanged(ID)).toBe(true);
    expect(isSessionChanged('other')).toBe(false);
    markSessionSaved(ID);
    expect(isSessionChanged(ID)).toBe(false);
  });

  it('notifies subscribers', () => {
    const listener = vi.fn();
    const unsubscribe = subscribeSessionChanges(listener);
    markSessionChanged(ID);
    markSessionSaved(ID);
    unsubscribe();
    markSessionChanged(ID);
    expect(listener).toHaveBeenCalledTimes(2);
  });
});

describe('observeFetchForChanges', () => {
  beforeEach(() => resetSessionChangesForTests());

  it('marks the session changed after a successful write', async () => {
    const fetchFn = vi.fn().mockResolvedValue(new Response('{}', { status: 200 }));
    const observed = observeFetchForChanges(fetchFn);
    await observed(url('/experiments'), { method: 'POST' });
    expect(isSessionChanged(ID)).toBe(true);
  });

  it('leaves the session unchanged when the write fails', async () => {
    const fetchFn = vi.fn().mockResolvedValue(new Response('{}', { status: 400 }));
    await observeFetchForChanges(fetchFn)(url('/experiments'), { method: 'POST' });
    expect(isSessionChanged(ID)).toBe(false);
  });

  it('reads the method from a Request object', async () => {
    const fetchFn = vi.fn().mockResolvedValue(new Response('{}', { status: 200 }));
    await observeFetchForChanges(fetchFn)(new Request(url('/model/train'), { method: 'POST' }));
    expect(isSessionChanged(ID)).toBe(true);
  });
});
