import { describe, it, expect, vi, beforeEach } from 'vitest';
import { AxiosError, AxiosHeaders } from 'axios';
import {
  chooseStartupSession,
  lookupSession,
  recoveryToOffer,
  needsRecoveryBackup,
  type RecoveryEntry,
} from './sessionResume';
import * as sessionAPI from '../api/endpoints/sessions';
import { getStoredSessionId, storeSessionId, clearStoredSessionId } from '../hooks/api/useSessions';

const A = '11111111-1111-4111-8111-111111111111';
const B = '22222222-2222-4222-8222-222222222222';

const found = vi.fn(async () => 'found' as const);

describe('chooseStartupSession', () => {
  beforeEach(() => found.mockClear());

  it('opens the session named in the URL first', async () => {
    const choice = await chooseStartupSession({ search: `?session=${A}`, storedId: B, lookup: found });
    expect(choice).toEqual({ sessionId: A, source: 'url', forgetStored: false });
    // The URL session is not looked up here: its own error stays visible.
    expect(found).not.toHaveBeenCalled();
  });

  it('reopens the stored session when the server still has it', async () => {
    const choice = await chooseStartupSession({ search: '', storedId: B, lookup: found });
    expect(choice).toEqual({ sessionId: B, source: 'stored', forgetStored: false });
    expect(found).toHaveBeenCalledWith(B);
  });

  it('forgets a stored session the server no longer has, and opens nothing', async () => {
    const choice = await chooseStartupSession({ search: '', storedId: B, lookup: async () => 'missing' });
    expect(choice).toEqual({ sessionId: null, source: null, forgetStored: true });
  });

  it('keeps a stored session it could not check (server error), and opens nothing', async () => {
    const choice = await chooseStartupSession({ search: '', storedId: B, lookup: async () => 'unavailable' });
    expect(choice).toEqual({ sessionId: null, source: null, forgetStored: false });
  });

  it('opens nothing on a first visit', async () => {
    const choice = await chooseStartupSession({ search: '?mode=monitor', storedId: null, lookup: found });
    expect(choice).toEqual({ sessionId: null, source: null, forgetStored: false });
    expect(found).not.toHaveBeenCalled();
  });
});

describe('lookupSession', () => {
  const httpError = (status: number) =>
    new AxiosError('failed', String(status), undefined, undefined, {
      status, statusText: '', headers: {}, config: { headers: new AxiosHeaders() }, data: {},
    });

  it('is found when the server answers', async () => {
    vi.spyOn(sessionAPI, 'getSession').mockResolvedValueOnce({} as any);
    expect(await lookupSession(A)).toBe('found');
  });

  it('is missing on a 404', async () => {
    vi.spyOn(sessionAPI, 'getSession').mockRejectedValueOnce(httpError(404));
    expect(await lookupSession(A)).toBe('missing');
  });

  it('is unavailable on any other failure', async () => {
    vi.spyOn(sessionAPI, 'getSession').mockRejectedValueOnce(httpError(503));
    expect(await lookupSession(A)).toBe('unavailable');
    vi.spyOn(sessionAPI, 'getSession').mockRejectedValueOnce(new Error('Network Error'));
    expect(await lookupSession(A)).toBe('unavailable');
  });
});

describe('recoveryToOffer', () => {
  const entry = (session_id: string): RecoveryEntry => ({
    session_id, session_name: 'x', backup_time: '2026-10-15T10:00:00', n_variables: 1, n_experiments: 9, model_trained: false,
  });

  it('offers the newest backup', () => {
    expect(recoveryToOffer([entry(A), entry(B)], null)?.session_id).toBe(A);
  });

  it('never offers the session that is open', () => {
    expect(recoveryToOffer([entry(A), entry(B)], A)?.session_id).toBe(B);
    expect(recoveryToOffer([entry(A)], A)).toBeNull();
  });

  it('offers nothing when there is nothing', () => {
    expect(recoveryToOffer([], null)).toBeNull();
    expect(recoveryToOffer(undefined, null)).toBeNull();
  });
});

describe('needsRecoveryBackup', () => {
  const withData = { variable_count: 2, experiment_count: 12 };

  it('backs up a session with unsaved changes', () => {
    expect(needsRecoveryBackup(withData, true)).toBe(true);
  });

  it('does not back up a session that matches its file', () => {
    // A backup taken after Save would be offered later as "unsaved work".
    expect(needsRecoveryBackup(withData, false)).toBe(false);
  });

  it('does not back up an empty session', () => {
    expect(needsRecoveryBackup({ variable_count: 0, experiment_count: 0 }, true)).toBe(false);
    expect(needsRecoveryBackup(undefined, true)).toBe(false);
  });
});

describe('stored session id', () => {
  beforeEach(() => localStorage.clear());

  it('round-trips through localStorage', () => {
    storeSessionId(A);
    expect(getStoredSessionId()).toBe(A);
    clearStoredSessionId();
    expect(getStoredSessionId()).toBeNull();
  });

  it('tolerates storage that throws (blocked site data)', () => {
    const spy = vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('denied'); });
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('denied'); });
    vi.spyOn(Storage.prototype, 'removeItem').mockImplementation(() => { throw new Error('denied'); });
    expect(getStoredSessionId()).toBeNull();
    expect(() => storeSessionId(A)).not.toThrow();
    expect(() => clearStoredSessionId()).not.toThrow();
    spy.mockRestore();
    vi.restoreAllMocks();
  });
});
