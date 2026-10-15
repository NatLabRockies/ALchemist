/**
 * Which session the app opens when it starts, and which recovery backup (if
 * any) it offers.
 *
 * The app reopens the session this browser last worked on, so a reload or a
 * relaunch of the server or desktop app comes back to the same session ID, and
 * with it the file its Save writes to (file links are kept per session ID; see
 * fileLinks.ts). The ID lives in this browser's localStorage, so on a shared
 * server every browser reopens its own session.
 */
import { isAxiosError } from 'axios';
import * as sessionAPI from '../api/endpoints/sessions';

/** 'unavailable': the server could not say (unreadable file, network error). */
export type SessionLookup = 'found' | 'missing' | 'unavailable';

export interface StartupChoice {
  sessionId: string | null;
  source: 'url' | 'stored' | null;
  /** The stored ID names a session the server no longer has. */
  forgetStored: boolean;
}

/**
 * `?session=` in the URL wins and is opened as is (its own errors stay
 * visible). Otherwise the stored session is reopened if the server still has
 * it; one it no longer has is forgotten, and one it cannot check right now is
 * kept for next time.
 */
export async function chooseStartupSession({
  search,
  storedId,
  lookup,
}: {
  search: string;
  storedId: string | null;
  lookup: (sessionId: string) => Promise<SessionLookup>;
}): Promise<StartupChoice> {
  const urlSessionId = new URLSearchParams(search).get('session');
  if (urlSessionId) return { sessionId: urlSessionId, source: 'url', forgetStored: false };
  if (!storedId) return { sessionId: null, source: null, forgetStored: false };
  const result = await lookup(storedId);
  if (result === 'found') return { sessionId: storedId, source: 'stored', forgetStored: false };
  return { sessionId: null, source: null, forgetStored: result === 'missing' };
}

export async function lookupSession(sessionId: string): Promise<SessionLookup> {
  try {
    await sessionAPI.getSession(sessionId);
    return 'found';
  } catch (error) {
    return isAxiosError(error) && error.response?.status === 404 ? 'missing' : 'unavailable';
  }
}

/** One row of GET /recovery/list, newest first. */
export interface RecoveryEntry {
  session_id: string;
  session_name: string;
  backup_time: string;
  n_variables: number;
  n_experiments: number;
  model_trained: boolean;
}

/** The newest backup, skipping the session that is open (its work is already here). */
export function recoveryToOffer(
  recoveries: RecoveryEntry[] | undefined,
  openSessionId: string | null,
): RecoveryEntry | null {
  return recoveries?.find(entry => entry.session_id !== openSessionId) ?? null;
}

/**
 * Back up only a session with data that differs from its saved file. A backup
 * taken after Save would come back later as "unsaved work".
 */
export function needsRecoveryBackup(
  session: { variable_count: number; experiment_count: number } | undefined,
  hasUnsavedChanges: boolean,
): boolean {
  if (!session || (session.variable_count === 0 && session.experiment_count === 0)) return false;
  return hasUnsavedChanges;
}
