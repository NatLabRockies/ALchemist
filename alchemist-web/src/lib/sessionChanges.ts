/**
 * Whether a session has changed since it was last saved to a file.
 *
 * The session's own `last_modified` cannot answer this: it only moves on locks,
 * saves and metadata edits, not when data is added. So the web app watches its
 * own writes instead. Every successful non-GET request to /sessions/{id}/...
 * marks that session changed, except requests that leave the session file as it
 * was (previews, design generators, predictions, plots, server-side save and
 * backup, lock and control). useSessionEvents also marks it changed when the live
 * connection reports a change made elsewhere (a controller adding data).
 *
 * Writes leave the app two ways, the axios client and plain fetch, so both are
 * observed (see installSessionChangeObservers), rather than asking each of the
 * call sites to remember.
 *
 * The state lives in sessionStorage so it survives a reload of the same tab.
 */
import type { AxiosInstance } from 'axios';

const SESSION_PATH = /\/sessions\/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})(\/[^?#]*)?/i;

/** Session sub-paths whose writes do not change what a saved session file contains. */
const UNCHANGING = [
  /^\/save$/,
  /^\/backup$/,
  /^\/lock$/,
  /^\/control$/,
  /^\/initial-design$/,
  /^\/optimal-design(\/info)?$/,
  /^\/split-plot-design$/,
  /^\/experiments\/preview$/,
  /^\/model\/predict$/,
  /^\/visualizations\//,
  /^\/acquisition\/find-optimum$/,
];

/** The session id a request changes, or null if it changes no session's file. */
export function sessionChangedByRequest(method: string | undefined, url: string): string | null {
  if (!method || method.toUpperCase() === 'GET' || method.toUpperCase() === 'HEAD') return null;
  const match = SESSION_PATH.exec(url);
  if (!match) return null;
  const path = (match[2] ?? '').replace(/\/$/, '');
  if (!path) return null; // the whole session: create, delete
  if (UNCHANGING.some(re => re.test(path))) return null;
  return match[1];
}

const STORAGE_KEY = 'alchemist-changed-sessions';
const listeners = new Set<() => void>();

function readChanged(): Set<string> {
  try {
    return new Set(JSON.parse(sessionStorage.getItem(STORAGE_KEY) ?? '[]') as string[]);
  } catch {
    return new Set();
  }
}

let changed = readChanged();

function commit(next: Set<string>) {
  changed = next;
  try { sessionStorage.setItem(STORAGE_KEY, JSON.stringify([...next])); } catch { /* in-memory copy stands */ }
  listeners.forEach(listener => listener());
}

export function markSessionChanged(sessionId: string): void {
  if (changed.has(sessionId)) return;
  commit(new Set(changed).add(sessionId));
}

export function markSessionSaved(sessionId: string): void {
  if (!changed.has(sessionId)) return;
  const next = new Set(changed);
  next.delete(sessionId);
  commit(next);
}

export function isSessionChanged(sessionId: string): boolean {
  return changed.has(sessionId);
}

export function subscribeSessionChanges(listener: () => void): () => void {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}

export function resetSessionChangesForTests(): void {
  changed = new Set();
  try { sessionStorage.removeItem(STORAGE_KEY); } catch { /* ignore */ }
}

/** Wrap a fetch so successful session writes mark the session changed. */
export function observeFetchForChanges(fetchFn: typeof fetch): typeof fetch {
  return async (input, init) => {
    const response = await fetchFn(input, init);
    if (response.ok) {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
      const method = init?.method ?? (input instanceof Request ? input.method : 'GET');
      const sessionId = sessionChangedByRequest(method, url);
      if (sessionId) markSessionChanged(sessionId);
    }
    return response;
  };
}

let installed = false;

/** Observe both ways the app writes: plain fetch and the axios client. Call once at startup. */
export function installSessionChangeObservers(apiClient: AxiosInstance): void {
  if (installed) return;
  installed = true;
  window.fetch = observeFetchForChanges(window.fetch.bind(window));
  apiClient.interceptors.response.use(response => {
    const { method, baseURL, url } = response.config;
    const full = url && /^https?:/.test(url) ? url : `${baseURL ?? ''}${url ?? ''}`;
    const sessionId = sessionChangedByRequest(method, full);
    if (sessionId) markSessionChanged(sessionId);
    return response;
  });
}
