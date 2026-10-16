/**
 * The server autosaves a session after every request that changes it and
 * reports a failed write in the `X-ALchemist-Saved: false` response header
 * (api/main.py). The change was applied in memory, so the request succeeded;
 * but the session file on the server is now stale. Warn once, not per request.
 */
import type { AxiosInstance } from 'axios';
import { toast } from './toast';

export const SAVED_HEADER = 'X-ALchemist-Saved';
export const SAVE_FAILED_MESSAGE = 'The server could not save the session file; use Save to keep a copy';
export const SAVE_WARNING_THROTTLE_MS = 60_000;

let lastWarnedAt = -Infinity;

/** Reads one response's header value; returns true when it warned. */
export function noteSavedHeader(value: string | null | undefined, now = Date.now()): boolean {
  if (value == null || String(value).toLowerCase() !== 'false') return false;
  if (now - lastWarnedAt < SAVE_WARNING_THROTTLE_MS) return false;
  lastWarnedAt = now;
  toast.warning(SAVE_FAILED_MESSAGE);
  return true;
}

export function resetSaveWarningsForTests(): void {
  lastWarnedAt = -Infinity;
}

/** Wrap a fetch so every response's save header is read. */
export function observeFetchForSaves(fetchFn: typeof fetch): typeof fetch {
  return async (input, init) => {
    const response = await fetchFn(input, init);
    noteSavedHeader(response.headers.get(SAVED_HEADER));
    return response;
  };
}

let installed = false;

/** Observe both ways the app writes: plain fetch and the axios client. Call once at startup. */
export function installSaveWarnings(apiClient: AxiosInstance): void {
  if (installed) return;
  installed = true;
  window.fetch = observeFetchForSaves(window.fetch.bind(window));
  apiClient.interceptors.response.use(response => {
    noteSavedHeader(response.headers?.[SAVED_HEADER.toLowerCase()]);
    return response;
  });
}
