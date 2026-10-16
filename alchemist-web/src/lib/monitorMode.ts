/**
 * Monitor mode: the read-only live view of a campaign. It is entered when a
 * controller locks the session, or when the page was opened with
 * `?mode=monitor` (a wall display), in which case it stays on after unlock.
 */
export function monitorModeFromUrl(search: string): boolean {
  return new URLSearchParams(search).get('mode') === 'monitor';
}

/** The mode after a lock-status change. */
export function monitoringModeFor(locked: boolean, fromUrl: boolean): boolean {
  return locked || fromUrl;
}
