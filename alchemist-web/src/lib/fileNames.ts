/**
 * Filenames for files the web app saves (session, variables, audit log).
 *
 * Files are named after the session title so they are recognisable on disk.
 * An untitled session falls back to its id, so two untitled sessions don't
 * both save as "Untitled Session.json".
 */

/** The name `SessionMetadata.create()` gives a session nobody has named. */
const DEFAULT_SESSION_NAME = 'Untitled Session';
const MAX_STEM_LENGTH = 120;
// Characters Windows forbids in filenames (a superset of what macOS forbids),
// plus control characters.
// eslint-disable-next-line no-control-regex
const FORBIDDEN = /[<>:"/\\|?*\u0000-\u001f]/g;
const WINDOWS_RESERVED = /^(con|prn|aux|nul|com[1-9]|lpt[1-9])$/i;

/** True when a session has no title of its own (blank, or the core's default). */
export function isUntitledSession(name: string | null | undefined): boolean {
  const title = (name ?? '').trim();
  return !title || title === DEFAULT_SESSION_NAME;
}

/** The name shown for a session in the header: its title, else "Untitled Session". */
export function sessionDisplayName(metadata: { name?: string | null } | null | undefined): string {
  const title = (metadata?.name ?? '').trim();
  return isUntitledSession(title) ? DEFAULT_SESSION_NAME : title;
}

/** Filename (without extension) for files belonging to a session. */
export function sessionFileStem(name: string | null | undefined, sessionId: string): string {
  const fallback = `alchemist_session_${sessionId.slice(0, 8)}`;
  if (isUntitledSession(name)) return fallback;
  const title = (name ?? '').trim();

  let stem = title
    .replace(/\s+/g, ' ')
    .replace(FORBIDDEN, '_')
    .slice(0, MAX_STEM_LENGTH)
    .replace(/[. ]+$/, '');
  if (!stem) return fallback;
  if (WINDOWS_RESERVED.test(stem)) stem += '_';
  return stem;
}
