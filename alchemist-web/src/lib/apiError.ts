/**
 * The message to show for a failed API call: the server's `detail` when it is
 * text, a readable line per field for a Pydantic 422 list, a plain sentence
 * when the server could not be reached, else the caller's fallback.
 */
import { isAxiosError } from 'axios';

export const SERVER_UNREACHABLE = 'Could not reach the server. Check that ALchemist is running and try again.';

export function apiErrorMessage(error: unknown, fallback: string): string {
  if (!isAxiosError(error)) return fallback;
  if (!error.response) return SERVER_UNREACHABLE;
  const data = error.response.data as { detail?: unknown; message?: unknown } | undefined;
  const detail = data?.detail ?? data?.message;
  if (typeof detail === 'string' && detail.trim()) return detail;
  if (Array.isArray(detail)) {
    const lines = detail.map((d) => {
      const item = d as { loc?: unknown; msg?: unknown };
      const loc = Array.isArray(item?.loc) ? item.loc.filter((x) => x !== 'body').join('.') : '';
      const msg = typeof item?.msg === 'string' ? item.msg : JSON.stringify(d);
      return loc ? `${loc}: ${msg}` : msg;
    });
    if (lines.length) return lines.join('; ');
  }
  return fallback;
}
