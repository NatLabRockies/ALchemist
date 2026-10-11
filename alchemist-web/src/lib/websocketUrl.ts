/**
 * WebSocket URL for an API path.
 *
 * The built app is served by the API itself (api/main.py mounts
 * alchemist-web/dist), so the socket belongs on the page's own host and port.
 * It used to be hard-wired to port 8000, so an API on any other port (e.g. a
 * second instance on 8010) never got live events, or worse, subscribed to a
 * different server's session. The Vite dev server (5173) does not proxy
 * WebSockets, so dev keeps talking to the backend on 8000 directly.
 */
export function websocketUrl(path: string): string {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = import.meta.env.DEV ? `${window.location.hostname}:8000` : window.location.host;
  return `${protocol}//${host}${path}`;
}
