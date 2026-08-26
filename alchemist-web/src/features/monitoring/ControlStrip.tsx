import { useEffect, useState } from 'react';
import { useControl, useRequestControl } from '../../hooks/api/useControl';

/** Older than this and the reported state is not trustworthy. 3x the
 *  controller's 10 s heartbeat, so an ordinary late beat does not flap. */
export const STALE_AFTER_S = 30;

function ageSeconds(iso: string | null, now: number): number | null {
  if (!iso) return null;
  const t = Date.parse(iso);
  return Number.isNaN(t) ? null : Math.max(0, Math.round((now - t) / 1000));
}

export function ControlStrip({ sessionId }: { sessionId: string }) {
  const { data } = useControl(sessionId);
  const request = useRequestControl(sessionId);

  // Local 1 s tick. The age must advance on its own: if the controller goes
  // quiet, silence has to LOOK like silence rather than a frozen "running".
  // This is why the strip needs no refetchInterval.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  if (!data) return null;

  const heard = ageSeconds(data.reported_at, now);
  const stale = heard === null || heard > STALE_AFTER_S;
  const asked = ageSeconds(data.requested_at, now);

  // Only `reported` may put a state on screen. `requested` never can.
  const pausedNow = !stale && data.reported === 'paused';
  const unacknowledged =
    !stale &&
    ((data.requested === 'pause' && data.reported !== 'paused') ||
     (data.requested === 'run' && data.reported === 'paused'));

  return (
    <div className="flex items-center gap-2 text-xs">
      <button
        onClick={() => request.mutate(pausedNow ? 'run' : 'pause')}
        disabled={request.isPending}
        className="px-2 py-1 rounded border hover:bg-muted disabled:opacity-50"
      >
        {pausedNow ? 'Resume' : 'Pause'}
      </button>

      {stale ? (
        <span className="text-amber-600">
          ⚠ no word from controller
          {heard === null ? '' : ` for ${heard}s`} — state unknown
        </span>
      ) : unacknowledged ? (
        // The reported state stays on screen here too. A pending request is
        // the one moment an operator most needs to know what the controller
        // is STILL doing -- dropping it would leave the strip showing only
        // what was asked for, which is the failure this whole split exists
        // to prevent.
        <span className="text-amber-600">
          controller: <span className="font-medium">{data.reported}</span>
          {' · ⚠ '}{data.requested === 'pause' ? 'pause' : 'resume'} requested
          {asked === null ? '' : ` ${asked}s ago`} — not yet acknowledged
        </span>
      ) : (
        <span className="text-muted-foreground">
          controller: <span className="font-medium">{data.reported}</span>
          {` · heard ${heard}s ago`}
          {data.detail ? ` · ${data.detail}` : ''}
        </span>
      )}
    </div>
  );
}
