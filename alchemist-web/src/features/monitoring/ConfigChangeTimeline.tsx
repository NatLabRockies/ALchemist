import { useConfigChanges } from '../../hooks/api/useQueue';

export function ConfigChangeTimeline({ sessionId }: { sessionId: string }) {
  const { data, isLoading } = useConfigChanges(sessionId);
  if (isLoading) return <div className="text-sm text-muted-foreground">Loading provenance…</div>;

  const changes = data?.changes ?? [];
  if (changes.length === 0) {
    return <div className="text-sm text-muted-foreground py-4">No config changes recorded.</div>;
  }

  return (
    <ul className="space-y-1">
      {changes.map((c, i) => (
        <li key={`${c.timestamp}-${i}`} className="rounded border p-2 text-sm">
          <div className="flex items-center justify-between">
            <span className="font-medium">{c.component}</span>
            <span className="text-xs text-muted-foreground">
              {c.iteration != null ? `iteration ${c.iteration}` : ''} · {c.timestamp}
            </span>
          </div>
          <div className="text-xs font-mono">
            {JSON.stringify(c.old)} → {JSON.stringify(c.new)}
          </div>
        </li>
      ))}
    </ul>
  );
}
