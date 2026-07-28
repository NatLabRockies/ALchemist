import { useExperimentQueue } from '../../hooks/api/useQueue';
import type { QueueItem, QueueStatus } from '../../api/endpoints/queue';

const STATUS_LABEL: Record<QueueStatus, string> = {
  pending: 'Pending',
  running: 'Running',
  done: 'Done',
  failed: 'Failed',
};

function formatInputs(inputs: Record<string, unknown>): string {
  return Object.entries(inputs)
    .map(([k, v]) => `${k}=${typeof v === 'number' ? v : String(v)}`)
    .join(', ');
}

export function QueueTimeline({ sessionId }: { sessionId: string }) {
  const { data, isLoading } = useExperimentQueue(sessionId);

  if (isLoading) return <div className="text-sm text-muted-foreground">Loading queue…</div>;

  const items: QueueItem[] = data?.items ?? [];
  if (items.length === 0) {
    return (
      <div className="text-sm text-muted-foreground py-6 text-center">
        No items yet — awaiting controller.
      </div>
    );
  }

  return (
    <ul className="space-y-1">
      {items.map((item) => (
        <li key={item.id} className="flex flex-col gap-0.5 rounded border p-2 text-sm">
          <div className="flex items-center justify-between">
            <span className="font-mono">{formatInputs(item.inputs)}</span>
            <span data-status={item.status}>{STATUS_LABEL[item.status]}</span>
          </div>
          {item.reason && <span className="text-xs text-muted-foreground">{item.reason}</span>}
          {item.status === 'done' && item.output != null && (
            <span className="text-xs">objective: {String(item.output)}</span>
          )}
          {item.status === 'failed' && item.error && (
            <span className="text-xs text-red-600">error: {item.error}</span>
          )}
        </li>
      ))}
    </ul>
  );
}
