import { useMemo, useState } from 'react';
import { ConfigTab } from './tabs/ConfigTab';
import { LiveTab } from './tabs/LiveTab';
import { HistoryTab } from './tabs/HistoryTab';
import { useExperimentQueue } from '../../hooks/api/useQueue';
import { useObjectiveMetadata } from '../../hooks/api/useObjectiveMetadata';

type TabKey = 'config' | 'live' | 'history';
const TABS: { key: TabKey; label: string }[] = [
  { key: 'config', label: 'Config' },
  { key: 'live', label: 'Live' },
  { key: 'history', label: 'History' },
];

function initialTab(): TabKey {
  const t = new URLSearchParams(window.location.search).get('tab');
  return t === 'config' || t === 'history' ? t : 'live';
}

export function LiveMonitor({ sessionId }: { sessionId: string }) {
  const [tab, setTab] = useState<TabKey>(initialTab);
  const { data: queue } = useExperimentQueue(sessionId);
  const isRunning = (queue?.n_running ?? 0) > 0;

  // Opaque objective label/unit for display (never interpreted).
  const objMeta = useObjectiveMetadata(sessionId);
  const objectiveLabel = useMemo(() => {
    const map = objMeta.data?.metadata ?? {};
    const first = Object.values(map)[0] as { label?: string; unit?: string } | undefined;
    if (!first?.label) return 'Objective';
    return first.unit ? `${first.label} (${first.unit})` : first.label;
  }, [objMeta.data]);

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between border-b px-4 py-2">
        <div className="flex gap-1" role="tablist">
          {TABS.map((t) => (
            <button
              key={t.key}
              role="tab"
              aria-selected={tab === t.key}
              onClick={() => setTab(t.key)}
              className={`px-3 py-1 text-sm rounded ${tab === t.key ? 'bg-primary text-primary-foreground' : 'hover:bg-muted'}`}
            >
              {t.label}
            </button>
          ))}
        </div>
        <div className="text-xs text-muted-foreground">
          objective: <span className="font-medium">{objectiveLabel}</span>
          {isRunning ? ' · running' : ' · idle'}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4">
        {tab === 'config' && <ConfigTab sessionId={sessionId} isRunning={isRunning} />}
        {tab === 'live' && <LiveTab sessionId={sessionId} objectiveLabel={objectiveLabel} goal="maximize" />}
        {tab === 'history' && <HistoryTab sessionId={sessionId} />}
      </div>
    </div>
  );
}
