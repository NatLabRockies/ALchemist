import { LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { usePlotTheme } from '../../components/visualizations/plotTheme';
import { useExperimentQueue } from '../../hooks/api/useQueue';
import type { QueueItem } from '../../api/endpoints/queue';

export interface TracePoint { index: number; value: number; best: number; }

/** Pure: cumulative-best objective trace over completed items. */
export function computeTrace(items: QueueItem[], goal: 'maximize' | 'minimize'): TracePoint[] {
  const done = items
    .filter((i) => i.status === 'done' && typeof i.output === 'number')
    .sort((a, b) => (a.completed_at ?? '').localeCompare(b.completed_at ?? ''));
  const trace: TracePoint[] = [];
  let best: number | null = null;
  done.forEach((i, idx) => {
    const value = i.output as number;
    if (best === null) best = value;
    else best = goal === 'maximize' ? Math.max(best, value) : Math.min(best, value);
    trace.push({ index: idx + 1, value, best });
  });
  return trace;
}

interface ObjectiveTraceProps {
  sessionId: string;
  goal?: 'maximize' | 'minimize';
  objectiveLabel?: string;
}

export function ObjectiveTrace({ sessionId, goal = 'maximize', objectiveLabel = 'Objective' }: ObjectiveTraceProps) {
  const { data } = useExperimentQueue(sessionId);
  const plot = usePlotTheme();
  const trace = computeTrace(data?.items ?? [], goal);

  if (trace.length === 0) {
    return <div className="text-sm text-muted-foreground py-6 text-center">No completed experiments yet.</div>;
  }

  return (
    <div>
      <div className="text-xs text-muted-foreground mb-1">{objectiveLabel}</div>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={trace}>
          <CartesianGrid {...plot.grid} />
          <XAxis dataKey="index" {...plot.axis} />
          <YAxis {...plot.axis} />
          <Tooltip {...plot.tooltip} />
          <Legend {...plot.legend} />
          {/* Two series in one colour were hard to tell apart: the blue points are the runs, the neutral step line is the best so far. */}
          <Line
            type="monotone"
            dataKey="value"
            name="per-experiment"
            stroke="#3b82f6"
            strokeWidth={1.5}
            dot={{ r: 3, fill: plot.colors.pointFill, stroke: '#3b82f6', strokeWidth: 2 }}
          />
          <Line type="stepAfter" dataKey="best" name="best so far" stroke={plot.colors.text} strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
