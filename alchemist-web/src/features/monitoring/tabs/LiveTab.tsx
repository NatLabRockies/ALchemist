import { QueueTimeline } from '../QueueTimeline';
import { ObjectiveTrace } from '../ObjectiveTrace';
import { MetricsPlot } from '../../../components/visualizations/MetricsPlot';
import { ParityPlot } from '../../../components/visualizations/ParityPlot';

interface LiveTabProps {
  sessionId: string;
  objectiveLabel: string;
  goal: 'maximize' | 'minimize';
}

export function LiveTab({ sessionId, objectiveLabel, goal }: LiveTabProps) {
  return (
    <div className="space-y-4">
      <section>
        <h3 className="text-sm font-semibold mb-2">Work Queue</h3>
        <QueueTimeline sessionId={sessionId} />
      </section>
      <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <MetricsPlot sessionId={sessionId} selectedMetric="R2" cvSplits={5} />
        <ParityPlot sessionId={sessionId} useCalibrated={false} sigmaMultiplier="2" />
      </section>
      <section>
        <h3 className="text-sm font-semibold mb-2">Objective so far</h3>
        <ObjectiveTrace sessionId={sessionId} goal={goal} objectiveLabel={objectiveLabel} />
      </section>
    </div>
  );
}
