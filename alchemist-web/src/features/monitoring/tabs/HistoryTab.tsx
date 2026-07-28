import { ConfigChangeTimeline } from '../ConfigChangeTimeline';
import { CalibrationCurve } from '../../../components/visualizations/CalibrationCurve';
import { QQPlot } from '../../../components/visualizations/QQPlot';
import { HyperparametersDisplay } from '../../../components/visualizations/HyperparametersDisplay';

export function HistoryTab({ sessionId }: { sessionId: string }) {
  return (
    <div className="space-y-4">
      <section>
        <h3 className="text-sm font-semibold mb-2">Config change provenance</h3>
        <ConfigChangeTimeline sessionId={sessionId} />
      </section>
      <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <CalibrationCurve sessionId={sessionId} useCalibrated={false} />
        <QQPlot sessionId={sessionId} useCalibrated={false} />
      </section>
      <HyperparametersDisplay sessionId={sessionId} />
    </div>
  );
}
