import { VariablesPanel } from '../../../features/variables/VariablesPanel';
import { GPRPanel } from '../../../features/models/GPRPanel';
import { AcquisitionPanel } from '../../../features/acquisition/AcquisitionPanel';
import { InitialDesignPanel } from '../../../features/experiments/InitialDesignPanel';

export function ConfigTab({ sessionId, isRunning }: { sessionId: string; isRunning: boolean }) {
  return (
    <div className="space-y-4">
      {isRunning && (
        <div className="rounded border border-amber-300 bg-amber-50 p-2 text-sm text-amber-900">
          Applies to the next suggestion the controller requests — ALchemist does not initiate cycles.
        </div>
      )}
      <VariablesPanel sessionId={sessionId} />
      <GPRPanel sessionId={sessionId} />
      <AcquisitionPanel sessionId={sessionId} />
      <InitialDesignPanel sessionId={sessionId} />
    </div>
  );
}
