/**
 * Visualizations Panel - Main container for model visualizations
 * Mimics desktop UI visualizations.py window structure
 */
import { useState } from 'react';
import { ParityPlot } from './ParityPlot';
import { MetricsPlot } from './MetricsPlot';
import { QQPlot } from './QQPlot';
import { CalibrationCurve } from './CalibrationCurve';
import { ContourPlot } from './ContourPlot';

interface VisualizationsPanelProps {
  sessionId: string;
  embedded?: boolean; // NEW: if true, render without modal wrapper
}

type PlotType = 'parity' | 'metrics' | 'qq' | 'calibration' | 'contour';
type MetricType = 'RMSE' | 'MAE' | 'MAPE' | 'R2';

export function VisualizationsPanel({ 
  sessionId, 
  embedded = false 
}: VisualizationsPanelProps) {
  const [activePlot, setActivePlot] = useState<PlotType>('parity');
  const [selectedMetric, setSelectedMetric] = useState<MetricType>('RMSE');
  const [sigmaMultiplier, setSigmaMultiplier] = useState<string>('1.96');
  const [useCalibrated, setUseCalibrated] = useState(false);

  const content = (
    <>
      {/* Controls Row 1 - Plot Selection */}
      <div className="p-3 border-b border-border bg-muted/30">
        <div className="flex flex-wrap gap-1.5">
            {/* Plot Type Buttons */}
            <button
              onClick={() => setActivePlot('parity')}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                activePlot === 'parity'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted hover:bg-muted/80'
              }`}
            >
              Plot Parity
            </button>
            <button
              onClick={() => setActivePlot('metrics')}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                activePlot === 'metrics'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted hover:bg-muted/80'
              }`}
            >
              Plot Metrics
            </button>
            <button
              onClick={() => setActivePlot('qq')}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                activePlot === 'qq'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted hover:bg-muted/80'
              }`}
            >
              Plot Q-Q
            </button>
            <button
              onClick={() => setActivePlot('calibration')}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                activePlot === 'calibration'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted hover:bg-muted/80'
              }`}
            >
              Plot Calibration
            </button>
            <button
              onClick={() => setActivePlot('contour')}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                activePlot === 'contour'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted hover:bg-muted/80'
              }`}
            >
              Plot Contour
            </button>

            {/* Metric Selector (for metrics plot) */}
            {activePlot === 'metrics' && (
              <select
                value={selectedMetric}
                onChange={(e) => setSelectedMetric(e.target.value as MetricType)}
                className="px-2.5 py-1.5 bg-background border border-input rounded-md text-xs"
              >
                <option value="RMSE">RMSE</option>
                <option value="MAE">MAE</option>
                <option value="MAPE">MAPE</option>
                <option value="R2">R²</option>
              </select>
            )}

            {/* Sigma Multiplier (for parity plot) */}
            {activePlot === 'parity' && (
              <>
                <span className="flex items-center text-xs text-muted-foreground ml-2">
                  Error bars:
                </span>
                <select
                  value={sigmaMultiplier}
                  onChange={(e) => setSigmaMultiplier(e.target.value)}
                  className="px-2.5 py-1.5 bg-background border border-input rounded-md text-xs"
                >
                  <option value="None">None</option>
                  <option value="1.0">1.0σ (68%)</option>
                  <option value="1.96">1.96σ (95%)</option>
                  <option value="2.0">2.0σ (95.4%)</option>
                  <option value="2.58">2.58σ (99%)</option>
                  <option value="3.0">3.0σ (99.7%)</option>
                </select>
              </>
            )}
          </div>
        </div>

        {/* Controls Row 2 - Calibration Toggle */}
        {(activePlot === 'parity' || activePlot === 'qq' || activePlot === 'calibration') && (
          <div className="px-3 py-1.5 border-b border-border bg-muted/20">
            <label className="flex items-center gap-2 text-xs">
              <input
                type="checkbox"
                checked={useCalibrated}
                onChange={(e) => setUseCalibrated(e.target.checked)}
                className="w-3.5 h-3.5 rounded border-gray-300"
              />
              <span>Use Calibrated Results</span>
            </label>
          </div>
        )}

        {/* Main Content Area - Plot Display */}
        <div className="flex-1 overflow-auto p-4">
          {activePlot === 'parity' && (
            <ParityPlot
              sessionId={sessionId}
              useCalibrated={useCalibrated}
              sigmaMultiplier={sigmaMultiplier}
            />
          )}
          {activePlot === 'metrics' && (
            <MetricsPlot
              sessionId={sessionId}
              selectedMetric={selectedMetric}
              cvSplits={5}
            />
          )}
          {activePlot === 'qq' && (
            <QQPlot
              sessionId={sessionId}
              useCalibrated={useCalibrated}
            />
          )}
          {activePlot === 'calibration' && (
            <CalibrationCurve
              sessionId={sessionId}
              useCalibrated={useCalibrated}
            />
          )}
          {/* ContourPlot always mounted but hidden to preserve state */}
          <div className={activePlot === 'contour' ? 'block h-full' : 'hidden'}>
            <ContourPlot sessionId={sessionId} />
          </div>
        </div>
    </>
  );

  // Embedded mode - render without modal wrapper
  if (embedded) {
    return <div className="h-full flex flex-col bg-card">{content}</div>;
  }

  // No modal mode supported anymore - visualizations are always embedded
  return null;
}
