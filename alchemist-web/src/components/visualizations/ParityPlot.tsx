/**
 * Parity Plot - Actual vs Predicted values with error bars
 * Mirrors desktop UI parity plot from visualizations.py
 */
import { useMemo } from 'react';
import {
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Line,
  ErrorBar,
  ComposedChart,
} from 'recharts';
import { useParityData } from '../../hooks/api/useVisualizations';
import { Loader2 } from 'lucide-react';
import { cvMetrics } from '../../lib/modelMetrics';
import { legendFormatter } from './plotTheme';

interface ParityPlotProps {
  sessionId: string;
  useCalibrated: boolean;
  sigmaMultiplier: string;
}

export function ParityPlot({ sessionId, useCalibrated, sigmaMultiplier }: ParityPlotProps) {
  const { data, isLoading, error } = useParityData(sessionId, useCalibrated);

  const chartData = useMemo(() => {
    if (!data) return [];

    const sigma = sigmaMultiplier === 'None' ? 0 : parseFloat(sigmaMultiplier);

    return data.y_true.map((actual, idx) => ({
      actual,
      predicted: data.y_pred[idx],
      errorLow: sigma > 0 && data.y_std ? sigma * data.y_std[idx] : 0,
      errorHigh: sigma > 0 && data.y_std ? sigma * data.y_std[idx] : 0,
    }));
  }, [data, sigmaMultiplier]);

  const parityLineData = useMemo(() => {
    if (!data || data.y_true.length === 0) return [];

    const allValues = [...data.y_true, ...data.y_pred];
    const minVal = Math.min(...allValues);
    const maxVal = Math.max(...allValues);

    return [
      { actual: minVal, parity: minVal },
      { actual: maxVal, parity: maxVal },
    ];
  }, [data]);

  // Confidence interval label
  const ciLabels: Record<string, string> = {
    '1.0': '68% CI',
    '1.96': '95% CI',
    '2.0': '95.4% CI',
    '2.58': '99% CI',
    '3.0': '99.7% CI',
  };
  const ciLabel = ciLabels[sigmaMultiplier] || `${sigmaMultiplier}σ`;

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <p className="text-destructive font-medium">Error loading parity plot</p>
          <p className="text-sm text-muted-foreground mt-2">{error.message}</p>
        </div>
      </div>
    );
  }

  if (!data || data.y_true.length === 0) {
    return (
      <div className="flex items-center justify-center h-96">
        <p className="text-muted-foreground">No data available for parity plot</p>
      </div>
    );
  }

  const showErrorBars = sigmaMultiplier !== 'None' && data.y_std;
  const resultsType = data.calibrated ? ' (Calibrated)' : ' (Uncalibrated)';
  const metrics = cvMetrics(data.metrics);
  const metricsLine = metrics
    ? `\nRMSE: ${metrics.rmse.toFixed(4)}, MAE: ${metrics.mae.toFixed(4)}, R²: ${metrics.r2.toFixed(4)}`
    : '';
  const title = showErrorBars
    ? `Cross-Validation Parity Plot${resultsType}${metricsLine}\nError bars: ±${sigmaMultiplier}σ (${ciLabel})`
    : `Cross-Validation Parity Plot${resultsType}${metricsLine}`;

  return (
    <div className="w-full h-full flex flex-col">
      {/* Title */}
      <div className="text-center mb-3">
        <h3 className="text-sm font-medium whitespace-pre-line leading-tight">{title}</h3>
      </div>

      {/* Chart */}
      <div className="flex-1 min-h-0 max-h-[450px]">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart margin={{ top: 20, right: 30, bottom: 50, left: 50 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="actual"
              type="number"
              domain={['auto', 'auto']}
              label={{ value: 'Actual Values', position: 'bottom', offset: 10 }}
            />
            <YAxis
              type="number"
              domain={['auto', 'auto']}
              label={{ value: 'Predicted Values', angle: -90, position: 'left', offset: 10 }}
            />
            <Tooltip
              formatter={(value: number) => value.toFixed(4)}
              labelFormatter={(value: number) => `Actual: ${value.toFixed(4)}`}
            />
            <Legend verticalAlign="top" height={36} formatter={legendFormatter} />

            {/* Parity Line (y=x) */}
            <Line
              data={parityLineData}
              dataKey="parity"
              stroke="#ef4444"
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={false}
              name="Parity line"
              isAnimationActive={false}
            />

            {/* Scatter with error bars */}
            <Scatter
              data={chartData}
              dataKey="predicted"
              fill="white"
              stroke="#3b82f6"
              strokeWidth={2}
              fillOpacity={1}
              name="Predictions"
            >
              {showErrorBars && (
                <ErrorBar
                  dataKey="errorLow"
                  direction="y"
                  width={0}
                  strokeWidth={1}
                  stroke="#3b82f6"
                />
              )}
              {showErrorBars && (
                <ErrorBar
                  dataKey="errorHigh"
                  direction="y"
                  width={0}
                  strokeWidth={1}
                  stroke="#3b82f6"
                />
              )}
            </Scatter>
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
