/**
 * Calibration Curve - Reliability diagram showing nominal vs empirical coverage
 * Mirrors desktop UI calibration plot from visualizations.py
 */
import { useMemo } from 'react';
import {
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ComposedChart,
} from 'recharts';
import { useCalibrationCurveData } from '../../hooks/api/useVisualizations';
import { Loader2 } from 'lucide-react';
import { usePlotTheme } from './plotTheme';

interface CalibrationCurveProps {
  sessionId: string;
  useCalibrated: boolean;
}

export function CalibrationCurve({ sessionId, useCalibrated }: CalibrationCurveProps) {
  const { data, isLoading, error } = useCalibrationCurveData(sessionId, useCalibrated);
  const plot = usePlotTheme();

  const chartData = useMemo(() => {
    if (!data) return [];

    return data.nominal_coverage.map((nominal, idx) => ({
      nominal,
      empirical: data.empirical_coverage[idx],
      overConfident: data.empirical_coverage[idx] < nominal ? data.empirical_coverage[idx] : nominal,
      underConfident: data.empirical_coverage[idx] >= nominal ? data.empirical_coverage[idx] : nominal,
    }));
  }, [data]);

  const perfectCalibrationData = useMemo(() => {
    return [
      { nominal: 0, perfect: 0 },
      { nominal: 1, perfect: 1 },
    ];
  }, []);

  // Calculate coverage metrics table - MUST be called before any early returns (Rules of Hooks)
  const coverageMetrics = useMemo(() => {
    if (!data || !data.confidence_levels) return [];
    
    return data.confidence_levels.map((level: string, idx: number) => {
      const nominal = data.nominal_probabilities ? data.nominal_probabilities[idx] : data.nominal_coverage[idx];
      const empirical = data.empirical_probabilities ? data.empirical_probabilities[idx] : data.empirical_coverage[idx];
      const diff = empirical - nominal;

      let status = '';
      if (Math.abs(diff) < 0.05) {
        status = '✓ Good';
      } else if (diff > 0.1) {
        status = '⚠ Under-conf';
      } else if (diff < -0.1) {
        status = '⚠ Over-conf';
      } else {
        status = '~ Acceptable';
      }

      return { level, nominal, empirical, diff, status };
    });
  }, [data]);

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
          <p className="text-destructive font-medium">Error loading calibration curve</p>
          <p className="text-sm text-muted-foreground mt-2">{error.message}</p>
        </div>
      </div>
    );
  }

  if (!data || data.nominal_coverage.length === 0) {
    return (
      <div className="flex items-center justify-center h-96">
        <p className="text-muted-foreground">No data available for calibration curve</p>
      </div>
    );
  }

  const resultsType = data.results_type === 'calibrated' ? ' (Calibrated)' : ' (Uncalibrated)';
  const title = `Calibration Curve (Reliability Diagram)${resultsType}\nN = ${data.n_samples}`;

  return (
    <div className="w-full h-full flex flex-col">
      {/* Title */}
      <div className="text-center mb-3">
        <h3 className="text-sm font-medium whitespace-pre-line leading-tight">{title}</h3>
        {data.n_samples < 30 && (
          <p className="text-xs text-yellow-700 dark:text-yellow-500 mt-1.5">
            ⚠ WARNING: Small sample size (N &lt; 30). Coverage estimates may be noisy.
          </p>
        )}
      </div>

      {/* Chart and Metrics Side by Side */}
      <div className="flex-1 flex flex-wrap gap-6">
        {/* Chart: takes the width the 20rem table leaves (as the old w-80 table did), and wraps above the table when that is under 22rem */}
        <div className="flex-[1_1_22rem] min-w-0 h-[420px]">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData} margin={{ top: 20, right: 30, bottom: 50, left: 50 }}>
              <CartesianGrid {...plot.grid} />
              <XAxis
                {...plot.axis}
                dataKey="nominal"
                type="number"
                domain={[0, 1]}
                label={plot.label('Nominal Coverage Probability', { position: 'bottom', offset: 10 })}
                tickFormatter={(value) => value.toFixed(1)}
              />
              <YAxis
                {...plot.axis}
                type="number"
                domain={[0, 1]}
                label={plot.label('Empirical Coverage Probability', { angle: -90, position: 'left', offset: 10 })}
                tickFormatter={(value) => value.toFixed(1)}
              />
              <Tooltip
                {...plot.tooltip}
                formatter={(value: number) => value.toFixed(3)}
                labelFormatter={(value: number) => `Nominal: ${value.toFixed(3)}`}
              />
              <Legend {...plot.legend} />

              {/* Perfect calibration line */}
              <Line
                data={perfectCalibrationData}
                dataKey="perfect"
                stroke="#ef4444"
                strokeWidth={2}
                strokeDasharray="5 5"
                dot={false}
                name="Perfect calibration"
                isAnimationActive={false}
              />

              {/* Empirical coverage line */}
              <Line
                dataKey="empirical"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={{ r: 4, fill: plot.colors.pointFill, stroke: '#3b82f6', strokeWidth: 2 }}
                name="Empirical coverage"
              />

              {/* Shaded regions for over/under confidence */}
              {/* Note: Recharts doesn't support conditional fill easily, 
                  so we use two separate Area components */}
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        {/* Metrics Table */}
        <div className="flex-[0_1_20rem] bg-muted/30 rounded-lg p-3 overflow-auto">
          <h4 className="font-semibold text-xs uppercase tracking-wide text-muted-foreground mb-2">Coverage Metrics</h4>
          <div className="text-xs">
            <table className="w-full">
              <thead className="border-b border-border">
                <tr className="text-left">
                  <th className="pb-2">Confidence</th>
                  <th className="pb-2 text-right">Nominal</th>
                  <th className="pb-2 text-right">Empirical</th>
                  <th className="pb-2 text-right">Diff</th>
                  <th className="pb-2">Status</th>
                </tr>
              </thead>
              <tbody>
                {coverageMetrics.map((metric: any, idx: number) => (
                  <tr key={idx} className="border-b border-border/50">
                    <td className="py-1">{metric.level}</td>
                    <td className="py-1 text-right">
                      {metric.nominal.toFixed(3)} ({(metric.nominal * 100).toFixed(1)}%)
                    </td>
                    <td className="py-1 text-right">
                      {metric.empirical.toFixed(3)} ({(metric.empirical * 100).toFixed(1)}%)
                    </td>
                    <td className={`py-1 text-right ${metric.diff > 0 ? 'text-blue-700 dark:text-blue-400' : 'text-orange-700 dark:text-orange-400'}`}>
                      {metric.diff > 0 ? '+' : ''}
                      {metric.diff.toFixed(3)} ({metric.diff > 0 ? '+' : ''}
                      {(metric.diff * 100).toFixed(1)}%)
                    </td>
                    <td className="py-1 text-xs">{metric.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
