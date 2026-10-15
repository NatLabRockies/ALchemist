/**
 * Q-Q Plot - Standardized Residuals vs Normal Distribution
 * Mirrors desktop UI Q-Q plot from visualizations.py
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
  ComposedChart,
  Area,
} from 'recharts';
import { useQQPlotData } from '../../hooks/api/useVisualizations';
import { Loader2 } from 'lucide-react';
import { usePlotTheme } from './plotTheme';

interface QQPlotProps {
  sessionId: string;
  useCalibrated: boolean;
}

export function QQPlot({ sessionId, useCalibrated }: QQPlotProps) {
  const { data, isLoading, error } = useQQPlotData(sessionId, useCalibrated);
  const plot = usePlotTheme();

  const chartData = useMemo(() => {
    if (!data) return [];

    return data.theoretical_quantiles.map((theoretical, idx) => ({
      theoretical,
      sample: data.sample_quantiles[idx],
    }));
  }, [data]);

  const parityLineData = useMemo(() => {
    if (!data || data.theoretical_quantiles.length === 0) return [];

    const allValues = [...data.theoretical_quantiles, ...data.sample_quantiles];
    const minVal = Math.min(...allValues);
    const maxVal = Math.max(...allValues);

    return [
      { theoretical: minVal, parity: minVal },
      { theoretical: maxVal, parity: maxVal },
    ];
  }, [data]);

  // Confidence band for small samples (±1.96/sqrt(n) around perfect line)
  const confidenceBandData = useMemo(() => {
    if (!data || data.n_samples >= 100) return [];

    const se = 1.96 / Math.sqrt(data.n_samples);
    
    // Create points along the perfect calibration line with upper/lower bounds
    return data.theoretical_quantiles.map(theoretical => ({
      theoretical,
      bounds: [theoretical - se, theoretical + se], // Store as tuple for Area component
    }));
  }, [data]);

  // Calculate axis limits based on perfect calibration ± 1.96 standard errors
  const axisLimits = useMemo(() => {
    if (!data) return { xMin: -3, xMax: 3, yMin: -3, yMax: 3, ticks: [-3, -2, -1, 0, 1, 2, 3] };
    
    const se = 1.96 / Math.sqrt(data.n_samples);
    
    // X-axis: use the theoretical quantiles range (perfect calibration)
    const xMin = Math.min(...data.theoretical_quantiles);
    const xMax = Math.max(...data.theoretical_quantiles);
    
    // Y-axis: use sample quantiles range ± 1.96*se
    const yMin = Math.min(...data.sample_quantiles) - se;
    const yMax = Math.max(...data.sample_quantiles) + se;
    
    // Calculate ticks based on perfect calibration line range
    const tickMin = Math.ceil(xMin);
    const tickMax = Math.floor(xMax);
    const ticks = Array.from(
      { length: tickMax - tickMin + 1 },
      (_, i) => tickMin + i
    );
    
    return { xMin, xMax, yMin, yMax, ticks };
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
          <p className="text-destructive font-medium">Error loading Q-Q plot</p>
          <p className="text-sm text-muted-foreground mt-2">{error.message}</p>
        </div>
      </div>
    );
  }

  if (!data || data.theoretical_quantiles.length === 0) {
    return (
      <div className="flex items-center justify-center h-96">
        <p className="text-muted-foreground">No data available for Q-Q plot</p>
      </div>
    );
  }

  const resultsType = data.results_type === 'calibrated' ? ' (Calibrated)' : ' (Uncalibrated)';
  const title = `Q-Q Plot: Standardized Residuals vs. Normal Distribution${resultsType}\nMean(z) = ${data.z_mean.toFixed(3)}, Std(z) = ${data.z_std.toFixed(3)}, N = ${data.n_samples}`;

  // Determine calibration quality
  const isWellCalibrated = Math.abs(data.z_mean) < 0.1 && Math.abs(data.z_std - 1.0) < 0.2;
  const isBiased = Math.abs(data.z_mean) > 0.2;
  const isUnderConfident = data.z_std < 0.7;
  const isOverConfident = data.z_std > 1.3;

  let calibrationMessage = '';
  if (isWellCalibrated) {
    calibrationMessage = '✓ Model appears well-calibrated (unbiased, good uncertainty)';
  } else if (isBiased) {
    calibrationMessage = '⚠ Model may be biased (mean(z) far from 0)';
  } else if (isUnderConfident) {
    calibrationMessage = '⚠ Model may be under-confident (std(z) < 1)';
  } else if (isOverConfident) {
    calibrationMessage = '⚠ Model may be over-confident (std(z) > 1)';
  }

  return (
    <div className="w-full h-full flex flex-col">
      {/* Title */}
      <div className="text-center mb-3">
        <h3 className="text-sm font-medium whitespace-pre-line leading-tight">{title}</h3>
        {calibrationMessage && (
          <p className={`text-xs mt-1.5 ${isWellCalibrated ? 'text-green-700 dark:text-green-500' : 'text-yellow-700 dark:text-yellow-500'}`}>
            {calibrationMessage}
          </p>
        )}
      </div>

      {/* Chart */}
      <div className="flex-1 min-h-0 max-h-[450px]">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart margin={{ top: 20, right: 30, bottom: 50, left: 50 }}>
            <CartesianGrid {...plot.grid} />
            <XAxis
              {...plot.axis}
              dataKey="theoretical"
              type="number"
              domain={[axisLimits.xMin, axisLimits.xMax]}
              ticks={axisLimits.ticks}
              label={plot.label('Theoretical Quantiles (Standard Normal)', { position: 'bottom', offset: 10 })}
            />
            <YAxis
              {...plot.axis}
              type="number"
              domain={[axisLimits.yMin, axisLimits.yMax]}
              ticks={axisLimits.ticks}
              label={plot.label('Observed Quantiles (Standardized Residuals)', { angle: -90, position: 'left', offset: 10 })}
            />
            <Tooltip
              {...plot.tooltip}
              formatter={(value: any) => typeof value === 'number' ? value.toFixed(3) : value}
              labelFormatter={(value: any) => typeof value === 'number' ? `Theoretical: ${value.toFixed(3)}` : value}
            />
            {/* Explicit entries: the points are hollow (card-coloured fill), so the automatic swatch would be invisible. */}
            <Legend
              {...plot.legend}
              payload={[
                ...(confidenceBandData.length > 0
                  ? [{ value: 'Approximate 95% CI', type: 'square' as const, color: plot.colors.band }]
                  : []),
                { value: 'Perfect calibration', type: 'line' as const, color: '#ef4444' },
                { value: 'Observations', type: 'circle' as const, color: '#3b82f6' },
              ]}
            />

            {/* Confidence band (if small sample) - rendered as shaded area */}
            {confidenceBandData.length > 0 && (
              <Area
                data={confidenceBandData}
                dataKey="bounds"
                stroke="none"
                fill={plot.colors.band}
                fillOpacity={1}
                name="Approximate 95% CI"
                isAnimationActive={false}
              />
            )}

            {/* Perfect calibration line */}
            <Line
              data={parityLineData}
              dataKey="parity"
              stroke="#ef4444"
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={false}
              name="Perfect calibration"
              isAnimationActive={false}
            />

            {/* Scatter points - rendered last so they're on top */}
            <Scatter
              data={chartData}
              dataKey="sample"
              fill={plot.colors.pointFill}
              stroke="#3b82f6"
              strokeWidth={2}
              fillOpacity={1}
              name="Observations"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
