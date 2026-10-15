/**
 * Metrics Plot - RMSE/MAE/MAPE/R² vs Number of Observations
 * Mirrors desktop UI metrics plot from visualizations.py
 */
import { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { useMetricsData } from '../../hooks/api/useVisualizations';
import { Loader2 } from 'lucide-react';
import { usePlotTheme } from './plotTheme';

interface MetricsPlotProps {
  sessionId: string;
  selectedMetric: 'RMSE' | 'MAE' | 'MAPE' | 'R2';
  cvSplits: number;
}

export function MetricsPlot({ sessionId, selectedMetric, cvSplits }: MetricsPlotProps) {
  const { data, isLoading, error } = useMetricsData(sessionId, cvSplits);
  const plot = usePlotTheme();

  const chartData = useMemo(() => {
    if (!data) return [];

    // Map metric name to data array
    const metricArrays: Record<string, (number | null)[]> = {
      RMSE: data.rmse,
      MAE: data.mae,
      MAPE: data.mape,
      R2: data.r2,
    };

    const values = metricArrays[selectedMetric];
    if (!values) return [];

    // X-axis starts at 5 (minimum observations for 5-fold CV)
    // Filter out null values (from NaN/Inf)
    return values
      .map((value, idx) => ({
        numObservations: idx + cvSplits,
        value: value !== null ? value : undefined,
      }))
      .filter(point => point.value !== undefined);
  }, [data, selectedMetric, cvSplits]);

  const metricLabels: Record<string, string> = {
    RMSE: 'RMSE',
    MAE: 'MAE',
    MAPE: 'MAPE (%)',
    R2: 'R²',
  };

  const metricTitles: Record<string, string> = {
    RMSE: 'RMSE vs Number of Observations',
    MAE: 'MAE vs Number of Observations',
    MAPE: 'MAPE vs Number of Observations',
    R2: 'R² vs Number of Observations',
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <Loader2 className="w-8 h-8 animate-spin text-primary mx-auto" />
          <p className="text-sm text-muted-foreground mt-2">
            Computing metrics... This may take 5-10 seconds
          </p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <p className="text-destructive font-medium">Error loading metrics plot</p>
          <p className="text-sm text-muted-foreground mt-2">{error.message}</p>
        </div>
      </div>
    );
  }

  if (!data || chartData.length === 0) {
    return (
      <div className="flex items-center justify-center h-96">
        <p className="text-muted-foreground">No data available for metrics plot</p>
      </div>
    );
  }

  return (
    <div className="w-full h-full flex flex-col">
      {/* Title */}
      <div className="text-center mb-3">
        <h3 className="text-sm font-medium">{metricTitles[selectedMetric]}</h3>
      </div>

      {/* Chart */}
      <div className="flex-1 min-h-0 max-h-[450px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 20, right: 30, bottom: 50, left: 50 }}>
            <CartesianGrid {...plot.grid} />
            <XAxis
              {...plot.axis}
              dataKey="numObservations"
              type="number"
              domain={['dataMin', 'dataMax']}
              label={plot.label('Number of Observations', { position: 'bottom', offset: 10 })}
            />
            <YAxis
              {...plot.axis}
              label={plot.label(metricLabels[selectedMetric], { angle: -90, position: 'left', offset: 10 })}
            />
            <Tooltip
              {...plot.tooltip}
              formatter={(value: number) => value.toFixed(4)}
              labelFormatter={(value: number) => `N = ${value}`}
            />
            <Legend {...plot.legend} />
            <Line
              type="monotone"
              dataKey="value"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={{ r: 4, fill: plot.colors.pointFill, stroke: '#3b82f6', strokeWidth: 2 }}
              name={metricLabels[selectedMetric]}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
