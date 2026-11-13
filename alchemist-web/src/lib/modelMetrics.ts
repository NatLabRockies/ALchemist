import type { ModelMetrics } from '../api/types';

/**
 * The model's cross-validation metrics, or null when it has none.
 *
 * A multi-objective model returns `metrics: {}` and its per-objective values in
 * `metrics_by_objective`, so code that formats RMSE/MAE/R² must check first;
 * calling .toFixed on the missing values blanked the app (W25).
 */
export function cvMetrics(metrics: Partial<ModelMetrics> | null | undefined): ModelMetrics | null {
  if (!metrics) return null;
  const { rmse, mae, r2 } = metrics;
  return [rmse, mae, r2].every(v => typeof v === 'number') ? (metrics as ModelMetrics) : null;
}
