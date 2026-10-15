/**
 * React Query hooks for visualization API endpoints
 */
import { useQuery } from '@tanstack/react-query';
import type { UseQueryResult } from '@tanstack/react-query';
import * as visualizationsApi from '../../api/endpoints/visualizations';
import type {
  ContourDataRequest,
  ContourDataResponse,
  ParityDataResponse,
  MetricsDataResponse,
  QQPlotDataResponse,
  CalibrationCurveDataResponse,
  HyperparametersResponse
} from '../../api/types';

/**
 * Hook to fetch contour plot data
 * Use enabled: false and refetch manually when plot parameters change
 */
export function useContourData(
  sessionId: string | null,
  request: ContourDataRequest,
  enabled: boolean = false
): UseQueryResult<ContourDataResponse> {
  return useQuery({
    queryKey: ['contour-data', sessionId, request],
    queryFn: () => visualizationsApi.getContourData(sessionId!, request),
    // Keep the last figure on screen while the next request loads, but never another session's.
    placeholderData: (previous, previousQuery) =>
      previousQuery?.queryKey[1] === sessionId ? previous : undefined,
    enabled: enabled && !!sessionId,
    staleTime: Infinity, // Never consider data stale
    gcTime: 5 * 60 * 1000, // Keep in cache for 5 minutes
    refetchOnWindowFocus: false,
    refetchOnMount: false,
    refetchOnReconnect: false,
  });
}

/**
 * Hook to fetch parity plot data
 */
export function useParityData(
  sessionId: string | null,
  useCalibrated: boolean = false,
  enabled: boolean = true
): UseQueryResult<ParityDataResponse> {
  return useQuery({
    queryKey: ['parity-data', sessionId, useCalibrated],
    queryFn: () => visualizationsApi.getParityData(sessionId!, useCalibrated),
    enabled: enabled && !!sessionId,
    staleTime: 30000,
    refetchOnWindowFocus: false,
  });
}

/**
 * Hook to fetch metrics data (CV performance over training size)
 */
export function useMetricsData(
  sessionId: string | null,
  cvSplits: number = 5,
  enabled: boolean = true
): UseQueryResult<MetricsDataResponse> {
  return useQuery({
    queryKey: ['metrics-data', sessionId, cvSplits],
    queryFn: () => visualizationsApi.getMetricsData(sessionId!, cvSplits),
    enabled: enabled && !!sessionId,
    staleTime: 60000, // 1 minute - this is expensive to compute
    refetchOnWindowFocus: false,
  });
}

/**
 * Hook to fetch Q-Q plot data
 */
export function useQQPlotData(
  sessionId: string | null,
  useCalibrated: boolean = false,
  enabled: boolean = true
): UseQueryResult<QQPlotDataResponse> {
  return useQuery({
    queryKey: ['qq-plot-data', sessionId, useCalibrated],
    queryFn: () => visualizationsApi.getQQPlotData(sessionId!, useCalibrated),
    enabled: enabled && !!sessionId,
    staleTime: 30000,
    refetchOnWindowFocus: false,
  });
}

/**
 * Hook to fetch calibration curve data
 */
export function useCalibrationCurveData(
  sessionId: string | null,
  useCalibrated: boolean = false,
  enabled: boolean = true
): UseQueryResult<CalibrationCurveDataResponse> {
  return useQuery({
    queryKey: ['calibration-curve-data', sessionId, useCalibrated],
    queryFn: () => visualizationsApi.getCalibrationCurveData(sessionId!, useCalibrated),
    enabled: enabled && !!sessionId,
    staleTime: 30000,
    refetchOnWindowFocus: false,
  });
}

/**
 * Hook to fetch model hyperparameters
 */
export function useHyperparameters(
  sessionId: string | null,
  enabled: boolean = true
): UseQueryResult<HyperparametersResponse> {
  return useQuery({
    queryKey: ['hyperparameters', sessionId],
    queryFn: () => visualizationsApi.getHyperparameters(sessionId!),
    enabled: enabled && !!sessionId,
    staleTime: 60000,
    refetchOnWindowFocus: false,
  });
}
