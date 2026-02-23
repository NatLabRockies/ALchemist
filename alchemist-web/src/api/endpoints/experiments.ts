/**
 * Experiments API endpoints
 */
import { apiClient } from '../client';
import type { 
  Experiment,
  ExperimentSummary,
  InitialDesignRequest,
  InitialDesignResponse,
  OptimalDesignInfoRequest,
  OptimalDesignInfoResponse,
  OptimalDesignRequest,
  OptimalDesignResponse,
} from '../types';

/**
 * Add a single experiment
 */
export const createExperiment = async (
  sessionId: string,
  experiment: Experiment
): Promise<{ message: string }> => {
  const response = await apiClient.post(
    `/sessions/${sessionId}/experiments`,
    experiment
  );
  return response.data;
};

/**
 * Add multiple experiments
 */
export const createExperimentBatch = async (
  sessionId: string,
  experiments: Experiment[]
): Promise<{ message: string; count: number }> => {
  const response = await apiClient.post(
    `/sessions/${sessionId}/experiments/batch`,
    { experiments }
  );
  return response.data;
};

/**
 * Upload experiments from CSV file
 */
export const uploadExperimentsCSV = async (
  sessionId: string,
  file: File,
  targetColumns?: string | string[]
): Promise<{ message: string; n_experiments: number }> => {
  const formData = new FormData();
  formData.append('file', file);
  
  // Build URL with target_columns as query parameter
  const targetParam = targetColumns 
    ? (Array.isArray(targetColumns) ? targetColumns.join(',') : targetColumns)
    : undefined;
  
  const url = targetParam
    ? `/sessions/${sessionId}/experiments/upload?target_columns=${encodeURIComponent(targetParam)}`
    : `/sessions/${sessionId}/experiments/upload`;
  
  const response = await apiClient.post(
    url,
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );
  return response.data;
};

/**
 * Preview CSV columns before upload to check for target columns
 */
export const previewCSVColumns = async (
  sessionId: string,
  file: File
): Promise<{
  columns: string[];
  available_targets: string[];
  has_output: boolean;
  recommended_target: string | null;
  n_rows: number;
}> => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await apiClient.post(
    `/sessions/${sessionId}/experiments/preview`,
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );
  return response.data;
};

/**
 * Get all experiments
 */
export const getExperiments = async (
  sessionId: string
): Promise<{ experiments: any[]; n_experiments: number }> => {
  const response = await apiClient.get(`/sessions/${sessionId}/experiments`);
  return response.data;
};

/**
 * Get experiment summary statistics
 */
export const getExperimentSummary = async (
  sessionId: string
): Promise<ExperimentSummary> => {
  const response = await apiClient.get<ExperimentSummary>(
    `/sessions/${sessionId}/experiments/summary`
  );
  return response.data;
};

/**
 * Generate initial experimental design (DoE)
 */
export const generateInitialDesign = async (
  sessionId: string,
  request: InitialDesignRequest
): Promise<InitialDesignResponse> => {
  const response = await apiClient.post<InitialDesignResponse>(
    `/sessions/${sessionId}/initial-design`,
    request
  );
  return response.data;
};

/**
 * Preview optimal design model terms and recommended run count (dry-run)
 */
export const getOptimalDesignInfo = async (
  sessionId: string,
  request: OptimalDesignInfoRequest
): Promise<OptimalDesignInfoResponse> => {
  const response = await apiClient.post<OptimalDesignInfoResponse>(
    `/sessions/${sessionId}/optimal-design/info`,
    request
  );
  return response.data;
};

/**
 * Generate a statistically optimal experimental design (D/A/I-optimal)
 */
export const generateOptimalDesign = async (
  sessionId: string,
  request: OptimalDesignRequest
): Promise<OptimalDesignResponse> => {
  const response = await apiClient.post<OptimalDesignResponse>(
    `/sessions/${sessionId}/optimal-design`,
    request
  );
  return response.data;
};
