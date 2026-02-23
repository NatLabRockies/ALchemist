/**
 * React Query hooks for experiments
 */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import * as experimentsAPI from '../../api/endpoints/experiments';
import type { Experiment, InitialDesignRequest, OptimalDesignInfoRequest, OptimalDesignRequest } from '../../api/types';
import { toast } from 'sonner';

/**
 * Hook to get all experiments
 */
export function useExperiments(sessionId: string | null) {
  return useQuery({
    queryKey: ['experiments', sessionId],
    queryFn: () => experimentsAPI.getExperiments(sessionId!),
    enabled: !!sessionId,
  });
}

/**
 * Hook to get experiments summary
 */
export function useExperimentsSummary(sessionId: string | null) {
  return useQuery({
    queryKey: ['experiments-summary', sessionId],
    queryFn: () => experimentsAPI.getExperimentSummary(sessionId!),
    enabled: !!sessionId,
  });
}

/**
 * Hook to add a single experiment
 */
export function useCreateExperiment(sessionId: string) {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (experiment: Experiment) => 
      experimentsAPI.createExperiment(sessionId, experiment),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['experiments', sessionId] });
      queryClient.invalidateQueries({ queryKey: ['experiments-summary', sessionId] });
      queryClient.invalidateQueries({ queryKey: ['session', sessionId] });
      toast.success('Experiment added successfully!');
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to add experiment');
    },
  });
}

/**
 * Hook to upload experiments from CSV file
 */
export function useUploadExperiments(sessionId: string) {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ file, targetColumns }: { file: File; targetColumns?: string | string[] }) => 
      experimentsAPI.uploadExperimentsCSV(sessionId, file, targetColumns),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['experiments', sessionId] });
      queryClient.invalidateQueries({ queryKey: ['experiments-summary', sessionId] });
      queryClient.invalidateQueries({ queryKey: ['session', sessionId] });
      toast.success(`Loaded ${data.n_experiments || 0} experiments successfully!`);
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to upload experiments');
    },
  });
}

/**
 * Hook to preview CSV columns before upload
 */
export function usePreviewCSV(sessionId: string) {
  return useMutation({
    mutationFn: (file: File) => 
      experimentsAPI.previewCSVColumns(sessionId, file),
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to preview CSV file');
    },
  });
}

/**
 * Hook to generate initial experimental design (DoE)
 */
export function useGenerateInitialDesign(sessionId: string) {
  return useMutation({
    mutationFn: (request: InitialDesignRequest) => 
      experimentsAPI.generateInitialDesign(sessionId, request),
    onSuccess: (data) => {
      toast.success(`Generated ${data.n_points} initial design points using ${data.method}`);
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to generate initial design');
    },
  });
}

/**
 * Hook to preview optimal design model terms (dry-run)
 */
export function useOptimalDesignInfo(sessionId: string) {
  return useMutation({
    mutationFn: (request: OptimalDesignInfoRequest) =>
      experimentsAPI.getOptimalDesignInfo(sessionId, request),
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to preview optimal design');
    },
  });
}

/**
 * Hook to generate optimal experimental design (D/A/I-optimal)
 */
export function useGenerateOptimalDesign(sessionId: string) {
  return useMutation({
    mutationFn: (request: OptimalDesignRequest) =>
      experimentsAPI.generateOptimalDesign(sessionId, request),
    onSuccess: (data) => {
      const dEff = data.design_info?.D_eff;
      const effStr = dEff != null ? ` (D_eff=${dEff.toFixed(1)}%)` : '';
      toast.success(`Generated ${data.n_points} optimal design points${effStr}`);
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to generate optimal design');
    },
  });
}
