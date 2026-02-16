/**
 * TypeScript types for ALchemist API
 * These interfaces match the Pydantic models in the FastAPI backend
 */

// ============================================================================
// Session Types
// ============================================================================

export interface Session {
  session_id: string;
  created_at: string;
  ttl_hours: number;
  expires_at: string;
  variable_count: number;
  experiment_count: number;
  model_trained: boolean;
  // Optional metadata may be present on session responses
  metadata?: {
    session_id?: string;
    name?: string;
    description?: string;
    tags?: string[];
    author?: string;
    created_at?: string;
    last_modified?: string;
  };
}

export interface CreateSessionRequest {
  ttl_hours?: number;
}

export interface CreateSessionResponse {
  session_id: string;
  created_at: string;
  ttl_hours: number;
  expires_at: string;
}

export interface UpdateTTLRequest {
  ttl_hours: number;
}

// ============================================================================
// Variable Types
// ============================================================================

export type VariableType = 'continuous' | 'discrete' | 'categorical';

// API expects these type values
export type APIVariableType = 'real' | 'integer' | 'categorical';

export interface Variable {
  name: string;
  type: VariableType;
  bounds?: [number, number];  // For continuous/discrete
  categories?: string[];      // For categorical
  unit?: string;
  description?: string;
}

// API request format (what backend expects for POST)
export interface APIVariable {
  name: string;
  type: APIVariableType;
  min?: number;              // For real/integer (API format)
  max?: number;              // For real/integer (API format)
  categories?: string[];     // For categorical
  unit?: string;
  description?: string;
}

// What backend returns in GET (includes bounds array)
export interface VariableDetail {
  name: string;
  type: APIVariableType;  // Backend returns 'real', 'integer', 'categorical'
  bounds?: [number, number] | null;
  categories?: string[] | null;
  unit?: string;
  description?: string;
}

export interface VariablesListResponse {
  variables: VariableDetail[];
  n_variables: number;  // Backend returns n_variables not count
}

// ============================================================================
// Experiment Types
// ============================================================================

export interface Experiment {
  inputs: Record<string, number | string>;
  output?: number;
  noise?: number;
  [key: string]: any;  // Allow indexing by string for dynamic column access
}

export interface ExperimentBatch {
  experiments: Experiment[];
}

export interface ExperimentSummary {
  n_experiments: number;
  has_data: boolean;
  has_noise?: boolean;
  target_stats?: {
    min: number;
    max: number;
    mean: number;
    std: number;
  };
  feature_names?: string[];
}

// ============================================================================
// Model Types
// ============================================================================

export type ModelBackend = 'sklearn' | 'botorch';
export type KernelType = 'RBF' | 'Matern' | 'RationalQuadratic' | 'IBNN';
export type MaternNu = '0.5' | '1.5' | '2.5' | 'inf';

// Sklearn-specific options
export type SklearnInputTransform = 'none' | 'minmax' | 'standard' | 'robust';
export type SklearnOutputTransform = 'none' | 'minmax' | 'standard' | 'robust';
export type SklearnOptimizer = 'CG' | 'BFGS' | 'L-BFGS-B' | 'TNC';

// BoTorch-specific options
export type BoTorchInputTransform = 'none' | 'normalize' | 'standardize';
export type BoTorchOutputTransform = 'none' | 'standardize';

export interface TrainModelRequest {
  backend: ModelBackend;
  kernel: KernelType;
  kernel_params?: {
    nu?: number;  // For Matern kernel
    [key: string]: any;
  };
  input_transform?: string;  // Transform type (backend-specific)
  output_transform?: string;  // Transform type (backend-specific)
  calibration_enabled?: boolean;
}

export interface ModelMetrics {
  rmse: number;
  mae: number;
  r2: number;
  mape?: number;
}

export interface TrainModelResponse {
  success: boolean;
  backend: ModelBackend;
  kernel: KernelType;
  hyperparameters: Record<string, any>;
  metrics: ModelMetrics;
  message: string;
}

export interface ModelInfo {
  backend: ModelBackend | null;
  hyperparameters: Record<string, any> | null;
  metrics: ModelMetrics | null;
  is_trained: boolean;
}

// ============================================================================
// Prediction Types
// ============================================================================

export interface PredictionRequest {
  inputs: Array<Record<string, number | string>>;
}

export interface PredictionResult {
  inputs: Record<string, number | string>;
  prediction: number;
  uncertainty: number;
}

export interface PredictionResponse {
  predictions: PredictionResult[];
  n_predictions: number;
}

// ============================================================================
// Acquisition Types
// ============================================================================

export type AcquisitionStrategy = 'EI' | 'PI' | 'UCB' | 'qEI' | 'qUCB' | 'qNIPV';
export type OptimizationGoal = 'maximize' | 'minimize';

export interface AcquisitionRequest {
  strategy: AcquisitionStrategy;
  goal: OptimizationGoal;
  n_suggestions?: number;
  xi?: number;      // For EI/PI
  kappa?: number;   // For UCB
}

export interface AcquisitionResponse {
  suggestions: Array<Record<string, any>>;
  n_suggestions: number;
}

// ============================================================================
// Find Optimum Types
// ============================================================================

export interface FindOptimumRequest {
  goal: OptimizationGoal;
}

export interface FindOptimumResponse {
  optimum: Record<string, any>;
  predicted_value: number;
  predicted_std: number | null;
  goal: string;
}

// ============================================================================
// Error Types
// ============================================================================

export interface APIError {
  detail: string;
}

// ============================================================================
// Visualization Types
// ============================================================================

export interface ContourDataRequest {
  x_var: string;
  y_var: string;
  fixed_values?: Record<string, any>;
  grid_resolution?: number;
  include_experiments?: boolean;
  include_suggestions?: boolean;
}

export interface ContourDataResponse {
  x_var: string;
  y_var: string;
  x_grid: number[][];
  y_grid: number[][];
  predictions: number[][];
  uncertainties: number[][];
  experiments?: {
    x: number[];
    y: number[];
    output: number[];
  } | null;
  suggestions?: {
    x: number[];
    y: number[];
  } | null;
  x_bounds: [number, number];
  y_bounds: [number, number];
  colorbar_bounds: [number, number];
}

export interface ParityDataResponse {
  y_true: number[];
  y_pred: number[];
  y_std: number[];
  metrics: {
    rmse: number;
    mae: number;
    r2: number;
    mape: number;
  };
  bounds: [number, number];
  calibrated: boolean;
}

export interface MetricsDataResponse {
  training_sizes: number[];
  rmse: (number | null)[];  // null for NaN/Inf values
  mae: (number | null)[];
  r2: (number | null)[];
  mape: (number | null)[];
}

export interface QQPlotDataResponse {
  theoretical_quantiles: number[];
  sample_quantiles: number[];
  z_mean: number;
  z_std: number;
  n_samples: number;
  bounds: [number, number];
  calibrated: boolean;
  results_type: string;  // 'calibrated' or 'uncalibrated'
}

export interface CalibrationCurveDataResponse {
  nominal_coverage: number[];
  empirical_coverage: number[];
  confidence_levels: string[];  // e.g., ['±1.0σ (68%)', '±1.96σ (95%)', ...]
  nominal_probabilities: number[];  // Same as nominal_coverage
  empirical_probabilities: number[];  // Same as empirical_coverage
  n_samples: number;
  calibrated: boolean;
  results_type: string;  // 'calibrated' or 'uncalibrated'
}

export interface HyperparametersResponse {
  hyperparameters: Record<string, any>;
  backend: string;
  kernel: string;
  input_transform: string | null;
  output_transform: string | null;
  calibration_enabled: boolean;
  calibration_factor: number | null;
}

// ============================================================================
// Autonomous Optimization Types
// ============================================================================

export type DoEMethod = 'random' | 'lhs' | 'sobol' | 'halton' | 'hammersly';
export type LHSCriterion = 'maximin' | 'correlation' | 'ratio';

export interface InitialDesignRequest {
  method: DoEMethod;
  n_points: number;
  random_seed?: number | null;
  lhs_criterion?: LHSCriterion;
}

export interface InitialDesignResponse {
  points: Array<Record<string, any>>;
  method: string;
  n_points: number;
}

export interface SessionStateResponse {
  session_id: string;
  n_variables: number;
  n_experiments: number;
  model_trained: boolean;
  last_suggestion: Record<string, any> | null;
}
