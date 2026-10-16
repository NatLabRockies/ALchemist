/**
 * The Model tab's controls, read back from a session's saved training config
 * (GET /model `training_config`), so after Join or Load the tab shows what the
 * model was trained with and Train does not silently switch backend (W31).
 */
import type {
  KernelType,
  MaternNu,
  ModelBackend,
  SklearnOptimizer,
  TrainingConfig,
} from '../api/types';

export interface PanelSettings {
  backend: ModelBackend;
  kernel: KernelType;
  maternNu: MaternNu | undefined;
  ibnnDepth: number | undefined;
  inputTransform: string;
  outputTransform: string;
  calibrateUncertainty: boolean;
  /** scikit-learn only; undefined when absent, unknown, or BoTorch. */
  optimizer: SklearnOptimizer | undefined;
}

const KERNELS: KernelType[] = ['RBF', 'Matern', 'RationalQuadratic', 'IBNN'];
const NUS: MaternNu[] = ['0.5', '1.5', '2.5', 'inf'];
const OPTIMIZERS: SklearnOptimizer[] = ['CG', 'BFGS', 'L-BFGS-B', 'TNC'];
const BOTORCH_DEFAULTS = { input: 'normalize', output: 'standardize' };

function kernelOf(name: unknown): KernelType | null {
  const text = String(name ?? '').toLowerCase();
  return KERNELS.find((k) => k.toLowerCase() === text) ?? null;
}

function nuOf(value: unknown): MaternNu | undefined {
  if (value === Infinity || value === 'inf') return 'inf';
  const text = typeof value === 'number' ? String(value) : String(value ?? '');
  return NUS.find((nu) => nu === text);
}

/**
 * A transform the way the panel's selects name it. The core keeps an absent
 * key apart from an explicit null: for BoTorch, absent means the backend
 * default while null means no transform; for sklearn both mean none.
 */
function transformOf(config: TrainingConfig, key: 'input_transform_type' | 'output_transform_type'): string {
  if (!(key in config)) {
    return config.backend === 'botorch'
      ? key === 'input_transform_type' ? BOTORCH_DEFAULTS.input : BOTORCH_DEFAULTS.output
      : 'none';
  }
  const value = config[key];
  return typeof value === 'string' && value ? value : 'none';
}

export function panelSettingsFromConfig(config: TrainingConfig | null | undefined): PanelSettings | null {
  if (!config || (config.backend !== 'sklearn' && config.backend !== 'botorch')) return null;
  const kernel = kernelOf(config.kernel);
  if (!kernel) return null;
  const params = config.kernel_params ?? {};
  const depth = Number(params.ibnn_depth);
  return {
    backend: config.backend,
    kernel,
    maternNu: kernel === 'Matern' ? nuOf(params.nu) : undefined,
    ibnnDepth: kernel === 'IBNN' && Number.isInteger(depth) && depth > 0 ? depth : undefined,
    inputTransform: transformOf(config, 'input_transform_type'),
    outputTransform: transformOf(config, 'output_transform_type'),
    calibrateUncertainty: config.calibration_enabled === true,
    optimizer:
      config.backend === 'sklearn' ? OPTIMIZERS.find((o) => o === config.optimizer) : undefined,
  };
}
