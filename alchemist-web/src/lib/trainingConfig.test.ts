import { describe, it, expect } from 'vitest';
import { panelSettingsFromConfig } from './trainingConfig';
import type { TrainingConfig } from '../api/types';

describe('panelSettingsFromConfig (W31)', () => {
  it('maps a BoTorch config the Model tab sent back onto its controls', () => {
    expect(panelSettingsFromConfig({
      backend: 'botorch', kernel: 'Matern', kernel_params: { nu: 2.5 },
      input_transform_type: 'normalize', output_transform_type: 'standardize', calibration_enabled: true,
    })).toEqual({
      backend: 'botorch', kernel: 'Matern', maternNu: '2.5', ibnnDepth: undefined,
      inputTransform: 'normalize', outputTransform: 'standardize', calibrateUncertainty: true,
    });
  });

  it('maps a scikit-learn config; an absent transform is none', () => {
    expect(panelSettingsFromConfig({
      backend: 'sklearn', kernel: 'RBF', kernel_params: null, calibration_enabled: false,
    })).toEqual({
      backend: 'sklearn', kernel: 'RBF', maternNu: undefined, ibnnDepth: undefined,
      inputTransform: 'none', outputTransform: 'none', calibrateUncertainty: false,
    });
  });

  it('for BoTorch, an absent transform is the backend default and an explicit null is none', () => {
    const absent = panelSettingsFromConfig({ backend: 'botorch', kernel: 'RBF' });
    expect(absent).toMatchObject({ inputTransform: 'normalize', outputTransform: 'standardize' });
    const explicitNone = panelSettingsFromConfig({
      backend: 'botorch', kernel: 'RBF', input_transform_type: null, output_transform_type: null,
    });
    expect(explicitNone).toMatchObject({ inputTransform: 'none', outputTransform: 'none' });
  });

  it('normalises kernel names the core accepts in lower case, and reads IBNN depth and Matern inf', () => {
    expect(panelSettingsFromConfig({ backend: 'sklearn', kernel: 'rbf' })?.kernel).toBe('RBF');
    expect(panelSettingsFromConfig({ backend: 'sklearn', kernel: 'matern', kernel_params: { nu: Infinity } }))
      .toMatchObject({ kernel: 'Matern', maternNu: 'inf' });
    expect(panelSettingsFromConfig({ backend: 'botorch', kernel: 'ibnn', kernel_params: { ibnn_depth: 4 } }))
      .toMatchObject({ kernel: 'IBNN', ibnnDepth: 4 });
  });

  it('returns null without a usable backend', () => {
    expect(panelSettingsFromConfig(null)).toBeNull();
    expect(panelSettingsFromConfig(undefined)).toBeNull();
    expect(panelSettingsFromConfig({ backend: 'tensorflow', kernel: 'RBF' } as unknown as TrainingConfig)).toBeNull();
  });
});
