# GPR Panel Component

React component for Gaussian Process Regression model training, matching the desktop UI `gpr_panel.py` functionality.

## Features

### Backend Selection
- **scikit-learn**: Traditional sklearn GPR implementation
- **BoTorch**: PyTorch-based Bayesian optimization backend

### Advanced Options
All advanced options are disabled by default and can be enabled via toggle switch:

#### Scikit-learn Options
- **Kernel Selection**: RBF, Matern, RationalQuadratic
- **Matern nu**: 0.5, 1.5, 2.5, inf (only shown when Matern is selected)
- **Optimizer**: CG, BFGS, L-BFGS-B, TNC
- **Input Scaling**: none, minmax, standard, robust
- **Output Scaling**: none, minmax, standard, robust
- **Calibrate Uncertainty**: Enable/disable uncertainty calibration

#### BoTorch Options
- **Continuous Kernel**: RBF, Matern
- **Matern nu**: 0.5, 1.5, 2.5 (only shown when Matern is selected)
- **Input Scaling**: none, normalize, standardize
- **Output Scaling**: none, standardize
- **Calibrate Uncertainty**: Enable/disable uncertainty calibration

## API Integration

The component uses the following REST API endpoints:

### GET `/api/v1/sessions/{session_id}/model`
Returns current model information if a model has been trained.

**Response:**
```typescript
{
  backend: 'sklearn' | 'botorch' | null,
  hyperparameters: { ... } | null,
  metrics: {
    rmse: number,
    mae: number,
    r2: number,
    mape?: number
  } | null,
  is_trained: boolean
}
```

### POST `/api/v1/sessions/{session_id}/model/train`
Trains a new model with the specified configuration.

**Request:**
```typescript
{
  backend: 'sklearn' | 'botorch',
  kernel: 'RBF' | 'Matern' | 'RationalQuadratic',
  kernel_params?: {
    nu?: number  // For Matern kernel
  },
  input_transform?: string,  // Backend-specific transform type
  output_transform?: string,  // Backend-specific transform type
  calibration_enabled?: boolean
}
```

**Response:**
```typescript
{
  success: boolean,
  backend: string,
  kernel: string,
  hyperparameters: { ... },
  metrics: {
    rmse: number,
    mae: number,
    r2: number
  },
  message: string
}
```

## Data Requirements

- Minimum **5 experiments** with output values required to train a model
- The component automatically checks experiment count and shows appropriate warnings
- Training is disabled until sufficient data is available

## Usage Example

```tsx
import { GPRPanel } from './features/models/GPRPanel';

function MyApp() {
  const sessionId = 'your-session-id';
  
  return (
    <GPRPanel sessionId={sessionId} />
  );
}
```

## State Management

The component uses React Query hooks from `useModels.ts`:

- `useModelInfo(sessionId)`: Fetches current model information
- `useTrainModel(sessionId)`: Mutation hook for training models

## UI/UX Features

1. **Progressive Disclosure**: Advanced options hidden by default
2. **Smart Defaults**: 
   - sklearn: RBF kernel, L-BFGS-B optimizer, no transforms
   - BoTorch: Matern kernel (nu=2.5), input normalize / output standardize (choose `none` to train unscaled)
3. **Visual Feedback**:
   - Loading spinner during training
   - Success message with green background
   - Warning for insufficient data (amber background)
   - Disabled state for buttons when conditions aren't met
4. **Conditional UI**: Matern nu option only appears when Matern kernel is selected
5. **Model Info Display**: Shows trained model metrics and hyperparameters

## Future Enhancements

- Visualization panel (currently disabled as "Coming Soon")
- Model comparison functionality
- Advanced hyperparameter tuning interface
- Real-time training progress updates
