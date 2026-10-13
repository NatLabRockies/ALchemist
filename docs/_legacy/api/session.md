# OptimizationSession Usage Guide

Complete usage guide for `alchemist_core.OptimizationSession` - the main Python class for Bayesian optimization workflows.

!!! tip "Auto-Generated Reference"
    For class/method signatures auto-generated from docstrings, see [OptimizationSession API Reference](session_class.md).

---

## Quick Start

**Complete workflow example**:
```python
from alchemist_core import OptimizationSession

# 1. Create session
session = OptimizationSession()

# 2. Define search space
session.add_variable('temperature', 'real', bounds=(20, 100), unit='°C')
session.add_variable('pressure', 'real', bounds=(1, 10), unit='bar')
session.add_variable('catalyst', 'categorical', categories=['A', 'B', 'C'])

# 3. Generate initial design
initial_points = session.generate_initial_design(method='lhs', n_points=15)

# 4. Run experiments and add data
for point in initial_points:
    output = my_experiment_function(**point)  # Your experiment
    session.add_experiment(point, output=output)

# 5. Train model
session.train_model(backend='botorch', kernel='Matern')

# 6. Generate next candidates
candidates = session.suggest_next(strategy='EI', n_suggestions=5, goal='maximize')

# 7. Save session
session.save_session('cache/sessions/')
```

---

## Session Initialization

### Basic Initialization

**Create empty session**:
```python
from alchemist_core import OptimizationSession

session = OptimizationSession()
```

**With metadata**:
```python
session = OptimizationSession()
session.metadata.name = "Catalyst Screening"
session.metadata.description = "Optimization of Pd catalyst loading"
session.metadata.tags = ["catalysis", "suzuki", "2025"]
session.metadata.author = "Jane Researcher"
```

### Loading Existing Session

**From file**:
```python
session = OptimizationSession.load_session('cache/sessions/my_session.json')

# Access session data
print(f"Session: {session.metadata.name}")
print(f"Variables: {len(session.search_space.variables)}")
print(f"Experiments: {len(session.experiment_manager.data)}")
```

**With custom components**:
```python
from alchemist_core.data.search_space import SearchSpace
from alchemist_core.events import EventEmitter

# Pre-configure components
space = SearchSpace()
space.add_variable('temp', 'real', min=20, max=100)

events = EventEmitter()
events.on('experiment_added', lambda data: print(f"Added: {data}"))

# Initialize with components
session = OptimizationSession(
    search_space=space,
    event_emitter=events
)
```

---

## Search Space Management

### Adding Variables

**Continuous (real) variables**:
```python
session.add_variable(
    name='temperature',
    var_type='real',
    bounds=(20.0, 100.0),
    unit='°C'
)

# Alternative syntax
session.add_variable('pressure', 'real', bounds=(1.0, 10.0), unit='bar')
```

**Integer variables** (contiguous whole-number range):
```python
session.add_variable(
    name='n_stages',
    var_type='integer',
    bounds=(1, 10)
)
```

**Discrete variables** (restricted to specific allowed values):
```python
session.add_variable(
    name='SAR',
    var_type='discrete',
    allowed_values=[80, 280, 450],
    unit='-',
    description='Silicon-to-alumina ratio (only synthesizable at specific values)'
)
```

**Categorical variables**:
```python
session.add_variable(
    name='catalyst',
    var_type='categorical',
    categories=['A', 'B', 'C', 'D']
)

# Alternative: use 'values' parameter
session.add_variable('solvent', 'categorical', values=['THF', 'DMF', 'toluene'])
```

### Search Space Summary

**Get variable information**:
```python
summary = session.get_search_space_summary()

print(f"Number of variables: {summary['n_variables']}")
for var in summary['variables']:
    print(f"{var['name']}: {var['type']}, bounds={var['bounds']}")
```

**Example output**:
```python
{
    'n_variables': 3,
    'variables': [
        {
            'name': 'temperature',
            'type': 'real',
            'bounds': [20.0, 100.0],
            'unit': '°C'
        },
        {
            'name': 'catalyst',
            'type': 'categorical',
            'categories': ['A', 'B', 'C']
        }
    ],
    'categorical_variables': ['catalyst']
}
```

---

## Initial Design Generation

### Design of Experiments (DOE)

**Latin Hypercube Sampling** (recommended):
```python
points = session.generate_initial_design(
    method='lhs',
    n_points=20,
    random_seed=42
)
# Returns list of dicts: [{'temp': 45.2, 'pressure': 3.1, 'catalyst': 'A'}, ...]
```

**Other methods**:
```python
# Random sampling
points = session.generate_initial_design('random', n_points=15)

# Sobol sequences (low discrepancy)
points = session.generate_initial_design('sobol', n_points=32)

# Halton sequences
points = session.generate_initial_design('halton', n_points=25)
```

**LHS with criteria**:
```python
points = session.generate_initial_design(
    method='lhs',
    n_points=20,
    lhs_criterion='maximin',  # 'maximin', 'correlation', 'ratio'
    random_seed=42
)
```

**Classical RSM methods** (run count determined by design structure):
```python
# Central Composite Design (requires continuous variables only)
points = session.generate_initial_design(
    method='ccd',
    ccd_alpha='orthogonal',
    ccd_face='circumscribed',
    n_center=1,
)

# Box-Behnken (3+ continuous variables, avoids corner combinations)
points = session.generate_initial_design(method='box_behnken')

# Plackett-Burman (ultra-efficient 2-level screening, continuous only)
points = session.generate_initial_design(method='plackett_burman')

# Generalized Subset Design (supports categorical variables)
points = session.generate_initial_design(method='gsd', gsd_reduction=2)

# Full Factorial with 3 levels per factor
points = session.generate_initial_design(method='full_factorial', n_levels=3)
```

**Optimal design** (specify model structure, get statistically efficient design):
```python
# Preview model terms and recommended run count before generating
info = session.get_optimal_design_info(model_type='quadratic')
print(f"{info['p_columns']} model terms, recommend {info['n_points_recommended']} runs")

# Generate D-optimal design for a quadratic model
points, design_info = session.generate_optimal_design(
    model_type='quadratic',
    p_multiplier=2.0,   # 2× as many runs as model columns
    criterion='D',
    algorithm='fedorov',
    random_seed=42,
)
print(f"D-efficiency: {design_info['D_eff']:.1f}%")

# Custom effects list
points, design_info = session.generate_optimal_design(
    effects=['Temperature', 'Pressure', 'Temperature*Pressure', 'Temperature**2'],
    n_points=20,
    criterion='D',
)
```

### Running Initial Experiments

**Evaluate and add results**:
```python
# Generate design
initial_points = session.generate_initial_design('lhs', n_points=15)

# Run experiments
for point in initial_points:
    # Your experiment function
    output = run_experiment(
        temperature=point['temperature'],
        pressure=point['pressure'],
        catalyst=point['catalyst']
    )
    
    # Add to session
    session.add_experiment(point, output=output, reason='LHS initial design')

print(f"Completed {len(initial_points)} initial experiments")
```

---

## Data Management

### Adding Experiments

**Single experiment**:
```python
session.add_experiment(
    inputs={'temperature': 60, 'pressure': 5, 'catalyst': 'A'},
    output=85.3,
    noise=1.2,  # Optional measurement uncertainty
    reason='Manual entry'
)
```

**Batch addition**:
```python
experiments = [
    {'temperature': 60, 'pressure': 5, 'catalyst': 'A'},
    {'temperature': 80, 'pressure': 3, 'catalyst': 'B'},
    {'temperature': 40, 'pressure': 7, 'catalyst': 'C'}
]
outputs = [85.3, 72.1, 68.9]

for inputs, output in zip(experiments, outputs):
    session.add_experiment(inputs, output=output)
```

### Loading from CSV

**Simple load**:
```python
session.load_data('experiments.csv', target_columns='yield')
```

**With noise column**:
```python
session.load_data(
    filepath='experiments.csv',
    target_columns='yield',
    noise_column='std_dev'
)
```

**CSV format**:
```csv
temperature,pressure,catalyst,yield,std_dev
60,5,A,85.3,1.2
80,3,B,72.1,0.9
40,7,C,68.9,1.5
```

### Data Summary

**Get statistics**:
```python
summary = session.get_data_summary()

print(f"Number of experiments: {summary['n_experiments']}")
print(f"Target range: {summary['target_stats']['min']:.2f} - {summary['target_stats']['max']:.2f}")
print(f"Target mean: {summary['target_stats']['mean']:.2f}")
print(f"Has noise data: {summary['has_noise']}")
```

---

## Model Training

### Basic Training

**BoTorch backend** (recommended):
```python
results = session.train_model(
    backend='botorch',
    kernel='Matern',
    kernel_params={'nu': 2.5}
)

print(f"R² = {results['metrics']['cv_r2']:.3f}")
print(f"RMSE = {results['metrics']['cv_rmse']:.3f}")
```

**Scikit-learn backend**:
```python
results = session.train_model(
    backend='sklearn',
    kernel='Matern',
    kernel_params={'nu': 2.5}
)
```

### Kernel Options

**Matern kernels** (most versatile):
```python
# Matern ν=1.5 (less smooth, more flexible)
session.train_model(backend='botorch', kernel='Matern', kernel_params={'nu': 1.5})

# Matern ν=2.5 (smooth, good default)
session.train_model(backend='botorch', kernel='Matern', kernel_params={'nu': 2.5})
```

**RBF kernel** (infinitely smooth):
```python
session.train_model(backend='botorch', kernel='RBF')
```

**Rational Quadratic** (mixture of lengthscales):
```python
session.train_model(backend='botorch', kernel='RationalQuadratic')
```

### Advanced Training Options

**BoTorch with transforms**:
```python
results = session.train_model(
    backend='botorch',
    kernel='Matern',
    kernel_params={'nu': 2.5},
    input_transform_type='normalize',      # Auto-applied by default
    output_transform_type='standardize',   # Auto-applied by default
    calibration_enabled=True               # Apply automatic calibration
)
```

**Sklearn with transforms**:
```python
results = session.train_model(
    backend='sklearn',
    kernel='Matern',
    kernel_params={'nu': 2.5},
    input_transform_type='minmax',    # 'minmax', 'standard', 'robust', 'none'
    output_transform_type='standard',  # 'standard' or 'none'
    n_restarts=10                      # Hyperparameter optimization restarts
)
```

### Training Results

**Inspect results**:
```python
results = session.train_model(backend='botorch', kernel='Matern')

# Cross-validation metrics
print("Cross-Validation Metrics:")
print(f"  R² = {results['metrics']['cv_r2']:.4f}")
print(f"  RMSE = {results['metrics']['cv_rmse']:.4f}")
print(f"  MAE = {results['metrics']['cv_mae']:.4f}")

# Calibration diagnostics
print("\nCalibration:")
print(f"  Mean(z) = {results['metrics']['mean_z']:.4f}")
print(f"  Std(z) = {results['metrics']['std_z']:.4f}")

# Hyperparameters
print("\nHyperparameters:")
print(f"  Lengthscales: {results['hyperparameters']['lengthscales']}")
print(f"  Outputscale: {results['hyperparameters']['outputscale']:.4f}")
print(f"  Noise: {results['hyperparameters']['noise']:.6f}")
```

---

## Acquisition Functions

### Generate Candidates

**Expected Improvement** (EI):
```python
candidates = session.suggest_next(
    strategy='EI',
    n_candidates=5,
    goal='maximize',
    xi=0.01  # Exploration parameter
)

# Returns DataFrame with candidates
print(candidates)
```

**Upper Confidence Bound** (UCB):
```python
candidates = session.suggest_next(
    strategy='UCB',
    n_candidates=3,
    goal='maximize',
    kappa=2.0  # Exploration weight
)
```

**Probability of Improvement** (PI):
```python
candidates = session.suggest_next(
    strategy='PI',
    n_candidates=5,
    goal='maximize',
    xi=0.01
)
```

**Thompson Sampling** (TS):
```python
candidates = session.suggest_next(
    strategy='ThompsonSampling',
    n_candidates=1,
    goal='maximize'
)
```

### Minimization vs Maximization

**Maximize** (e.g., yield):
```python
candidates = session.suggest_next(strategy='EI', goal='maximize', n_candidates=3)
```

**Minimize** (e.g., cost, error):
```python
candidates = session.suggest_next(strategy='EI', goal='minimize', n_candidates=3)
```

### Working with Candidates

**Extract candidate values**:
```python
candidates = session.suggest_next('EI', n_candidates=3, goal='maximize')

for i, row in candidates.iterrows():
    print(f"Candidate {i+1}:")
    print(f"  Temperature: {row['temperature']:.2f}")
    print(f"  Pressure: {row['pressure']:.2f}")
    print(f"  Catalyst: {row['catalyst']}")
    print()
```

**Convert to list of dicts**:
```python
candidates = session.suggest_next('EI', n_candidates=3, goal='maximize')
candidate_dicts = candidates.to_dict('records')

# Each element is a dict: {'temperature': 75.3, 'pressure': 4.2, 'catalyst': 'A'}
for point in candidate_dicts:
    output = run_experiment(**point)
    session.add_experiment(point, output=output, reason='EI')
```

---

## Staged Experiments Workflow

**Purpose**: Manage experiments awaiting evaluation

**Pattern**:
```python
# 1. Generate and stage candidates
candidates = session.suggest_next('EI', n_candidates=5, goal='maximize')
for _, row in candidates.iterrows():
    session.add_staged_experiment(row.to_dict())

# 2. Get staged experiments
staged = session.get_staged_experiments()
print(f"{len(staged)} experiments staged")

# 3. Run experiments
outputs = []
for point in staged:
    output = run_experiment(**point)
    outputs.append(output)

# 4. Move to dataset in batch
session.move_staged_to_experiments(
    outputs=outputs,
    reason='Expected Improvement - Batch 3'
)

print(f"Added {len(outputs)} experiments to dataset")
```

---

## Predictions

### Make Predictions

**Single point**:
```python
# Must train model first
session.train_model(backend='botorch', kernel='Matern')

# Predict
point = {'temperature': 65, 'pressure': 4, 'catalyst': 'A'}
mean, std = session.predict(point)

print(f"Predicted: {mean:.2f} ± {std:.2f}")
```

**Multiple points**:
```python
points = [
    {'temperature': 65, 'pressure': 4, 'catalyst': 'A'},
    {'temperature': 75, 'pressure': 5, 'catalyst': 'B'}
]

means, stds = session.predict_batch(points)

for point, mean, std in zip(points, means, stds):
    print(f"{point}: {mean:.2f} ± {std:.2f}")
```

### Prediction with Confidence Intervals

**Custom confidence level**:
```python
from scipy.stats import norm

mean, std = session.predict(point)

# 95% confidence interval
z_95 = 1.96
ci_lower = mean - z_95 * std
ci_upper = mean + z_95 * std

print(f"Prediction: {mean:.2f}")
print(f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
```

---

## Session Persistence

### Saving Sessions

**Simple save**:
```python
filepath = session.save_session()
print(f"Saved to: {filepath}")
```

**Custom location**:
```python
session.save_session(directory='cache/sessions/', filename='my_optimization.json')
```

**With directory creation**:
```python
import os
os.makedirs('results/sessions', exist_ok=True)
session.save_session('results/sessions/')
```

### Loading Sessions

**Load from file**:
```python
session = OptimizationSession.load_session('cache/sessions/my_optimization.json')
```

**Check session contents**:
```python
session = OptimizationSession.load_session('path/to/session.json')

print(f"Session: {session.metadata.name}")
print(f"Created: {session.metadata.created_at}")
print(f"Variables: {session.get_search_space_summary()['n_variables']}")
print(f"Experiments: {session.get_data_summary()['n_experiments']}")
print(f"Model trained: {session.model is not None}")
```

---

## Audit Logs

### Locking Decisions

**Lock data**:
```python
# Add experimental data
session.load_data('experiments.csv', target_columns='yield')

# Lock data state
session.lock_data(notes="Initial dataset after LHS design")
```

**Lock model**:
```python
# Train model
session.train_model(backend='botorch', kernel='Matern')

# Lock model state
session.lock_model(notes="Production model for batch 3")
```

**Lock acquisition**:
```python
# Generate candidates
candidates = session.suggest_next('EI', n_candidates=5, goal='maximize')

# Lock acquisition decision
session.lock_acquisition(
    strategy='EI',
    candidates=candidates,
    notes="Batch 3 - targeting optimal region"
)
```

### Viewing Audit Log

**Get all entries**:
```python
entries = session.audit_log.entries

for entry in entries:
    print(f"{entry.timestamp}: {entry.entry_type}")
    print(f"  Notes: {entry.notes}")
    print(f"  Hash: {entry.hash[:16]}...")
```

**Filter by type**:
```python
model_entries = [e for e in session.audit_log.entries if e.entry_type == 'model_locked']
print(f"Found {len(model_entries)} model lock entries")
```

**Verify integrity**:
```python
is_valid = session.audit_log.verify_integrity()
if is_valid:
    print("✓ Audit log verified - no tampering detected")
else:
    print("✗ Audit log corrupted - integrity check failed")
```

---

## Event Handling

### Subscribe to Events

**Listen for events**:
```python
def on_experiment_added(data):
    print(f"New experiment: {data['inputs']} → {data['output']}")

def on_model_trained(data):
    print(f"Model trained: R² = {data['metrics']['cv_r2']:.3f}")

session.events.on('experiment_added', on_experiment_added)
session.events.on('model_trained', on_model_trained)

# Now events will trigger callbacks
session.add_experiment({'temp': 60}, output=85)
session.train_model(backend='botorch')
```

**Available events**:

- `variable_added`

- `data_loaded`

- `experiment_added`

- `initial_design_generated`

- `model_trained`

- `acquisition_generated`

- `staged_experiments_cleared`

---

## Complete Example Workflows

### Basic Optimization Loop

```python
from alchemist_core import OptimizationSession

# Initialize
session = OptimizationSession()
session.metadata.name = "Process Optimization"

# Define variables
session.add_variable('temperature', 'real', bounds=(20, 100))
session.add_variable('pressure', 'real', bounds=(1, 10))

# Initial design
initial_points = session.generate_initial_design('lhs', n_points=10)
for point in initial_points:
    output = my_experiment(**point)
    session.add_experiment(point, output=output)

# Optimization loop
for iteration in range(10):
    # Train model
    session.train_model(backend='botorch', kernel='Matern')
    
    # Get candidates
    candidates = session.suggest_next('EI', n_candidates=3, goal='maximize')
    
    # Evaluate
    for _, row in candidates.iterrows():
        point = row.to_dict()
        output = my_experiment(**point)
        session.add_experiment(point, output=output, iteration=iteration+1)
    
    # Check progress
    summary = session.get_data_summary()
    best = summary['target_stats']['max']
    print(f"Iteration {iteration+1}: Best = {best:.2f}")

# Save final session
session.save_session()
```

### Batch Processing with Staging

```python
# Generate batch of candidates
batch_size = 10
candidates = session.suggest_next('UCB', n_candidates=batch_size, goal='maximize')

# Stage all candidates
for _, row in candidates.iterrows():
    session.add_staged_experiment(row.to_dict())

# Run experiments (possibly in parallel)
staged = session.get_staged_experiments()
outputs = [run_experiment(**point) for point in staged]

# Add all results at once
session.move_staged_to_experiments(
    outputs=outputs,
    reason=f'UCB Batch {batch_number}'
)

# Re-train with updated data
session.train_model(backend='botorch', kernel='Matern')
```

### Reproducible Research Workflow

```python
# Set random seed for reproducibility
random_seed = 42

# Create session with metadata
session = OptimizationSession()
session.metadata.name = "Manuscript Optimization"
session.metadata.description = "Results for Journal of X"
session.metadata.author = "Jane Researcher"
session.metadata.tags = ["publication", "2025"]

# Define search space
session.add_variable('var1', 'real', bounds=(0, 1))
session.add_variable('var2', 'real', bounds=(0, 1))

# Generate reproducible initial design
initial = session.generate_initial_design('lhs', n_points=20, random_seed=random_seed)

# Add data and lock
for point in initial:
    output = my_experiment(**point)
    session.add_experiment(point, output=output)
session.lock_data(notes="Initial LHS design, n=20, seed=42")

# Train model and lock
results = session.train_model(backend='botorch', kernel='Matern')
session.lock_model(notes=f"Production model, R²={results['metrics']['cv_r2']:.3f}")

# Generate candidates and lock
candidates = session.suggest_next('EI', n_candidates=5, goal='maximize')
session.lock_acquisition(strategy='EI', candidates=candidates, notes="Batch 1")

# Save with audit trail
session.save_session('manuscript_data/sessions/')
session.audit_log.export('manuscript_data/audit_log.json')
```

---

## Configuration Options

### Session Configuration

**Access config**:
```python
# View current config
print(session.config)

# Modify settings
session.config['random_state'] = 123
session.config['verbose'] = True
```

**Available options**:
```python
{
    'random_state': 42,           # Random seed
    'verbose': True,              # Logging verbosity
    'auto_train': False,          # Auto-train after adding data
    'auto_train_threshold': 5     # Min experiments before auto-train
}
```

---

## Best Practices

### Recommended Workflow

1. **Define complete search space** before generating designs
2. **Generate space-filling initial design** (LHS with 5-10× dimensions)
3. **Lock data** before training production models
4. **Compare multiple kernels** during exploration
5. **Use BoTorch backend** for most applications
6. **Monitor calibration** (Std(z) ≈ 1.0)
7. **Save sessions frequently** at milestones
8. **Lock decisions** for reproducibility

### Performance Tips

**For large datasets** (> 100 points):

- Use BoTorch backend (GPU acceleration if available)

- Consider subset for cross-validation

- Batch candidate generation

**For many variables** (> 10):

- Use ARD lengthscales (enabled by default)

- Increase initial design size (10-15× dimensions)

- Consider variable screening

**For expensive experiments**:

- Start with smaller initial design

- Use conservative acquisition (low ξ/κ)

- Validate model calibration carefully

---

## Further Reading

- [REST API](rest.md) - HTTP interface to Session API
- [Session Management](../reproducibility/sessions.md) - Save/load sessions
- [Audit Logs](../reproducibility/audit_logs.md) - Reproducibility tracking
- [Web Application](../setup/web_app.md) - Browser interface
- [BoTorch Backend](../modeling/botorch.md) - Model training options

---

**Key Takeaway**: The Core Session API provides complete programmatic control over Bayesian optimization workflows. Use it for automation, custom integrations, and reproducible research.
