# ALchemist FastAPI Backend

REST API wrapper for the `alchemist_core` Session API. Provides language-agnostic access to Bayesian optimization functionality.

## Quick Start

### Installation

Install FastAPI dependencies:

```bash
pip install -e .
```

Or install just the API dependencies:

```bash
pip install "fastapi>=0.109.0" "uvicorn[standard]>=0.27.0" "pydantic>=2.5.0" "python-multipart>=0.0.6"
```

### Running the Server

Development mode (with auto-reload):

```bash
python api/main.py
```

Or using uvicorn directly:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Production mode:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### API Documentation

Once running, visit:

- **Interactive Docs (Swagger UI)**: http://localhost:8000/api/docs
- **Alternative Docs (ReDoc)**: http://localhost:8000/api/redoc
- **OpenAPI Schema**: http://localhost:8000/api/openapi.json
- **Health Check**: http://localhost:8000/health

## API Overview

### Base URL

```
http://localhost:8000/api/v1
```

### Workflow Example

```python
import requests

BASE_URL = "http://localhost:8000/api/v1"

# 1. Create a session
response = requests.post(f"{BASE_URL}/sessions", json={})
session_id = response.json()["session_id"]

# 2. Define search space
requests.post(
    f"{BASE_URL}/sessions/{session_id}/variables",
    json={
        "name": "temperature",
        "type": "real",
        "bounds": [100, 500],
        "unit": "°C"
    }
)

requests.post(
    f"{BASE_URL}/sessions/{session_id}/variables",
    json={
        "name": "pressure",
        "type": "real",
        "bounds": [1, 10],
        "unit": "bar"
    }
)

# 3. Add experimental data
requests.post(
    f"{BASE_URL}/sessions/{session_id}/experiments",
    json={
        "inputs": {"temperature": 250, "pressure": 5},
        "output": 0.85
    }
)

# Or upload CSV
with open("experiments.csv", "rb") as f:
    requests.post(
        f"{BASE_URL}/sessions/{session_id}/experiments/upload",
        files={"file": f}
    )

# 4. Train surrogate model
response = requests.post(
    f"{BASE_URL}/sessions/{session_id}/model/train",
    json={
        "backend": "botorch",
        "kernel": "rbf",
        "output_transform": "standardize"
    }
)

# 5. Get next experiment suggestions
response = requests.post(
    f"{BASE_URL}/sessions/{session_id}/acquisition/suggest",
    json={
        "strategy": "qEI",
        "goal": "maximize",
        "n_suggestions": 3
    }
)
suggestions = response.json()["suggestions"]

# 6. Make predictions
response = requests.post(
    f"{BASE_URL}/sessions/{session_id}/model/predict",
    json={
        "inputs": [
            {"temperature": 300, "pressure": 7},
            {"temperature": 400, "pressure": 3}
        ]
    }
)
predictions = response.json()["predictions"]
```

## Endpoints

### Sessions

- `POST /sessions` - Create new session
- `GET /sessions/{session_id}` - Get session info
- `DELETE /sessions/{session_id}` - Delete session

### Variables (Search Space)

- `POST /sessions/{session_id}/variables` - Add single variable
- `GET /sessions/{session_id}/variables` - List all variables
- `POST /sessions/{session_id}/variables/load` - Load from JSON file

### Experiments (Data)

- `POST /sessions/{session_id}/experiments` - Add single experiment
- `POST /sessions/{session_id}/experiments/batch` - Add multiple experiments
- `GET /sessions/{session_id}/experiments` - List all experiments
- `POST /sessions/{session_id}/experiments/upload` - Upload CSV file
- `GET /sessions/{session_id}/experiments/summary` - Get data summary

### Staged Experiments (Workflow Queue)

- `POST /sessions/{session_id}/experiments/staged` - Stage experiment for later execution
- `POST /sessions/{session_id}/experiments/staged/batch` - Stage multiple experiments
- `GET /sessions/{session_id}/experiments/staged` - Get staged experiments
- `DELETE /sessions/{session_id}/experiments/staged` - Clear staged experiments
- `POST /sessions/{session_id}/experiments/staged/complete` - Complete staged experiments with outputs

### Models

- `POST /sessions/{session_id}/model/train` - Train surrogate model
- `GET /sessions/{session_id}/model` - Get model info
- `POST /sessions/{session_id}/model/predict` - Make predictions

### Acquisition

- `POST /sessions/{session_id}/acquisition/suggest` - Suggest next experiments

## Data Formats

### Variable Definition

```json
{
  "name": "temperature",
  "type": "real",  // or "integer", "categorical"
  "bounds": [100, 500],  // for real/integer
  "categories": null,    // for categorical: ["low", "medium", "high"]
  "unit": "°C",
  "description": "Reaction temperature"
}
```

### Experiment Data

```json
{
  "inputs": {
    "temperature": 250,
    "pressure": 5
  },
  "output": 0.85
}
```

### CSV Format

Upload CSV files with columns for each variable + "output":

```csv
temperature,pressure,output
250,5,0.85
300,7,0.92
200,3,0.71
```

### Model Training Request

```json
{
  "backend": "botorch",           // or "sklearn"
  "kernel": "rbf",                // "matern", "periodic", etc.
  "kernel_params": null,          // optional: {"nu": 2.5}
  "input_transform": "normalize", // or "standardize", null
  "output_transform": "standardize", // or "normalize", null
  "calibration_enabled": false
}
```

### Acquisition Request

```json
{
  "strategy": "qEI",      // "EI", "PI", "UCB", "qUCB", "qNIPV"
  "goal": "maximize",     // or "minimize"
  "n_suggestions": 3,
  "xi": null,            // optional: exploration parameter for EI/PI
  "kappa": null          // optional: exploration parameter for UCB
}
```

### Prediction Request

```json
{
  "inputs": [
    {"temperature": 300, "pressure": 7},
    {"temperature": 400, "pressure": 3}
  ]
}
```

### Prediction Response

```json
{
  "predictions": [
    {
      "inputs": {"temperature": 300, "pressure": 7},
      "prediction": 0.89,
      "uncertainty": 0.05
    },
    {
      "inputs": {"temperature": 400, "pressure": 3},
      "prediction": 0.78,
      "uncertainty": 0.12
    }
  ],
  "n_predictions": 2
}
```

## Architecture

### Session Management

- **Storage**: In-memory session store with UUID-based keys
- **Persistence**: Sessions persist to disk under `cache/sessions/` and are restored on startup

### Model Training

- **Synchronous**: Training is fast (seconds) and runs synchronously
- **Storage**: Models stored in session memory
- **Backends**: sklearn (Gaussian Process) or BoTorch (state-of-the-art)
- **Transforms**: Input/output normalization for better performance

### Error Handling

All errors return consistent JSON format:

```json
{
  "detail": "Human-readable error message"
}
```

HTTP status codes:
- `400` - Invalid request data
- `404` - Session or resource not found
- `422` - Validation error
- `500` - Server error

## CORS Configuration

Configured for React development servers:
- http://localhost:3000 (Create React App)
- http://localhost:5173 (Vite)

Update `api/main.py` to add additional origins.

## Testing

Run integration tests:

```bash
pytest tests/api/
```

Test individual endpoints:

```bash
# Create session
curl -X POST http://localhost:8000/api/v1/sessions

# Health check
curl http://localhost:8000/health
```

## Deployment

### Docker (Optional)

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -e .

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t alchemist-api .
docker run -p 8000:8000 alchemist-api
```

### Production Considerations

- Use process manager (e.g., systemd, supervisord)
- Configure multiple workers for concurrent requests
- Add authentication/authorization if needed
- Consider Redis for session persistence
- Add rate limiting for public APIs
- Use HTTPS with reverse proxy (nginx, caddy)

---

## Autonomous Optimization Workflow

ALchemist supports **fully autonomous, human-out-of-the-loop optimization** for real-time process control. This section describes the autonomous workflow using the API.

### Overview

The autonomous workflow enables:
1. **Initial Experimental Design (DoE)** - Generate space-filling initial points
2. **Auto-Training** - Automatically retrain models as new data arrives
3. **Continuous Suggestions** - Get next experiment suggestions without manual intervention
4. **Progress Monitoring** - Query session state for dashboard/logging

### Workflow Steps

#### 1. Create Session and Define Search Space

```python
import requests

BASE_URL = "http://localhost:8000/api/v1"

# Create session
response = requests.post(f"{BASE_URL}/sessions")
session_id = response.json()["session_id"]

# Define variables
variables = [
    {"name": "temperature", "type": "real", "min": 300, "max": 500},
    {"name": "flow_rate", "type": "real", "min": 1, "max": 10},
    {"name": "catalyst", "type": "categorical", "categories": ["A", "B", "C"]}
]

for var in variables:
    requests.post(f"{BASE_URL}/sessions/{session_id}/variables", json=var)
```

#### 2. Generate Initial Experimental Design

```python
# Generate 10 LHS points for initial exploration
response = requests.post(
    f"{BASE_URL}/sessions/{session_id}/initial-design",
    json={
        "method": "lhs",           # Latin Hypercube Sampling
        "n_points": 10,
        "random_seed": 42,         # For reproducibility
        "lhs_criterion": "maximin" # Space-filling criterion
    }
)

initial_points = response.json()["points"]
# Returns: [{"temperature": 350.2, "flow_rate": 4.5, "catalyst": "A"}, ...]
```

**Available DoE Methods**:
- `lhs` - Latin Hypercube Sampling (recommended for most cases)
- `sobol` - Sobol quasi-random sequences
- `halton` - Halton sequences
- `hammersly` - Hammersly sequences
- `random` - Uniform random sampling

#### 3. Run Initial Experiments and Upload Data

```python
# Autonomous controller runs initial experiments
# (hardware interaction happens here)

# Upload results with auto-training enabled
for point in initial_points:
    output = run_physical_experiment(**point)  # Your hardware interface
    
    response = requests.post(
        f"{BASE_URL}/sessions/{session_id}/experiments",
        json={"inputs": point, "output": output},
        params={
            "auto_train": True,           # NEW: Auto-train after adding data
            "training_backend": "sklearn",
            "training_kernel": "rbf"
        }
    )
    
    # Check if model was trained
    if response.json().get("model_trained"):
        print(f"Model trained! RMSE: {response.json()['training_metrics']['rmse']:.3f}")
```

**Auto-Training**:
- Model retrains automatically after each data point (if `auto_train=True`)
- Only trains if ≥5 experiments exist
- Returns training metrics in response
- Falls back gracefully if training fails

#### 4. Autonomous Optimization Loop

```python
# Continuous optimization loop
while not convergence_criteria_met():
    # Get next suggestion
    response = requests.post(
        f"{BASE_URL}/sessions/{session_id}/acquisition/suggest",
        json={
            "strategy": "qEI",
            "goal": "maximize",
            "n_suggestions": 1
        }
    )
    
    next_point = response.json()["suggestions"][0]
    
    # Run experiment
    output = run_physical_experiment(**next_point)
    
    # Add data and auto-train
    response = requests.post(
        f"{BASE_URL}/sessions/{session_id}/experiments",
        json={"inputs": next_point, "output": output},
        params={"auto_train": True}
    )
    
    # Check progress
    state = requests.get(f"{BASE_URL}/sessions/{session_id}/state").json()
    print(f"Experiments: {state['n_experiments']}, Model trained: {state['model_trained']}")
```

#### 5. Monitor Progress

```python
# Query session state for monitoring dashboard
response = requests.get(f"{BASE_URL}/sessions/{session_id}/state")
state = response.json()

print(f"Session: {state['session_id']}")
print(f"Variables: {state['n_variables']}")
print(f"Experiments: {state['n_experiments']}")
print(f"Model trained: {state['model_trained']}")
if state['last_suggestion']:
    print(f"Last suggestion: {state['last_suggestion']}")
```

### Web-Based Monitoring

**Normal Mode** (full interface):
```
http://localhost:5174
```

**Monitoring Mode** (read-only, auto-refresh):
```
http://localhost:5174?mode=monitor
```

The monitoring dashboard:
- Auto-refreshes every 90 seconds
- Displays real-time session metrics
- Shows last suggested experiment
- All controls disabled (read-only)
- Suitable for wall displays or remote observation

### Example: Complete Autonomous Script

```python
#!/usr/bin/env python3
"""
Autonomous reactor optimization controller.
Connects ALchemist to hardware via MQTT.
"""
import requests
import time

BASE_URL = "http://localhost:8000/api/v1"

# 1. Setup
response = requests.post(f"{BASE_URL}/sessions")
session_id = response.json()["session_id"]

# Define reactor variables
requests.post(f"{BASE_URL}/sessions/{session_id}/variables", 
              json={"name": "temperature", "type": "real", "min": 300, "max": 500})
requests.post(f"{BASE_URL}/sessions/{session_id}/variables",
              json={"name": "flow_rate", "type": "real", "min": 1, "max": 10})

# 2. Initial design
response = requests.post(f"{BASE_URL}/sessions/{session_id}/initial-design",
                        json={"method": "lhs", "n_points": 8, "random_seed": 42})
initial_points = response.json()["points"]

# 3. Run initial experiments
for i, point in enumerate(initial_points):
    print(f"Initial experiment {i+1}/{len(initial_points)}")
    
    # Set hardware via MQTT/HMI
    set_reactor_conditions(**point)
    wait_for_steady_state()
    
    # Measure output
    output = measure_spectral_feature()
    
    # Upload with auto-train (trains after 5th point)
    requests.post(f"{BASE_URL}/sessions/{session_id}/experiments",
                 json={"inputs": point, "output": output},
                 params={"auto_train": True, "training_backend": "sklearn"})

# 4. Optimization loop
for iteration in range(50):  # Maximum 50 iterations
    print(f"\n=== Optimization Iteration {iteration+1} ===")
    
    # Get suggestion
    response = requests.post(f"{BASE_URL}/sessions/{session_id}/acquisition/suggest",
                            json={"strategy": "qEI", "goal": "maximize", "n_suggestions": 1})
    next_point = response.json()["suggestions"][0]
    
    # Apply to hardware
    set_reactor_conditions(**next_point)
    wait_for_steady_state()
    output = measure_spectral_feature()
    
    # Update model
    response = requests.post(f"{BASE_URL}/sessions/{session_id}/experiments",
                            json={"inputs": next_point, "output": output},
                            params={"auto_train": True})
    
    # Log progress
    state = requests.get(f"{BASE_URL}/sessions/{session_id}/state").json()
    print(f"Total experiments: {state['n_experiments']}")
    print(f"Current output: {output:.3f}")
    
    # Check convergence (example)
    if iteration > 10 and abs(output - previous_output) < 0.01:
        print("Converged!")
        break
    
    previous_output = output
    time.sleep(300)  # 5 min between experiments

print(f"\nOptimization complete! View results at: http://localhost:5174?mode=monitor")
```

### Best Practices

1. **Start with DoE**: Always generate initial design points for good space coverage
2. **Use Auto-Train**: Enable `auto_train=True` to keep model updated
3. **Monitor State**: Poll `/state` endpoint for progress tracking
4. **Error Handling**: Wrap API calls in try/except for robustness
5. **Steady-State Detection**: Only upload data when process is stable
6. **Safety Validation**: Validate suggestions before applying to hardware
7. **Session Persistence**: Export session periodically for backup
8. **Use Staged Experiments**: Track pending experiments with staging endpoints

### API Endpoints Summary

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/sessions/{id}/initial-design` | POST | Generate DoE points |
| `/sessions/{id}/experiments?auto_train=true` | POST | Add data + train |
| `/sessions/{id}/experiments/staged` | POST | Stage experiment for execution |
| `/sessions/{id}/experiments/staged` | GET | Get pending experiments |
| `/sessions/{id}/experiments/staged/complete` | POST | Complete staged with outputs |
| `/sessions/{id}/state` | GET | Query progress |
| `/sessions/{id}/acquisition/suggest` | POST | Get next point |

See complete API documentation: http://localhost:8000/api/docs

---

## Development

### Project Structure

```
api/
├── main.py                    # FastAPI app entry point
├── dependencies.py            # Shared dependency injection
├── models/
│   ├── requests.py           # Pydantic request schemas
│   └── responses.py          # Pydantic response schemas
├── routers/
│   ├── sessions.py           # Session lifecycle
│   ├── variables.py          # Search space management
│   ├── experiments.py        # Data management
│   ├── models.py             # Model training & prediction
│   └── acquisition.py        # Next experiment suggestions
├── services/
│   └── session_store.py      # Session storage & management
└── middleware/
    └── error_handlers.py     # Custom exceptions & handlers
```

### Adding New Endpoints

1. Define request/response models in `api/models/`
2. Add route handler in appropriate router
3. Update this README with endpoint documentation
4. Add tests in `tests/api/`

### Code Style

- Use type hints throughout
- Add docstrings to all public functions
- Follow REST conventions (resource-based URLs)
- Return appropriate HTTP status codes
- Include examples in Pydantic schemas

## License

BSD 3-Clause (same as alchemist_core)
