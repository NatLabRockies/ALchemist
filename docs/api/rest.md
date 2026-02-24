# REST API Reference

The **ALchemist REST API** provides HTTP access to the Core Session API, enabling language-agnostic integration with Bayesian optimization workflows. Built with FastAPI, the API offers interactive documentation, type validation, and WebSocket support for real-time updates.

!!! info "Python API vs REST API"
    - **Python API**: Direct Python import (`from alchemist_core import OptimizationSession`) for notebooks and scripts. [See Python API docs](python_overview.md)
    - **REST API**: HTTP endpoints for web applications, remote access, and non-Python clients (this page)

---

## Overview

**Base URL**: `http://localhost:8000/api/v1` (default)  
**API Version**: v1  
**Content Type**: `application/json`  
**Interactive Docs**: http://localhost:8000/api/docs (Swagger UI)  
**Alternative Docs**: http://localhost:8000/api/redoc (ReDoc)  
**Web Application**: http://localhost:8000

**Architecture**:
```
HTTP Client → FastAPI REST API → Core Session API → BoTorch/Sklearn
```

---

## Getting Started

### Starting the API Server

**Development mode** (auto-reload):
```bash
python -m api.run_api
```

**Production mode** (with uvicorn):
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

**With custom settings**:
```bash
HOST=0.0.0.0 PORT=8080 python -m api.run_api
```

### Testing the API

**Browser**:

- API Documentation: http://localhost:8000/docs

- Web Application: http://localhost:8000/app

- Use interactive Swagger UI to test endpoints

**cURL**:
```bash
# Health check
curl http://localhost:8000/health

# Create session
curl -X POST http://localhost:8000/api/v1/sessions \
  -H "Content-Type: application/json"
```

**Python**:
```python
import requests

BASE_URL = "http://localhost:8000/api/v1"

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())  # {"status": "healthy"}

# Create session
response = requests.post(f"{BASE_URL}/sessions")
session_id = response.json()["session_id"]
print(f"Created session: {session_id}")
```

---

## Authentication

**Current version**: No authentication (local development)

**Future versions**: Support for:

- API keys (header-based)

- OAuth2 (for web deployment)

- JWT tokens

**For production**: Use reverse proxy (nginx) with authentication layer

---

## Common Patterns

### Error Handling

**HTTP Status Codes**:

| Code | Meaning | When Used |
|------|---------|-----------|
| 200 | OK | Successful GET/PATCH request |
| 201 | Created | Successful POST creating resource |
| 204 | No Content | Successful DELETE |
| 400 | Bad Request | Invalid input data |
| 404 | Not Found | Session or resource doesn't exist |
| 422 | Unprocessable Entity | Validation error (Pydantic) |
| 500 | Internal Server Error | Server-side error |

**Error Response Format**:
```json
{
  "detail": "Session abc123 not found"
}
```

**Validation Error** (422):
```json
{
  "detail": [
    {
      "loc": ["body", "temperature"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### Pagination

**Large result sets** (experiments, sessions):

**Request**:
```http
GET /sessions/{session_id}/experiments?skip=0&limit=50
```

**Parameters**:

- `skip`: Number of items to skip (default: 0)

- `limit`: Maximum items to return (default: 100, max: 1000)

**Response**:
```json
{
  "items": [...],
  "total": 500,
  "skip": 0,
  "limit": 50
}
```

### Filtering and Sorting

**Experiments endpoint**:
```http
GET /sessions/{session_id}/experiments?sort_by=yield&order=desc
```

**Parameters**:

- `sort_by`: Column name to sort by

- `order`: `asc` or `desc`

- `filter`: JSON filter expression (advanced)

---

## Sessions API

### Create Session

**Endpoint**: `POST /sessions`

**Purpose**: Create a new optimization session

**Request Body** (optional):
```json
{
  "metadata": {
    "name": "Catalyst Screening",
    "description": "Pd catalyst optimization",
    "tags": ["catalysis", "suzuki"]
  }
}
```

**Response** (201 Created):
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2025-12-12T10:00:00Z",
  "expires_at": "2025-12-13T10:00:00Z"
}
```

**Example**:
```python
response = requests.post(f"{BASE_URL}/sessions")
session_id = response.json()["session_id"]
```

### Get Session Details

**Endpoint**: `GET /sessions/{session_id}`

**Purpose**: Get complete session information

**Response** (200 OK):
```json
{
  "session_id": "550e8400-...",
  "metadata": {...},
  "search_space": {
    "n_variables": 3,
    "variables": [...]
  },
  "experiments": {
    "count": 25,
    "target_column": "yield"
  },
  "model": {
    "trained": true,
    "backend": "botorch",
    "kernel": "Matern"
  }
}
```

### Save Session

**Endpoint**: `POST /sessions/{session_id}/save`

**Purpose**: Persist session to server-side storage

**Response** (200 OK):
```json
{
  "message": "Session persisted to server storage"
}
```

### Export Session

**Endpoint**: `GET /sessions/{session_id}/export`

**Purpose**: Download session as JSON file

**Response**: JSON file with `Content-Disposition: attachment`

**Example**:
```python
response = requests.get(f"{BASE_URL}/sessions/{session_id}/export")
with open('my_session.json', 'wb') as f:
    f.write(response.content)
```

### Import Session

**Endpoint**: `POST /sessions/import`

**Purpose**: Upload and create session from JSON file

**Request**: Multipart form data with file

**Response** (201 Created):
```json
{
  "session_id": "new-session-id",
  "created_at": "2025-12-12T10:00:00Z",
  "expires_at": "2025-12-13T10:00:00Z"
}
```

### Delete Session

**Endpoint**: `DELETE /sessions/{session_id}`

**Purpose**: Remove session from memory

**Response** (204 No Content)

---

## Variables API

### Add Variable

**Endpoint**: `POST /sessions/{session_id}/variables`

**Purpose**: Add variable to search space

**Request Body** (continuous):
```json
{
  "name": "temperature",
  "type": "real",
  "bounds": [20.0, 100.0],
  "unit": "°C"
}
```

**Request Body** (categorical):
```json
{
  "name": "solvent",
  "type": "categorical",
  "categories": ["THF", "DMF", "toluene"]
}
```

**Request Body** (discrete — restricted numeric values):
```json
{
  "name": "SAR",
  "type": "discrete",
  "allowed_values": [80, 280, 450],
  "unit": "-",
  "description": "Silicon-to-alumina ratio"
}
```

**Response** (201 Created):
```json
{
  "name": "temperature",
  "type": "real",
  "bounds": [20.0, 100.0],
  "unit": "°C"
}
```

**Example**:
```python
# Add continuous variable
requests.post(
    f"{BASE_URL}/sessions/{session_id}/variables",
    json={
        "name": "temperature",
        "type": "real",
        "bounds": [20, 100],
        "unit": "°C"
    }
)

# Add categorical variable
requests.post(
    f"{BASE_URL}/sessions/{session_id}/variables",
    json={
        "name": "solvent",
        "type": "categorical",
        "categories": ["THF", "DMF", "toluene"]
    }
)
```

### Get All Variables

**Endpoint**: `GET /sessions/{session_id}/variables`

**Response** (200 OK):
```json
{
  "variables": [
    {
      "name": "temperature",
      "type": "real",
      "bounds": [20.0, 100.0],
      "unit": "°C"
    },
    {
      "name": "solvent",
      "type": "categorical",
      "categories": ["THF", "DMF", "toluene"]
    }
  ],
  "count": 2
}
```

### Delete Variable

**Endpoint**: `DELETE /sessions/{session_id}/variables/{variable_name}`

**Response** (204 No Content)

---

## Experiments API

### Add Experiment

**Endpoint**: `POST /sessions/{session_id}/experiments`

**Purpose**: Add single experiment with results

**Request Body**:
```json
{
  "inputs": {
    "temperature": 60.0,
    "catalyst_loading": 2.5,
    "solvent": "THF"
  },
  "output": 85.3,
  "noise": 1.2
}
```

**Response** (201 Created):
```json
{
  "experiment_id": 1,
  "inputs": {...},
  "output": 85.3,
  "added_at": "2025-12-12T10:15:00"
}
```

### Add Multiple Experiments

**Endpoint**: `POST /sessions/{session_id}/experiments/batch`

**Purpose**: Add multiple experiments at once

**Request Body**:
```json
{
  "experiments": [
    {
      "inputs": {"temperature": 60, "solvent": "THF"},
      "output": 85.3
    },
    {
      "inputs": {"temperature": 80, "solvent": "DMF"},
      "output": 72.1
    }
  ]
}
```

**Response** (201 Created):
```json
{
  "added": 2,
  "total_experiments": 27
}
```

### Get All Experiments

**Endpoint**: `GET /sessions/{session_id}/experiments`

**Query Parameters**:

- `skip`: Pagination offset (default: 0)

- `limit`: Max results (default: 100)

**Response** (200 OK):
```json
{
  "experiments": [
    {
      "temperature": 60.0,
      "catalyst_loading": 2.5,
      "solvent": "THF",
      "yield": 85.3
    }
  ],
  "total": 25,
  "target_column": "yield"
}
```

### Upload CSV

**Endpoint**: `POST /sessions/{session_id}/experiments/upload`

**Purpose**: Upload experiments from CSV file

**Request**: Multipart form data

**cURL Example**:
```bash
# Single-objective
curl -X POST \
  "http://localhost:8000/api/v1/sessions/{session_id}/experiments/upload?target_columns=yield" \
  -F "file=@experiments.csv"

# Multi-objective (comma-separated)
curl -X POST \
  "http://localhost:8000/api/v1/sessions/{session_id}/experiments/upload?target_columns=yield,selectivity" \
  -F "file=@experiments.csv"
```

**Python Example**:
```python
with open('experiments.csv', 'rb') as f:
    response = requests.post(
        f"{BASE_URL}/sessions/{session_id}/experiments/upload",
        files={'file': f},
        params={'target_columns': 'yield'}  # or 'yield,selectivity' for multi-objective
    )
```

**Response** (200 OK):
```json
{
  "added": 50,
  "total_experiments": 50,
  "message": "Successfully uploaded 50 experiments"
}
```

### Export Experiments

**Endpoint**: `GET /sessions/{session_id}/experiments/export`

**Purpose**: Download experiments as CSV

**Response**: CSV file with `Content-Disposition: attachment`

**Example**:
```python
response = requests.get(
    f"{BASE_URL}/sessions/{session_id}/experiments/export"
)
with open('experiments.csv', 'wb') as f:
    f.write(response.content)
```

### Delete All Experiments

**Endpoint**: `DELETE /sessions/{session_id}/experiments`

**Response** (204 No Content)

---

## Design of Experiments (DoE) API

### Generate Initial Design

**Endpoint**: `POST /sessions/{session_id}/initial-design`

**Purpose**: Generate initial experimental design using space-filling, classical RSM, or screening methods.

**Space-filling methods** (require `n_points`):
```json
{
  "method": "lhs",
  "n_points": 20,
  "random_seed": 42,
  "lhs_criterion": "maximin"
}
```

**Classical methods** (run count from design structure; `n_points` ignored):
```json
{
  "method": "ccd",
  "ccd_alpha": "orthogonal",
  "ccd_face": "circumscribed",
  "n_center": 1
}
```

Available methods: `random`, `lhs`, `sobol`, `halton`, `hammersly`, `full_factorial`, `fractional_factorial`, `ccd`, `box_behnken`, `plackett_burman`, `gsd`

**Response** (200 OK):
```json
{
  "points": [
    {"Temperature": 225.4, "Pressure": 3.7, "Catalyst": "Ni"},
    {"Temperature": 310.2, "Pressure": 8.1, "Catalyst": "Pt"}
  ],
  "method": "lhs",
  "n_points": 20,
  "design_info": {
    "run_count": 20,
    "method": "lhs"
  }
}
```

### Preview Optimal Design

**Endpoint**: `POST /sessions/{session_id}/optimal-design/info`

**Purpose**: Dry-run inspection — see model terms and recommended run count without generating the design.

**Request Body**:
```json
{
  "model_type": "quadratic"
}
```

Or with explicit effects:
```json
{
  "effects": ["Temperature", "Pressure", "Temperature*Pressure", "Temperature**2"]
}
```

**Response** (200 OK):
```json
{
  "p_columns": 6,
  "model_terms": ["Intercept", "Temperature", "Pressure", "Temperature*Pressure", "Temperature**2", "Pressure**2"],
  "n_points_minimum": 6,
  "n_points_recommended": 12
}
```

### Generate Optimal Design

**Endpoint**: `POST /sessions/{session_id}/optimal-design`

**Purpose**: Generate a statistically optimal design (D/A/I-optimal) for a user-specified model.

**Request Body**:
```json
{
  "model_type": "quadratic",
  "p_multiplier": 2.0,
  "criterion": "D",
  "algorithm": "fedorov",
  "random_seed": 42
}
```

With explicit effects and fixed run count:
```json
{
  "effects": ["Temperature", "Pressure", "Temperature*Pressure"],
  "n_points": 15,
  "criterion": "D",
  "algorithm": "detmax"
}
```

| Parameter | Options | Default |
|---|---|---|
| `model_type` | `"linear"`, `"interaction"`, `"quadratic"` | — |
| `effects` | list of effect strings | — |
| `n_points` | absolute integer | — |
| `p_multiplier` | float ≥ 1.0 | — |
| `criterion` | `"D"`, `"A"`, `"I"` | `"D"` |
| `algorithm` | `"sequential"`, `"simple_exchange"`, `"fedorov"`, `"modified_fedorov"`, `"detmax"` | `"fedorov"` |
| `n_levels` | int 2–20 | `5` |
| `max_iter` | int 10–10000 | `200` |
| `random_seed` | int | — |

**Response** (200 OK):
```json
{
  "points": [...],
  "n_points": 12,
  "design_info": {
    "D_eff": 94.2,
    "A_eff": 91.7,
    "model_terms": ["Intercept", "Temperature", "Pressure", ...],
    "p_columns": 6
  }
}
```

---

## LLM API

### Suggest Effects (Streaming)

**Endpoint**: `POST /llm/suggest-effects/{session_id}`

**Purpose**: Stream AI-suggested model terms for an optimal design. Requires `pip install 'alchemist-nrel[llm]'`.

**Response type**: `text/event-stream` (Server-Sent Events)

**Request Body**:
```json
{
  "structuring_provider": {
    "provider": "openai",
    "model": "gpt-4o",
    "api_key": "sk-..."
  },
  "system_context": "Fischer-Tropsch synthesis over supported Co catalysts. Maximizing C5+ selectivity.",
  "edison_config": {
    "job_type": "literature",
    "api_key": "edi-..."
  }
}
```

Without Edison (Ollama example):
```json
{
  "structuring_provider": {
    "provider": "ollama",
    "model": "llama3.2",
    "base_url": "http://localhost:11434"
  },
  "system_context": "Pd-catalyzed Suzuki coupling. Maximizing yield."
}
```

**SSE event stream** (one JSON object per `data:` line):
```
data: {"status": "starting", "message": "Submitting Edison search..."}
data: {"status": "edison_searching", "trajectory_url": "https://..."}
data: {"status": "structuring", "message": "LLM structuring call in progress..."}
data: {"status": "complete", "result": {"effects": ["Temperature", "Temperature*Pressure", ...], "reasoning": [...], "confidence": [...], "sources": [...]}}
```

### Get LLM Config

**Endpoint**: `GET /llm/config`

**Response**:
```json
{
  "openai": {"api_key": "sk-..."},
  "ollama": {"base_url": "http://localhost:11434"},
  "edison": {"api_key": "edi-..."}
}
```

### Save LLM Config

**Endpoint**: `POST /llm/config`

**Request Body** (any subset of providers):
```json
{
  "openai": {"api_key": "sk-..."},
  "ollama": {"base_url": "http://localhost:11434"}
}
```

### List Ollama Models

**Endpoint**: `GET /llm/ollama/models?base_url=http://localhost:11434`

**Response**:
```json
{
  "models": ["llama3.2", "mistral", "phi3"],
  "error": null
}
```

---

## Staged Experiments API

Staged experiments provide a workflow queue for autonomous optimization. Use these endpoints to track which experiments are pending execution.

### Stage Experiment

**Endpoint**: `POST /sessions/{session_id}/experiments/staged`

**Purpose**: Queue an experiment for later execution

**Request Body**:
```json
{
  "inputs": {
    "temperature": 375.2,
    "catalyst_loading": 3.1,
    "solvent": "DMF"
  },
  "reason": "qEI"
}
```

**Response** (200 OK):
```json
{
  "message": "Experiment staged successfully",
  "n_staged": 1,
  "staged_inputs": {"temperature": 375.2, "catalyst_loading": 3.1, "solvent": "DMF"}
}
```

### Stage Multiple Experiments

**Endpoint**: `POST /sessions/{session_id}/experiments/staged/batch`

**Purpose**: Queue multiple experiments at once (e.g., from batch acquisition)

**Request Body**:
```json
{
  "experiments": [
    {"temperature": 375.2, "catalyst_loading": 3.1, "solvent": "DMF"},
    {"temperature": 412.5, "catalyst_loading": 1.8, "solvent": "THF"}
  ],
  "reason": "qEI batch"
}
```

**Response** (200 OK):
```json
{
  "experiments": [
    {"temperature": 375.2, "catalyst_loading": 3.1, "solvent": "DMF"},
    {"temperature": 412.5, "catalyst_loading": 1.8, "solvent": "THF"}
  ],
  "n_staged": 2,
  "reason": "qEI batch"
}
```

### Get Staged Experiments

**Endpoint**: `GET /sessions/{session_id}/experiments/staged`

**Purpose**: Retrieve all experiments awaiting execution

**Response** (200 OK):
```json
{
  "experiments": [
    {"temperature": 375.2, "catalyst_loading": 3.1, "solvent": "DMF"}
  ],
  "n_staged": 1,
  "reason": "qEI"
}
```

### Clear Staged Experiments

**Endpoint**: `DELETE /sessions/{session_id}/experiments/staged`

**Purpose**: Remove all staged experiments

**Response** (200 OK):
```json
{
  "message": "Staged experiments cleared",
  "n_cleared": 2
}
```

### Complete Staged Experiments

**Endpoint**: `POST /sessions/{session_id}/experiments/staged/complete`

**Purpose**: Finalize staged experiments by providing output values

**Query Parameters**:

- `auto_train`: Auto-retrain model (default: false)
- `training_backend`: "sklearn" or "botorch"
- `training_kernel`: Kernel type

**Request Body**:
```json
{
  "outputs": [85.3, 78.9],
  "noises": [1.2, 0.8],
  "iteration": 5,
  "reason": "qEI"
}
```

**Response** (200 OK):
```json
{
  "message": "Staged experiments completed and added to dataset",
  "n_added": 2,
  "n_experiments": 27,
  "model_trained": true,
  "training_metrics": {
    "rmse": 2.3,
    "r2": 0.94,
    "backend": "botorch"
  }
}
```

### Autonomous Workflow Example

```python
import requests

BASE_URL = "http://localhost:8000/api/v1"
session_id = "your-session-id"

# 1. Get suggestion from acquisition
response = requests.post(
    f"{BASE_URL}/sessions/{session_id}/acquisition/suggest",
    json={"strategy": "qEI", "goal": "maximize", "n_suggestions": 2}
)
suggestions = response.json()["suggestions"]

# 2. Stage suggestions
for suggestion in suggestions:
    requests.post(
        f"{BASE_URL}/sessions/{session_id}/experiments/staged",
        json={"inputs": suggestion, "reason": "qEI"}
    )

# 3. Get staged experiments (e.g., for hardware controller)
response = requests.get(f"{BASE_URL}/sessions/{session_id}/experiments/staged")
pending = response.json()["experiments"]

# 4. Execute experiments and collect outputs
outputs = [run_experiment(**exp) for exp in pending]

# 5. Complete staged experiments with outputs
response = requests.post(
    f"{BASE_URL}/sessions/{session_id}/experiments/staged/complete",
    json={"outputs": outputs},
    params={"auto_train": True}
)
```

---

## Models API

### Train Model

**Endpoint**: `POST /sessions/{session_id}/model/train`

**Purpose**: Train Gaussian Process model

**Request Body**:
```json
{
  "backend": "botorch",
  "kernel": "Matern",
  "kernel_params": {
    "nu": 2.5
  },
  "target_column": "yield",
  "goal": "maximize"
}
```

**Response** (200 OK):
```json
{
  "success": true,
  "backend": "botorch",
  "kernel": "Matern",
  "hyperparameters": {
    "lengthscales": [0.15, 0.22, 0.18],
    "outputscale": 1.45,
    "noise": 0.05
  },
  "metrics": {
    "cv_rmse": 3.2,
    "cv_r2": 0.93,
    "cv_mae": 2.1,
    "mean_z": 0.02,
    "std_z": 1.05
  },
  "training_time_s": 2.34
}
```

**Example**:
```python
response = requests.post(
    f"{BASE_URL}/sessions/{session_id}/models/train",
    json={
        "backend": "botorch",
        "kernel": "Matern",
        "target_column": "yield",
        "goal": "maximize"
    }
)
metrics = response.json()["metrics"]
print(f"R² = {metrics['cv_r2']:.3f}")
```

### Get Model Info

**Endpoint**: `GET /sessions/{session_id}/model`

**Response** (200 OK):
```json
{
  "trained": true,
  "backend": "botorch",
  "kernel": "Matern",
  "hyperparameters": {...},
  "metrics": {...},
  "trained_at": "2025-12-12T10:30:00"
}
```

### Get Predictions

**Endpoint**: `POST /sessions/{session_id}/model/predict`

**Purpose**: Make predictions at new points

**Request Body**:
```json
{
  "inputs": [
    {"temperature": 65, "catalyst_loading": 3.0, "solvent": "THF"},
    {"temperature": 75, "catalyst_loading": 2.0, "solvent": "DMF"}
  ]
}
```

**Response** (200 OK):
```json
{
  "predictions": [
    {
      "inputs": {"temperature": 65, ...},
      "mean": 87.5,
      "std": 2.3,
      "confidence_interval_95": [82.9, 92.1]
    },
    {
      "inputs": {"temperature": 75, ...},
      "mean": 79.2,
      "std": 3.1,
      "confidence_interval_95": [73.1, 85.3]
    }
  ]
}
```

---

## Acquisition API

### Suggest Next Experiments

**Endpoint**: `POST /sessions/{session_id}/acquisition/suggest`

**Purpose**: Generate next experiment candidates

**Request Body**:
```json
{
  "strategy": "EI",
  "n_suggestions": 5,
  "goal": "maximize",
  "xi": 0.01
}
```

**Available Strategies**:

- `EI` - Expected Improvement

- `PI` - Probability of Improvement

- `UCB` - Upper Confidence Bound

- `qEI` - Batch Expected Improvement

- `qUCB` - Batch Upper Confidence Bound

- `qNIPV` - Negative Integrated Posterior Variance (exploration)

**Response** (200 OK):
```json
{
  "suggestions": [
    {
      "temperature": 75.3,
      "catalyst_loading": 2.8,
      "solvent": "THF"
    }
  ],
  "n_suggestions": 1
}
```

**Example**:
```python
response = requests.post(
    f"{BASE_URL}/sessions/{session_id}/acquisition/suggest",
    json={
        "strategy": "EI",
        "n_suggestions": 3,
        "goal": "maximize"
    }
)
suggestions = response.json()["suggestions"]
for i, cand in enumerate(candidates, 1):
    print(f"Candidate {i}: {cand}")
```

---

## Visualizations API

### Get Parity Plot Data

**Endpoint**: `GET /sessions/{session_id}/visualizations/parity`

**Response** (200 OK):
```json
{
  "actual": [85.3, 72.1, 90.2, ...],
  "predicted": [86.1, 71.5, 89.8, ...],
  "std": [2.3, 3.1, 2.0, ...],
  "metrics": {
    "rmse": 3.2,
    "mae": 2.1,
    "r2": 0.93
  }
}
```

### Get Q-Q Plot Data

**Endpoint**: `GET /sessions/{session_id}/visualizations/qq_plot`

**Response** (200 OK):
```json
{
  "theoretical_quantiles": [-2.5, -2.0, ...],
  "sample_quantiles": [-2.4, -1.9, ...],
  "mean_z": 0.02,
  "std_z": 1.05,
  "calibration_status": "well_calibrated"
}
```

### Get Calibration Curve Data

**Endpoint**: `GET /sessions/{session_id}/visualizations/calibration`

**Response** (200 OK):
```json
{
  "confidence_levels": [0.68, 0.90, 0.95, 0.99],
  "observed_coverage": [0.70, 0.89, 0.94, 0.98],
  "confidence_bands_lower": [0.62, 0.85, 0.90, 0.96],
  "confidence_bands_upper": [0.78, 0.94, 0.98, 0.99]
}
```

---

## WebSocket API

### Real-Time Updates

**Endpoint**: `WS /ws/{session_id}`

**Purpose**: Subscribe to session events

**Events**:

- `experiment_added`: New experimental data

- `model_trained`: Model training completed

- `acquisition_generated`: New candidates available

**Example** (Python with websockets):
```python
import websockets
import asyncio

async def listen_to_session(session_id):
    uri = f"ws://localhost:8000/ws/{session_id}"
    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            event = json.loads(message)
            print(f"Event: {event['type']}")
            print(f"Data: {event['data']}")

asyncio.run(listen_to_session(session_id))
```

---

## Rate Limiting

**Current**: No rate limiting (local development)

**Future**: Configurable rate limits for production:

- 100 requests/minute per IP (general)

- 10 model training requests/hour per session

- 1000 requests/hour for read-only endpoints

---

## CORS Configuration

**Default** (development):
```python
allow_origins=["http://localhost:5173", "http://localhost:3000"]
```

**Production**: Configure allowed origins in environment:
```bash
ALLOWED_ORIGINS=https://myapp.com,https://app.mycompany.com
```

---

## Best Practices

### Session Lifecycle

**Recommended pattern**:
```python
# 1. Create session
session = requests.post(f"{BASE_URL}/sessions").json()
session_id = session["session_id"]

# 2. Define variables
for var in variables:
    requests.post(f"{BASE_URL}/sessions/{session_id}/variables", json=var)

# 3. Add experiments
requests.post(f"{BASE_URL}/sessions/{session_id}/experiments/batch", json=experiments)

# 4. Train model
requests.post(f"{BASE_URL}/sessions/{session_id}/model/train", json=model_config)

# 5. Get suggestions
suggestions = requests.post(
    f"{BASE_URL}/sessions/{session_id}/acquisition/suggest",
    json={"strategy": "EI", "n_suggestions": 5}
).json()

# 6. Save session
requests.post(f"{BASE_URL}/sessions/{session_id}/save")
```

### Error Handling

**Always check status codes**:
```python
response = requests.post(url, json=data)
if response.status_code == 201:
    result = response.json()
elif response.status_code == 422:
    errors = response.json()["detail"]
    print(f"Validation errors: {errors}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

### Performance Optimization

**Batch operations**:

- Use `/experiments/batch` instead of multiple single adds

- Upload CSV for large datasets

- Minimize round-trips

**Caching**:

- Cache session_id after creation

- Reuse trained models

- Store experiment data locally

---

## Migration from Core Session API

**Convert Session API code to REST API**:

**Before** (Core Session API):
```python
from alchemist_core import OptimizationSession

session = OptimizationSession()
session.add_variable('temp', 'real', bounds=(20, 100))
session.add_experiment({'temp': 60}, output=85.3)
session.train_model(backend='botorch')
next_point = session.suggest_next(strategy='EI')
```

**After** (REST API):
```python
import requests

BASE_URL = "http://localhost:8000"

# Create session
r = requests.post(f"{BASE_URL}/sessions")
session_id = r.json()["session_id"]

# Add variable
requests.post(
    f"{BASE_URL}/sessions/{session_id}/variables",
    json={"name": "temp", "type": "real", "bounds": [20, 100]}
)

# Add experiment
requests.post(
    f"{BASE_URL}/sessions/{session_id}/experiments",
    json={"inputs": {"temp": 60}, "output": 85.3}
)

# Train model
requests.post(
    f"{BASE_URL}/sessions/{session_id}/model/train",
    json={"backend": "botorch"}
)

# Get suggestions
r = requests.post(
    f"{BASE_URL}/sessions/{session_id}/acquisition/suggest",
    json={"strategy": "EI", "n_suggestions": 1}
)
next_point = r.json()["suggestions"][0]
```

---

## Further Reading

- [Core Session API](session.md) - Python interface (no HTTP)
- [Web Application](../setup/web_app.md) - Browser interface using this API
- [Session Management](../reproducibility/sessions.md) - Session lifecycle and storage
- [Audit Logs](../reproducibility/audit_logs.md) - Reproducibility tracking

---

**Key Takeaway**: The REST API provides language-agnostic access to ALchemist's Bayesian optimization capabilities. Use it for web applications, remote integrations, or non-Python clients.
