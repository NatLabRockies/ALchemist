# Python API Overview

The **ALchemist Python API** (`alchemist_core`) is the core library for Bayesian optimization in Python. Import it directly in scripts, Jupyter notebooks, or Python applications for offline, programmatic optimization workflows.

---

## Python API vs REST API vs Web/Desktop App

| Feature | Python API | REST API | Web/Desktop App |
|---------|-----------|----------|-----------------|
| **Interface** | Python import | HTTP requests | Browser/GUI |
| **Use Case** | Scripts, notebooks | Remote access, web apps | No-code interface |
| **Requires Server** | No | Yes | Yes (web) / No (desktop) |
| **Language** | Python only | Any (HTTP) | N/A |
| **Offline** | Yes | No | Desktop only |
| **Best For** | Automation, analysis | Integration, remote | Interactive exploration |

---

## Core Classes

The Python API is organized around these main classes:

**Orchestration**:
- **[OptimizationSession](session_class.md)** - Main class coordinating all optimization workflows

**Data & Space**:
- **[SearchSpace](search_space.md)** - Variable space definition and management
- **[ExperimentManager](session_class.md#alchemist_core.data.experiment_manager.ExperimentManager)** - Experimental data storage

**Modeling**:
- **[BoTorchModel](models.md#alchemist_core.models.botorch_model.BoTorchModel)** - PyTorch-based Gaussian Process models
- **[SklearnModel](models.md#alchemist_core.models.sklearn_model.SklearnModel)** - Scikit-learn Gaussian Process models

**Acquisition**:
- **[BoTorchAcquisition](acquisition.md#alchemist_core.acquisition.botorch_acquisition.BoTorchAcquisition)** - BoTorch acquisition functions
- **[SkoptAcquisition](acquisition.md#alchemist_core.acquisition.skopt_acquisition.SkoptAcquisition)** - Scikit-optimize acquisition functions

---

## Quick Example

```python
from alchemist_core import OptimizationSession

session = OptimizationSession()
session.add_variable('temperature', 'real', bounds=(20, 100))
session.add_variable('pressure', 'real', bounds=(1, 10))

# Generate design, add data, train, optimize
points = session.generate_initial_design('lhs', n_points=15)
# ... run experiments and add results ...
session.train_model(backend='botorch', kernel='Matern')
candidates = session.suggest_next(strategy='EI', n_suggestions=5, goal='maximize')
```

For complete workflows and detailed examples, see the [OptimizationSession API Reference](session_class.md).

---

## Documentation Structure

- **[OptimizationSession](session_class.md)** - Auto-generated API reference from docstrings
- **[SearchSpace](search_space.md)** - Variable management reference
- **[Models](models.md)** - Gaussian Process model reference
- **[Acquisition](acquisition.md)** - Acquisition function reference

---

## Key Differences from REST API

| Aspect | Python API | REST API |
|--------|-----------|----------|
| **Import** | `from alchemist_core import OptimizationSession` | Not applicable |
| **Session** | Python object in memory | Server-side, accessed by ID |
| **Method Call** | `session.add_variable(...)` | `POST /sessions/{id}/variables` |
| **Return Type** | Python objects (DataFrame, dict) | JSON over HTTP |
| **Concurrency** | Single process | Multi-client via HTTP |
