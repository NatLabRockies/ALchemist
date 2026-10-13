# When to Use ALchemist

ALchemist is designed for researchers and engineers who need to optimize complex systems with expensive experiments, where reducing experimental overhead is critical.

---

## What ALchemist Can Be Used For

**ALchemist is ideal when you're optimizing (or modeling) a system with many variables**, where each experiment is time-consuming or expensive. It's particularly well-suited for:

- **Multi-variable optimization**: 2-10+ input variables with complex interactions

- **Expensive experiments**: Each test costs significant time, money, or resources

- **Black-box systems**: You can measure outputs but don't have explicit mathematical models

- **Uncertainty quantification**: You need confidence estimates and risk management

**Key Strengths:**

- **Tracks your iterations**: Complete audit logs and session management for reproducibility

- **Detailed statistical insights**: Uncertainty quantification, calibration diagnostics, and model performance metrics

- **No advanced coding required**: User-friendly desktop and web interfaces (though programmatic access is available)

- **Free and open source**: Unlike commercial tools like JMP, ALchemist is freely available with full source code

---

## Example Application Domains

| Domain | Example Variables | Example Objectives |
|--------|------------------|-------------------|
| **Chemical Synthesis** | Temperature, pressure, catalyst loading, reagent ratios, reaction time | Maximize yield, minimize byproducts, optimize selectivity |
| **Energy Materials & Batteries** | Composition (cathode/anode/electrolyte), processing conditions, electrode thickness, charging protocols | Maximize energy density, improve cycle life, reduce charging time |
| **Materials Discovery** | Elemental composition, heat treatment schedule, sintering parameters, microstructure | Optimize mechanical properties, maximize conductivity, minimize defects |
| **Process Manufacturing** | Operating conditions (temp/flow/pressure), equipment settings, material ratios, timing | Maximize throughput, minimize energy use, improve quality, reduce waste |
| **Biological Systems** | Growth conditions (temp/light/humidity), nutrient concentrations, treatment protocols | Maximize yield, optimize metabolite production, improve growth rates |
| **Engineering Design** | Design parameters, control tuning (PID gains), calibration parameters, operating set points | Minimize settling time, reduce overshoot, improve efficiency, match experimental data |

---

## Workflow Modes

### 1. Interactive Optimization

**Best for:** Exploratory research, learning Bayesian optimization, hands-on control

**How it works:**

- Use desktop GUI or web UI to manually review suggestions

- Visualize progress in real-time

- Decide when to run experiments

**Example workflow:**
```
1. Define variable space (e.g., 3 variables: temp, pressure, catalyst)
2. Generate initial experiments (Latin Hypercube sampling)
3. Run experiments in lab
4. Load results into ALchemist
5. Train GP model, view diagnostics
6. Run acquisition function (Expected Improvement)
7. Review suggested next experiment
8. Repeat steps 3-7 until satisfied
```

---

### 2. Programmatic Workflows

**Best for:** Batch processing, integration with analysis pipelines, reproducible research

**Example code:**
```python
from alchemist_core.session import OptimizationSession

# Create session
session = OptimizationSession()

# Define variables
session.add_variable("temperature", "real", min=200, max=400)
session.add_variable("pressure", "real", min=1, max=10)

# Load experimental data
session.load_experiments("data.csv")

# Train model
session.train_model(backend="botorch", kernel="matern")

# Get next suggestion
next_point = session.suggest_next(strategy="qEI", goal="maximize")

# Save session
session.save_session("my_optimization.json")
```

---

### 3. Autonomous Optimization

**Best for:** Real-time process control, automated laboratory systems, self-driving experiments

**Example autonomous loop:**
```python
import requests

BASE_URL = "http://localhost:8000"
SESSION_ID = "abc-123"

while not converged:
    # Get next experiment from ALchemist
    response = requests.post(
        f"{BASE_URL}/sessions/{SESSION_ID}/acquisition/suggest",
        json={"strategy": "qEI", "goal": "maximize"}
    )
    next_point = response.json()["suggestions"][0]
    
    # Execute experiment on hardware
    result = run_experiment_on_reactor(next_point)
    
    # Send result back to ALchemist
    requests.post(
        f"{BASE_URL}/sessions/{SESSION_ID}/experiments",
        params={"auto_train": True},
        json={"inputs": next_point, "output": result}
    )
```

---

## Interface Options

ALchemist offers multiple interfaces for different workflows:

- **Desktop GUI**: Offline work, local files, ideal for learning (no coding required)
- **Web UI**: Collaboration, remote access, browser-based (no coding required)
- **Python Session API**: Batch processing, custom workflows, notebooks (Python scripting)
- **REST API**: Real-time control, automation, integration with other systems (API integration)

For a detailed comparison, see the [Python API Overview](api/python_overview.md).

---

## Getting Started

**First-Time Users:**

1. Start with desktop or web GUI for learning
2. Try a simple optimization (2-3 variables, 20-50 experiments)
3. Explore visualizations to understand model behavior

**Experienced Users:**

1. Use Session API for scripted workflows
2. Leverage session save/load for reproducibility
3. Consider REST API for automated systems

---

## Next Steps

- [Bayesian Optimization Intro](background/bayesian_optimization.md) - Learn the theory

- [Web UI Guide](setup/web_app.md) - Browser-based workflows

- [Session API Guide](api/session.md) - Programmatic workflows

- [Variable Space Setup](setup/variable_space.md) - Define your search space

- [Model Training](modeling/botorch.md) - Train Gaussian Process models

---

**Questions?** [Open an issue on GitHub](https://github.com/NatLabRockies/ALchemist/issues) or contact ccoatney@nrel.gov
